# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""HF Token Generation Pipeline"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    runtime_checkable,
)

from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.profiler import Tracer, traced

from .config import PipelineConfig
from .context import InputContext
from .interfaces import TokenGenerator
from .kv_cache import KVCacheManager, KVCacheStrategy
from .response import LogProbabilities, TextResponse
from .sampling import token_sampler

ARCH_SAFE_VRAM_USAGE_LIMIT = {
    "DeepseekCoder": 0.96,
    "ExaoneForCausalLM": 0.96,
    "LlamaForCausalLM": 0.96,
    "MistralForCausalLM": 0.96,
}


class ModelInputs:
    """
    Base class for model inputs.
    Use this class to encapsulate inputs for your model.
    You may store any number of dataclass fields

    Example:
        >>> class ReplitInputs(ModelInputs):
        ...     tokens: Tensor
        ...     input_row_offsets: Tensor
        ...
        ...     def __init__(self, tokens: Tensor, input_row_offsets: Tensor):
        ...         self.tokens = tokens
        ...         self.input_row_offsets = input_row_offsets
        ...
        >>> # Create tensors
        >>> tokens = Tensor.zeros((1, 2, 3), DType.int64)
        >>> input_row_offsets = Tensor.zeros((1, 1, 1), DType.int64)
        >>> # Initialize inputs
        >>> inputs = ReplitInputs(tokens=tokens, input_row_offsets=input_row_offsets)
        >>> # Access tensors
        >>> list(inputs) == [tokens, input_row_offsets]
        True
    """


@dataclass(frozen=True)
class ModelOutputs:
    next_token_logits: Tensor | None = None
    """Logits for just the next token."""

    logits: Tensor | None = None
    """Logits for the entire token sequence."""


T = TypeVar("T", bound=InputContext)


class PipelineModel(ABC):
    """A pipeline model with setup, input preparation and execution methods."""

    def __init__(
        self, pipeline_config: PipelineConfig, session: InferenceSession
    ) -> None:
        self.pipeline_config = pipeline_config
        self.available_cache_memory = self.estimate_memory_footprint()

        if isinstance(self, KVCacheMixin):
            self.kv_manager = self.load_kv_manager(
                session, self.available_cache_memory
            )

    def estimate_memory_footprint(self) -> int:
        """Calculates the estimated memory consumption of our engine and
        returns the estimated available space to store the KVCache."""

        def to_mib(bytes):
            return round(bytes / 1024 / 1024)

        total_size = 0
        weights_size = self.pipeline_config.weights_size()
        if weights_size is not None:
            total_size += weights_size

        free_memory = None
        try:
            # TODO(AIPIPE-200): change this back to free_memory.
            # Currently, free_memory leads to issues with calculation on
            # repeated model loads (like seen in tests). Based on the outputs,
            # it seems like we are not freeing memory from a model until the
            # next model is loading weights. As such, there is not enough free
            # memory when running this check. Memory likely should be getting
            # cleaned up much earlier.
            free_memory = self.pipeline_config.device.stats["total_memory"]
        except:
            pass

        model_arch = self.pipeline_config.architecture
        current_batch_size = self.pipeline_config.max_cache_batch_size
        current_seq_len = self.pipeline_config.max_length
        vram_usage_limit_scale = ARCH_SAFE_VRAM_USAGE_LIMIT.get(
            model_arch or "", 0.75
        )

        available_cache_memory = (
            (free_memory - weights_size)
            * self.pipeline_config.gpu_memory_utilization
            if free_memory
            else 0
        )
        if isinstance(self, KVCacheMixin):
            kv_size = self.estimate_kv_cache_size(available_cache_memory)
        else:
            kv_size = 0

        max_batch_size = (
            int(
                (free_memory * vram_usage_limit_scale - weights_size)
                / (kv_size / current_batch_size)
            )
            if free_memory and kv_size > 0
            else None
        )

        total_size += kv_size

        weights_str = ""
        if weights_size:
            weights_str = (
                f"\n\t    Weights:                {to_mib(weights_size)} MiB"
            )

        if free_memory:
            free_memory_str = f" / {to_mib(free_memory)} MiB free"
        logging_str = (
            "\n"
            f"\n\tEstimated memory consumption:"
            f"{weights_str}"
            f"\n\t    KVCache allocation:     {to_mib(kv_size)} MiB"
            f"\n\t    Total estimated:        {to_mib(total_size)} MiB used{free_memory_str}\n"
            f"\n\tCurrent batch size: {current_batch_size}"
            f"\n\tCurrent max sequence length: {current_seq_len}"
        )
        if (
            self.pipeline_config.cache_strategy != KVCacheStrategy.PAGED
            and max_batch_size is not None
        ):
            # max batch size is less relevant for paged attention, given that we over-subscribe the
            # number of in-flight batches to available cache space.
            logging_str += f"\n\tMax recommended batch size for current sequence length: {max_batch_size}\n"
        logging.info(logging_str)

        if isinstance(free_memory, (int, float)):
            if total_size > free_memory:
                msg = f"Estimated model and kv cache memory use exceeds available memory ({to_mib(total_size)} / {free_memory_str} MiB)"

                if self.pipeline_config.cache_strategy == KVCacheStrategy.PAGED:
                    msg += ". Try reducing --gpu-memory-consumption to a smaller value."
                else:
                    max_batch_size_rec_str = (
                        f" to {max_batch_size} " if max_batch_size else " "
                    )
                    msg += f". Try reducing your --max-cache-batch-size{max_batch_size_rec_str}or reducing the value passed to --max-seq-len."

                raise RuntimeError(msg)
            elif total_size > vram_usage_limit_scale * free_memory:
                logging.warning(
                    "Estimated model and kv cache memory use nears available memory. You may experience errors."
                )

        return available_cache_memory

    @abstractmethod
    def execute(
        self,
        model_inputs: ModelInputs,
        # TODO(zheng): This should be wrapped in a class called KVCacheInputs in the future.
        kv_cache_inputs: Sequence[Tensor] | None = None,
    ) -> ModelOutputs:
        """Executes the graph with the given inputs.

        Args:
            model_inputs: The model inputs to execute, containing tensors and any other
                required data for model execution.
            kv_cache_inputs: The kv cache inputs to execute, containing tensors and any other
                required data for model execution.

        Returns:
            ModelOutputs containing the pipeline's output tensors.

        This is an abstract method that must be implemented by concrete PipelineModels
        to define their specific execution logic.
        """

    @abstractmethod
    def prepare_initial_token_inputs(
        self, context_batch: Sequence[T]
    ) -> ModelInputs:
        """Prepares the initial inputs to be passed to `.execute()`.

        The inputs and functionality of this method can vary per model.
        For example, the model inputs could include:
        - Encoded tensors
        - A unique IDs for each tensor if this model uses a KV Cache manager.

        This function would batch the encoded tensors, claim a slot in the kv
        cache if the ID hasn't been seen before, and return the inputs and
        caches as a list of tensors."""
        ...

    @abstractmethod
    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> ModelInputs:
        """Prepares the secondary inputs to be passed to `.execute()`.

        While `prepare_initial_token_inputs` is responsible for managing the initial inputs.
        This function is responsible for updating the inputs, for each step in a multi-step execution pattern.
        """
        ...

    def compute_log_probabilities(
        self,
        model_inputs: ModelInputs,
        model_outputs: ModelOutputs,
        next_tokens: Tensor,
        batch_top_n: list[int],
        batch_echo: list[bool],
    ) -> list[LogProbabilities | None] | None:
        """Optional method that can be overridden to compute log probabilities.

        Args:
            model_inputs: Inputs to the model returned by
                `prepare_*_token_inputs()`.
            model_outputs: Outputs returned by `execute()`.
            next_tokens: Sampled tokens. Should have shape=[batch size]
            batch_top_n: Number of top log probabilities to return per input in
                the batch. For any element where `top_n == 0`, the
                LogProbabilities is skipped.
            batch_echo: Whether to include input tokens in the returned log
                probabilities.

        Returns:
            List of log probabilities.
        """
        raise NotImplementedError(
            f"Log probabilities not implemented for {type(self)}."
        )


@runtime_checkable
class KVCacheMixin(Protocol):
    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        """Provided a PipelineConfig and InferenceSession, loads the KV manager.

        Args:
            session: Inference session to compile and init the KV cache.
            available_cache_memory: Amount of memory available to the KV cache,
                in bytes.

        Returns:
            Either a single KV cache manager or a tuple of KV cache managers:
            one per input modality.
        """
        ...

    def estimate_kv_cache_size(self, available_cache_memory: int) -> int:
        """Estimates the size of the kv cache in bytes."""
        ...


class TextGenerationPipeline(TokenGenerator[T]):
    """Generalized token generator pipeline."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: Type[PipelineModel],
        # TODO: This should be removed.
        eos_token_id: int,
    ) -> None:
        self._pipeline_config = pipeline_config

        # Expand eos tokens if more are provided in pipeline_config
        if "eos_token_id" in pipeline_config.huggingface_config:
            eos_tokens = pipeline_config.huggingface_config.eos_token_id
            if isinstance(eos_tokens, int):
                if eos_tokens != eos_token_id:
                    msg = f"eos_token_id provided in huggingface config ({eos_tokens}), does not match provided eos_token_id ({eos_token_id}), using provided eos_token_id"
                    logging.warning(msg)

                self._eos_token_id = set([eos_tokens])
            elif isinstance(eos_tokens, list):
                if eos_token_id in eos_tokens:
                    self._eos_token_id = set(eos_tokens)
                else:
                    self._eos_token_id = set([eos_token_id])
            else:
                msg = f"eos_token_id in huggingface_config, is neither int or list: {eos_tokens}"
                logging.warning(msg)
                self._eos_token_id = set([eos_token_id])

        else:
            self._eos_token_id = set([eos_token_id])

        # Initialize Session.
        session = InferenceSession(devices=self._pipeline_config.devices)

        # Load model.
        self._pipeline_model = pipeline_model(
            pipeline_config=self._pipeline_config, session=session
        )

        # Load sampler.
        self._sampler = session.load(
            token_sampler(
                self._pipeline_config.top_k,
                # Logits are at index 0 of model outputs.
                in_dtype=self._pipeline_model.model.output_metadata[0].dtype,  # type: ignore
                # Logits returned from the sampler are always float32 for now.
                out_dtype=DType.float32,
            )
        )

    @traced
    def next_token(
        self,
        batch: dict[str, T],
        num_steps: int = 1,
    ) -> list[dict[str, Any]]:
        """Provided a batch, process batch inputs, execute the graph for num_steps in a multi-step scenario,
        then decode the tokens holistically and return the list of decoded tokens.
        """
        tracer: Tracer = Tracer("compute_parameters")
        # Flatten our batch for consistent indexing.
        context_batch = list(batch.values())

        # Get extra compute parameters for each input.
        batch_top_n = [context.log_probabilities for context in context_batch]
        compute_log_probabilities = any(batch_top_n)
        batch_echo: list[bool] = [
            context.log_probabilities_echo for context in context_batch
        ]

        tracer.next("claim_cache_rows")
        # Claim cache rows for our batch.
        for context in context_batch:
            # this is effectively: max_seq_len - (num_tokens_in_kv_cache + num_new_tokens) - num_new_tokens
            num_available_steps = self._pipeline_config.max_length - (
                context.current_length - context.seq_len
            )
            if num_available_steps <= 0:
                raise ValueError(
                    f"Request {context.cache_seq_id} length ({context.current_length}) is larger than or equal to the configured max_length ({self._pipeline_config.max_length})"
                )

            num_steps = (
                num_steps
                if num_available_steps > num_steps
                else num_available_steps
            )
            if not self._pipeline_model.kv_manager.contains(
                context.cache_seq_id
            ):
                self._pipeline_model.kv_manager.external_claim(
                    [context.cache_seq_id]
                )

        seq_ids_and_prompts = {
            ctx.cache_seq_id: ctx.next_tokens for ctx in context_batch
        }
        seq_ids_and_untrimmed_lengths = {
            ctx.cache_seq_id: len(ctx.next_tokens) for ctx in context_batch
        }

        # `fetch` mutates the seq_ids_and_prompts input in place when tokens are
        # retrieved from the cache. This shortens the prompt in the event that
        # some tokens have backing KV cache entries.
        tracer.next("fetch_kv_cache")
        kv_cache_inputs = self._pipeline_model.kv_manager.fetch(
            seq_ids_and_prompts, num_steps
        )

        # Update the context with the new possibly shortened prompt.
        for ctx in context_batch:
            seq_id = ctx.cache_seq_id
            untrimmed_length = seq_ids_and_untrimmed_lengths[seq_id]
            trimmed_length = len(seq_ids_and_prompts[seq_id])
            ctx.trim_prompt(untrimmed_length - trimmed_length)

        tracer.next("prepare_initial_token_inputs")
        # Prepare inputs for the first token in multistep execution.
        model_inputs = self._pipeline_model.prepare_initial_token_inputs(
            context_batch
        )

        # Multistep execution loop.
        tracer.next("allocate_generated_tokens")
        generated_tokens = Tensor.zeros(
            (len(context_batch), 0),
            dtype=DType.int64,
            device=self._pipeline_config.device,
        )

        curr_step_inputs = model_inputs
        batch_log_probabilities = []
        tracer.next(f"multistep_execution_loop_{num_steps}_steps")
        for i in range(num_steps):
            tracer.push(f"step_{i}")
            # Extract the kv-cache inputs and flatten before pushing to execute
            kv_cache_inputs_tuple = [
                inp
                for kv_cache_input in kv_cache_inputs
                for inp in kv_cache_input
            ]
            # Execute the model and get next tokens.
            model_outputs = self._pipeline_model.execute(
                model_inputs=curr_step_inputs,
                kv_cache_inputs=kv_cache_inputs_tuple,
            )
            assert model_outputs.next_token_logits is not None
            next_token_logits = model_outputs.next_token_logits
            tracer.next("sample_next_token")
            new_tokens, new_generated_tokens = self._sampler(
                next_token_logits, generated_tokens
            )[:2]
            assert isinstance(new_tokens, Tensor)
            assert isinstance(new_generated_tokens, Tensor)
            generated_tokens = new_generated_tokens

            if compute_log_probabilities:
                try:
                    tracer.next("compute_log_probabilities")
                    batch_log_probabilities.append(
                        self._pipeline_model.compute_log_probabilities(
                            curr_step_inputs,
                            model_outputs,
                            new_tokens,
                            batch_top_n,
                            batch_echo,
                        )
                    )
                except NotImplementedError:
                    warnings.warn(
                        "Unable to compute log probabilities for"
                        f" {self._pipeline_config.short_name}"
                    )
                    batch_log_probabilities.append(None)
            # Check if we're on our last iteration. If so, skip preparing the next batch
            if i == num_steps - 1:
                tracer.pop()  # pops f"step_{i}"
                break
            # Prepare inputs for the next token in multistep execution
            tracer.next("increment_cache_lengths")  # pops sample_next_token
            # Unpack model inputs for execute() call by getting all fields
            curr_step_inputs_tuple = tuple(
                getattr(curr_step_inputs, field)
                for field in vars(curr_step_inputs)
            )
            kv_cache_inputs = self._pipeline_model.kv_manager.increment_cache_lengths(
                kv_cache_inputs,
                # TODO(zheng): Due to a circular import in kv_cache/manager.py,
                # we cannot import ModelInputs here. We just unroll all fields
                # as an iterable one like this.
                curr_step_inputs_tuple,
            )
            tracer.next("prepare_next_token_inputs")  # pops inc_cache_lengths
            curr_step_inputs = self._pipeline_model.prepare_next_token_inputs(
                new_tokens,
                curr_step_inputs,
            )
            tracer.pop()  # pops step_{i}

        # Do the copy to host for each token generated.
        tracer.next(
            "generated_tokens.to(CPU())"
        )  # pops multistep_execution_loop_steps
        generated_tokens_host = generated_tokens.to(CPU()).to_numpy()

        # Actually update the cache lengths in our kv_cache manager
        tracer.next("kv_manager.step")  # pops generated_tokens.to(CPU())
        seq_ids_and_new_tokens = {
            ctx.cache_seq_id: generated_tokens_host[i]
            for i, ctx in enumerate(context_batch)
        }
        self._pipeline_model.kv_manager.step(seq_ids_and_new_tokens)
        tracer.pop()  # pops kv_manager.step

        # Prepare the response, pruning away completed requests as we go.
        res: list[dict[str, Any]] = [{} for _ in range(num_steps)]
        tracer.push("prepare_response")
        for batch_index, (request_id, context) in enumerate(batch.items()):
            step = 0
            while step < num_steps:
                # Convert to a Python scalar to improve serialization performance.
                next_token = int(generated_tokens_host[batch_index, step])

                if (
                    next_token in self._eos_token_id
                    or (context.current_length + step) >= context.max_length
                ):
                    step += 1
                    break

                # Set up TextResponse
                log_probs: Optional[LogProbabilities] = None
                if compute_log_probabilities and (
                    log_probs_for_step := batch_log_probabilities[step]
                ):
                    log_probs = log_probs_for_step[batch_index]

                # Removing the positional arguments here, go about 100us faster.
                res[step][request_id] = TextResponse(next_token, log_probs)

                step += 1

            # Update the context once, just at the end.
            tracer.push(f"update_batch_{batch_index}")
            context.update(
                new_token=next_token,
                num_steps=step,
            )
            tracer.pop()  # pops update_batch_{batch_index}
        return res

    def release(self, context: T) -> None:
        """Mark the context as complete, releasing the cache slot from the KV manager."""
        self._pipeline_model.kv_manager.release(context.cache_seq_id)
