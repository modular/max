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
from typing import Any, Optional, Sequence, Type, TypeVar

import numpy as np
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.profiler import Tracer, traced

from .config import PipelineConfig
from .context import InputContext
from .interfaces import TokenGenerator
from .kv_cache import KVCacheManager
from .response import LogProbabilities, TextResponse
from .sampling import token_sampler

ARCH_SAFE_VRAM_USAGE_LIMIT = {
    "DeepseekCoder": 0.96,
    "LlamaForCausalLM": 0.96,
    "MistralForCausalLM": 0.96,
}


@dataclass(frozen=True)
class ModelOutputs:
    next_token_logits: Tensor
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
        self.estimate_memory_footprint()
        self.kv_manager = self.load_kv_manager(session)
        self.model = self.load_model(session)

    def estimate_memory_footprint(self):
        def to_mib(bytes):
            return round(bytes / 1024 / 1024)

        total_size = 0
        weights_size = self.pipeline_config.weights_size()
        if weights_size is not None:
            total_size += weights_size

        model_arch = self.pipeline_config.architecture
        current_batch_size = self.pipeline_config.max_cache_batch_size
        current_seq_len = self.pipeline_config.max_length
        kv_size = self.estimate_kv_cache_size()
        total_size += kv_size

        free_memory = None
        max_batch_size = None
        try:
            # TODO(AIPIPE-200): change this back to free_memory.
            # Currently, free_memory leads to issues with calculation on
            # repeated model loads (like seen in tests). Based on the outputs,
            # it seems like we are not freeing memory from a model until the
            # next model is loading weights. As such, there is not enough free
            # memory when running this check. Memory likely should be getting
            # cleaned up much earlier.
            free_memory = self.pipeline_config.device.stats["total_memory"]
            vram_usage_limit_scale = ARCH_SAFE_VRAM_USAGE_LIMIT.get(
                model_arch or "", 0.75
            )
            max_batch_size = int(
                (free_memory * vram_usage_limit_scale - weights_size)
                / (kv_size / current_batch_size)
            )
        except:
            pass

        weights_str = str(to_mib(weights_size)) if weights_size else "unknown"
        free_memory_str = (
            str(to_mib(free_memory)) if free_memory is not None else "unknown"
        )
        max_batch_size_str = (
            str(max_batch_size) if max_batch_size is not None else "unknown"
        )
        logging.info(
            "\n"
            f"\n\tEstimated memory consumption:"
            f"\n\t    Weights:                {weights_str} MiB"
            f"\n\t    KVCache allocation:     {to_mib(kv_size)} MiB"
            f"\n\t    Total estimated:        {to_mib(total_size)} MiB used / {free_memory_str} MiB free\n"
            f"\n\tCurrent batch size: {current_batch_size}"
            f"\n\tCurrent max sequence length: {current_seq_len}"
            f"\n\tMax recommended batch size for current sequence length: {max_batch_size_str}\n"
        )

        if isinstance(free_memory, (int, float)):
            if total_size > free_memory:
                raise RuntimeError(
                    "Estimated model and kv cache memory use exceeds available memory."
                )
            elif total_size > 0.75 * free_memory:
                logging.warning(
                    "Estimated model and kv cache memory use nears available memory. You may experience errors."
                )

    @abstractmethod
    def execute(self, *model_inputs: Tensor) -> ModelOutputs:
        """Runs the graph."""
        ...

    @abstractmethod
    def prepare_initial_token_inputs(
        self, context_batch: Sequence[T]
    ) -> tuple[Tensor, ...]:
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
        prev_model_inputs: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        """Prepares the secondary inputs to be passed to `.execute()`.

        While `prepare_initial_token_inputs` is responsible for managing the initial inputs.
        This function is responsible for updating the inputs, for each step in a multi-step execution pattern.
        """
        ...

    @abstractmethod
    def load_kv_manager(self, session: InferenceSession) -> KVCacheManager:
        """Provided a PipelineConfig and InferenceSession, load the kv manager."""
        ...

    @abstractmethod
    def estimate_kv_cache_size(self) -> int:
        """Estimates the size of the kv cache in bytes."""
        ...

    @abstractmethod
    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        """Provided a PipelineConfig and InferenceSession, build and load the model graph."""
        ...

    def compute_log_probabilities(
        self,
        model_inputs: Sequence[Tensor],
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


class TextGenerationPipeline(TokenGenerator[T]):
    """Generalized token generator pipeline."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: Type[PipelineModel],
        # TODO: This should be removed.
        eos_token_id: int,
    ):
        self._pipeline_config = pipeline_config
        self._eos_token_id = eos_token_id

        # Initialize Session.
        session = InferenceSession(devices=[self._pipeline_config.device])

        # Load model.
        self._pipeline_model = pipeline_model(
            pipeline_config=self._pipeline_config, session=session
        )

        # Load sampler.
        self._sampler = session.load(
            token_sampler(
                self._pipeline_config.top_k,
                # Logits are at index 0 of model ouputs.
                in_dtype=self._pipeline_model.model.output_metadata[0].dtype,
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
            if not self._pipeline_model.kv_manager.contains(
                context.cache_seq_id
            ):
                self._pipeline_model.kv_manager.external_claim(
                    [context.cache_seq_id]
                )

        # Get cache seq ids for batch.
        valid_lengths = {
            ctx.cache_seq_id: ctx.seq_len + num_steps - 1
            for ctx in context_batch
        }

        tracer.next("prepare_initial_token_inputs")
        # Prepare inputs for the first token in multistep execution.
        model_inputs = self._pipeline_model.prepare_initial_token_inputs(
            context_batch
        )
        tracer.next("fetch_kv_cache")
        kv_cache_inputs = self._pipeline_model.kv_manager.fetch(valid_lengths)

        # Multistep execution loop.
        tracer.next("Tensor.from_numpy")
        generated_tokens = Tensor.from_numpy(
            np.zeros((len(context_batch), 0), dtype=np.int64)
        ).to(self._pipeline_config.device)

        curr_step_inputs = model_inputs
        batch_log_probabilities = []
        tracer.next(f"multistep_execution_loop_{num_steps}_steps")
        for i in range(num_steps):
            tracer.push(f"step_{i}")
            # Assuming 1 device, get first KVCache
            kv_cache_inputs_tuple = kv_cache_inputs[0]
            # Execute the model and get next tokens.
            model_outputs = self._pipeline_model.execute(
                *curr_step_inputs, *kv_cache_inputs_tuple
            )
            next_token_logits = model_outputs.next_token_logits
            tracer.next("sample_next_token")
            new_tokens, generated_tokens = self._sampler(  # type: ignore
                next_token_logits, generated_tokens
            )[:2]
            assert isinstance(new_tokens, Tensor)

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
            kv_cache_inputs = (
                self._pipeline_model.kv_manager.increment_cache_lengths(
                    kv_cache_inputs,
                    curr_step_inputs,
                )
            )
            tracer.next("prepare_next_token_inputs")  # pops inc_cache_lengths
            curr_step_inputs = self._pipeline_model.prepare_next_token_inputs(
                new_tokens,
                curr_step_inputs,
            )
            tracer.pop()  # pops step_{i}

        # Actually update the cache lengths in our kv_cache manager
        tracer.next("kv_manager.step")  # pops multistep_execution_loop_steps
        self._pipeline_model.kv_manager.step(valid_lengths=valid_lengths)

        # Do the copy to host for each token generated.
        tracer.next("generated_tokens.to(CPU())")  # pops kv_manager.step
        generated_tokens_host = generated_tokens.to(CPU()).to_numpy()
        tracer.pop()  # pops generated_tokens.to(CPU())

        # Prepare the response, pruning away completed requests as we go.
        res: list[dict[str, Any]] = [{} for _ in range(num_steps)]
        tracer.push("prepare_response")
        for batch_index, (request_id, context) in enumerate(batch.items()):
            step = 0
            while step < num_steps:
                # Convert to a Python scalar to improve serialization performance.
                next_token = int(generated_tokens_host[batch_index, step])

                if (
                    next_token == self._eos_token_id
                    or (context.current_length + step) >= context.max_length
                ):
                    step += 1
                    break

                # Set up TextResponse
                log_probs: Optional[LogProbabilities] = None
                if compute_log_probabilities and batch_log_probabilities[step]:
                    log_probs = batch_log_probabilities[step][batch_index]  # type: ignore

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
