# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import logging
import time
import warnings
from typing import List, Sequence, Union, cast

import numpy as np
from dataprocessing import batch_padded_tokens_and_mask
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import GGUFWeights
from max.pipelines import (
    LogProbabilities,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
    TextContext,
)
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheParams,
    estimate_kv_cache_size,
    load_kv_manager,
)
from nn.compute_log_probabilities import compute_log_probabilities

from .gguf import distributed_transformer_opaque, transformer


class Llama3Inputs(ModelInputs):
    """A class representing inputs for the Llama3 model.

    This class encapsulates the input tensors required for the Llama3 model execution:
    - tokens: A tensor containing the input token IDs
    - input_row_offsets_or_attn_mask: A tensor containing the offsets for each row in the ragged input sequence,
    or the attention mask for the padded input sequence
    """

    tokens: Tensor
    input_row_offsets_or_attn_mask: Tensor

    def __init__(
        self,
        tokens: Tensor,
        input_row_offsets_or_attn_mask: Tensor,
    ) -> None:
        self.tokens = tokens
        self.input_row_offsets_or_attn_mask = input_row_offsets_or_attn_mask


class Llama3Model(PipelineModel):
    def __init__(
        self, pipeline_config: PipelineConfig, session: InferenceSession
    ) -> None:
        super().__init__(pipeline_config, session)
        self.model = self.load_model(session)

    @classmethod
    def get_kv_params(cls, pipeline_config: PipelineConfig) -> KVCacheParams:
        cache_dtype = (
            DType.float32
            if pipeline_config.quantization_encoding.quantization_encoding
            is not None
            else pipeline_config.dtype
        )
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=pipeline_config.huggingface_config.num_key_value_heads,
            head_dim=(
                pipeline_config.huggingface_config.hidden_size
                // pipeline_config.huggingface_config.num_attention_heads
            ),
            page_size=pipeline_config.kv_cache_page_size,
            cache_strategy=pipeline_config.cache_strategy,
            enable_prefix_caching=pipeline_config.enable_prefix_caching,
        )

    @classmethod
    def get_num_layers(cls, pipeline_config: PipelineConfig) -> int:
        return pipeline_config.huggingface_config.num_hidden_layers

    def execute(
        self,
        model_inputs: ModelInputs,
        kv_cache_inputs: Sequence[Tensor] | None = None,
    ) -> ModelOutputs:
        model_inputs = cast(Llama3Inputs, model_inputs)
        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets_or_attn_mask,
            *kv_cache_inputs,
            copy_inputs_to_device=(
                not self.pipeline_config.cache_strategy.uses_opaque()
            ),
        )

        if self.pipeline_config.enable_echo:
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[1],
            )
        else:
            return ModelOutputs(next_token_logits=model_outputs[0])

    def _prepare_ragged_initial_token_inputs(
        self, context_batch: Sequence[TextContext]
    ) -> Llama3Inputs:
        # Get input_row_offsets: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offsets = np.cumsum(
            [0] + [ctx.seq_len for ctx in context_batch],
            dtype=np.uint32,
        )

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])

        return Llama3Inputs(
            tokens=Tensor.from_numpy(tokens).to(
                self.pipeline_config.devices[0]
            ),
            input_row_offsets_or_attn_mask=Tensor.from_numpy(
                input_row_offsets
            ).to(self.pipeline_config.devices[0]),
        )

    def _prepare_padded_initial_token_inputs(
        self, context_batch: Sequence[TextContext]
    ) -> Llama3Inputs:
        # Get tokens and seq_ids
        tokens = [ctx.next_tokens for ctx in context_batch]

        # Pad tokens and compute attention mask for the batch.
        max_seq_len = self.kv_manager.max_sequence_length
        start_pos = [max_seq_len] * len(context_batch)
        next_tokens_batch, _, attn_mask = batch_padded_tokens_and_mask(
            start_pos=start_pos,
            tokens=tokens,
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
        )

        return Llama3Inputs(
            tokens=next_tokens_batch,
            input_row_offsets_or_attn_mask=attn_mask,
        )

    def prepare_initial_token_inputs(
        self, context_batch: Sequence[TextContext]
    ) -> Llama3Inputs:
        """Prepare the inputs for the first pass in multistep execution."""
        if self.pipeline_config.cache_strategy.uses_opaque():
            return self._prepare_ragged_initial_token_inputs(context_batch)
        else:
            return self._prepare_padded_initial_token_inputs(context_batch)

    def _prepare_ragged_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: Llama3Inputs,
    ) -> Llama3Inputs:
        row_offsets_size = (
            prev_model_inputs.input_row_offsets_or_attn_mask.shape[0]
        )
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]

        return Llama3Inputs(
            tokens=next_tokens,
            input_row_offsets_or_attn_mask=next_row_offsets,
        )

    def _prepare_padded_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: Llama3Inputs,
    ) -> Llama3Inputs:
        batch_size = prev_model_inputs.tokens.shape[0]
        start_pos = [
            prev_model_inputs.input_row_offsets_or_attn_mask.shape[-1]
        ] * batch_size
        next_tokens_batch, _, attn_mask = batch_padded_tokens_and_mask(
            start_pos=start_pos,
            tokens=next_tokens,
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
        )
        return Llama3Inputs(
            tokens=next_tokens_batch,
            input_row_offsets_or_attn_mask=attn_mask,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> Llama3Inputs:
        """Prepare the inputs for the next token in multistep execution.
        This should avoid any device synchronization or copy operations.
        """
        prev_model_inputs = cast(Llama3Inputs, prev_model_inputs)
        if self.pipeline_config.cache_strategy.uses_opaque():
            return self._prepare_ragged_next_token_inputs(
                next_tokens, prev_model_inputs
            )
        else:
            return self._prepare_padded_next_token_inputs(
                next_tokens, prev_model_inputs
            )

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        return load_kv_manager(
            params=self.get_kv_params(self.pipeline_config),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=self.pipeline_config.huggingface_config.max_seq_len,
            num_layers=self.pipeline_config.huggingface_config.num_hidden_layers,
            devices=self.pipeline_config.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.pipeline_config.kv_cache_page_size,
            session=session,
        )

    @classmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: List[Device],
    ) -> int:
        """Estimates the size of the kv cache in bytes."""
        return estimate_kv_cache_size(
            params=cls.get_kv_params(pipeline_config),
            max_cache_batch_size=pipeline_config.max_cache_batch_size,
            max_seq_len=pipeline_config.huggingface_config.max_seq_len,
            num_layers=pipeline_config.huggingface_config.num_hidden_layers,
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        assert (
            self.pipeline_config.max_cache_batch_size
        ), "Expected max_cache_batch_size to be set"
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(
                self.pipeline_config.max_cache_batch_size + 1, dtype=np.uint32
            )
        ).to(self.pipeline_config.devices[0])

        # Read in weights.
        self._weights = self.pipeline_config.load_weights()

        if serialized_path := self.pipeline_config.serialized_model_path:
            # Hydrate all weights to be referenced by the serialized path.
            weights_registry = {}
            for name, weight in self._weights.items():
                weights_registry[name] = weight.raw_tensor()

            logging.info("Loading serialized model from %s", serialized_path)

            return session.load(
                serialized_path, weights_registry=weights_registry
            )

        else:
            logging.info("Building model...")
            graph = self._build_graph(self._weights)
            logging.info("Compiling...")
            before = time.perf_counter()
            model = session.load(
                graph, weights_registry=self._weights.allocated_weights
            )
            after = time.perf_counter()
            logging.info(f"Compiling model took {after - before:.6f} seconds")
            if (
                export_path
                := self.pipeline_config.save_to_serialized_model_path
            ):
                logging.info("Exporting serialized model to %s", export_path)
                model._export_mef(export_path)
            return model

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[Tensor]
    ) -> List[tuple[Tensor, ...]]:
        kv_params = self.get_kv_params(self.pipeline_config)
        n_devices = kv_params.n_devices
        fetch_types = self.kv_manager.input_symbols()
        len_of_kv_tuple_per_dev = len(fetch_types[0])
        kv_caches_per_dev = [
            tuple(
                kv_inputs_flat[
                    i * len_of_kv_tuple_per_dev : (i + 1)
                    * len_of_kv_tuple_per_dev
                ]
            )
            for i in range(n_devices)
        ]
        return kv_caches_per_dev

    def _flatten_kv_inputs(
        self, kv_caches_per_dev: List[tuple[Union[Tensor, TensorType], ...]]
    ) -> Sequence[Union[Tensor, TensorType]]:
        return [item for sublist in kv_caches_per_dev for item in sublist]

    def _build_opaque_graph(self, weights: GGUFWeights) -> Graph:
        device0 = self.pipeline_config.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        # NOTE: input_row_offsets_len should be batch_size + 1.
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )

        if len(self.pipeline_config.devices) > 1:
            kv_cache_args = self.kv_manager.input_symbols()
            flattened_kv_types = self._flatten_kv_inputs(kv_cache_args)

            with Graph(
                "llama3",
                input_types=[
                    tokens_type,
                    input_row_offsets_type,
                    *flattened_kv_types,
                ],
            ) as graph:
                model = distributed_transformer_opaque(
                    graph,
                    self.pipeline_config,
                    weights,
                    self.get_kv_params(self.pipeline_config),
                )
                tokens, input_row_offsets, *kv_cache = graph.inputs
                kv_caches_per_dev = self._unflatten_kv_inputs(kv_cache)

                outputs = model(
                    tokens,
                    kv_caches_per_dev,
                    input_row_offsets=input_row_offsets,
                )
                graph.output(*outputs)
                return graph
        else:
            kv_cache_args = self.kv_manager.input_symbols()[0]

            with Graph(
                "llama3",
                input_types=[
                    tokens_type,
                    input_row_offsets_type,
                    *kv_cache_args,
                ],
            ) as graph:
                model = transformer(
                    graph,
                    self.pipeline_config,
                    weights,
                    self.get_kv_params(self.pipeline_config),
                )
                tokens, input_row_offsets, *kv_cache = graph.inputs
                outputs = model(
                    tokens,
                    kv_cache,
                    input_row_offsets=input_row_offsets,
                )
                graph.output(*outputs)
                return graph

    def _build_graph(self, weights: GGUFWeights) -> Graph:
        if self.pipeline_config.cache_strategy.uses_opaque():
            return self._build_opaque_graph(weights)

        tokens_type = TensorType(DType.int64, shape=["batch_size", "seq_len"])
        attn_mask_type = TensorType(
            DType.float32, shape=["batch_size", "seq_len", "post_seq_len"]
        )

        if len(self.pipeline_config.devices) > 1:
            raise ValueError(
                "Naive mode does not support distributed execution"
            )

        kv_inputs = self.kv_manager.input_symbols()[0]

        with Graph(
            "llama3",
            input_types=[
                tokens_type,
                attn_mask_type,
                *kv_inputs,
            ],
        ) as graph:
            model = transformer(
                graph,
                self.pipeline_config,
                weights,
                self.get_kv_params(self.pipeline_config),
            )
            tokens, attention_mask, k_cache, v_cache, start_pos, _ = (
                graph.inputs
            )
            mask_dtype = (
                self.pipeline_config.dtype
                if self.pipeline_config.quantization_encoding
                in [
                    SupportedEncoding.float32,
                    SupportedEncoding.bfloat16,
                ]
                else DType.float32
            )
            logits = model(
                tokens,
                attention_mask.cast(mask_dtype),
                k_cache,
                v_cache,
                start_pos,
            )[0]

            if self.pipeline_config.enable_echo:
                graph.output(logits[:, -1], logits)
            else:
                graph.output(logits[:, -1])

            return graph

    def compute_log_probabilities(
        self,
        model_inputs: ModelInputs,
        model_outputs: ModelOutputs,
        next_tokens: Tensor,
        batch_top_n: list[int],
        batch_echo: list[bool],
    ) -> list[LogProbabilities | None] | None:
        if any(echo for echo in batch_echo):
            if model_outputs.logits is None:
                warnings.warn(
                    "Could not get logprobs with echo because the full logits"
                    f" were not returned by {self.pipeline_config.short_name}"
                    " model. Please ensure that this model is started with "
                    "`--enable-echo`."
                )
                assert (
                    not self.pipeline_config.enable_echo
                ), "Echo was enabled but logits were not returned."
                return None
            logits = model_outputs.logits.to_numpy()
        next_token_logits = model_outputs.next_token_logits.to_numpy()

        sampled_tokens = next_tokens.to_numpy()
        if self.pipeline_config.cache_strategy.uses_opaque():
            # Handle the ragged inputs
            model_inputs = cast(Llama3Inputs, model_inputs)
            tokens = model_inputs.tokens.to_numpy()
            input_row_offsets = model_inputs.input_row_offsets.to(
                CPU()
            ).to_numpy()

            def _get_logits_and_samples(
                batch_index: int, echo: bool
            ) -> tuple[np.ndarray, np.ndarray]:
                if echo:
                    start_offset = input_row_offsets[batch_index]
                    end_offset = input_row_offsets[batch_index + 1]
                    batch_logits = logits[start_offset:end_offset]
                    samples = np.concatenate(
                        (
                            tokens[start_offset + 1 : end_offset],
                            sampled_tokens[batch_index : batch_index + 1],
                        )
                    )
                else:
                    batch_logits = next_token_logits[
                        batch_index : batch_index + 1
                    ]
                    samples = sampled_tokens[batch_index : batch_index + 1]
                return batch_logits, samples

        else:
            # Handle batched inputs. Llama pads them to the right so the seq
            # lengths can be computed by finding the first 0 token.
            tokens = model_inputs.tokens
            seq_lens = np.sum(tokens > 0, axis=1)

            def _get_logits_and_samples(
                batch_index: int, echo: bool
            ) -> tuple[np.ndarray, np.ndarray]:
                if echo:
                    seq_len = seq_lens[batch_index]
                    padded_tokens = tokens[batch_index]

                    batch_logits = logits[batch_index, :seq_len, :]
                    samples = np.concatenate(
                        (
                            padded_tokens[1:seq_len],
                            sampled_tokens[batch_index : batch_index + 1],
                        )
                    )
                else:
                    batch_logits = next_token_logits[
                        batch_index : batch_index + 1, :
                    ]
                    samples = sampled_tokens[batch_index : batch_index + 1]
                return batch_logits, samples

        return compute_log_probabilities(
            _get_logits_and_samples, batch_top_n, batch_echo
        )
