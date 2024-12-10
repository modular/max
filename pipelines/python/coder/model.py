# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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

import numpy as np
from dataprocessing import batch_padded_tokens_and_mask
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.pipelines import ModelOutputs, PipelineModel, TextContext
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    estimate_kv_cache_size,
    load_kv_manager,
)

from .graph import _build_graph


class CoderModel(PipelineModel):
    def execute(self, *model_inputs: Tensor) -> ModelOutputs:
        model_outputs = self.model.execute(
            *model_inputs,
            copy_inputs_to_device=(
                not self.pipeline_config.cache_strategy.uses_opaque()
            ),
        )

        if self.pipeline_config.enable_echo:
            return ModelOutputs(
                next_token_logits=model_outputs[0], logits=model_outputs[1]
            )
        else:
            return ModelOutputs(next_token_logits=model_outputs[0])

    def _prepare_ragged_initial_token_inputs(
        self, context_batch: list[TextContext]
    ) -> tuple[Tensor, ...]:
        # Get tokens and seq_ids
        tokens = [ctx.next_tokens for ctx in context_batch]

        # Get input_row_offsets: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offsets = Tensor.from_numpy(
            np.cumsum(
                [0] + [ctx.seq_len for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.pipeline_config.device)

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        next_tokens_batch = np.concatenate(tokens)
        next_tokens_batch = Tensor.from_numpy(next_tokens_batch).to(
            self.pipeline_config.device
        )

        return (next_tokens_batch, input_row_offsets)

    def _prepare_padded_initial_token_inputs(
        self, context_batch: list[TextContext]
    ) -> tuple[Tensor, ...]:
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

        return (next_tokens_batch, attn_mask)

    def prepare_initial_token_inputs(
        self, context_batch: list[TextContext]
    ) -> tuple[Tensor, ...]:
        """Prepare the inputs for the first pass in multistep execution."""
        if self.pipeline_config.cache_strategy == KVCacheStrategy.CONTINUOUS:
            return self._prepare_ragged_initial_token_inputs(context_batch)
        else:
            return self._prepare_padded_initial_token_inputs(context_batch)

    def _prepare_continuous_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: tuple[Tensor, ...],
    ):
        _, old_row_offsets = prev_model_inputs
        row_offsets_size = old_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]
        next_token_inputs = (next_tokens, next_row_offsets)

        return next_token_inputs

    def _prepare_naive_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: tuple[Tensor, ...],
    ):
        prev_tokens, prev_attn_mask = prev_model_inputs
        batch_size = prev_tokens.shape[0]
        start_pos = [prev_attn_mask.shape[-1]] * batch_size
        next_tokens_batch, _, attn_mask = batch_padded_tokens_and_mask(
            start_pos=start_pos,
            tokens=next_tokens,
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
        )
        next_token_inputs = (next_tokens_batch, attn_mask)

        return next_token_inputs

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        """Prepare the inputs for the next token in multistep execution.
        This should avoid any device synchronization or copy operations.
        """
        if self.pipeline_config.cache_strategy == KVCacheStrategy.CONTINUOUS:
            return self._prepare_continuous_next_token_inputs(
                next_tokens, prev_model_inputs
            )
        else:
            return self._prepare_naive_next_token_inputs(
                next_tokens, prev_model_inputs
            )

    def _get_kv_params(self) -> KVCacheParams:
        cache_dtype = (
            DType.float32
            if self.pipeline_config.quantization_encoding.quantization_encoding
            is not None
            else self.pipeline_config.dtype
        )
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=self.pipeline_config.huggingface_config.num_key_value_heads,
            head_dim=self.pipeline_config.huggingface_config.hidden_size
            // self.pipeline_config.huggingface_config.num_attention_heads,
            cache_strategy=self.pipeline_config.cache_strategy,
        )

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        return load_kv_manager(
            params=self._get_kv_params(),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=self.pipeline_config.huggingface_config.max_seq_len,
            num_layers=self.pipeline_config.huggingface_config.num_hidden_layers,
            devices=[self.pipeline_config.device],
            available_cache_memory=available_cache_memory,
            session=session,
        )

    def estimate_kv_cache_size(self, available_cache_memory: int) -> int:
        return estimate_kv_cache_size(
            params=self._get_kv_params(),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=self.pipeline_config.huggingface_config.max_seq_len,
            num_layers=self.pipeline_config.huggingface_config.num_hidden_layers,
            available_cache_memory=available_cache_memory,
            devices=[self.pipeline_config.device],
        )

    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(
                self.pipeline_config.max_cache_batch_size + 1, dtype=np.uint32
            )
        ).to(self.pipeline_config.device)

        # Read in weights.
        self._weights = self.pipeline_config.load_weights()

        if serialized_path := self.pipeline_config.serialized_model_path:
            # Hydrate all weights to be referenced by the serialized path.
            weights_registry = {}
            for name, tensor in self._weights._tensors.items():
                weights_registry[name] = tensor.data

            logging.info("Loading serialized model from ", serialized_path)

            return session.load(
                serialized_path, weights_registry=weights_registry
            )

        else:
            logging.info("Building model...")
            graph = _build_graph(
                self.pipeline_config,
                self._weights,
                self._get_kv_params(),
                kv_manager=self.kv_manager,
            )
            logging.info("Compiling...")
            model = session.load(
                graph,
                weights_registry=self._weights.allocated_weights,  # type: ignore
            )
            if (
                export_path
                := self.pipeline_config.save_to_serialized_model_path
            ):
                logging.info("Exporting serialized model to %s", export_path)
                model._export_mef(export_path)
            return model

    def compute_log_probabilities(
        self,
        model_inputs: Sequence[Tensor],
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
            logits = model_outputs.logits.to(CPU()).to_numpy()
        next_token_logits = model_outputs.next_token_logits.to(CPU()).to_numpy()

        sampled_tokens = next_tokens.to(CPU()).to_numpy()
        # Handle batched inputs.
        token_tensor, _, valid_length_tensor = model_inputs
        tokens = token_tensor.to(CPU()).to_numpy()
        valid_lengths = valid_length_tensor.to(CPU()).to_numpy()

        def _get_logits_and_samples(
            batch_index: int, echo: bool
        ) -> tuple[np.ndarray, np.ndarray]:
            if echo:
                seq_len = valid_lengths[batch_index]
                padded_tokens = tokens[batch_index]
                assert model_outputs.logits is not None
                batch_logits = logits[batch_index, :seq_len]
                samples = np.concatenate(
                    (
                        padded_tokens[1:seq_len],
                        sampled_tokens[batch_index : batch_index + 1],
                    )
                )
            else:
                batch_logits = next_token_logits[batch_index : batch_index + 1]
                samples = sampled_tokens[batch_index : batch_index + 1]
            return batch_logits, samples

        return compute_log_probabilities(
            _get_logits_and_samples, batch_top_n, batch_echo
        )
