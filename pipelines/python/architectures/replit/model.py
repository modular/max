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
import time
import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
from dataprocessing import causal_attention_mask_with_alibi, collate_batch
from max.driver import CPU, DeviceSpec, Tensor
from max.engine import InferenceSession, Model
from max.graph.weights import GGUFWeights
from max.pipelines import (
    LogProbabilities,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    TextContext,
)
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheParams,
    estimate_kv_cache_size,
    load_kv_manager,
)
from nn.compute_log_probabilities import compute_log_probabilities

from .graph import _build_graph


class ReplitModel(PipelineModel):
    def __init__(
        self, pipeline_config: PipelineConfig, session: InferenceSession
    ) -> None:
        if pipeline_config.device_specs[0] == DeviceSpec.cpu():
            msg = "Replit currently only supported on gpu."
            raise ValueError(msg)

        super().__init__(pipeline_config, session)
        self.model = self.load_model(session)

    def execute(self, *model_inputs: Tensor) -> ModelOutputs:  # type: ignore
        model_outputs = self.model.execute(
            *model_inputs, copy_inputs_to_device=False
        )
        if self.pipeline_config.enable_echo:
            assert len(model_outputs) == 2
            assert isinstance(model_outputs[0], Tensor)
            assert isinstance(model_outputs[1], Tensor)
            return ModelOutputs(
                next_token_logits=model_outputs[0], logits=model_outputs[1]
            )
        else:
            assert len(model_outputs) == 1
            assert isinstance(model_outputs[0], Tensor)
            return ModelOutputs(next_token_logits=model_outputs[0])

    def prepare_initial_token_inputs(
        self,
        context_batch: list[TextContext],  # type: ignore
    ) -> tuple[Tensor, ...]:
        # Get tokens and seq_ids.
        tokens = [ctx.next_tokens for ctx in context_batch]

        unpadded_lengths = [ctx.seq_len for ctx in context_batch]

        # Pad tokens and compute attention mask for the batch.
        max_seq_len = self.kv_manager.max_sequence_length
        start_pos = [max_seq_len] * len(context_batch)
        next_tokens_batch, _ = collate_batch(
            tokens,
            batch_size=len(tokens),
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
        )

        attention_mask = causal_attention_mask_with_alibi(
            original_start_pos=start_pos,
            original_seq_len=[len(t) for t in tokens],
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
            alibi_bias_max=self.pipeline_config.huggingface_config.attn_config[
                "alibi_bias_max"
            ],
            n_heads=self.pipeline_config.huggingface_config.n_heads,
        )

        return (
            Tensor.from_numpy(next_tokens_batch).to(
                self.pipeline_config.device
            ),
            Tensor.from_numpy(attention_mask).to(self.pipeline_config.device),
            Tensor.from_numpy(np.array(unpadded_lengths, np.uint32)).to(
                self.pipeline_config.device
            ),
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        # Update valid_lengths by one for all inputs.
        prev_tokens, prev_attention_mask, valid_lengths = prev_model_inputs
        valid_lengths += 1  # type: ignore

        batch_size = prev_tokens.shape[0]
        start_pos = [prev_attention_mask.shape[-1]] * batch_size
        next_tokens_batch = collate_batch(
            next_tokens,  # type: ignore
            batch_size=batch_size,
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
        )

        attention_mask = causal_attention_mask_with_alibi(
            original_start_pos=start_pos,
            original_seq_len=[len(t) for t in next_tokens],  # type: ignore
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
            alibi_bias_max=self.pipeline_config.huggingface_config.attn_config[
                "alibi_bias_max"
            ],
            n_heads=self.pipeline_config.huggingface_config.n_heads,
        )

        # I believe, next_tokens_batch & valid_lengths, should already be resident on the GPU.
        # The attention mask is a new tensor, and thus has to be moved over.
        return (
            Tensor.from_numpy(next_tokens_batch),  # type: ignore
            Tensor.from_numpy(attention_mask).to(self.pipeline_config.device),
            Tensor.from_numpy(valid_lengths),  # type: ignore
        )

    def _get_kv_params(self) -> KVCacheParams:
        return KVCacheParams(
            dtype=self.pipeline_config.dtype,
            n_kv_heads=self.pipeline_config.huggingface_config.attn_config[
                "kv_n_heads"
            ],
            head_dim=self.pipeline_config.huggingface_config.d_model
            // self.pipeline_config.huggingface_config.n_heads,
            cache_strategy=self.pipeline_config.cache_strategy,
            enable_prefix_caching=self.pipeline_config.enable_prefix_caching,
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
            num_layers=self.pipeline_config.huggingface_config.n_layers,
            devices=self.pipeline_config.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.pipeline_config.kv_cache_page_size,
            session=session,
        )

    def estimate_kv_cache_size(self, available_cache_memory: int) -> int:
        return estimate_kv_cache_size(
            params=self._get_kv_params(),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=self.pipeline_config.huggingface_config.max_seq_len,
            num_layers=self.pipeline_config.huggingface_config.n_layers,
            available_cache_memory=available_cache_memory,
            devices=self.pipeline_config.devices,
        )

    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        # TODO: AIPIPE-235 - Support Multistep scheduling for Replit
        if self.pipeline_config.max_num_steps > 1:
            msg = "Replit pipeline does not support max_num_steps > 1"
            raise ValueError(msg)

        # Read in weights.
        weights = self.pipeline_config.load_weights()
        if not isinstance(weights, GGUFWeights):
            msg = "only gguf weights supported in Replit."
            raise ValueError(msg)

        self._weights = weights

        if serialized_path := self.pipeline_config.serialized_model_path:
            # Hydrate all weights to be referenced by the serialized path.
            weights_registry = {}
            for name, weight in self._weights.items():
                weights_registry[name] = weight.raw_tensor()

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

    def compute_log_probabilities(
        self,
        model_inputs: Iterable[Any],
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
        assert model_outputs.next_token_logits
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
