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
from typing import Sequence

import numpy as np
from max.driver import Tensor
from max.engine import InferenceSession, Model
from max.graph.weights import SafetensorWeights
from max.pipelines import (
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

from .graph import _build_graph


class MistralModel(PipelineModel):
    def __init__(
        self, pipeline_config: PipelineConfig, session: InferenceSession
    ) -> None:
        super().__init__(pipeline_config, session)
        self.model = self.load_model(session)

    def execute(self, *model_inputs: Tensor) -> ModelOutputs:
        """Runs the graph."""
        model_outputs = self.model.execute(
            *model_inputs, copy_inputs_to_device=False
        )
        assert isinstance(model_outputs[0], Tensor)
        return ModelOutputs(next_token_logits=model_outputs[0])

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],  # type: ignore
    ) -> tuple[Tensor, ...]:
        # Get tokens and seq ids
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

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        _, old_row_offsets = prev_model_inputs
        row_offsets_size = old_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]
        next_token_inputs = (next_tokens, next_row_offsets)

        return next_token_inputs

    def _get_kv_params(self) -> KVCacheParams:
        return KVCacheParams(
            dtype=self.pipeline_config.dtype,
            n_kv_heads=self.pipeline_config.huggingface_config.num_key_value_heads,
            head_dim=self.pipeline_config.huggingface_config.head_dim,
            cache_strategy=self.pipeline_config.cache_strategy,
            enable_prefix_caching=self.pipeline_config.enable_prefix_caching,
        )

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        assert (
            self.pipeline_config.devices
        ), "devices must be provided to load kv manager."
        return load_kv_manager(
            params=self._get_kv_params(),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=self.pipeline_config.huggingface_config.max_seq_len,
            num_layers=self.pipeline_config.huggingface_config.num_hidden_layers,
            devices=self.pipeline_config.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.pipeline_config.kv_cache_page_size,
            session=session,
        )

    def estimate_kv_cache_size(self, available_cache_memory: int) -> int:
        assert (
            self.pipeline_config.devices
        ), "devices must be provided to estimate kv cache size."
        return estimate_kv_cache_size(
            params=self._get_kv_params(),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=self.pipeline_config.huggingface_config.max_seq_len,
            num_layers=self.pipeline_config.huggingface_config.num_hidden_layers,
            available_cache_memory=available_cache_memory,
            devices=self.pipeline_config.devices,
        )

    def load_model(self, session: InferenceSession) -> Model:
        if self.pipeline_config.enable_echo:
            msg = "Mistral model does not currently implement enable echo."
            raise ValueError(msg)

        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(
                self.pipeline_config.max_cache_batch_size + 1, dtype=np.uint32
            )
        ).to(self.pipeline_config.device)

        self._weights = self.pipeline_config.load_weights()

        if not isinstance(self._weights, SafetensorWeights):
            msg = (
                "only safetensors weights are currently supported in Mistral"
                " models."
            )
            raise ValueError(msg)

        if serialized_path := self.pipeline_config.serialized_model_path:
            # Hydrate all weights to be referenced by the serialized graph.
            weights_registry = {}
            for name, weight in self._weights.items():
                weights_registry[name] = weight.raw_tensor()
            logging.info(
                "Loading serialized model from ", serialized_path, "..."
            )
            return session.load(
                serialized_path,
                weights_registry=weights_registry,
            )
        else:
            logging.info("Building model...")
            graph = _build_graph(
                self.pipeline_config,
                self._weights,
                self._get_kv_params(),
                self.kv_manager,
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
