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

import numpy as np
from max.driver import Tensor
from max.engine import InferenceSession, Model
from max.graph.weights import SafetensorWeights
from max.pipelines import (
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    TextAndVisionContext,
)
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheParams,
    estimate_kv_cache_size,
    load_kv_manager,
)

from .model.graph import _build_text_graph, _build_vision_graph
from .vision_encoder.attention_utils import causal_attention_mask_2d_from_imgs


class PixtralModel(PipelineModel):
    """The overall interface to the Pixtral model."""

    def __init__(
        self, pipeline_config: PipelineConfig, session: InferenceSession
    ) -> None:
        super().__init__(pipeline_config, session)
        self.vision_model, self.language_model = self.load_model(session)
        # Note that in a multimodal model, the language model is the last model in the
        # pipeline. Unfortunately, self.model is still being used (and exposed)
        # in the token generation code, so we still need to set it here.
        self.model = self.language_model

    def execute(self, *model_inputs: Tensor) -> ModelOutputs:  # type: ignore
        model_input_list = list(model_inputs)

        if (
            len(model_input_list) == 8
        ):  # vision_graph has 4 inputs: pixel_values and attention_mask
            image_embeds = self.vision_model.execute(
                *model_input_list[2:4], copy_inputs_to_device=False
            )[0]
            # drop pixel_values and attention_mask from inputs for the language model
            model_input_list = model_input_list[:2] + model_input_list[4:]
        else:
            # batch_size * num_concurrent_media * num_patches are set to 0 here to imitate a dummy tensor (used in text-only mode).
            image_embeds = Tensor.zeros(
                shape=[
                    0,
                    0,
                    self.pipeline_config.huggingface_config.text_config.hidden_size,
                ],
                dtype=self.pipeline_config.dtype,
            ).to(self.pipeline_config.device)

        model_input_list.insert(1, image_embeds)  # type: ignore
        model_outputs = self.language_model.execute(
            *model_input_list, copy_inputs_to_device=False
        )
        assert not self.pipeline_config.enable_echo
        assert isinstance(model_outputs[0], Tensor)
        return ModelOutputs(next_token_logits=model_outputs[0])

    def prepare_initial_token_inputs(
        self,
        context_batch: list[TextAndVisionContext],  # type: ignore
    ) -> tuple[Tensor, ...]:
        # Input row offset type: ["input_row_offsets_len"], UInt32
        input_row_offsets = Tensor.from_numpy(
            np.cumsum(
                [0] + [ctx.seq_len for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.pipeline_config.device)

        # Input Ids: ["total_seq_len"], Int64
        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.ascontiguousarray(
            np.concatenate([ctx.next_tokens for ctx in context_batch])
        )
        input_ids = Tensor.from_numpy(tokens).to(self.pipeline_config.device)

        # TODO: change this to work with all contexts in the batch.
        if context_batch[
            0
        ].pixel_values:  # check if the request has pixel_values
            # Get first image in first batch and permute the order to (HWC).
            # Pixtral processor returns CHW images.
            image = np.ascontiguousarray(
                np.transpose(context_batch[0].pixel_values[0], (1, 2, 0))
            )
            pixel_values = Tensor.from_numpy(image).to(
                self.pipeline_config.device
            )
            # TODO(KERN-782): This should be -inf but softmax saturates with NaNs.
            fill_val = -10000.0
            attention_mask = causal_attention_mask_2d_from_imgs(
                [image],
                self.pipeline_config.huggingface_config.vision_config.patch_size,
                1,
                fill_val,
            )
            attention_mask = Tensor.from_numpy(attention_mask).to(
                self.pipeline_config.device
            )
            return (
                input_ids,
                input_row_offsets,
                pixel_values,
                attention_mask,
            )

        return (
            input_ids,
            input_row_offsets,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        # input_ids, old_row_offsets, Optional: [pixel_values, attention_mask]
        old_row_offsets = prev_model_inputs[1]

        row_offsets_size = old_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]
        # In multi-step execution, don't re-pass the pixel_values.
        return (
            next_tokens,
            next_row_offsets,
        )

    def _get_kv_params(self) -> KVCacheParams:
        return KVCacheParams(
            dtype=self.pipeline_config.dtype,
            n_kv_heads=self.pipeline_config.huggingface_config.text_config.num_key_value_heads,
            head_dim=self.pipeline_config.huggingface_config.text_config.head_dim,
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
            num_layers=self.pipeline_config.huggingface_config.text_config.num_hidden_layers,
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
            num_layers=self.pipeline_config.huggingface_config.text_config.num_hidden_layers,
            available_cache_memory=available_cache_memory,
            devices=self.pipeline_config.devices,
        )

    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
        if self.pipeline_config.enable_echo:
            msg = "Pixtral model does not currently implement enable echo."
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
                "only safetensors weights are currently supported in Pixtral"
                " models."
            )
            raise ValueError(msg)

        logging.info("Building vision model...")
        vision_graph = _build_vision_graph(
            self.pipeline_config,
            self._weights,
        )
        logging.info("Compiling...")
        before = time.perf_counter()
        vision_model = session.load(
            vision_graph, weights_registry=self._weights.allocated_weights
        )
        after = time.perf_counter()
        logging.info(f"Compiling model took {after - before:.6f} seconds")
        if export_path := self.pipeline_config.save_to_serialized_model_path:
            logging.info("Exporting serialized model to %s", export_path)
            vision_model._export_mef(export_path)

        logging.info("Building text model...")
        text_graph = _build_text_graph(
            self.pipeline_config,
            self._weights,
            self._get_kv_params(),
            self.kv_manager,
        )
        logging.info("Compiling...")
        before = time.perf_counter()
        text_model = session.load(
            text_graph, weights_registry=self._weights.allocated_weights
        )
        after = time.perf_counter()
        logging.info(f"Compiling model took {after - before:.6f} seconds")
        if export_path := self.pipeline_config.save_to_serialized_model_path:
            logging.info("Exporting serialized model to %s", export_path)
            text_model._export_mef(export_path)

        return vision_model, text_model
