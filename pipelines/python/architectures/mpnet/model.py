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
"""Defines the MPNet pipeline model.

Implementation is based on MPNetModel from the transformers library.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence, cast

import numpy as np
from dataprocessing import collate_batch
from max.driver import Tensor
from max.engine import InferenceSession, Model
from max.pipelines import (
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    TextContext,
)
from max.pipelines.kv_cache import KVCacheParams

from .graph import build_graph

PAD_VALUE = 1


class MPNetInputs(ModelInputs):
    """A class representing inputs for the MPNet model.

    This class encapsulates the input tensors required for the MPNet model execution:
    - next_tokens_batch: A tensor containing the input token IDs
    - extended_attention_mask: A tensor containing the extended attention mask
    """

    next_tokens_batch: Tensor
    extended_attention_mask: Tensor

    def __init__(
        self,
        next_tokens_batch: Tensor,
        extended_attention_mask: Tensor,
    ) -> None:
        self.next_tokens_batch = next_tokens_batch
        self.extended_attention_mask = extended_attention_mask


class MPNetPipelineModel(PipelineModel):
    def __init__(
        self, pipeline_config: PipelineConfig, session: InferenceSession
    ) -> None:
        super().__init__(pipeline_config, session)
        self.model = self.load_model(session)

    @classmethod
    def get_kv_params(cls, pipeline_config: PipelineConfig) -> KVCacheParams:
        return KVCacheParams(
            dtype=pipeline_config.dtype,
            n_kv_heads=pipeline_config.huggingface_config.num_attention_heads,
            head_dim=(
                pipeline_config.huggingface_config.hidden_size
                // pipeline_config.huggingface_config.num_attention_heads
            ),
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
        model_inputs = cast(MPNetInputs, model_inputs)
        assert kv_cache_inputs is None, "MPNet does not have KV cache inputs"
        model_outputs = self.model.execute(
            model_inputs.next_tokens_batch,
            model_inputs.extended_attention_mask,
            copy_inputs_to_device=False,
        )
        assert isinstance(model_outputs[0], Tensor)
        return ModelOutputs(logits=model_outputs[0])

    def prepare_initial_token_inputs(
        self,
        context_batch: list[TextContext],  # type: ignore
    ) -> MPNetInputs:
        # Get tokens and seq_ids.
        tokens = [ctx.next_tokens for ctx in context_batch]

        # Pad tokens for the batch.
        next_tokens_batch, _ = collate_batch(
            tokens,
            pad_value=PAD_VALUE,
            batch_size=len(tokens),
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
        )

        # Compute and extend attention mask.
        attention_mask = (next_tokens_batch != PAD_VALUE).astype(np.int64)
        extended_attention_mask = _get_extended_attention_mask(attention_mask)

        return MPNetInputs(
            next_tokens_batch=Tensor.from_numpy(next_tokens_batch).to(
                self.pipeline_config.device
            ),
            extended_attention_mask=Tensor.from_numpy(
                extended_attention_mask
            ).to(self.pipeline_config.device),
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> MPNetInputs:
        raise NotImplementedError(
            "MPNet does not support preparing next tokens inputs."
        )

    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        if self.pipeline_config.max_num_steps > 1:
            msg = "MPNet does not support max_num_steps > 1"
            raise ValueError(msg)

        # Read in weights.
        weights = self.pipeline_config.load_weights()
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
            graph = build_graph(
                self.pipeline_config,
                self._weights,
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


def _get_extended_attention_mask(attention_mask: np.ndarray):
    extended_attention_mask = attention_mask[:, None, None, :]
    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.astype(dtype=np.float32)
    extended_attention_mask = (1.0 - extended_attention_mask) * np.finfo(
        np.float32
    ).min
    return extended_attention_mask
