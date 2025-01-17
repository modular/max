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
import math
import time

import numpy as np
from dataprocessing import collate_batch
from max.driver import Tensor
from max.engine import InferenceSession, Model
from max.pipelines import (
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    TextContext,
)

from .graph import build_graph

PAD_VALUE = 1


class MPNetPipelineModel(PipelineModel):
    def __init__(
        self, pipeline_config: PipelineConfig, session: InferenceSession
    ) -> None:
        super().__init__(pipeline_config, session)
        self.model = self.load_model(session)

    def execute(self, *model_inputs: Tensor) -> ModelOutputs:  # type: ignore
        model_outputs = self.model.execute(
            *model_inputs, copy_inputs_to_device=False
        )
        assert isinstance(model_outputs[0], Tensor)
        return ModelOutputs(logits=model_outputs[0])

    def prepare_initial_token_inputs(
        self,
        context_batch: list[TextContext],  # type: ignore
    ) -> tuple[Tensor, ...]:
        # Get tokens and seq_ids.
        tokens = [ctx.next_tokens for ctx in context_batch]

        # Pad tokens for the batch.
        next_tokens_batch, _ = collate_batch(
            tokens,
            pad_value=PAD_VALUE,
            batch_size=len(tokens),
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
        )

        # Compute position ids and relative position bucket.
        mask = (next_tokens_batch != PAD_VALUE).astype(np.int64)
        incremental_indices = np.cumsum(mask, axis=1) * mask
        position_ids = incremental_indices + PAD_VALUE
        relative_position_bucket = compute_relative_position_bucket(
            position_ids,
            num_buckets=self.pipeline_config.huggingface_config.relative_attention_num_buckets,
        )

        # Compute and extend attention mask.
        attention_mask = np.expand_dims(mask, (1, 2)).astype(
            self.pipeline_config.dtype.to_numpy()
        )

        return (
            Tensor.from_numpy(next_tokens_batch).to(
                self.pipeline_config.device
            ),
            Tensor.from_numpy(attention_mask).to(self.pipeline_config.device),
            Tensor.from_numpy(position_ids).to(self.pipeline_config.device),
            Tensor.from_numpy(relative_position_bucket).to(
                self.pipeline_config.device
            ),
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
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


def compute_relative_position_bucket(
    position_ids: np.ndarray, num_buckets=32, max_distance=128
):
    context_position = position_ids[:, :, None]
    memory_position = position_ids[:, None, :]
    relative_position = memory_position - context_position

    ret = 0
    n = -relative_position

    num_buckets //= 2
    ret += (n < 0).astype(np.int64) * num_buckets
    n = np.abs(n)

    max_exact = num_buckets // 2
    is_small = n < max_exact

    val_if_large = max_exact + (
        np.log(n.astype(np.float32) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(np.int64)

    val_if_large = np.minimum(
        val_if_large, np.full_like(val_if_large, num_buckets - 1)
    )
    return ret + np.where(is_small, n, val_if_large)
