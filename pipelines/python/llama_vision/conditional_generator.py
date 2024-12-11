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
"""The Llama 3.2 model which consists of a vision encoder and a language model."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import Dim, TensorValue, ops
from max.pipelines import PipelineConfig
from nn import Linear
from nn.layer import Layer

from .language_model import CausalLanguageModel
from .vision_model import VisionModel


@dataclass
class ConditionalGenerator(Layer):
    """
    The Llama model which consists of a vision encoder and a language model.
    """

    pipeline_config: PipelineConfig
    vision_model: VisionModel
    multi_modal_projector: Linear
    language_model: CausalLanguageModel

    def __call__(
        self,
        pixel_values: TensorValue,
        aspect_ratio_ids: TensorValue,
        aspect_ratio_mask: TensorValue,
        input_ids: TensorValue,
        hidden_input_row_offsets: TensorValue,
        cross_input_row_offsets: TensorValue,
        kv_cache_inputs: tuple[TensorValue, ...],
    ) -> TensorValue:
        if pixel_values is not None:
            if aspect_ratio_ids is None:
                msg = (
                    "`aspect_ratio_ids` must be provided if `pixel_values` is "
                    "provided"
                )
                raise ValueError(msg)

            # Get vision tokens from vision model.
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
            )
            cross_attention_states = vision_outputs[0]

            num_patches = cross_attention_states.shape[-2]

            cross_attention_states = self.multi_modal_projector(
                cross_attention_states
            ).reshape(
                [
                    Dim("batch_size")
                    * Dim("num_concurrent_media")
                    * self.pipeline_config.huggingface_config.vision_config.max_num_tiles
                    * num_patches,
                    self.pipeline_config.huggingface_config.text_config.hidden_size,
                ]
            )

        logits = self.language_model(
            kv_cache_inputs=kv_cache_inputs,
            input_ids=input_ids,
            hidden_input_row_offsets=hidden_input_row_offsets,
            cross_attention_states=cross_attention_states,
            cross_input_row_offsets=cross_input_row_offsets,
        )
        # Always return float32 logits, no matter the activation type
        return ops.cast(logits, DType.float32)
