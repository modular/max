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

from dataclasses import dataclass
from typing import Union

from max.graph import TensorValue, TensorValueLike, ops
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheParams,
)
from max.pipelines.nn import (
    Conv1D,
    Embedding,
    LPLayerNorm,
    TransformerBlock,
)
from max.pipelines.nn.layer import Layer


@dataclass
class WhisperEncoder(Layer):
    """A Transformer consisting of a stem, positional embeddings, and self attention layers.

    The differences between this transformer and `nn.Transformer` are:
        1. Whisper uses this transformer to pass the input through a stem of:
        Two convolution layers with a filter width of 3 and the GELU activation
        function where the second convolution layer has a stride of two.

        2. After that, Sinusoidal position embeddings are then added to the output of the stem.

        After that, the usual Transformer blocks (with pre-activation residual blocks) are applied.

        3. No final linear layer "output".
    """

    conv1: Conv1D
    conv2: Conv1D
    embed_positions: Embedding
    layers: list[TransformerBlock]
    norm: LPLayerNorm  # TODO: Is LayerNorm here not the same as nn.LayerNorm

    kv_params: KVCacheParams
    kv_collection_constructor: Union[
        FetchContinuousBatchingKVCacheCollection, FetchPagedKVCacheCollection
    ]
    all_logits: bool = False

    def __call__(
        self,
        input_features: TensorValueLike,
        kv_cache_inputs: tuple[
            TensorValue, TensorValue, TensorValue, TensorValue
        ],
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        """
        Args:
            input_features: Tensor of shape (batch_size, feature_size, sequence_length)
            expected_seq_length = config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]

        """
        # Encoder stem: two convolution layers and the GELU activation function.
        inputs_embeds = ops.gelu(self.conv1(input_features))
        inputs_embeds = ops.gelu(self.conv2(inputs_embeds))

        # self.embed_positions.weights layers is of shape = (1500, 1280)
        # TODO: Do we need the reshape to (batch_size, sequence_length, feature_size) or is it already in the right shape?
        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        # Add sinusoidal position embeddings to the output of the stem
        h = inputs_embeds + self.embed_positions.weights

        kv_collection = self.kv_collection_constructor(*kv_cache_inputs)

        for _, layer in enumerate(self.layers):
            h = layer(h, kv_collection, **kwargs)

        # # A final layer normalization is applied to the encoder output
        normalized = self.norm(h)

        return normalized
