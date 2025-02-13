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

import math
from dataclasses import dataclass

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops
from max.pipelines.nn import (
    Conv1D,
    Embedding,
    LayerNorm,
    Linear,
    Sequential,
)
from max.pipelines.nn.layer import Layer


@dataclass
class WhisperSdpaAttention(Layer):
    n_heads: int
    head_dim: int

    wq: Linear
    wk: Linear
    wv: Linear
    wo: Linear

    def scaled_dot_product_attention(
        self,
        xq: TensorValueLike,
        xk: TensorValueLike,
        xv: TensorValueLike,
    ) -> TensorValue:
        xq = TensorValue(xq)
        xk = TensorValue(xk)
        xv = TensorValue(xv)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scale = math.sqrt(1.0 / self.head_dim)
        scores = xq @ ops.transpose(xk, 2, 3)
        # Note, the graph compiler currently requires the order of operands
        # to be `scores * scale` in order to pattern match the fused attention
        # operator.
        return ops.softmax(scores * scale) @ xv

    def __call__(
        self,
        x: TensorValue,
        **kwargs,
    ) -> TensorValue:
        """Computes attention on x.

        Args:
            x: Activations with shape (batch, seq_len, dim).

        Returns the result of WhisperSdpaAttention self attention on the input.
        """
        x = TensorValue(x)
        batch, seq_len = x.shape[0], x.shape[1]
        # matmul weights
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = ops.reshape(xq, [batch, seq_len, self.n_heads, self.head_dim])

        xk = ops.reshape(
            xk,
            [
                batch,
                seq_len,
                self.n_heads,
                self.head_dim,
            ],
        )
        xv = ops.reshape(
            xv,
            [
                batch,
                seq_len,
                self.n_heads,
                self.head_dim,
            ],
        )

        output = (
            self.scaled_dot_product_attention(xq, xk, xv)
            .transpose(1, 2)
            .reshape([batch, seq_len, -1])
        )
        return self.wo(output)


@dataclass
class WhisperEncoderLayer(Layer):
    """Stack of Attention, FeedForward, and LayerNorm layers."""

    attention: WhisperSdpaAttention
    mlp: Sequential
    attention_norm: LayerNorm
    mlp_norm: LayerNorm

    def __call__(
        self,
        x: TensorValue,
        **kwargs,
    ) -> TensorValue:
        attn_out = self.attention(
            self.attention_norm(x),
            **kwargs,
        )

        h = x + attn_out
        h = h + self.mlp(self.mlp_norm(h))

        return h


@dataclass
class WhisperEncoder(Layer):
    """A Transformer consisting of a stem, positional embeddings, and self attention layers.

    The differences between this transformer and `nn.Transformer` are:
        1. Whisper passes the input through a stem of:
        Two convolution layers with a filter width of 3 and the GELU activation
        function where the second convolution layer has a stride of two.

        2. After that, Sinusoidal position embeddings are then added to the output of the stem.

        After that, the usual Transformer blocks (with pre-activation residual blocks) are applied.
        We use naive attention where the linear projections have a bias.

        3. No final transformer linear layer "output".
    """

    conv1: Conv1D
    conv2: Conv1D
    embed_positions: Embedding
    layers: list[WhisperEncoderLayer]
    norm: LayerNorm  # TODO: Is LayerNorm here not the same as nn.LayerNorm

    all_logits: bool = False

    def __call__(
        self,
        input_features: TensorValueLike,
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
        # inputs_embeds = ops.permute(inputs_embeds, [0, 2, 1])

        # Add sinusoidal position embeddings to the output of the stem
        h = inputs_embeds + self.embed_positions.weights

        for _, layer in enumerate(self.layers):
            h = layer(h, **kwargs)

        # # A final layer normalization is applied to the encoder output
        normalized = self.norm(h)

        if "input_row_offsets" in kwargs:
            # Ragged inputs/activations
            last_indices = kwargs["input_row_offsets"][1:] - 1
            last_tokens = ops.gather(normalized, last_indices, axis=0)
        else:
            # Dense padded inputs/activations
            valid_lengths = kwargs["valid_lengths"]
            # TODO: Remove once `gather_nd` works with nonstatic last dims.
            indices = ops.unsqueeze(valid_lengths - 1, -1)
            last_tokens = ops.gather_nd(normalized, indices, batch_dims=1)

        # Always return float32 logits, no matter the activation type.
        last_token_logits = ops.cast(last_tokens, DType.float32)

        if self.all_logits:
            all_logits = ops.cast(normalized, DType.float32)
            return (last_token_logits, all_logits)

        return (last_token_logits,)
