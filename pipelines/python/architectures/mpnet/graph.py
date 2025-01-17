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

import math

from max.dtype import DType
from max.graph import Graph, TensorType, TensorValue, ops
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import Weights
from max.pipelines import PipelineConfig
from nn import (
    Embedding,
    Linear,
    LPLayerNorm,
    Sequential,
)
from nn.layer import Layer


def _quantization_encoding(
    pipeline_config: PipelineConfig,
) -> QuantizationEncoding | None:
    if supported_encoding := pipeline_config.quantization_encoding:
        return supported_encoding.quantization_encoding
    return None


class MPNetEmbeddings(Layer):
    """An embeddings layer that combines the tokens embeddings and positions
    embeddings."""

    def __init__(self, pipeline_config: PipelineConfig, weights: Weights):
        config = pipeline_config.huggingface_config
        self.word_embeddings = Embedding(
            weights.word_embeddings.weight.allocate(
                pipeline_config.dtype,
                [
                    config.vocab_size,
                    config.hidden_size,
                ],
                _quantization_encoding(pipeline_config),
            )
        )
        self.position_embeddings = Embedding(
            weights.position_embeddings.weight.allocate(
                pipeline_config.dtype,
                [
                    config.max_position_embeddings,
                    config.hidden_size,
                ],
            )
        )
        self.layer_norm = LPLayerNorm(
            weight=weights.LayerNorm.weight.allocate(
                pipeline_config.dtype,
                [config.hidden_size],
            ),
            bias=weights.LayerNorm.bias.allocate(
                pipeline_config.dtype, [config.hidden_size]
            ),
            eps=config.layer_norm_eps,
        )
        self.position_ids = weights.position_ids.allocate(
            DType.int64,
            [
                1,
                config.max_position_embeddings,
            ],
        )

    def __call__(
        self, input_ids: TensorValue, position_ids: TensorValue
    ) -> TensorValue:
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        return self.layer_norm(embeddings)


class MPNetSelfAttention(Layer):
    """Self-attention layer with position compensation."""

    def __init__(self, pipeline_config: PipelineConfig, weights: Weights):
        config = pipeline_config.huggingface_config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.q = Linear(
            weights.q.weight.allocate(
                pipeline_config.dtype,
                [self.all_head_size, config.hidden_size],
                _quantization_encoding(pipeline_config),
            ),
            bias=weights.q.bias.allocate(
                pipeline_config.dtype,
                [self.all_head_size],
                _quantization_encoding(pipeline_config),
            ),
        )
        self.k = Linear(
            weights.k.weight.allocate(
                pipeline_config.dtype,
                [self.all_head_size, config.hidden_size],
                _quantization_encoding(pipeline_config),
            ),
            bias=weights.k.bias.allocate(
                pipeline_config.dtype,
                [self.all_head_size],
                _quantization_encoding(pipeline_config),
            ),
        )
        self.v = Linear(
            weights.v.weight.allocate(
                pipeline_config.dtype,
                [self.all_head_size, config.hidden_size],
                _quantization_encoding(pipeline_config),
            ),
            bias=weights.v.bias.allocate(
                pipeline_config.dtype,
                [self.all_head_size],
                _quantization_encoding(pipeline_config),
            ),
        )
        self.o = Linear(
            weights.o.weight.allocate(
                pipeline_config.dtype,
                [config.hidden_size, config.hidden_size],
                _quantization_encoding(pipeline_config),
            ),
            bias=weights.o.bias.allocate(
                pipeline_config.dtype,
                [config.hidden_size],
                _quantization_encoding(pipeline_config),
            ),
        )

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [
            self.num_attention_heads,
            self.attention_head_size,
        ]
        x = ops.reshape(x, new_x_shape)
        return ops.permute(x, [0, 2, 1, 3])

    def __call__(
        self,
        hidden_states,
        attention_mask: TensorValue,
        position_bias: TensorValue,
    ):
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = q @ k.transpose(-1, -2)
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size
        )

        # Apply relative position embedding (precomputed in MPNetEncoder).
        attention_scores += position_bias

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # TODO: Check that ops.softmax(v) == nn.functional.softmax(v, dim=-1)
        attention_probs = ops.softmax(attention_scores)

        c = attention_probs @ v

        c = ops.permute(c, [0, 2, 1, 3])
        new_c_shape = c.shape[:-2] + [self.all_head_size]
        c = ops.reshape(c, new_c_shape)

        return self.o(c)


class MPNetAttention(Layer):
    """Container for the attention and attention output layer norm layers."""

    def __init__(self, pipeline_config: PipelineConfig, weights: Weights):
        config = pipeline_config.huggingface_config
        self.attn = MPNetSelfAttention(pipeline_config, weights.attn)
        self.layer_norm = LPLayerNorm(
            weight=weights.LayerNorm.weight.allocate(
                DType.float32, [config.hidden_size]
            ),
            bias=weights.LayerNorm.bias.allocate(
                DType.float32, [config.hidden_size]
            ),
            eps=config.layer_norm_eps,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue,
        position_bias: TensorValue,
    ):
        attn_output = self.attn(
            hidden_states,
            attention_mask,
            position_bias,
        )
        return self.layer_norm(attn_output + hidden_states)


_ACTIVATIONS = {
    "gelu": ops.gelu,
    "relu": ops.relu,
    "silu": ops.silu,
    "sigmoid": ops.sigmoid,
    "tanh": ops.tanh,
}


class MPNetIntermediate(Layer):
    """Fully connected layer with an activation function."""

    def __init__(self, pipeline_config: PipelineConfig, weights: Weights):
        config = pipeline_config.huggingface_config
        self.dense = Linear(
            weights.dense.weight.allocate(
                pipeline_config.dtype,
                [config.intermediate_size, config.hidden_size],
                _quantization_encoding(pipeline_config),
            ),
            bias=weights.dense.bias.allocate(
                pipeline_config.dtype,
                [config.intermediate_size],
                _quantization_encoding(pipeline_config),
            ),
        )
        self.intermediate_act_fn = _ACTIVATIONS[config.hidden_act]

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class MPNetOutput(Layer):
    """Layer that combines the outputs of the intermediate and attention layers."""

    def __init__(self, pipeline_config: PipelineConfig, weights: Weights):
        config = pipeline_config.huggingface_config
        self.dense = Linear(
            weights.dense.weight.allocate(
                pipeline_config.dtype,
                [config.hidden_size, config.intermediate_size],
                _quantization_encoding(pipeline_config),
            ),
            bias=weights.dense.bias.allocate(
                pipeline_config.dtype,
                [config.hidden_size],
                _quantization_encoding(pipeline_config),
            ),
        )
        self.layer_norm = LPLayerNorm(
            weight=weights.LayerNorm.weight.allocate(
                DType.float32, [config.hidden_size]
            ),
            bias=weights.LayerNorm.bias.allocate(
                DType.float32, [config.hidden_size]
            ),
            eps=config.layer_norm_eps,
        )

    def __call__(
        self, hidden_states: TensorValue, input_tensor: TensorValue
    ) -> TensorValue:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class MPNetLayer(Layer):
    """An Encoder layer block."""

    def __init__(self, pipeline_config: PipelineConfig, weights: Weights):
        self.attention = MPNetAttention(pipeline_config, weights.attention)
        self.intermediate = MPNetIntermediate(
            pipeline_config, weights.intermediate
        )
        self.output = MPNetOutput(pipeline_config, weights.output)

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue,
        position_bias: TensorValue,
    ) -> TensorValue:
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            position_bias=position_bias,
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class MPNetEncoder(Layer):
    """Encoder that contains stacks of MPNetLayers."""

    def __init__(self, pipeline_config: PipelineConfig, weights: Weights):
        config = pipeline_config.huggingface_config
        self.n_heads = config.num_attention_heads
        num_hidden_layers = config.num_hidden_layers
        self.layer = Sequential(
            [
                MPNetLayer(pipeline_config, weights.layer[n])
                for n in range(num_hidden_layers)
            ]
        )
        self.relative_attention_bias = Embedding(
            weights.relative_attention_bias.weight.allocate(
                pipeline_config.dtype,
                [
                    config.relative_attention_num_buckets,
                    config.num_attention_heads,
                ],
            )
        )
        self.num_attention_heads = config.num_attention_heads

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue,
        relative_position_bucket: TensorValue,
    ):
        position_bias = self.compute_position_bias(
            hidden_states, relative_position_bucket
        )
        for layer in self.layer.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask,
                position_bias,
            )
        return hidden_states

    def compute_position_bias(
        self, hidden_states, relative_position_bucket
    ) -> TensorValue:
        shape = hidden_states.shape
        bsz, qlen, klen = shape[0], shape[1], shape[1]
        values = self.relative_attention_bias(relative_position_bucket)
        values = ops.permute(values, [0, 3, 1, 2])
        values = ops.broadcast_to(
            values,
            [bsz, self.num_attention_heads, qlen, klen],
        )
        return values


class MPNetModel(Layer):
    """The MPNet encoder model.

    Based on the MPNetModel transformers implementation."""

    def __init__(self, pipeline_config: PipelineConfig, weights: Weights):
        self.embeddings = MPNetEmbeddings(pipeline_config, weights.embeddings)
        self.encoder = MPNetEncoder(pipeline_config, weights.encoder)
        # This model doesn't contain a pooler, since the pooled outputs
        # are not used.

    def __call__(
        self,
        input_ids: TensorValue,
        attention_mask: TensorValue,
        position_ids: TensorValue,
        relative_position_bucket: TensorValue,
    ) -> TensorValue:
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
        )
        return self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            relative_position_bucket=relative_position_bucket,
        )


def build_graph(
    pipeline_config: PipelineConfig,
    weights: Weights,
) -> Graph:
    # Graph input types.
    input_ids_type = TensorType(DType.int64, shape=["batch_size", "seq_len"])
    attention_mask_type = TensorType(
        pipeline_config.dtype, shape=["batch_size", 1, 1, "seq_len"]
    )
    position_ids_type = TensorType(DType.int64, shape=["batch_size", "seq_len"])
    relative_position_bucket_type = TensorType(
        DType.int64,
        shape=["batch_size", "seq_len", "seq_len"],
    )

    # Initialize Graph.
    return Graph(
        "mpnet",
        MPNetModel(pipeline_config, weights),
        input_types=[
            input_ids_type,
            attention_mask_type,
            position_ids_type,
            relative_position_bucket_type,
        ],
    )
