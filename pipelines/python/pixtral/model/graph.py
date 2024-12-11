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

from max.dtype import DType
from max.graph import Graph, TensorType
from max.graph.weights import SafetensorWeights
from max.pipelines import PipelineConfig
from max.pipelines.kv_cache import KVCacheManager, KVCacheParams

# from mistral.model.graph import _transformer
from nn import Linear

from ..llava.llava import LlavaConditionalGeneration
from ..llava.llava_projector import LlavaMultiModalConnector
from ..vision_encoder.graph import _vision_encoder
from .mistral_graph import _transformer


def _linear(
    dtype: DType,
    in_features: int,
    out_features: int,
    weights: SafetensorWeights,
) -> Linear:
    """Unlike the vision encoder's version, this linear layer has a bias.
    This linear layer is used by the LlavaMultiModalConnector
    """
    # TODO: How to init bias?
    return Linear(
        weights.weight.allocate(dtype, [in_features, out_features], None),
        bias=weights.bias.allocate(dtype, [in_features], None),
    )


def _multi_modal_projector(
    dtype: DType,
    params: PipelineConfig,
    weights: SafetensorWeights,
) -> LlavaMultiModalConnector:
    """Connects the vision encoder to the text decoder.
    This MLP projects the patch embeddings to the text-encoder's embeddings space.
    Input shape:
    Output shape:
    """
    return LlavaMultiModalConnector(
        _linear(
            dtype,
            params.huggingface_config.text_config.hidden_size,
            params.huggingface_config.vision_config.hidden_size,
            weights.linear_1,
        ),
        _linear(
            dtype,
            params.huggingface_config.text_config.hidden_size,
            params.huggingface_config.text_config.hidden_size,
            weights.linear_2,
        ),
    )


def _llava(
    graph: Graph,
    params: PipelineConfig,
    weights: SafetensorWeights,
    kv_params: KVCacheParams,
) -> LlavaConditionalGeneration:
    # params for vision encoder in pixtral config.json are under vision_config.
    # vision encoder params missing from pixtral config.json:
    # num_attention_heads, num_channels, hidden_size, attention_dropout, intermediate_size, num_hidden_layers

    vision_encoder = _vision_encoder(graph, params, weights)
    multi_modal_projector = _multi_modal_projector(
        params.dtype, params, weights.multi_modal_projector
    )
    # Weights of pixtral have the same names and shapes as weights of mistral.
    language_model = _transformer(graph, params, weights, kv_params)

    return LlavaConditionalGeneration(
        vision_encoder=vision_encoder,
        multi_modal_projector=multi_modal_projector,
        language_model=language_model,
        vocab_size=params.huggingface_config.text_config.vocab_size,
        image_token_index=params.huggingface_config.image_token_index,
        vision_feature_layer=params.huggingface_config.vision_feature_layer,
        vision_feature_select_strategy=params.huggingface_config.vision_feature_select_strategy,
        image_seq_length=params.huggingface_config.image_seq_length,
    )


def _build_graph(
    params: PipelineConfig,
    weights: SafetensorWeights,
    kv_params: KVCacheParams,
    kv_manager: KVCacheManager,
) -> Graph:
    # TODO: Make this work for multiple devices. Now getting the types for device [0]
    kv_cache_types = kv_manager.input_symbols()[0]

    # TODO: Do we need text_token_type and input_row_offsets_type from mistral?
    input_ids_type = TensorType(
        # DType.int64, total_seq_len=sum(len(batch) for batch in input_ids)
        DType.int64,
        ["total_seq_len"],
    )
    # TODO: should be changed to add "batch_size", "n_images" dims when working with multiple images
    pixel_values_type = TensorType(
        DType.bfloat16,
        [304, 400, 3],  # ["height", "width", "num_channels"]
    )
    # Type of start and end position of each batch in the combined total_seq_len dimension.
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"]
    )

    # TODO: Use symbolic dims.
    # Initialize Graph.
    with Graph(
        "pixtral",
        input_types=[
            input_ids_type,
            pixel_values_type,
            input_row_offsets_type,
            *kv_cache_types,
        ],
    ) as graph:
        model = _llava(graph, params, weights, kv_params)
        input_ids, pixel_values, input_row_offsets, *kv_cache_inputs = (
            graph.inputs
        )
        logits = model(
            input_ids=input_ids,  # type: ignore
            pixel_values=pixel_values,  # type: ignore
            kv_cache_inputs=kv_cache_inputs,  # type: ignore
            input_row_offsets=input_row_offsets,
        )
        graph.output(logits)
        return graph
