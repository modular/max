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

from max.graph.weights import WeightsFormat
from max.pipelines import (
    PipelineTask,
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)
from max.pipelines.kv_cache import KVCacheStrategy

from ..llama3 import weight_adapters
from .model import GraniteModel

granite_arch = SupportedArchitecture(
    name="GraniteForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "ibm-granite/granite-3.1-8b-instruct",
        "ibm-granite/granite-3.1-8b-base",
    ],
    default_weights_format=WeightsFormat.gguf,
    default_encoding=SupportedEncoding.float32,
    supported_encodings={
        SupportedEncoding.float32: [
            KVCacheStrategy.PAGED,
            KVCacheStrategy.CONTINUOUS,
            KVCacheStrategy.NAIVE,
        ],
        SupportedEncoding.bfloat16: [
            KVCacheStrategy.PAGED,
            KVCacheStrategy.CONTINUOUS,
            KVCacheStrategy.NAIVE,
        ],
    },
    pipeline_model=GraniteModel,
    tokenizer=TextTokenizer,
    rope_type=RopeType.normal,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
)
