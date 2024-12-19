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

from max.pipelines import (
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
    WeightsFormat,
)
from max.pipelines.kv_cache import KVCacheStrategy

from .model import Llama3Model
from .safetensor_converter import LlamaSafetensorWeights

llama_arch = SupportedArchitecture(
    name="LlamaForCausalLM",
    default_encoding=SupportedEncoding.q4_k,
    supported_encodings={
        SupportedEncoding.q4_k: [KVCacheStrategy.NAIVE],
        SupportedEncoding.q4_0: [KVCacheStrategy.NAIVE],
        SupportedEncoding.q6_k: [KVCacheStrategy.NAIVE],
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
    pipeline_model=Llama3Model,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.gguf,
    weight_converters={WeightsFormat.safetensors: LlamaSafetensorWeights},
)
