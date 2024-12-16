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
    HuggingFaceFile,
    SupportedArchitecture,
    SupportedEncoding,
    SupportedVersion,
    TextTokenizer,
    WeightsFormat,
)
from max.pipelines.kv_cache import KVCacheStrategy

from .model import MistralModel

mistral_arch = SupportedArchitecture(
    name="MistralForCausalLM",
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: [KVCacheStrategy.CONTINUOUS]
    },
    versions=[
        SupportedVersion(
            name="default",
            encodings={
                SupportedEncoding.bfloat16: (
                    [
                        HuggingFaceFile(
                            "mistralai/Mistral-Nemo-Instruct-2407", f
                        )
                        for f in [
                            "model-00001-of-00005.safetensors",
                            "model-00002-of-00005.safetensors",
                            "model-00003-of-00005.safetensors",
                            "model-00004-of-00005.safetensors",
                            "model-00005-of-00005.safetensors",
                        ]
                    ],
                    [KVCacheStrategy.CONTINUOUS],
                )
            },
            default_encoding=SupportedEncoding.bfloat16,
        )
    ],
    default_version="default",
    pipeline_model=MistralModel,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
)
