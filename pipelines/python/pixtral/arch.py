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
    TextAndVisionTokenizer,
    WeightsFormat,
)
from max.pipelines.kv_cache import KVCacheStrategy

from .pixtral import PixtralModel

pixtral_arch = SupportedArchitecture(
    name="LlavaForConditionalGeneration",
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
                            "mistral-community/pixtral-12b",
                            "model-00001-of-00006.safetensors",
                        ),
                        HuggingFaceFile(
                            "mistral-community/pixtral-12b",
                            "model-00002-of-00006.safetensors",
                        ),
                        HuggingFaceFile(
                            "mistral-community/pixtral-12b",
                            "model-00003-of-00006.safetensors",
                        ),
                        HuggingFaceFile(
                            "mistral-community/pixtral-12b",
                            "model-00004-of-00006.safetensors",
                        ),
                        HuggingFaceFile(
                            "mistral-community/pixtral-12b",
                            "model-00005-of-00006.safetensors",
                        ),
                        HuggingFaceFile(
                            "mistral-community/pixtral-12b",
                            "model-00006-of-00006.safetensors",
                        ),
                    ],
                    [KVCacheStrategy.CONTINUOUS],
                )
            },
            default_encoding=SupportedEncoding.bfloat16,
        )
    ],
    default_version="default",
    pipeline_model=PixtralModel,
    tokenizer=TextAndVisionTokenizer,
    default_weights_format=WeightsFormat.safetensors,
)
