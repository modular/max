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

from .llama_vision import LlamaVision

llama_vision_arch = SupportedArchitecture(
    name="MllamaForConditionalGeneration",
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: [KVCacheStrategy.CONTINUOUS]
    },
    versions=[
        SupportedVersion(
            name="3.2",
            encodings={
                SupportedEncoding.bfloat16: (
                    [
                        HuggingFaceFile(
                            "meta-llama/Llama-3.2-11B-Vision-Instruct",
                            "model-00001-of-00005.safetensors",
                        ),
                        HuggingFaceFile(
                            "meta-llama/Llama-3.2-11B-Vision-Instruct",
                            "model-00002-of-00005.safetensors",
                        ),
                        HuggingFaceFile(
                            "meta-llama/Llama-3.2-11B-Vision-Instruct",
                            "model-00003-of-00005.safetensors",
                        ),
                        HuggingFaceFile(
                            "meta-llama/Llama-3.2-11B-Vision-Instruct",
                            "model-00004-of-00005.safetensors",
                        ),
                        HuggingFaceFile(
                            "meta-llama/Llama-3.2-11B-Vision-Instruct",
                            "model-00005-of-00005.safetensors",
                        ),
                    ],
                    [KVCacheStrategy.CONTINUOUS],
                )
            },
            default_encoding=SupportedEncoding.bfloat16,
        )
    ],
    default_version="3.2",
    pipeline_model=LlamaVision,
    tokenizer=TextAndVisionTokenizer,
    default_weights_format=WeightsFormat.safetensors,
)
