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
"""All configurable parameters for Llama3."""

from max.pipelines import HuggingFaceFile, SupportedEncoding


def get_llama_huggingface_file(
    version: str, encoding: SupportedEncoding
) -> HuggingFaceFile:
    if version == "3":
        if encoding == SupportedEncoding.bfloat16:
            return HuggingFaceFile(
                "modularai/llama-3",
                "llama-3-8b-instruct-bf16.gguf",
            )
        elif encoding == SupportedEncoding.float32:
            return HuggingFaceFile(
                "modularai/llama-3",
                "llama-3-8b-f32.gguf",
            )
        elif encoding == SupportedEncoding.q4_k:
            return HuggingFaceFile(
                "modularai/llama-3",
                "llama-3-8b-instruct-q4_k_m.gguf",
            )
        elif encoding == SupportedEncoding.q4_0:
            return HuggingFaceFile(
                "modularai/llama-3",
                "llama-3-8b-instruct-q4_0.gguf",
            )
        elif encoding == SupportedEncoding.q6_k:
            return HuggingFaceFile(
                "modularai/llama-3",
                "llama-3-8b-instruct-q6_k.gguf",
            )
        else:
            raise ValueError(
                f"encoding does not have default hf file: {encoding}"
            )

    elif version == "3.1":
        if encoding == SupportedEncoding.bfloat16:
            return HuggingFaceFile(
                "modularai/llama-3.1",
                "llama-3.1-8b-instruct-bf16.gguf",
            )
        elif encoding == SupportedEncoding.float32:
            return HuggingFaceFile(
                "modularai/llama-3.1",
                "llama-3.1-8b-instruct-f32.gguf",
            )
        elif encoding == SupportedEncoding.q4_k:
            return HuggingFaceFile(
                "modularai/llama-3.1",
                "llama-3.1-8b-instruct-q4_k_m.gguf",
            )
        elif encoding == SupportedEncoding.q4_0:
            return HuggingFaceFile(
                "modularai/llama-3.1",
                "llama-3.1-8b-instruct-q4_0.gguf",
            )
        elif encoding == SupportedEncoding.q6_k:
            return HuggingFaceFile(
                "modularai/llama-3.1",
                "llama-3.1-8b-instruct-q6_k.gguf",
            )
        else:
            raise ValueError(f"encoding does not hf file: {encoding}")

    else:
        raise ValueError(f"version {version} not supported for llama")
