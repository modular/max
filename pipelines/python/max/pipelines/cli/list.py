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
"""Utilities for exploring supported pipelines."""

from max.pipelines import PIPELINE_REGISTRY


def list_pipelines_to_console():
    print()
    for arch in PIPELINE_REGISTRY.architectures.values():
        print()
        print(f"    Architecture: {arch.name}")
        print()
        print("         Example Huggingface Repo Ids: ")
        for repo_id in arch.example_repo_ids:
            print(f"              {repo_id}")

        print()
        for (
            encoding_name,
            kv_cache_strategies,
        ) in arch.supported_encodings.items():
            print(
                f"         Encoding Supported: {encoding_name}, with Cache Strategies: {kv_cache_strategies}"
            )

    print()
