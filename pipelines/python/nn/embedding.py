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

from dataclasses import dataclass

from max.graph import TensorValue, TensorValueLike, Weight, ops

from .layer import Layer


@dataclass
class Embedding(Layer):
    weights: TensorValueLike

    def __call__(self, indices: TensorValueLike) -> TensorValue:
        result = ops.gather(self.weights, indices, axis=0)
        if (
            isinstance(self.weights, Weight)
            and self.weights.quantization_encoding is not None
        ):
            result = ops.dequantize(self.weights.quantization_encoding, result)
        return result
