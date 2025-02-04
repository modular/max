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
"""A General sequential layer, each layer is executed with the outputs of the previous."""

from collections.abc import Callable
from dataclasses import dataclass

from max.graph import TensorValue

from .layer import Layer


@dataclass
class Sequential(Layer):
    layers: list[Callable[..., TensorValue]]

    def __post_init__(self):
        if len(self.layers) == 0:
            raise ValueError(
                "more than one layer must be provided to sequential."
            )

    def __call__(self, *args, **kwargs) -> TensorValue:
        x = self.layers[0](*args, **kwargs)
        for layer in self.layers[1:]:
            x = layer(x)

        return x
