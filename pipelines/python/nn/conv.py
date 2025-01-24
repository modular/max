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
from typing import Tuple, Union

from max.graph import TensorValue, TensorValueLike, Weight, ops

from .layer import Layer


@dataclass
class Conv2D(Layer):
    """A 2D convolution over an input signal composed of several input
    planes.
    """

    filter: TensorValueLike

    stride: Union[int, Tuple[int, int]] = (1, 1)
    padding: Union[int, Tuple[int, int, int, int]] = (0, 0, 0, 0)
    dilation: Union[int, Tuple[int, int]] = (1, 1)
    groups: int = 1
    bias: bool = False

    def __call__(self, x: TensorValue) -> TensorValue:
        # These need to be casted as the underlying ops.conv2d call
        # expects them to only be tuple types.
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)

        if isinstance(self.padding, int):
            self.padding = (
                self.padding,
                self.padding,
                self.padding,
                self.padding,
            )

        if isinstance(self.dilation, int):
            self.dilation = (self.dilation, self.dilation)

        if (
            isinstance(self.filter, Weight)
            and self.filter.quantization_encoding is not None
        ):
            return ops.conv2d(
                x,
                self.filter.quantization_encoding,  # type: ignore
                self.stride,
                self.dilation,
                self.padding,
                self.groups,
            )
        return ops.conv2d(
            x,
            self.filter,
            self.stride,
            self.dilation,
            self.padding,
            self.groups,
        )
