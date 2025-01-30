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

"""Multi-layer Perceptron."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, TensorValueLike, Weight, ops

from .kernels import swish_glu
from .layer import Layer


@dataclass
class Linear(Layer):
    """A fully connected layer."""

    weight: TensorValue
    bias: TensorValue | None = None

    def __call__(self, x: TensorValue) -> TensorValue:
        weight = self.weight
        if (
            isinstance(self.weight, Weight)
            and self.weight.quantization_encoding is not None
        ):
            res = ops.qmatmul(self.weight.quantization_encoding, x, weight)
            if self.bias is not None:
                res += self.bias
            return res

        res = x @ weight.T
        if self.bias is not None:
            res += self.bias
        return res


class LinearV2(Layer):
    """
    Applies a linear transformation to incoming data: :math:`y = xW^T + b`.

    This layer implements a fully connected layer where inputs are multiplied
    by a weight matrix and optionally added with a bias vector.
    Both weights and bias initially reside on CPU, and the model init phase
    moves them to :obj:`device`.

    Example:

    .. code-block:: python

        linear_layer = LinearV2(
            in_dim=256,
            out_dim=128,
            dtype=DType.float32,
            device=DeviceRef.GPU(),
            name="linear",
            has_bias=True
        )

        # Input tensor of shape: [batch, ..., 256]
        input_tensor: TensorValue
        output = linear_layer(input_tensor)
    """

    weight: Weight
    """The weight matrix stored on CPU with shape (out_dim, in_dim).
    Model init transposes the weight and moves it to :obj:`device`."""

    bias: Weight | None = None
    """The optional bias vector stored on CPU with shape (out_dim,).
    Model init moves the bias to :obj:`device` if present."""

    device: DeviceRef
    """The device where matrix operations are performed."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType,
        device: DeviceRef,
        name: str,
        has_bias: bool = False,
    ) -> None:
        """Initializes the linear layer with weights and optional bias.

        Args:
            in_dim: The dimensionality of the input space.
            out_dim: The dimensionality of the output space.
            dtype: The data type for both weights and bias.
            device: The target device for computation.
                Weights remain on CPU until moved during computation.
            name: Base name for weights (appended with ``.weight`` and
                ``.bias`` if applicable).
            has_bias: When :obj:`True`, adds a bias vector to the layer.
                Defaults to :obj:`False`.
        """
        super().__init__()

        self.weight = Weight(
            name=f"{name}.weight",
            dtype=dtype,
            shape=(out_dim, in_dim),
            device=DeviceRef.CPU(),
        )
        self.device = device

        if has_bias:
            self.bias = Weight(
                name=f"{name}.bias",
                dtype=dtype,
                shape=(out_dim,),
                device=DeviceRef.CPU(),
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Applies a linear transformation to the input data.

        Args:
            x: Input tensor of shape ``(..., in_dim)``.
                The last dimension must match the layer's ``in_dim``.
                The input tensor must reside on :obj:`device`.

        Returns:
            Output tensor of shape ``(..., out_dim)``.
            The result resides on the device specified in :obj:`device`.

        Raises:
            ValueError: If the last dimension of ``x`` doesn't match ``in_dim``.
        """
        weight = TensorValue(self.weight).to(self.device)

        res = x @ weight.T
        if self.bias is not None:
            res += TensorValue(self.bias).to(self.device)
        return res


@dataclass
class MLP(Layer):
    """
    Simple multi-layer perceptron composed of three linear layers.
    Uses SiLU activation function.
    """

    gate_proj: Linear
    down_proj: Linear
    up_proj: Linear

    def __call__(self, x: TensorValueLike) -> TensorValue:
        if (
            self.gate_proj.bias is None
            and self.up_proj.bias is None
            and TensorValue(x).rank == 2
            and TensorValue(x).device is not None
            and TensorValue(x).device != DeviceRef.CPU()
            and False  # GEX-1476: This causes elaboration errors - disable swish_glu pathway.
        ):
            return self.down_proj(
                swish_glu(
                    x,
                    self.gate_proj.weight,
                    self.up_proj.weight,
                )
            )

        return self.down_proj(ops.silu(self.gate_proj(x)) * self.up_proj(x))  # type: ignore


@dataclass
class DistributedMLP(Layer):
    list_of_mlps: list[MLP]
    num_devices: int

    def __call__(self, x: list[TensorValue]) -> list[TensorValue]:
        mlp_outs = [self.list_of_mlps[i](x[i]) for i in range(self.num_devices)]
        return ops.allreduce.sum(mlp_outs)  # type: ignore
