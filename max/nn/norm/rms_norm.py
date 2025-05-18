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

"""Normalization layer."""

from dataclasses import dataclass

from max.dtype import DType
from max.graph import (
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    TensorValueLike,
    Weight,
    _ChainType,
    ops,
)

from ..layer import Layer, Module


@dataclass
class RMSNormV1(Layer):
    """Computes the Root Mean Square normalization on inputs.

    Deprecated: Use `RMSNorm` instead.
    """

    weight: TensorValueLike
    eps: float = 1e-6
    weight_offset: float = 0.0
    multiply_before_cast: bool = True

    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.custom(
            "rms_norm",
            [
                x,
                ops.cast(self.weight, x.dtype),
                ops.cast(self.eps, x.dtype),
                ops.constant(
                    self.weight_offset, dtype=x.dtype, device=DeviceRef.CPU()
                ),
            ],
            [TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
            parameters={"multiply_before_cast": self.multiply_before_cast},
        )[0].tensor


class RMSNorm(Module):
    """Computes the Root Mean Square normalization on inputs.

    Args:
        dim: Size of last dimension of the expected input.
        eps: Value added to denominator for numerical stability.
        weight_offset: Constant offset added to the learned weights at runtime.
            For Gemma-style RMSNorm, this should be set to 1.0.
        multiply_before_cast: True if we multiply the inputs by the learned
            weights before casting to the input type (Gemma3-style). False if we
            cast the inputs to the input type first, then multiply by the learned
            weights (Llama-style).
    """

    def __init__(
        self,
        dim: int,
        dtype: DType,
        eps: float = 1e-6,
        weight_offset: float = 0.0,
        multiply_before_cast: bool = True,
    ):
        super().__init__()
        self.weight = Weight("weight", dtype, [dim], device=DeviceRef.CPU())
        self.eps = eps
        self.weight_offset = weight_offset
        self.multiply_before_cast = multiply_before_cast

    def build_subgraph(self, name: str, x_type: TensorType) -> Module:
        weight_type = TensorType(
            dtype=x_type.dtype, shape=self.weight.shape, device=x_type.device
        )
        graph_inputs = [_ChainType(), x_type, weight_type]
        subgraph = Graph.current._subgraphs.get(name)

        if subgraph is None:
            with Graph.current.add_subgraph(
                name, input_types=graph_inputs
            ) as subgraph:
                subgraph._current_chain._mlir_value = subgraph.inputs[
                    0
                ]._mlir_value
                x = subgraph.inputs[1]
                weight = subgraph.inputs[2]

                result = ops.custom(
                    "rms_norm",
                    [
                        x,
                        weight,
                        ops.constant(
                            self.eps, dtype=x.dtype, device=DeviceRef.CPU()
                        ),
                        ops.constant(
                            self.weight_offset,
                            dtype=x.dtype,
                            device=DeviceRef.CPU(),
                        ),
                    ],
                    [TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
                )[0].tensor

                subgraph.output(subgraph._current_chain, result)

        outer_self = self

        class RMSNormSubgraph(Module):
            def __call__(self, x: TensorValueLike) -> TensorValue:
                x = TensorValue(x)
                weight: TensorValue = ops.cast(outer_self.weight, x.dtype)
                if x.device:
                    weight = weight.to(x.device)
                return ops.call(subgraph, x, weight)[0].tensor

        return RMSNormSubgraph()

    def __call__(self, x: TensorValue) -> TensorValue:
        weight: TensorValue = ops.cast(self.weight, x.dtype)
        if x.device:
            weight = weight.to(x.device)

        return ops.custom(
            "rms_norm",
            [
                x,
                weight,
                ops.constant(self.eps, dtype=x.dtype, device=DeviceRef.CPU()),
                ops.constant(
                    self.weight_offset, dtype=x.dtype, device=DeviceRef.CPU()
                ),
            ],
            [TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
            parameters={"multiply_before_cast": self.multiply_before_cast},
        )[0].tensor


class DistributedRMSNorm(RMSNorm):
    def __init__(self, *args, devices: list[DeviceRef], **kwargs):
        super().__init__(*args, **kwargs)
        self.num_devices = len(devices)

        clone_weight = lambda weight, i: weight
        self.weight.set_sharding_strategy(clone_weight)
        # Create a separate RMS layer for each device.
        self.rms_norms = []
        for n, device in enumerate(devices):
            layer = RMSNorm(*args, **kwargs)
            layer.weight = self.weight.shard(n, device)
            self.rms_norms.append(layer)

    def build_subgraph(self, name: str, x_type: list[TensorType]) -> Module:  # type: ignore[override]
        rms_norm_subgraphs = []
        for i, rms_norm in enumerate(self.rms_norms):
            rms_norm_subgraphs.append(
                rms_norm.build_subgraph(f"{name}_rms_norm_{i}", x_type[i])
            )

        class DistributedRMSNormSubgraph(Module):
            def __call__(self, x: list[TensorValue]) -> list[TensorValue]:
                rms_norm_outs = [
                    rms_norm(x[i])
                    for i, rms_norm in enumerate(rms_norm_subgraphs)
                ]
                return rms_norm_outs

        return DistributedRMSNormSubgraph()

    def __call__(self, xs: list[TensorValue]) -> list[TensorValue]:  # type: ignore[override]
        return [self.rms_norms[i](xs[i]) for i in range(self.num_devices)]
