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

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from typing import Any, Callable, Dict, Iterable, Tuple

from max.graph import DeviceRef, TensorValue, Weight

from ._identity import IdentitySet


@dataclass
class ShardingStrategy:
    """Defines how to load and shard a weight onto multiple devices."""

    host_device: DeviceRef
    shard_value: Callable[[Weight], tuple[TensorValue, ...]]


class Layer:
    """Base Layer class.

    Currently, only functionality is for adding hooks to the call function of
    each layer to support testing, debugging or profiling.
    """

    def __init_subclass__(cls):
        if cls.__name__ == "LayerV2":
            # LayerV2 subclasses Layer, but we don't want to apply
            # _call_with_hooks to it.
            return
        # Check `__dict__` instead of `hasattr` because `hasattr` passes on
        # subclasses that don't implement the method.
        if "__call__" in cls.__dict__:
            setattr(cls, "__call__", _call_with_hooks(cls.__dict__["__call__"]))


class LayerV2(Layer, ABC):
    """(new) Base class for model layers with weight and device management.

    This will be merged with the above class once all layers have been moved to
    V2.
    """

    def __init__(self):
        self._sublayers: dict[str, LayerV2] = {}
        self._layer_weights: dict[str, Weight] = {}

        # Update device list and sharding strategy to a singular object
        # similar to a partition spec.
        self._devices: tuple[DeviceRef, ...] = ()
        self._sharding_strategy: ShardingStrategy | None = None

    def __setattr__(self, name, value):
        if isinstance(value, LayerV2):
            self._sublayers[name] = value
            if self._devices or self._sharding_strategy:
                value.to(
                    *self._devices, sharding_strategy=self._sharding_strategy
                )

        elif isinstance(value, Weight):
            self._layer_weights[name] = value

    def to(
        self,
        *devices: DeviceRef,
        sharding_strategy: ShardingStrategy | None = None,
    ) -> None:
        if len(self._devices) > 1 and not sharding_strategy:
            raise ValueError(
                "Must provide a sharding strategy if multiple "
                " devices are provided."
            )

        for _, layer in recursive_named_layers(self):
            layer._devices = devices
            layer._sharding_strategy = sharding_strategy

    @property
    def layer_weights(self) -> dict[str, Weight]:
        return self._layer_weights

    @property
    def sublayers(self) -> dict[str, LayerV2]:
        return self._sublayers

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Defines the forward function of this layer.

        Subclasses must override this function. There is no exact signature that a
        call function must follow, but inputs/outputs should generally be
        `max.graph.TensorValue`. Non-`TensorValue` inputs are fine, but
        cannot be updated once the graph is built.
        """


def recursive_named_layers(parent: LayerV2) -> Iterable[tuple[str, LayerV2]]:
    """Recursively walks through the layers and generates names."""
    seen = IdentitySet()
    queue: deque[tuple[str, LayerV2]] = deque()
    queue.append(("", parent))

    while queue:
        name, layer = queue.popleft()
        if layer in seen:
            continue
        seen.add(layer)

        yield (name, layer)
        prefix = f"{name}." if name else ""
        for local_name, layer in layer.sublayers.items():
            queue.append((f"{prefix}{local_name}", layer))


_LOCAL = threading.local()
_LAYER_HOOKS = _LOCAL._layer_hooks = []


def add_layer_hook(
    fn: Callable[[Layer, Tuple[Any, ...], Dict[str, Any], Any], Any],
) -> None:
    """Adds a hook to call a function after each layer's `__call__`.

    The function will be passed four inputs: the layer, input_args,
    input_kwargs and outputs. The function can either return `None` or new
    outputs that will replace the layer returned outputs.

    Note that input and outputs contain graph Values, which show limited
    information (like shape and dtype). You can still see the computed values
    if you include the Value in the `graph.output` op, or call `value.print`.

    Example of printing debug inputs:

    .. code-block:: python

        def print_info(layer, args, kwargs, outputs):
            print("Layer:", type(layer).__name__)
            print("Input args:", args)
            print("Input kwargs:", kwargs)
            print("Outputs:", outputs)
            return outputs

        add_layer_hook(print_info)
    """
    _LAYER_HOOKS.append(fn)


def clear_hooks():
    """Remove all hooks."""
    _LAYER_HOOKS.clear()


def _call_with_hooks(call_fn):
    @wraps(call_fn)
    def __call_with_hooks(layer, *args, **kwargs):
        # Hide this wrapper from rich traceback.
        _rich_traceback_omit = True

        outputs = call_fn(layer, *args, **kwargs)
        # Use the inspect lib to ensure that args and kwargs are passed
        # to the hook as defined in the function signature.
        bound_args = signature(call_fn).bind(layer, *args, **kwargs)
        for hook in _LAYER_HOOKS:
            # Call the hook. Note that the first argument in `bound_args.args`
            # is the layer, so it is skipped.
            hook_outputs = hook(
                layer, bound_args.args[1:], bound_args.kwargs, outputs
            )
            if hook_outputs is not None:
                outputs = hook_outputs
        return outputs

    return __call_with_hooks
