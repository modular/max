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

import threading
from functools import wraps
from inspect import signature
from typing import Any, Callable, Dict, Tuple


class Layer:
    """Base Layer class.

    Currently, only functionality is for adding hooks to the call function of
    each layer to support testing, debugging or profiling.
    """

    def __init_subclass__(cls):
        # Check `__dict__` instead of `hasattr` because `hasattr` passes on
        # subclasses that don't implement the method.
        if "__call__" in cls.__dict__:
            setattr(cls, "__call__", _call_with_hooks(cls.__dict__["__call__"]))


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

    ```python
    def print_info(layer, args, kwargs, outputs):
        print("Layer:", type(layer).__name__)
        print("Input args:", args)
        print("Input kwargs:", kwargs)
        print("Outputs:", outputs)
        return outputs

    add_layer_hook(print_info)
    ```
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
