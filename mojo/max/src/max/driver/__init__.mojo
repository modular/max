# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""APIs to interact with devices.

Although there are several modules in this `max.driver` package, you'll get
everything you need from this top-level `driver` namespace, so you don't need
to import each module.

For example, the basic code you need to create tensor on CPU looks like this:

```mojo
from max.driver import Tensor, cpu_device
from testing import assert_equal
from max.tensor import TensorShape

def main():
    tensor = Tensor[DType.float32, rank=2](TensorShape(1,2))
    tensor[0, 0] = 1.0

    # You can also explicitly set the devices.
    device = cpu_device()
    new_tensor = Tensor[DType.float32, rank=2](TensorShape(1,2), device)
    new_tensor[0, 0] = 1.0

    # You can also create slices of tensor
    subtensor = tensor[:, :1]
    assert_equal(subtensor[0, 0], tensor[0, 0])
```
"""

from .anytensor import AnyTensor, AnyMemory, AnyMojoValue
from .device import Device, cpu_device
from .device_memory import DeviceMemory, DeviceTensor
from .tensor import Tensor
from .tensor_slice import TensorSlice
from max.tensor import StaticTensorSpec
from max._tensor_utils import ManagedTensorSlice
