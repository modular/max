# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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

import os
from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops


def create_simple_graph() -> Graph:
    """Configure a simple, one-input, one-operation graph."""
    dtype = DType.float32
    with Graph(
        "addition",
        input_types=[
            TensorType(dtype, shape=[5, 10]),
        ],
    ) as graph:
        # Take in the single input to the graph.
        x, *_ = graph.inputs

        # The custom Mojo operation is referenced by its string name, and we
        # need to provide inputs as a list as well as expected output types.
        result = ops.custom(
            name="add_one_custom",
            values=[x],
            out_types=[TensorType(dtype=dtype, shape=x.tensor.shape)],
        )[0].tensor

        # Return the result of the custom operation as the output of the graph.
        graph.output(result)
        return graph


if __name__ == "__main__":
    # This is necessary only in specific build environments.
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    path = Path(__file__).parent / "kernels.mojopkg"

    # Configure our simple graph.
    graph = create_simple_graph()

    # Set up an inference session that runs the graph on a GPU, if available.
    session = InferenceSession(
        devices=[CPU() if accelerator_count() == 0 else Accelerator()],
        custom_extensions=path,
    )
    # Compile the graph.
    model = session.load(graph)

    # Fill an input matrix with random values.
    x = np.random.uniform(size=(5, 10)).astype(np.float32)

    # Perform inference on the target device. The input NumPy array is
    # transparently copied to the device.
    result = model.execute(x)[0]

    # Copy values back to the CPU to be read.
    assert isinstance(result, Tensor)
    result = result.to(CPU())

    print("Graph result:")
    print(result.to_numpy())
    print()

    print("Expected result:")
    print(x + 1)
