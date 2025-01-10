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


def create_simple_graph(width: int, height: int, max_iterations: int) -> Graph:
    """Configure a graph to run a Mandelbrot kernel."""
    input_dtype = DType.float32
    output_dtype = DType.int32
    # output_dtype = DType.float32
    with Graph(
        "mandelbrot",
        input_types=[
            TensorType(input_dtype, shape=[height, width]),
            TensorType(input_dtype, shape=[height, width]),
        ],
    ) as graph:
        # Take in the inputs to the graph.
        cx, cy = graph.inputs

        # The custom Mojo operation is referenced by its string name, and we
        # need to provide inputs as a list as well as expected output types.
        result = ops.custom(
            name="mandelbrot",
            values=[ops.constant(max_iterations, dtype=DType.int32), cx, cy],
            out_types=[TensorType(dtype=output_dtype, shape=cx.tensor.shape)],
        )[0].tensor

        # Return the result of the custom operation as the output of the graph.
        graph.output(result)
        return graph


if __name__ == "__main__":
    # This is necessary only in specific build environments.
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    path = Path(__file__).parent / "kernels.mojopkg"

    width = 15
    height = 15
    max_iterations = 100
    min_x = -1.5
    max_x = 0.7
    min_y = -1.12
    max_y = 1.12

    # Configure our simple graph.
    graph = create_simple_graph(width, height, max_iterations)

    # Set up an inference session that runs the graph on a GPU, if available.
    session = InferenceSession(
        devices=[CPU() if accelerator_count() == 0 else Accelerator()],
        custom_extensions=path,
    )
    # Compile the graph.
    model = session.load(graph)

    # Fill the initial complex values.
    cx = np.zeros(shape=(height, width), dtype=np.float32)
    cy = np.zeros(shape=(height, width), dtype=np.float32)

    scale_x = (max_x - min_x) / width
    scale_y = (max_y - min_y) / height

    for row in range(height):
        for col in range(width):
            cx[row, col] = min_x + col * scale_x
            cy[row, col] = min_y + row * scale_y

    # Perform inference on the target device. The input NumPy arrays are
    # transparently copied to the device.
    result = model.execute(cx, cy)[0]

    # Copy values back to the CPU to be read.
    assert isinstance(result, Tensor)
    result = result.to(CPU())

    print("Iterations to escape:")
    print(result.to_numpy())
    print()
