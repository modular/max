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

import os
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

import numpy as np
from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import Graph, TensorType, ops
from numpy.typing import NDArray


class NextWordFrequency:
    def __init__(self, text):
        # nested `DefaultDict` to create the keys when first indexed
        # Structure looks like: {"word": {"next_word": count}}
        self.word_frequencies: DefaultDict[str, DefaultDict[str, int]] = (
            defaultdict(lambda: defaultdict(int))
        )

        # Build word frequencies
        words = text.lower().split()
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            self.word_frequencies[current_word][next_word] += 1

    def next_word_probabilities(self, current_word) -> NDArray[np.float32]:
        if current_word not in self.word_frequencies:
            return np.empty(0, dtype=np.float32)

        frequencies = self.word_frequencies[current_word]
        freq_list = np.array(
            [value for value in frequencies.values()], dtype=np.float32
        )
        freq_list /= freq_list.sum()
        return freq_list

    def get_key(self, word, idx):
        return list(self.word_frequencies[word].keys())[idx]

    def __getitem__(self, idx):
        return self.word_frequencies[idx]


# Example usage
if __name__ == "__main__":
    # This is necessary only for Modular internal CI.
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    path = Path(__file__).parent / "kernels.mojopkg"

    input_text = """
    The quick rabbit runs past the brown fox
    The quick rabbit jumps over the brown dog
    The quick dog chases past the lazy fox
    The quick dog runs through the tall trees
    The quick brown fox jumps over the lazy dog
    The brown dog sleeps under the shady tree
    The brown rabbit hops under the tall tree
    The brown fox runs through the forest trees
    The brown fox watches the sleeping rabbit
    The lazy fox watches over the sleeping dog
    The lazy dog watches the quick rabbit
    The shady tree shelters the brown rabbit
    The shady fox sleeps under the old tree
    The sleeping fox rests beside the shady tree
    The lazy rabbit rests beside the brown fox
    """

    # initial word to predict the next word for
    first_word = "the"

    # Initialize the next word frequency for each unique word
    frequencies = NextWordFrequency(input_text)

    # Get probabilities of each next word after `first_word`
    logit_values = frequencies.next_word_probabilities(first_word)

    batch_size = 1
    token_length = len(logit_values)
    # The amount of top results to find and sort
    k = 10
    # Make sure we don't have a higher k than elements passed in
    k = min(k, token_length)
    dtype = DType.float32

    # Configure our simple one-operation graph.
    with Graph(
        "sampler",
        input_types=[
            TensorType(dtype, shape=[batch_size, token_length]),
        ],
    ) as graph:
        # Take in the single input to the graph.
        x, *_ = graph.inputs

        # The custom Mojo operation is referenced by its string name, and we
        # need to provide inputs as a list as well as expected output types.
        results = ops.custom(
            # Can change this to use `mo.top_k` which is the MAX internal
            # implementation. This one is using the implementation from
            # `./examples/kernels/top_k_sampler.mojo` which is more concise for
            # learning purposes.
            name="top_k_custom",
            values=[
                x,
                ops.constant(k, dtype=DType.int64),
                ops.constant(1, dtype=DType.int64),  # axis
                ops.constant(True, dtype=DType.bool),  # sorted
            ],
            out_types=[
                TensorType(
                    dtype=x.tensor.dtype, shape=x.tensor.shape
                ),  # values
                TensorType(dtype=DType.int64, shape=x.tensor.shape),  # indices
            ],
        )
        graph.output(*results)

    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()

    # Set up an inference session for running the graph.
    session = InferenceSession(
        devices=[device],
        custom_extensions=path,
    )

    # Compile the graph.
    model = session.load(graph)

    # Create a driver tensor from the next word probabilities, adding a rank
    # for the batch size of 1, and move it to the accelerator.
    logits = Tensor.from_numpy(logit_values.reshape(1, -1)).to(device)

    # Perform the calculation on the target device.
    values, indices = model.execute(logits)

    # Copy values and indices back to the CPU to be read.
    assert isinstance(values, Tensor)
    values = values.to(CPU())
    np_values = values.to_numpy()[0]

    assert isinstance(indices, Tensor)
    indices = indices.to(CPU())
    np_indices = indices.to_numpy()[0]

    num_values = min(len(np_indices), k)

    print(f"Predicting word after `{first_word}`")
    print("------------------------------")
    print("| word         | confidence  |")
    print("------------------------------")
    for i in range(num_values):
        print(
            f"| {frequencies.get_key(first_word, np_indices[i]):<13}| {np_values[i]:<11.8} |"
        )
    print("------------------------------")
