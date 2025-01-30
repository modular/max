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

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, TensorValueLike, Weight, ops

from .layer import Layer


@dataclass
class Embedding(Layer):
    weights: TensorValueLike

    def __call__(self, indices: TensorValueLike) -> TensorValue:
        result = ops.gather(self.weights, indices, axis=0)
        if (
            isinstance(self.weights, Weight)
            and self.weights.quantization_encoding is not None
        ):
            result = ops.dequantize(self.weights.quantization_encoding, result)
        return result


class EmbeddingV2(Layer):
    """
    A lookup table for embedding integer indices into dense vectors.

    This layer maps each integer index to a dense vector of fixed size.
    Embedding weights are stored on the CPU but are moved to the specified
    device during the model init phase.

    Example:

    .. code-block:: python

        embedding_layer = EmbeddingV2(
            vocab_size=1000,
            hidden_dim=256,
            dtype=DType.float32,
            device=DeviceRef.GPU(),
            name="embeddings",
        )

        # Token indices of shape: [batch, ..., num_indices].
        token_indices: TensorValueLike
        embeddings = embedding_layer(token_indices)
    """

    weight: Weight
    """The embedding weight matrix stored on the CPU.
    Model init moves weights to the device specified in :obj:`device`."""

    device: DeviceRef
    """The device on which embedding lookup is performed."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        dtype: DType,
        device: DeviceRef,
        name: str,
    ) -> None:
        """Initializes the embedding layer with the given arguments.

        Args:
            vocab_size: The number of unique items in the vocabulary.
                Indices must be in the range ``[0, vocab_size)``.
            hidden_dim: The dimensionality of each embedding vector.
            dtype: The data type of the embedding weights.
            device: The device where embedding lookups are executed.
                Model init transfers the initially CPU-resident weights to this
                device.
            name: The name identifier for the embedding weight matrix.
        """
        super().__init__()

        self.device = device
        self.weight = Weight(
            name, dtype, shape=(vocab_size, hidden_dim), device=DeviceRef.CPU()
        )

    def __call__(self, indices: TensorValueLike) -> TensorValue:
        """Embeds the input indices by looking up corresponding vectors.

        Args:
            indices: A tensor of integer indices to look up.
                Each index must be in the range ``[0, vocab_size)``.

        Returns:
            A tensor containing the embeddings corresponding to the input
            indices.
            The result resides on the device specified in :obj:`device`.
        """
        return ops.gather(
            TensorValue(self.weight).to(self.device), indices, axis=0
        )
