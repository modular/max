# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Abstract base class for KVCacheManager for KV Cache."""

from abc import ABC, abstractmethod
from typing import List

from max.driver import Device, Tensor
from max.dtype import DType
from max.graph import TensorType
from max.engine import InferenceSession

from .cache_params import KVCacheParams


class KVCacheManager(ABC):
    def __init__(
        self,
        params: KVCacheParams,
        max_cache_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        devices: List[Device],
        session: InferenceSession,
    ) -> None:
        self.params = params
        self.max_cache_batch_size = max_cache_batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.devices = devices
        self.session = session

        # Attributes for managing available slots.
        self.available = set(range(self.max_cache_batch_size))
        self.cache_lengths: dict[int, int] = {}

        # Allocate boolean tensors
        # allocating once up top, ensures we are not re-allocating
        # new memory to store a boolean on each fetch call.
        self.true_tensor = Tensor.zeros((1,), DType.bool)
        self.true_tensor[0] = True

        self.false_tensor = Tensor.zeros((1,), DType.bool)
        self.false_tensor[0] = False

    @abstractmethod
    def estimated_memory_size(
        cls,
        params: KVCacheParams,
        max_cache_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        devices: List[Device],
    ) -> int:
        """Returns the estimated total memory usage of the kv cache."""
        ...

    @abstractmethod
    def fetch(
        self, seq_ids: list[int]
    ) -> List[tuple[Tensor, Tensor, Tensor, Tensor]]: ...

    @abstractmethod
    def input_symbols(
        self,
    ) -> List[tuple[TensorType, TensorType, TensorType, TensorType]]: ...

    def claim(self, n: int) -> List[int]:
        """Claims `n` blocks of memory in the cache for incoming requests.

        This returns a list of sequence ids, which identify a sequence's
        location within the cache. This sequence id can then be passed
        in the fetch function to return the ContinuousBatchingKVCacheCollection
        for those sequences.
        """
        seq_ids = []

        for _ in range(n):
            id = self.available.pop()
            seq_ids.append(id)
            self.cache_lengths[id] = 0

        return seq_ids

    def external_claim(self, seq_ids: List[int]) -> None:
        """Variant of the above where sequence ids are reserved externally."""
        for seq_id in seq_ids:
            self.available.remove(seq_id)
            self.cache_lengths[seq_id] = 0

    def step(self, valid_lengths: dict[int, int]) -> None:
        """Update the `cache_lengths` objects to not that a new
        kv projection step has occurred, and that the underlying memory
        has been written to. This `cache_lengths` value is then used
        downstream in `fetch` to track what section of memory should
        be used in the kernels.
        """

        for id, length in valid_lengths.items():
            if id not in self.cache_lengths:
                raise ValueError(f"seq_id: {id} not in cache.")

            self.cache_lengths[id] += length

    def release(self, seq_id: int) -> None:
        """Release `seq_id` provided, marking this sequence as complete.
        This returns the seq_id back to the available pool of cache memory,
        allowing it to be reused when a new sequence is claimed.
        """

        if seq_id not in self.cache_lengths:
            raise ValueError(f"seq_id: {id} not in cache.")

        self.available.add(seq_id)
        del self.cache_lengths[seq_id]

    @property
    def slots_remaining(self) -> set[int]:
        """The outstanding cache slots available."""
        return self.available

    @property
    def max_sequence_length(self) -> int:
        """The maximum sequence length in current cache."""
        return max(self.cache_lengths.values())

    @abstractmethod
    def increment_cache_lengths(
        self,
        kv_cache_inputs: List[tuple[Tensor, Tensor, Tensor, Tensor]],
        prev_model_inputs: tuple[Tensor, ...],
    ) -> List[tuple[Tensor, Tensor, Tensor, Tensor]]:
        """
        Prepare the inputs for a multistep execution, generally by incrementing
        the cache lengths. This should not require a device synchronization,
        as this would defeat the purpose of multistep execution.

        This should also not update the cache lengths in our manager, this batch is
        still considered in-progress.
        """
        ...
