# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""PagedAttention-enabled KV cache for the Transformer leveraging the mo.opaque pattern."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, Iterator
from functools import reduce
from operator import mul

import numpy as np

from max.dtype import DType
from max.graph import (
    TensorType,
    TensorValue,
    _OpaqueType,
    _OpaqueValue,
    ops,
    Device as GraphDevice,
)
from max.engine import InferenceSession
from max.driver import Device, Tensor

from .cache_params import KVCacheParams
from .manager import KVCacheManager


def ceildiv(n: int, d: int) -> int:
    """Compute ceil(n/d) using strictly integer arithmetic."""
    q, r = divmod(n, d)
    return q + bool(r)


class PagedKVCacheType(_OpaqueType):
    """PagedAttention Mojo KV Cache graph type."""

    def __init__(self) -> None:
        """Creates an opaque type containing a paged KV Cache."""
        super().__init__("PagedKVCache")


class PagedKVCacheCollectionType(_OpaqueType):
    """The graph type for a "view" of the cache for the given sequences in the
    batch.

    This object does not own the underlying buffers in k_cache and v_cache,
    it's borrowing them from the BlockWrappers in our ContinuousKVCacheManager.
    It does own the Pointer[NDBuffer[type, 3]] and valid_lengths buffer
    """

    def __init__(self) -> None:
        """Creates an opaque type containing a paged KV cache collection."""
        super().__init__("PagedKVCacheCollection")


class PagedKVCache(_OpaqueValue):
    """PagedAttention Mojo KV cache graph value."""


class PagedKVCacheCollection(_OpaqueValue):
    """The graph value for a view of the KV cache."""


class FetchPagedKVCacheCollection:
    def __init__(self, kv_params: KVCacheParams) -> None:
        self.kv_params = kv_params

    def __call__(
        self,
        blocks: TensorValue,  # NDBuffer[type, 6, Self.blocks_shape]
        cache_lengths: TensorValue,  # NDBuffer[DType.uint32, 1],
        lookup_table: TensorValue,  # NDBuffer[DType.uint32, 2],
        is_cache_empty: TensorValue,
    ) -> PagedKVCacheCollection:
        """Constructs a PagedKVCacheCollection for use downstream."""

        # Explicit validation.
        if blocks.dtype != self.kv_params.dtype:
            msg = (
                f"expected blocks to be dtype: {self.kv_params.dtype}, got"
                f" {blocks.dtype}"
            )
            raise ValueError(msg)

        if blocks.rank != 6:
            msg = f"expected blocks to be of rank 6, got {blocks.rank}"
            raise ValueError(msg)

        # For all tensors other than the blocks tensor, the length should be equivalent
        # to batch size, which is unknown within the graph at this stage.
        if cache_lengths.dtype != DType.uint32:
            msg = (
                "expected cache lengths to be dtype: uint32, got"
                f" {cache_lengths.dtype}"
            )
            raise ValueError(msg)

        if cache_lengths.rank != 1:
            msg = (
                "expected cache lengths to be of rank 1, got"
                f" {cache_lengths.rank}"
            )
            raise ValueError(msg)

        if lookup_table.dtype != DType.uint32:
            msg = (
                "expected lookup_table to be dtype: uint32, got"
                f" {lookup_table.dtype}"
            )
            raise ValueError(msg)

        if lookup_table.rank != 2:
            msg = (
                "expected lookup_table to be of rank 2, got"
                f" {lookup_table.rank}"
            )
            raise ValueError(msg)

        op_name = f"paged_kv_cache_collection_h{self.kv_params.n_kv_heads}_d{self.kv_params.head_dim}_bshd"
        return PagedKVCacheCollection(
            ops.custom(
                op_name,
                values=[blocks, cache_lengths, lookup_table, is_cache_empty],
                out_types=[PagedKVCacheCollectionType()],
            )[0].opaque
        )


@dataclass
class _PagedCacheMetadata:
    committed_blocks: list[int] = field(default_factory=list)
    inflight_blocks: list[int] = field(default_factory=list)

    @property
    def all_assigned_blocks(self) -> Iterator[int]:
        return chain(self.committed_blocks, self.inflight_blocks)


class PagedKVCacheManager(KVCacheManager):
    def __init__(
        self,
        params: KVCacheParams,
        max_cache_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        devices: list[Device],
        session: InferenceSession,
        page_size: int = 512,
    ) -> None:
        self.page_size = page_size
        self.total_num_blocks = self._compute_total_num_blocks(
            max_cache_batch_size, max_seq_len, page_size
        )

        super().__init__(
            params=params,
            max_cache_batch_size=max_cache_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            devices=devices,
            session=session,
            is_ragged=True,
        )

        self.available_blocks = set(range(self.total_num_blocks))
        self.blocks: list[Tensor] = []
        for device in self.devices:
            self.blocks.append(
                Tensor.zeros(
                    self._block_shape(
                        self.params,
                        self.total_num_blocks,
                        self.page_size,
                        self.num_layers,
                    ),
                    self.params.dtype,
                    device=device,
                )
            )

        self.active_requests: Dict[int, _PagedCacheMetadata] = {}

    @classmethod
    def _compute_total_num_blocks(
        cls, max_cache_batch_size: int, max_seq_len: int, page_size: int
    ) -> int:
        return (max_cache_batch_size * max_seq_len) // page_size

    @classmethod
    def estimated_memory_size(
        cls,
        params: KVCacheParams,
        max_cache_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        devices: list[Device],
    ) -> int:
        # page_size and total_num_blocks cancel out, we can set page_size to 1 here and
        # still get an accurate estimate of how much memory is used
        page_size = 1
        total_num_blocks = cls._compute_total_num_blocks(
            max_cache_batch_size, max_seq_len, page_size
        )
        return (
            reduce(
                mul,
                cls._block_shape(
                    params, total_num_blocks, page_size, num_layers
                ),
            )
            * params.dtype.size_in_bytes
        )

    @classmethod
    def _block_shape(
        cls,
        params: KVCacheParams,
        total_num_blocks: int,
        page_size: int,
        num_layers: int,
    ) -> list[int]:
        # split k and v caches across a single dim
        # 0 = key
        # 1 = value
        kv_dim = 2
        return [
            num_layers,
            kv_dim,
            total_num_blocks,
            page_size,
            params.n_kv_heads_per_device,
            params.head_dim,
        ]

    def fetch(
        self,
        seq_ids_and_lengths: Dict[int, int],
    ) -> list[tuple[Tensor, Tensor, Tensor, Tensor]]:
        """This method identifies available blocks to service the given requests and marks them as inflight.
        They're assigned to the request as "in-flight" until step is called."""

        batch_size = len(seq_ids_and_lengths)

        max_seq_len_in_batch = -1
        # before we start making any changes, validate that we won't over-write the cache
        for batch_idx, (seq_id, num_tokens) in enumerate(
            seq_ids_and_lengths.items()
        ):
            curr_seq_len = num_tokens + self.cache_lengths[seq_id]
            if curr_seq_len > max_seq_len_in_batch:
                max_seq_len_in_batch = curr_seq_len

            if curr_seq_len <= self.max_seq_len:
                continue

            msg = f"seq_id: {seq_id} would overrun the max cache length of {self.max_seq_len} with {num_tokens} new tokens. Existing length: {self.cache_lengths[seq_id]}"
            raise ValueError(msg)

        max_num_pages = ceildiv(max_seq_len_in_batch, self.page_size)

        # Allocate the buffers containing metadata about the batch.
        lut_table_np = np.zeros((batch_size, max_num_pages), dtype=np.uint32)
        cache_lengths_np = np.zeros((batch_size,), dtype=np.uint32)

        is_cache_empty = True

        # Iterate over requests in the batch.
        for batch_idx, (seq_id, num_tokens) in enumerate(
            seq_ids_and_lengths.items()
        ):
            # Ensure we've called claim for this sequence id.
            if seq_id not in self.active_requests:
                raise ValueError(f"seq_id: {seq_id} not in active requests.")

            # Validate there aren't other inflight requests for this sequence.
            inflight_metadata = self.active_requests[seq_id]
            if inflight_metadata.inflight_blocks:
                # TODO we need a way to invalidate "in-flight" blocks if something goes wrong during execution.
                # probably via a ``release_failed`` method.
                raise ValueError(
                    f"seq_id: {seq_id} already has inflight blocks."
                )

            # Get the existing cache length for this sequence.
            cache_length = self.cache_lengths[seq_id]
            cache_lengths_np[batch_idx] = cache_length
            if cache_length > 0:
                is_cache_empty = False

            # Compute the total sequence length and the number of pages required to store it.
            total_sequence_length = cache_length + num_tokens
            num_pages_required = ceildiv(total_sequence_length, self.page_size)

            # Compute the number of *new* pages we need to allocate.
            # Note that the last page in committed_blocks may only be partially filled.
            num_new_pages = num_pages_required - len(
                inflight_metadata.committed_blocks
            )

            # Assign some new pages to this request.
            for _ in range(num_new_pages):
                next_block = self.available_blocks.pop()
                inflight_metadata.inflight_blocks.append(next_block)

            # Populate the lookup table with the new pages.
            for i, block_idx in enumerate(
                inflight_metadata.all_assigned_blocks
            ):
                lut_table_np[batch_idx, i] = block_idx

        lut_table_host = Tensor.from_numpy(lut_table_np)
        cache_lengths_host = Tensor.from_numpy(cache_lengths_np)
        ret_list = []
        for i, device in enumerate(self.devices):
            ret_list.append(
                (
                    self.blocks[i],
                    cache_lengths_host.to(device=device),
                    lut_table_host.to(device=device),
                    self.true_tensor if is_cache_empty else self.false_tensor,
                )
            )
        return ret_list

    def input_symbols(
        self,
    ) -> list[tuple[TensorType, TensorType, TensorType, TensorType]]:
        return [
            (
                # kv_blocks
                TensorType(
                    self.params.dtype,
                    shape=[
                        self.num_layers,
                        2,
                        self.total_num_blocks,
                        self.page_size,
                        self.params.n_kv_heads_per_device,
                        self.params.head_dim,
                    ],
                    device=GraphDevice(
                        self.devices[i].label, self.devices[i].id
                    ),
                ),
                # cache_lengths
                TensorType(
                    DType.uint32,
                    shape=["batch_size"],
                    device=GraphDevice(
                        self.devices[i].label, self.devices[i].id
                    ),
                ),
                # lookup_table
                TensorType(
                    DType.uint32,
                    shape=["batch_size", "max_num_pages"],
                    device=GraphDevice(
                        self.devices[i].label, self.devices[i].id
                    ),
                ),
                # is_cache_empty
                TensorType(DType.bool, shape=[1]),
            )
            for i in range(len(self.devices))
        ]

    def claim(self, n: int) -> list[int]:
        """Claims `n` blocks of memory in the cache for incoming requests.

        This returns a list of sequence ids, which identify a sequence's
        location within the cache. This sequence id can then be passed
        in the fetch function to return the ContinuousBatchingKVCacheCollection
        for those sequences.
        """
        seq_ids = super().claim(n)
        for seq_id in seq_ids:
            self.active_requests[seq_id] = _PagedCacheMetadata()
        return seq_ids

    def external_claim(self, seq_ids: list[int]) -> None:
        """Variant of the above where sequence ids are reserved externally."""
        super().external_claim(seq_ids)
        for seq_id in seq_ids:
            self.active_requests[seq_id] = _PagedCacheMetadata()

    def release(self, seq_id: int) -> None:
        """Release `seq_id` provided, marking this sequence as complete.
        This returns the seq_id back to the available pool of cache memory,
        allowing it to be reused when a new sequence is claimed.
        """
        super().release(seq_id)
        request_metadata = self.active_requests[seq_id]
        for block in request_metadata.all_assigned_blocks:
            self.available_blocks.add(block)
        del self.active_requests[seq_id]

    def step(self, valid_lengths: dict[int, int]) -> None:
        """Update the `cache_lengths` objects to not that a new
        kv projection step has occurred, and that the underlying memory
        has been written to. This `cache_lengths` value is then used
        downstream in `fetch` to track what section of memory should
        be used in the kernels.
        """
        for seq_id, length in valid_lengths.items():
            if seq_id not in self.active_requests:
                raise ValueError(f"seq_id: {seq_id} not in active requests.")
            request_metadata = self.active_requests[seq_id]
            expected_num_pages = ceildiv(
                length + self.cache_lengths[seq_id], self.page_size
            )
            actual_num_pages = len(request_metadata.inflight_blocks) + len(
                request_metadata.committed_blocks
            )
            if expected_num_pages != actual_num_pages:
                raise ValueError(
                    f"Mismatch between expected and actual number of pages for seq_id: {seq_id}. Expected: {expected_num_pages}, Actual: {actual_num_pages}  "
                )

            request_metadata.committed_blocks.extend(
                request_metadata.inflight_blocks
            )
            request_metadata.inflight_blocks.clear()

        super().step(valid_lengths)
