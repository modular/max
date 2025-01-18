# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""PagedAttention-enabled KV cache for the Transformer leveraging the mo.opaque pattern."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, Iterator, Optional

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DeviceRef,
    TensorType,
    TensorValue,
    _OpaqueType,
    _OpaqueValue,
    ops,
)

from .cache_params import KVCacheParams
from .manager import KVCacheManager
from .radix_trie import RadixTrie, TrieNode

PERCENTAGE_BLOCKS_TO_EVICT = 0.05


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

        op_name = f"mo.kv_collection_ctor.paged.nhead_{self.kv_params.n_kv_heads_per_device}.hdim_{self.kv_params.head_dim}"
        return PagedKVCacheCollection(
            ops.custom(
                op_name,
                values=[blocks, cache_lengths, lookup_table, is_cache_empty],
                out_types=[PagedKVCacheCollectionType()],
            )[0].opaque
        )


@dataclass
class _PagedCacheMetadata:
    # Committed blocks are part of the radix trie and can be shared by many sequences.
    # They are used by the current sequence and possibly other sequences.
    committed_blocks: list[int] = field(default_factory=list)
    # Inflight blocks are not part of the radix trie and are not shared.
    # They are only used by the current sequence.
    inflight_blocks: list[int] = field(default_factory=list)

    # Leftover tokens from a prior call to step that were not committed because
    # they were in a partially filled block.
    previous_uncommitted_tokens: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )

    # This is a pointer into the radix trie indicating which prefix of the sequence
    # has been cached and committed into the radix trie.
    node: Optional[TrieNode] = None

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
        cache_memory: int,
        page_size: int = 512,
    ) -> None:
        self.page_size = page_size

        single_page_size_bytes = (
            2
            * num_layers
            * params.n_kv_heads_per_device
            * params.head_dim
            * page_size
            * params.dtype.size_in_bytes
        )
        self.total_num_blocks = int((cache_memory) // single_page_size_bytes)

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

        self.radix_trie: Optional[RadixTrie] = None
        if params.enable_prefix_caching:
            self.radix_trie = RadixTrie(page_size=self.page_size)

    def evict_blocks(self, percentage_to_evict: float = 1.0):
        if self.radix_trie is None:
            return

        # Evict a percentage of all blocks according to a LRU policy on the
        # trie leaves.
        evicted_blocks = self.radix_trie.evict_blocks(
            desired_num_evicted=int(
                max(1, self.total_num_blocks * percentage_to_evict)
            )
        )

        for block in evicted_blocks:
            assert block not in self.available_blocks
            self.available_blocks.add(block)

    def alloc_block(self) -> int:
        if len(self.available_blocks) == 0:
            self.evict_blocks(percentage_to_evict=PERCENTAGE_BLOCKS_TO_EVICT)

        if len(self.available_blocks) == 0:
            raise RuntimeError(
                "Available KVCache pages have been exhausted! You must restart your process"
                " and set a smaller batch size or max seq len."
            )

        block = self.available_blocks.pop()
        return block

    def release_block(self, block: int, is_committed: bool = False) -> None:
        """We can release a block if prefix caching is disabled or if it is not committed.

        If it is committed, it may be in the radix tree and in use by other sequences.
        This means it can't be safely released without further checks.
        """
        if self.radix_trie is None or not is_committed:
            self.available_blocks.add(block)

    @classmethod
    def estimated_memory_size(
        cls,
        params: KVCacheParams,
        max_cache_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: list[Device],
    ) -> int:
        return available_cache_memory

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

    def _fetch(
        self, seq_ids_and_prompts: dict[int, np.ndarray], num_steps: int = 1
    ) -> Sequence[tuple[Tensor, ...]]:
        """This method identifies available blocks to service the given requests and marks them as inflight.
        They're assigned to the request as "in-flight" until step is called.

        Generally the prompt length is n for prefill, and 1 for decode step. Additionally, there is not a
        kv entry associated with each token in the prompt.

        When prefix caching is enabled, and KV entries can be retrieved for some tokens in the prompt, the
        input `seq_ids_and_prompts` will be modified. Each prompt will be shortened to only include the tokens
        for which we do not have a cached KV entry. Note that we will never return a empty prompt.
        """

        batch_size = len(seq_ids_and_prompts)

        max_seq_len_in_batch = -1
        # before we start making any changes, validate that we won't over-write the cache
        for batch_idx, (seq_id, prompt) in enumerate(
            seq_ids_and_prompts.items()
        ):
            curr_seq_len = (
                self.cache_lengths[seq_id] + len(prompt) + num_steps - 1
            )
            if curr_seq_len > max_seq_len_in_batch:
                max_seq_len_in_batch = curr_seq_len

            assert curr_seq_len <= self.max_seq_len, (
                f"seq_id: {seq_id} would overrun the max cache length of {self.max_seq_len} "
                f"with {len(prompt)} new tokens. Existing length: {self.cache_lengths[seq_id]}"
            )

        max_num_pages = ceildiv(max_seq_len_in_batch, self.page_size)

        # Allocate the buffers containing metadata about the batch.
        lut_table_np = np.zeros((batch_size, max_num_pages), dtype=np.uint32)
        cache_lengths_np = np.zeros((batch_size,), dtype=np.uint32)

        max_seq_length = 0
        max_cache_length = 0

        # Iterate over requests in the batch.
        for batch_idx, (seq_id, prompt) in enumerate(
            seq_ids_and_prompts.items()
        ):
            # Ensure we've called claim for this sequence id.
            if seq_id not in self.active_requests:
                raise ValueError(f"seq_id: {seq_id} not in active requests.")

            # Validate there aren't other inflight requests for this sequence.
            assert seq_id not in self.fetch_metadata

            # There can at most be one partially filled inflight block.
            inflight_metadata = self.active_requests[seq_id]
            if len(inflight_metadata.inflight_blocks) > 1:
                # TODO we need a way to invalidate "in-flight" blocks if something goes wrong during execution.
                # probably via a ``release_failed`` method.
                raise ValueError(
                    f"seq_id: {seq_id} already has {len(inflight_metadata.inflight_blocks)} inflight blocks."
                )

            # Extend the kv cache for given request with any cached prefixes.
            if self.radix_trie is not None and len(prompt) > 1:
                # Attempt to match all but the last token in the prompt. This is
                # because the model expects a prompt of length at least 1.
                inflight_metadata.node, prefix_blocks = (
                    self.radix_trie.match_prefix(
                        prompt[:-1], node=inflight_metadata.node
                    )
                )

                # Add the prefix blocks to the request's cached blocks.
                inflight_metadata.committed_blocks.extend(prefix_blocks)
                self.cache_lengths[seq_id] += (
                    len(prefix_blocks) * self.page_size
                )

                # Shorten the prompt to only include tokens that were not cached
                # by mutating the input dict.
                prompt = prompt[len(prefix_blocks) * self.page_size :]
                seq_ids_and_prompts[seq_id] = prompt

                # Mark the prefix blocks we retrieved from the radix trie cache as
                # in use by this sequence so they don't get evicted prematurely.
                assert inflight_metadata.node is not None
                self.radix_trie.mark_in_use_by(inflight_metadata.node, seq_id)

            # Get the existing cache length for this sequence.
            cache_length = self.cache_lengths[seq_id]
            cache_lengths_np[batch_idx] = cache_length

            # Compute the total sequence length and the number of pages required to store it.
            total_sequence_length = cache_length + len(prompt) + num_steps - 1
            num_pages_required = ceildiv(total_sequence_length, self.page_size)

            # Compute the number of *new* pages we need to allocate.
            # Note that inflight_blocks may contain one partially filled block.
            assert len(inflight_metadata.inflight_blocks) <= 1
            num_new_pages = (
                num_pages_required
                - len(inflight_metadata.committed_blocks)
                - len(inflight_metadata.inflight_blocks)
            )

            # Assign some new pages to this request.
            for _ in range(num_new_pages):
                next_block = self.alloc_block()
                inflight_metadata.inflight_blocks.append(next_block)

            # Populate the lookup table with the new pages.
            for i, block_idx in enumerate(
                inflight_metadata.all_assigned_blocks
            ):
                lut_table_np[batch_idx, i] = block_idx

            # Update the maximum lengths seen so far.
            max_seq_length = max(max_seq_length, len(prompt))
            max_cache_length = max(max_cache_length, cache_length)

        # Build a tensor of maximum lengths. Each step slices the first row to
        # advance to the values for the next row.
        max_lengths_np = np.empty((num_steps, 2), np.uint32)
        step_max_seq_length = max_seq_length
        step_max_cache_length = max_cache_length
        for step in range(num_steps):
            max_lengths_np[step, 0] = step_max_seq_length
            max_lengths_np[step, 1] = step_max_cache_length
            step_max_cache_length += step_max_seq_length
            step_max_seq_length = 1
        max_lengths_host = Tensor.from_numpy(max_lengths_np)

        lut_table_host = Tensor.from_numpy(lut_table_np)
        cache_lengths_host = Tensor.from_numpy(cache_lengths_np)

        ret_list = []
        for i, device in enumerate(self.devices):
            ret_list.append(
                (
                    self.blocks[i],
                    cache_lengths_host.to(device=device),
                    lut_table_host.to(device=device),
                    max_lengths_host,
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
                    device=DeviceRef(self.devices[i].label, self.devices[i].id),
                ),
                # cache_lengths
                TensorType(
                    DType.uint32,
                    shape=["batch_size"],
                    device=DeviceRef(self.devices[i].label, self.devices[i].id),
                ),
                # lookup_table
                TensorType(
                    DType.uint32,
                    shape=["batch_size", "max_num_pages"],
                    device=DeviceRef(self.devices[i].label, self.devices[i].id),
                ),
                # max_lengths
                TensorType(DType.uint32, shape=["steps_remaining", 2]),
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

        if self.radix_trie is not None:
            # mark the prefix blocks as not in use by this sequence so they can
            # potentially be evicted when we need more memory
            assert request_metadata.node is not None
            self.radix_trie.mark_not_in_use_by(request_metadata.node, seq_id)

        for block in request_metadata.committed_blocks:
            self.release_block(block, is_committed=True)
        for block in request_metadata.inflight_blocks:
            self.release_block(block, is_committed=False)
        del self.active_requests[seq_id]

    def _step(
        self,
        seq_ids_and_new_tokens: dict[int, np.ndarray],
    ) -> None:
        """Update the `cache_lengths` objects to not that a new
        kv projection step has occurred, and that the underlying memory
        has been written to. This `cache_lengths` value is then used
        downstream in `fetch` to track what section of memory should
        be used in the kernels.
        """

        for seq_id, new_tokens in seq_ids_and_new_tokens.items():
            if seq_id not in self.active_requests:
                raise ValueError(f"seq_id: {seq_id} not in active requests.")

            request_metadata = self.active_requests[seq_id]
            fetch_metadata = self.fetch_metadata[seq_id]
            prompt = fetch_metadata.prompt
            num_steps = fetch_metadata.num_steps
            assert len(new_tokens) == num_steps

            num_tokens_with_kv_entry = (
                self.cache_lengths[seq_id] + len(prompt) + len(new_tokens) - 1
            )
            num_tokens_in_partially_filled_block = (
                num_tokens_with_kv_entry % self.page_size
            )

            # Now that we wrote to the inflight blocks, we will try to commit
            # them to the radix trie.
            if self.radix_trie is not None:
                # Match the prefix of the prompt that is already cached in the
                # radix trie
                request_metadata.node, existing_blocks = (
                    self.radix_trie.match_prefix(
                        prompt, node=request_metadata.node
                    )
                )

                # If we computed a kv entry for a token that was already cached,
                # we will just release that block we just computed.
                for b0, b1 in zip(
                    existing_blocks, request_metadata.inflight_blocks
                ):
                    if b0 != b1:
                        self.release_block(b1, is_committed=False)

                # Replace the inflight blocks with the existing prefix blocks.
                request_metadata.inflight_blocks[: len(existing_blocks)] = (
                    existing_blocks
                )

                # Commit the rest of the tokens in the trie for use by future
                # sequences.
                uncommitted_blocks = request_metadata.inflight_blocks[
                    len(existing_blocks) :
                ]
                uncommitted_prompt = prompt[
                    len(existing_blocks) * self.page_size :
                ]
                # All but the last newly generated token should have a kv block.
                uncommitted_new_tokens = new_tokens[:-1]
                all_uncommitted_tokens = np.concatenate(
                    [
                        request_metadata.previous_uncommitted_tokens,
                        uncommitted_prompt,
                        uncommitted_new_tokens,
                    ]
                )

                # round the number of uncommitted new tokens to the nearest
                # multiple of the page size if not aligned
                blocks_to_commit = uncommitted_blocks
                tokens_to_commit = all_uncommitted_tokens
                if num_tokens_in_partially_filled_block > 0:
                    prefix, suffix = (
                        tokens_to_commit[
                            :-num_tokens_in_partially_filled_block
                        ],
                        tokens_to_commit[
                            -num_tokens_in_partially_filled_block:
                        ],
                    )
                    tokens_to_commit = prefix
                    blocks_to_commit = blocks_to_commit[:-1]
                    request_metadata.previous_uncommitted_tokens = suffix
                else:
                    # Clear out the previous uncommitted tokens
                    request_metadata.previous_uncommitted_tokens = np.array(
                        [], dtype=np.int64
                    )

                assert (
                    len(tokens_to_commit)
                    == len(blocks_to_commit) * self.page_size
                )

                # If there are any tokens to commit, insert them into the radix
                # trie.
                if len(tokens_to_commit) > 0:
                    request_metadata.node = self.radix_trie.insert(
                        tokens_to_commit,
                        blocks_to_commit,
                        node=request_metadata.node,
                    )

                # Mark the recently committed blocks as in use by this sequence
                # so they don't get evicted prematurely.
                assert request_metadata.node is not None
                self.radix_trie.mark_in_use_by(request_metadata.node, seq_id)

            expected_num_pages = ceildiv(
                num_tokens_with_kv_entry,
                self.page_size,
            )

            actual_num_pages = len(request_metadata.inflight_blocks) + len(
                request_metadata.committed_blocks
            )

            if expected_num_pages != actual_num_pages:
                raise ValueError(
                    f"Mismatch between expected and actual number of pages for seq_id: {seq_id}. Expected: {expected_num_pages}, Actual: {actual_num_pages}  "
                )

            if num_tokens_in_partially_filled_block > 0:
                # Leave one partially filled block in the inflight blocks
                # and finish committing the rest.
                partially_filled_block = request_metadata.inflight_blocks[-1]
                request_metadata.committed_blocks.extend(
                    request_metadata.inflight_blocks[:-1]
                )
                # This mutates the list in place.
                request_metadata.inflight_blocks.clear()
                request_metadata.inflight_blocks.append(partially_filled_block)
            else:
                # Commit all of the inflight blocks.
                request_metadata.committed_blocks.extend(
                    request_metadata.inflight_blocks
                )
                request_metadata.inflight_blocks.clear()
