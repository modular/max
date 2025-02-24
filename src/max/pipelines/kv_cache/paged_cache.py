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

"""PagedAttention-enabled KV cache for the Transformer leveraging the mo.opaque pattern."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Dict, Optional

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    _OpaqueType,
    _OpaqueValue,
    ops,
)

from ._utils import build_max_lengths_tensor
from .cache_params import KVCacheParams
from .manager import KVCacheInputSymbols, KVCacheManager
from .radix_trie import RadixTrie, TrieNode

logger = logging.getLogger("max.pipelines")

PERCENTAGE_BLOCKS_TO_EVICT = 0.05


def ceildiv(n: int, d: int) -> int:
    """Compute ceil(n/d) using strictly integer arithmetic."""
    q, r = divmod(n, d)
    return q + bool(r)


@dataclass
class PagedCacheInputSymbols(KVCacheInputSymbols):
    kv_blocks: TensorType
    cache_lengths: TensorType
    lookup_table: TensorType
    max_lengths: TensorType


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
            msg = f"expected cache lengths to be dtype: uint32, got {cache_lengths.dtype}"
            raise ValueError(msg)

        if cache_lengths.rank != 1:
            msg = f"expected cache lengths to be of rank 1, got {cache_lengths.rank}"
            raise ValueError(msg)

        if lookup_table.dtype != DType.uint32:
            msg = f"expected lookup_table to be dtype: uint32, got {lookup_table.dtype}"
            raise ValueError(msg)

        if lookup_table.rank != 2:
            msg = f"expected lookup_table to be of rank 2, got {lookup_table.rank}"
            raise ValueError(msg)

        return PagedKVCacheCollection(
            ops.custom(
                "mo.kv_collection_ctor.paged",
                values=[blocks, cache_lengths, lookup_table, is_cache_empty],
                out_types=[PagedKVCacheCollectionType()],
                parameters={
                    "num_heads": self.kv_params.n_kv_heads_per_device,
                    "head_dim": self.kv_params.head_dim,
                    "page_size": int(blocks.shape[3]),
                },
            )[0].opaque
        )


def construct_cow_strided_memcpy_graph(
    block_shape: list[int | str], dtype: DType, devices: list[Device]
) -> Graph:
    """
    Returns a graph for performing COW operations on the KV cache.
    """

    assert len(block_shape) == 6
    device_refs = [DeviceRef(device.label, device.id) for device in devices]
    blocks_ty = [
        BufferType(dtype, shape=block_shape, device=device_ref)
        for device_ref in device_refs
    ]
    block_src_idx_ty = TensorType(DType.uint32, shape=[])
    block_dst_idx_ty = TensorType(DType.uint32, shape=[])
    num_tokens_ty = TensorType(DType.uint32, shape=[])

    with Graph(
        "mo.kv_collection_cow_strided_memcpy.paged",
        input_types=[
            block_dst_idx_ty,
            block_src_idx_ty,
            num_tokens_ty,
            *blocks_ty,
        ],
        output_types=[],
    ) as graph:
        block_dst_idx, block_src_idx, num_tokens, *all_blocks = graph.inputs
        for blocks in all_blocks:
            ops.inplace_custom(
                "mo.kv_collection_cow_strided_memcpy.paged",
                values=[blocks, block_dst_idx, block_src_idx, num_tokens],
                out_types=[],
            )
        graph.output()

    return graph


class _PagedCacheMetadata:
    """Metadata for a single sequence in the paged KV cache.

    Token array layout:

    - Committed tokens have been inserted into the prefix cache. This must be a
      multiple of the page size.
    - Cached tokens are tokens that have a backing KV projection in the cache.
      Uncached tokens will be written to the cache when the model is run.
    - Committable tokens are tokens that are not yet committed but have a known
      value (i.e. not inflight). We can query the prefix cache for such tokens.
    - Inflight tokens are slots allocated for tokens that have not been generated
      yet. Such tokens have a filler value of 0. After `fetch`, there should be
      `num_steps - 1` inflight tokens. They will be replaced with actual tokens
      in `step`.

                  committed_idx V                              seq_len V
       +----------------------------------------------------------------+
       | committed              | uncommitted                           |
       |----------------------------------------------------------------|
       |                        | committable            |              |
       |----------------------------------------------------------------|
       | cached                      | uncached                         |
       |----------------------------------------------------------------|
       |                             | prompt            | inflight     |
       +----------------------------------------------------------------+
                          cached_idx ^      inflight_idx ^
    """

    def __init__(self, page_size: int, max_seq_len: int) -> None:
        self.page_size = page_size
        self.committed_idx: int = 0
        self.cached_idx: int = 0
        self.inflight_idx: int = 0
        self.seq_len: int = 0
        self.blocks: list[int] = []
        self.tokens: np.ndarray = np.full((max_seq_len,), 0, dtype=np.int64)
        # This is a pointer into the radix trie indicating the prefix of the sequence
        # that has been committed into the radix trie.
        self.node: Optional[TrieNode] = None

    @property
    def committed_blocks(self) -> list[int]:
        return self.blocks[: self.committed_idx // self.page_size]

    @property
    def uncommitted_blocks(self) -> list[int]:
        return self.blocks[self.committed_idx // self.page_size : self.seq_len]

    @property
    def committed_tokens(self) -> np.ndarray:
        return self.tokens[: self.committed_idx]

    @property
    def uncached_tokens(self) -> np.ndarray:
        return self.tokens[self.cached_idx : self.seq_len]

    @property
    def prompt_tokens(self) -> np.ndarray:
        return self.tokens[self.cached_idx : self.inflight_idx]

    @property
    def inflight_tokens(self) -> np.ndarray:
        return self.tokens[self.inflight_idx : self.seq_len]

    @property
    def committable_tokens_aligned(self) -> np.ndarray:
        # Return all tokens that are committable and part of a block that only
        # contains committable tokens.
        inflight_idx = self.inflight_idx
        partial_tokens = inflight_idx % self.page_size
        if partial_tokens > 0:
            inflight_idx -= partial_tokens
        return self.tokens[self.committed_idx : inflight_idx]

    @property
    def committable_blocks_aligned(self) -> list[int]:
        # Returns all blocks that only contain committable tokens.
        return self.blocks[
            self.committed_idx // self.page_size : self.inflight_idx
            // self.page_size
        ]

    @property
    def committable_tokens(self) -> np.ndarray:
        # Returns all tokens that are committable.
        return self.tokens[self.committed_idx : self.inflight_idx]

    @property
    def committable_blocks(self) -> list[int]:
        # Returns any block that contains at least one committable token.
        return self.blocks[
            self.committed_idx // self.page_size : ceildiv(
                self.inflight_idx, self.page_size
            )
        ]

    def _validate_indices(self):
        assert (
            0
            <= self.committed_idx
            <= self.cached_idx
            <= self.inflight_idx
            <= self.seq_len
        ), "The indices must be in the correct order"
        assert self.seq_len <= len(self.tokens), (
            "Sequence has exceeded the max sequence length"
        )
        assert self.committed_idx % self.page_size == 0, (
            "The committed_idx must be a multiple of the page size since we "
            "can't commit a partial page into the prefix cache"
        )

    def fetch(self, prompt: np.ndarray, num_steps: int) -> None:
        """Add prompt to token array and reserve space for inflight tokens."""
        self._validate_indices()
        assert len(self.prompt_tokens) == 0, (
            "At the start of fetch, there should be no prompt tokens"
        )
        assert len(self.inflight_tokens) == 0, (
            "At the start of fetch, there should be no inflight tokens"
        )
        assert len(prompt) > 0, (
            "The prompt provided to fetch should be non-empty"
        )
        num_inflight_tokens = num_steps - 1
        self.inflight_idx += len(prompt)
        self.seq_len += len(prompt) + num_inflight_tokens
        self.tokens[self.cached_idx : self.inflight_idx] = prompt
        self._validate_indices()

    def step(self, new_tokens: np.ndarray) -> None:
        """Write new tokens into inflight token slots. Also update the cached_idx."""
        self._validate_indices()
        assert len(self.prompt_tokens) > 0, (
            "We could not have executed the model without at least one prompt token"
        )
        num_inflight_tokens = len(new_tokens) - 1
        assert len(self.inflight_tokens) == num_inflight_tokens, (
            "The existing slots for inflight tokens should correspond to all but the last newly generated token"
        )
        self.tokens[self.inflight_idx : self.seq_len] = new_tokens[:-1]
        self.cached_idx = self.seq_len
        self.inflight_idx = self.seq_len
        assert len(self.uncached_tokens) == 0, (
            "After step, all tokens should have a backing KV projection in the cache"
        )
        self._validate_indices()


class PagedKVCacheManager(KVCacheManager):
    def __init__(
        self,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        devices: list[Device],
        session: InferenceSession,
        cache_memory: int,
        page_size: int = 128,
        enable_runtime_checks: bool = False,
    ):
        """
        Args:
            params: The KVCacheParams for the given pipeline.
            max_batch_size: The maximum number of active
                requests that the manager should support.
            max_seq_len: The maximum sequence length we will generate.
            num_layers: The number of layers in the model.
            devices: The devices on which the manager will allocate memory.
            session: The inference session to load ops from.
            cache_memory: The total amount of memory available for caching.
                This is aggregated across all devices.
            page_size: The number of tokens that will be stored in a single page.
            enable_runtime_checks: Whether to enable runtime correctness checks.
        """
        # The number of tokens in a single page.
        self.page_size = page_size

        # The number of bytes that a single page will occupy.
        single_page_size_bytes = (
            2
            * num_layers
            * params.n_kv_heads_per_device
            * params.head_dim
            * page_size
            * params.dtype.size_in_bytes
        )

        # Normalize cache_memory across all devices.
        cache_memory_per_device = cache_memory // len(devices)

        # The total number of pages we'll have per-device.
        self.total_num_pages = int(
            cache_memory_per_device // single_page_size_bytes
        )

        if self.total_num_pages == 0:
            raise RuntimeError(
                f"Insufficient cache memory to allocate even a single page.\n"
                f"One page requires {single_page_size_bytes} bytes but only {cache_memory_per_device} bytes are available."
            )

        if max_batch_size > self.total_num_pages:
            logger.warning(
                f"Insufficient cache memory to support a batch containing {max_batch_size} requests with one token per request. "
                f"Need to allocate at least {max_batch_size} blocks, but only have enough memory for {self.total_num_pages} blocks. "
                f"One page requires {single_page_size_bytes} bytes but only {cache_memory_per_device} bytes are available."
            )

        blocks_needed_for_max_seq_len = ceildiv(max_seq_len, page_size)
        if blocks_needed_for_max_seq_len > self.total_num_pages:
            logger.warning(
                f"Insufficient cache memory to support a batch containing one request at the max sequence length of {max_seq_len} tokens. "
                f"Need to allocate at least {blocks_needed_for_max_seq_len} blocks, but only have enough memory for {self.total_num_pages} blocks. "
                f"One page requires {single_page_size_bytes} bytes but only {cache_memory_per_device} bytes are available."
            )

        # call our base class constructor
        super().__init__(
            params=params,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            devices=devices,
            session=session,
            is_ragged=True,
        )

        # Initialize the set of available blocks.
        self.available_blocks = set(range(self.total_num_pages))

        # Initialize the blocks for each device.
        self.blocks: list[Tensor] = []
        for device in self.devices:
            self.blocks.append(
                Tensor.zeros(
                    self.block_shape(),  # type: ignore
                    self.params.dtype,
                    device=device,
                )
            )

        self.active_requests: Dict[int, _PagedCacheMetadata] = {}

        # Initialize the radix trie if prefix caching is enabled.
        self.radix_trie: Optional[RadixTrie] = None
        self.cow_strided_memcpy_graph: Optional[Model] = None
        if params.enable_prefix_caching:
            self.radix_trie = RadixTrie(page_size=self.page_size)
            # Load single op graph for performing memory transfers needed for COW
            if self.page_size > 1:
                self.cow_strided_memcpy_graph = session.load(
                    construct_cow_strided_memcpy_graph(
                        self.block_shape(is_parameterized=True),
                        params.dtype,
                        devices,
                    ),
                )
        self.ce_all_tokens = 0
        self.ce_cache_hit_tokens = 0

        # Whether to enable runtime correctness checks. These correctness checks
        # are expensive and should only be used in tests.
        self.enable_runtime_checks = enable_runtime_checks

    def _runtime_check(self) -> None:
        if not self.enable_runtime_checks:
            return
        assert self._count_all_pages() == self.total_num_pages
        if self.radix_trie is None:
            return
        for data in self.active_requests.values():
            # Check that the committed tokens and blocks match what was actually
            # committed into the radix trie.
            if data.node is not None:
                # Climb up the trie from the given node, accumulating all the
                # prefix tokens and blocks.
                tokens, blocks = data.node.get_prefix_tokens_and_blocks()
            else:
                tokens, blocks = np.array([], dtype=np.int64), []
            assert (tokens == data.committed_tokens).all()
            assert blocks == data.committed_blocks

    def cache_hit_rate(self) -> float:
        """Returns the prefix cache hit rate.

        This is the number of CE prompt tokens that were read from the prefix
        cache divided by the total number of CE prompt tokens.
        """
        if self.ce_all_tokens == 0:
            return 0.0
        return self.ce_cache_hit_tokens / self.ce_all_tokens

    def evict_blocks(self, percentage_to_evict: float = 1.0):
        if self.radix_trie is None:
            return

        # Evict a percentage of all blocks according to a LRU policy on the
        # trie leaves.
        evicted_blocks = self.radix_trie.evict_blocks(
            desired_num_evicted=int(
                max(1, self.total_num_pages * percentage_to_evict)
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
                f"All {self.total_num_pages} KVCache pages have been exhausted! "
                "You must restart your process and set a smaller batch size or max seq len."
            )

        block = self.available_blocks.pop()
        return block

    def release_block(self, block: int) -> None:
        """We can release a block if prefix caching is disabled or if it is not committed.

        If it is committed, it may be in the radix tree and in use by other sequences.
        This means it can't be safely released without further checks.
        """
        if self.radix_trie is None:
            self.available_blocks.add(block)
            return

        # We can only add the block to the available set if it is not committed
        # to the prefix cache.
        if block not in self.radix_trie.get_all_blocks():
            self.available_blocks.add(block)

    @classmethod
    def _block_size_per_token(
        cls, params: KVCacheParams, num_layers: int
    ) -> int:
        return (
            reduce(mul, cls._block_shape(params, 1, 1, num_layers), 1)
            * params.dtype.size_in_bytes
        )

    @classmethod
    def estimated_memory_size(
        cls,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: list[Device],
        **kwargs: Any,
    ) -> int:
        # Determine how much size is necessary to store the full cache based on max_batch_size and max_seq_len.
        # If that's less than available_cache_memory, return that.
        # Otherwise, return available_cache_memory.
        # This is to prevent over-allocation on devices with a large amount of free memory (e.g. CPUs).
        assert params.page_size is not None
        block_size_per_token = cls._block_size_per_token(
            params, num_layers
        ) * len(devices)

        # round up our max_seq_len to the nearest page_size
        max_seq_len_round_up = (
            math.ceil(max_seq_len / params.page_size) * params.page_size
        )
        size_to_support_full_cache = (
            block_size_per_token * max_batch_size * max_seq_len_round_up
        )

        # return the minimum of the two
        return min(available_cache_memory, size_to_support_full_cache)

    @classmethod
    def infer_optimal_batch_size(
        cls,
        params: KVCacheParams,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: list[Device],
        **kwargs: Any,
    ) -> int:
        # We just hard-code a default of 512 for paged attention.
        # The worst case scenario if this is too high is that we'll evict
        # requests at an elevated rate. We print warnings in that case so users
        # are aware of what needs to be tweaked/changed.
        return 512

    def block_shape(
        self,
        is_parameterized: bool = False,
    ) -> list[int | str]:
        return self._block_shape(
            self.params,
            self.total_num_pages,
            self.page_size,
            self.num_layers,
            is_parameterized,
        )

    @classmethod
    def _block_shape(
        cls,
        params: KVCacheParams,
        total_num_pages: int,
        page_size: int,
        num_layers: int,
        is_parameterized: bool = False,
    ) -> list[int | str]:
        # split k and v caches across a single dim
        # 0 = key
        # 1 = value
        kv_dim = 2
        return [
            num_layers,
            kv_dim,
            "total_num_pages" if is_parameterized else total_num_pages,
            page_size,
            params.n_kv_heads_per_device,
            params.head_dim,
        ]

    def get_num_free_blocks(self) -> int:
        if self.radix_trie is None:
            return len(self.available_blocks)
        else:
            return len(self.available_blocks) + len(
                self.radix_trie.get_evictable_blocks()
            )

    def can_fetch(
        self, seq_ids_and_prompts: dict[int, np.ndarray], num_steps: int = 1
    ) -> bool:
        """Checks if there are sufficient KV pages to run `fetch` on given batch.

        It is OK if some seq_id are not in the cache. We assume the cache lengths
        are zero in those cases.
        """

        total_blocks_to_allocate = 0
        all_cache_hit_blocks: set[int] = set()

        for seq_id, prompt in seq_ids_and_prompts.items():
            data = self.active_requests.get(
                seq_id, _PagedCacheMetadata(self.page_size, self.max_seq_len)
            )

            # Extend the kv cache for given request with any cached prefixes.
            cached_blocks: list[int] = []
            if self.radix_trie is not None:
                # Attempt to match all but the last token in the prompt. This is
                # because the model expects a prompt of length at least 1.
                _, cached_blocks = self.radix_trie.match_prefix(
                    prompt[:-1], node=data.node
                )

            # Compute the total sequence length and the number of pages required to store it.
            total_sequence_length = (
                data.cached_idx + len(prompt) + num_steps - 1
            )
            num_pages_required = ceildiv(total_sequence_length, self.page_size)

            # Compute the number of *new* pages we need to allocate.
            blocks_to_allocate = (
                num_pages_required - len(data.blocks) - len(cached_blocks)
            )

            total_blocks_to_allocate += blocks_to_allocate
            all_cache_hit_blocks.update(cached_blocks)

        num_evictable_blocks = 0
        if self.radix_trie is not None:
            # the blocks in the prefix cache that will be used by sequences in
            # this batch are no longer eligible for eviction / allocation.
            num_evictable_blocks = len(
                self.radix_trie.get_evictable_blocks() - all_cache_hit_blocks
            )

        num_free_blocks = len(self.available_blocks) + num_evictable_blocks

        return total_blocks_to_allocate <= num_free_blocks

    def get_num_cached_tokens(self, prompt: np.ndarray) -> int:
        """Returns the number of tokens in the CE prompt that are found in the
        prefix cache.
        """
        if self.radix_trie is None:
            return 0
        _, cached_blocks = self.radix_trie.match_prefix(prompt[:-1])
        return len(cached_blocks) * self.page_size

    def _fetch_prefix_caching(
        self,
        seq_id: int,
        data: _PagedCacheMetadata,
    ) -> list[int]:
        """Extend the kv cache for given request with any cached prefixes.

        This will increment the committed_idx and cached_idx if there is a cache
        hit. The prompt will be trimmed in the event that cached_idx is bumped.
        """
        assert self.radix_trie is not None

        # If there is only one committable token, that means that the prompt
        # is one token. We cannot reduce the prompt length any further since
        # the model expects a prompt of length at least 1.
        committable_tokens = data.committable_tokens[:-1]
        if len(committable_tokens) == 0:
            return []

        # Query trie for all but last token.
        data.node, prefix_blocks = self.radix_trie.match_prefix(
            committable_tokens, node=data.node
        )

        # Mark the prefix blocks we retrieved from the radix trie cache as
        # in use by this sequence so they don't get evicted prematurely.
        assert data.node is not None
        self.radix_trie.mark_in_use_by(data.node, seq_id)

        # Update the cache hit rate metrics.
        num_cache_hit_tokens = len(prefix_blocks) * self.page_size
        self.ce_cache_hit_tokens += num_cache_hit_tokens
        self.ce_all_tokens += len(committable_tokens)

        # If there is a block with partially cached tokens, we should release it
        # if the cache hit blocks already contain these tokens and more
        if data.committed_idx < data.cached_idx and num_cache_hit_tokens > 0:
            assert data.committed_idx + num_cache_hit_tokens > data.cached_idx
            partial_blocks = data.committable_blocks
            assert len(partial_blocks) == 1
            self.release_block(partial_blocks[0])
            data.blocks.pop()
            partial_tokens = data.cached_idx - data.committed_idx
            assert 0 < partial_tokens < self.page_size
            data.cached_idx -= partial_tokens
            assert data.committed_idx == data.cached_idx

        data.blocks.extend(prefix_blocks)
        # Bump the committed_idx since we got cache hits
        data.committed_idx += num_cache_hit_tokens
        data.cached_idx += num_cache_hit_tokens
        return prefix_blocks

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
        self._runtime_check()

        max_seq_len_in_batch = -1
        # before we start making any changes, validate that we won't over-write the cache
        for batch_idx, (seq_id, prompt) in enumerate(
            seq_ids_and_prompts.items()
        ):
            # Validate there aren't other inflight requests for this sequence.
            assert seq_id not in self.fetch_metadata

            # Add prompt and inflight tokens to the token array
            data = self.active_requests[seq_id]
            data.fetch(prompt, num_steps)

            # Compute the total sequence length
            if data.seq_len > max_seq_len_in_batch:
                max_seq_len_in_batch = data.seq_len

            assert data.seq_len <= self.max_seq_len, (
                f"seq_id: {seq_id} would overrun the max cache length of {self.max_seq_len} "
                f"with {len(prompt)} new tokens. Existing length: {data.cached_idx}"
            )

        max_num_pages = ceildiv(max_seq_len_in_batch, self.page_size)

        # Allocate the buffers containing metadata about the batch.
        # [0, total_num_pages) are the valid block ids and total_num_pages
        # denotes an unassigned block.
        batch_size = len(seq_ids_and_prompts)
        lut_table_np = np.full(
            (batch_size, max_num_pages), self.total_num_pages, dtype=np.uint32
        )
        cache_lengths_np = np.zeros((batch_size,), dtype=np.uint32)

        # Iterate over requests and query prefix cache
        all_cache_hit_blocks: set[int] = set()
        for batch_idx, (seq_id, prompt) in enumerate(
            seq_ids_and_prompts.items()
        ):
            # Ensure we've called claim for this sequence id.
            if seq_id not in self.active_requests:
                raise ValueError(f"seq_id: {seq_id} not in active requests.")

            if self.radix_trie is not None:
                data = self.active_requests[seq_id]
                # bump the committed_idx, and possibly the cached_idx
                prefix_blocks = self._fetch_prefix_caching(seq_id, data)
                all_cache_hit_blocks.update(prefix_blocks)
                # Possibly trim the input prompt.
                seq_ids_and_prompts[seq_id] = data.prompt_tokens

        # Determine the number of pages required for each sequence.
        max_seq_length = 0
        max_cache_length = 0
        total_sequence_length = 0
        total_blocks_to_allocate = 0
        blocks_to_allocate_by_seq = {}
        for batch_idx, (seq_id, prompt) in enumerate(
            seq_ids_and_prompts.items()
        ):
            data = self.active_requests[seq_id]

            # Get the existing cache length for this sequence.
            cache_length = data.cached_idx
            cache_lengths_np[batch_idx] = cache_length

            # Update the maximum lengths seen so far.
            max_seq_length = max(max_seq_length, len(prompt))
            max_cache_length = max(max_cache_length, cache_length)

            # Compute the total sequence length and the number of pages required to store it.
            total_sequence_length += data.seq_len
            num_pages_required = ceildiv(data.seq_len, self.page_size)

            # Compute the number of *new* pages we need to allocate.
            num_new_pages = num_pages_required - len(data.blocks)
            assert num_new_pages >= 0
            blocks_to_allocate_by_seq[seq_id] = num_new_pages
            total_blocks_to_allocate += num_new_pages

        # Check if we have enough free blocks to service all requests.
        num_evictable_blocks = 0
        if self.radix_trie is not None:
            # the blocks in the prefix cache that will be used by sequences in
            # this batch are no longer eligible for eviction / allocation.
            num_evictable_blocks = len(
                self.radix_trie.get_evictable_blocks() - all_cache_hit_blocks
            )
        num_free_blocks = len(self.available_blocks) + num_evictable_blocks
        if total_blocks_to_allocate > num_free_blocks:
            raise RuntimeError(
                f"Not enough free blocks to service all {len(seq_ids_and_prompts)} requests.\n"
                f"Need an additional {total_blocks_to_allocate} blocks to store KV projections for all {total_sequence_length} tokens.\n"
                f"But only {num_free_blocks} out of {self.total_num_pages} cache blocks are available to be allocated.\n"
                f"You must restart your process and set a smaller batch size or max sequence length.\n"
            )

        # Allocate additional pages for each request in the batch
        for batch_idx, (seq_id, num_new_pages) in enumerate(
            blocks_to_allocate_by_seq.items()
        ):
            data = self.active_requests[seq_id]

            # Assign some new pages to this request.
            for _ in range(num_new_pages):
                next_block = self.alloc_block()
                data.blocks.append(next_block)

            # Populate the lookup table with the new pages.
            for i, block_idx in enumerate(data.blocks):
                lut_table_np[batch_idx, i] = block_idx

        # Build a tensor of maximum lengths. Each step slices the first row to
        # advance to the values for the next row.
        max_lengths_host = build_max_lengths_tensor(
            num_steps, max_seq_length, max_cache_length
        )

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

        self._runtime_check()

        return ret_list

    def input_symbols(
        self,
    ) -> list[PagedCacheInputSymbols]:
        return [
            PagedCacheInputSymbols(
                kv_blocks=TensorType(
                    self.params.dtype,
                    shape=self.block_shape(is_parameterized=True),
                    device=DeviceRef(self.devices[i].label, self.devices[i].id),
                ),
                cache_lengths=TensorType(
                    DType.uint32,
                    shape=["batch_size"],
                    device=DeviceRef(self.devices[i].label, self.devices[i].id),
                ),
                lookup_table=TensorType(
                    DType.uint32,
                    shape=["batch_size", "max_num_pages"],
                    device=DeviceRef(self.devices[i].label, self.devices[i].id),
                ),
                max_lengths=TensorType(
                    DType.uint32, shape=["steps_remaining", 2]
                ),
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
            self.active_requests[seq_id] = _PagedCacheMetadata(
                self.page_size, self.max_seq_len
            )
        return seq_ids

    def external_claim(self, seq_ids: list[int]) -> None:
        """Variant of the above where sequence ids are reserved externally."""
        super().external_claim(seq_ids)
        for seq_id in seq_ids:
            self.active_requests[seq_id] = _PagedCacheMetadata(
                self.page_size, self.max_seq_len
            )

    def _count_all_pages(self) -> int:
        available_blocks = self.available_blocks
        prefix_cache_blocks = set()
        if self.radix_trie is not None:
            prefix_cache_blocks = self.radix_trie.get_all_blocks()
        uncommitted_blocks = set()
        for seq_id in self.active_requests:
            uncommitted_blocks.update(
                self.active_requests[seq_id].uncommitted_blocks
            )
        return len(available_blocks | prefix_cache_blocks | uncommitted_blocks)

    def release(self, seq_id: int) -> None:
        """Release `seq_id` provided, marking this sequence as complete.
        This returns the seq_id back to the available pool of cache memory,
        allowing it to be reused when a new sequence is claimed.
        """
        super().release(seq_id)
        data = self.active_requests[seq_id]

        if self.radix_trie is not None:
            # mark the prefix blocks as not in use by this sequence so they can
            # potentially be evicted when we need more memory
            assert data.node is not None
            self.radix_trie.mark_not_in_use_by(data.node, seq_id)

        for block in data.blocks:
            self.release_block(block)
        del self.active_requests[seq_id]

    def _step_prefix_caching(
        self, seq_id: int, data: _PagedCacheMetadata
    ) -> None:
        """Now that we have written to the inflight blocks, we will try to commit
        them to the radix trie.

        This increments the committed_idx. We guarantee that the number of committed
        tokens will be a multiple of the page size. There may be some uncommitted
        tokens left over due to there being a partial page at the end. Thus the
        number of uncommitted tokens will always be less than the page size.
        """
        assert self.radix_trie is not None

        committable_tokens = data.committable_tokens_aligned
        data.node, existing_blocks = self.radix_trie.match_prefix(
            committable_tokens, node=data.node
        )

        # If we computed a kv entry for a token that was already cached,
        # we will just release that block we just computed.
        for b0, b1 in zip(existing_blocks, data.committable_blocks_aligned):
            if b0 != b1:
                self.release_block(b1)

        # Replace the inflight blocks with the existing prefix blocks.
        committed_block_idx = data.committed_idx // self.page_size
        data.blocks[
            committed_block_idx : committed_block_idx + len(existing_blocks)
        ] = existing_blocks
        data.committed_idx += len(existing_blocks) * self.page_size

        committable_tokens = data.committable_tokens_aligned
        committable_blocks = data.committable_blocks_aligned
        assert len(committable_tokens) % self.page_size == 0
        assert (
            len(committable_tokens) == len(committable_blocks) * self.page_size
        )

        # If there are any tokens to commit, insert them into the prefix cache.
        data.node = self.radix_trie.insert(
            committable_tokens,
            committable_blocks,
            node=data.node,
        )
        data.committed_idx += len(committable_tokens)

        # Mark the recently committed blocks as in use by this sequence
        # so they don't get evicted prematurely.
        assert data.node is not None
        self.radix_trie.mark_in_use_by(data.node, seq_id)

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
        self._runtime_check()

        for seq_id, new_tokens in seq_ids_and_new_tokens.items():
            if seq_id not in self.active_requests:
                raise ValueError(f"seq_id: {seq_id} not in active requests.")

            # Write the new tokens into the token array and bump the cached_idx
            data = self.active_requests[seq_id]
            data.step(new_tokens)

            if self.radix_trie is not None:
                # Bump the committed_idx
                self._step_prefix_caching(seq_id, data)

            expected_num_pages = ceildiv(data.seq_len, self.page_size)
            actual_num_pages = len(data.blocks)
            if expected_num_pages != actual_num_pages:
                raise ValueError(
                    f"Mismatch between expected and actual number of pages for seq_id: {seq_id}. Expected: {expected_num_pages}, Actual: {actual_num_pages}"
                )

        self._runtime_check()
