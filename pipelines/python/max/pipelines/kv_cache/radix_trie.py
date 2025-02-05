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

import heapq
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

TokenId = Any
BlockId = Any
SeqId = int


def _token_prefix_match_len(
    tokens0: np.ndarray, tokens1: np.ndarray, page_size: int
) -> int:
    """Computes the length of maximum shared prefix of two tokens, aligned by
    `page_size`.

    e.g: _token_prefix_match_len(["i", "like", "dogs"], ["i", "like", "cats"], page_size = 1) => 2
         _token_prefix_match_len(["i", "like", "dogs"], ["we", "like", "cats"], page_size = 1) => 0
         _token_prefix_match_len(["i", "like", "dogs"], ["i", "like", "dogs", "and", "cats"], page_size = 1) => 3
    """
    assert len(tokens0) % page_size == 0
    assert len(tokens1) % page_size == 0
    shorter_len = min(len(tokens0), len(tokens1))
    for i in range(0, shorter_len, page_size):
        if (tokens0[i : i + page_size] != tokens1[i : i + page_size]).any():
            return i
    return shorter_len


def _token_to_key(tokens: np.ndarray, page_size: int) -> tuple[TokenId, ...]:
    assert len(tokens) >= page_size, (
        f"tokens must be at least page_size ({page_size}) long but is only {len(tokens)} tokens"
    )
    return tuple(tokens[:page_size])


class TrieNode:
    """A TrieNode consists of a list of tokens and blocks.

    - Tokens are the ids of the tokens in the sequence.
    - Blocks are the offsets into the KVCache region that back the KV entries
      for a given token. I.e: the page index
    """

    def __init__(self) -> None:
        """Constructs a TrieNode."""
        self.children: Dict[tuple[TokenId, ...], TrieNode] = {}
        # Typically in a map, we would have keys mapping to values.
        # To avoid collision with KV cache terminology, we call them tokens and blocks.
        #
        # Only the root should have empty tokens/blocks
        self.tokens: np.ndarray = np.array([])
        self.blocks: List[BlockId] = []
        # Only the root should have a null parent
        self.parent: Optional[TrieNode] = None
        # Sequences that are using the blocks owned by this trie node
        # The node can only be evicted if self.active_seqs is empty
        self.active_seqs: Set[SeqId] = set()
        # Last access time is used to determine which nodes to evict first
        self.last_access_time: float = time.time()

    def __lt__(self, other):
        """Comparison function for use by heapq"""
        return self.last_access_time < other.last_access_time


class RadixTrie:
    """This RadixTrie is specially designed for prefix caching in paged attention.

    The RadixTrie allows for efficient insertion and matching of sequences. It
    matches each prefix of tokens in a sequence to its corresponding blocks.
    Compared to a naive trie, the RadixTrie allows storing multiple tokens in a
    single node for less indirection and faster access.

    Blocks in the RadixTrie should be immutable and committed. If it is in the
    RadixTrie, it is eligible for sharing. An inflight or uncommitted block that
    is being written to by a sequence should not be in the RadixTrie.

    The RadixTrie allows for an LRU eviction policy for its leaves. We only allow
    evictions if no active sequences are using the node.

    Currently, the RadixTrie assumes that the paged KVCache page size is 1.

    This implementation is based off of SGLang:
        - https://github.com/sgl-project/sglang/blob/337fe53ac41c68d6f171ef3b446f55eb0e98f77c/python/sglang/srt/mem_cache/radix_cache.py#L58
    """

    def __init__(self, page_size: int = 1) -> None:
        """Constructs a RadixTrie."""
        self.root = TrieNode()
        self.page_size = page_size
        self.evictable_blocks: set[BlockId] = set()
        self.all_blocks: set[BlockId] = set()

    def _check_node_valid(self, node: TrieNode):
        """Rudimentary checks of data structure invariants for TrieNode."""
        if self.root == node:
            assert len(node.tokens) == 0
            assert len(node.blocks) == 0
            assert not node.parent
        else:
            assert len(node.tokens) > 0
            assert len(node.blocks) > 0
            assert node.parent
            assert len(node.tokens) % self.page_size == 0
            assert len(node.tokens) // self.page_size == len(node.blocks)

    def insert(
        self,
        tokens: Union[np.ndarray, List[TokenId]],
        blocks: List[BlockId],
        node: Optional[TrieNode] = None,
    ) -> TrieNode:
        """Inserts `tokens` and `blocks` into the trie.

        We assume that each block contains exactly one token so the length of both
        input lists must match.

        Args:
            tokens: Tokens to insert into trie
            blocks: KV cache block for each token
            node: Node to begin insertion at. If this is not a leaf node, blocks
                  in the tree are overwritten.
        Return:
            trie_node: Node corresponding to end of the sequence where future
                       generated tokens can be inserted
        """

        if isinstance(tokens, list):
            tokens = np.array(tokens)

        def insert_helper(
            prev: TrieNode, tokens: np.ndarray, blocks: List[BlockId]
        ) -> TrieNode:
            if len(tokens) == 0:
                return prev

            key = _token_to_key(tokens, self.page_size)
            if key not in prev.children:
                # insert new node
                curr = TrieNode()
                curr.parent = prev
                curr.tokens = tokens
                curr.blocks = blocks
                prev.children[key] = curr
                self.evictable_blocks.update(blocks)
                self.all_blocks.update(blocks)

            curr = prev.children[key]
            prefix_len = _token_prefix_match_len(
                curr.tokens, tokens, self.page_size
            )

            if prefix_len == len(curr.tokens) and prefix_len == len(tokens):
                return curr

            unmatched_tokens = tokens[prefix_len:]
            unmatched_blocks = blocks[prefix_len // self.page_size :]
            if prefix_len == len(curr.tokens):
                return insert_helper(curr, unmatched_tokens, unmatched_blocks)

            # this means that we got a partial match and must split the curr node
            #   (prev) -> (curr)
            # becomes:
            #   (prev) -> (parent) -> (child)
            (parent, _) = self._split_node(curr, prefix_len)
            unmatched_tokens = tokens[prefix_len:]
            unmatched_blocks = blocks[prefix_len // self.page_size :]
            return insert_helper(parent, unmatched_tokens, unmatched_blocks)

        if len(tokens) % self.page_size != 0:
            msg = f"Insertion failed: the number of tokens is not divisible by the page size. len(tokens) == {len(tokens)} but page_size == {self.page_size}."
            raise ValueError(msg)
        if len(tokens) // self.page_size != len(blocks):
            msg = f"Insertion failed: the number of tokens and blocks do not match. len(tokens) // self.page_size == {len(tokens)} // {self.page_size} == {len(tokens) // self.page_size} but len(blocks) == {len(blocks)}."
            raise ValueError(msg)
        if len(tokens) == 0:
            msg = "Insertion failed: Attempted to insert 0 tokens into trie. Please provide at least one token to insert."
            raise ValueError(msg)

        # clone to avoid mutating the original lists
        tokens = tokens.copy()
        blocks = blocks.copy()

        if node is None:
            node = self.root
        return insert_helper(node, tokens, blocks)

    def match_prefix(
        self,
        tokens: Union[np.ndarray, List[TokenId]],
        node: Optional[TrieNode] = None,
    ) -> Tuple[TrieNode, List[BlockId]]:
        """Matches the input `tokens` with the contents of the trie.

        Args:
            tokens: tokens to search the trie for
            node: Node to begin matching at.
        Return:
            Tuple containing:
                - trie_node: Node corresponding to end of matched prefix where
                             future generated tokens can be inserted. This is
                             a leaf node.
                - block_list: KV cache blocks for matched prefix
        """
        if isinstance(tokens, list):
            tokens = np.array(tokens)

        def match_prefix_helper(
            prev: TrieNode, tokens: np.ndarray, blocks: List[BlockId]
        ) -> TrieNode:
            if len(tokens) == 0:
                return prev

            key = _token_to_key(tokens, self.page_size)
            if key not in prev.children:
                return prev

            curr = prev.children[key]
            prefix_len = _token_prefix_match_len(
                curr.tokens, tokens, self.page_size
            )
            if prefix_len < len(curr.tokens):
                #   (prev) -> (curr)
                # becomes:
                #   (prev) -> (parent) -> (child)
                (parent, _) = self._split_node(curr, prefix_len)
                blocks.extend(parent.blocks)
                return parent
            else:
                blocks.extend(curr.blocks)
                return match_prefix_helper(curr, tokens[prefix_len:], blocks)

        if len(tokens) == 0:
            msg = "Match failed: Attempted to match 0 tokens in trie. Please provide at least one token to match."
            raise ValueError(msg)

        # AIPIPE-323: We should support partial block matches
        # truncate tokens to be divisible by page size
        tokens = tokens[: len(tokens) // self.page_size * self.page_size]

        blocks: List[BlockId] = []
        if node is None:
            node = self.root
        leaf_node = match_prefix_helper(node, tokens, blocks)
        return leaf_node, blocks

    def _split_node(
        self, node: TrieNode, split_len: int
    ) -> Tuple[TrieNode, TrieNode]:
        """Splits the provided node into two.

        The resulting parent node receives exactly `split_len` tokens/blocks, and
        the child receives the remainder.

           before   │  after splitting w/ `split_len` = 2
                    │  ┌────────┐
                    │  │  ab    │ (parent)
        ┌────────┐  │  └───▲────┘
        │ abcdef │  │      │
        └────────┘  │  ┌───▼────┐
                    │  │  cdef  │ (child)
                    │  └────────┘
        """
        assert node != self.root
        assert split_len > 0
        assert split_len % self.page_size == 0

        parent = TrieNode()
        child = node
        parent.tokens, child.tokens = (
            child.tokens[:split_len],
            child.tokens[split_len:],
        )
        parent.blocks, child.blocks = (
            child.blocks[: split_len // self.page_size],
            child.blocks[split_len // self.page_size :],
        )

        parent.parent = child.parent
        assert parent.parent is not None
        assert len(parent.tokens) > 0
        parent_key = _token_to_key(parent.tokens, self.page_size)
        parent.parent.children[parent_key] = parent
        child_key = _token_to_key(child.tokens, self.page_size)
        parent.children = {child_key: child}
        child.parent = parent

        parent.last_access_time = child.last_access_time
        parent.active_seqs = child.active_seqs.copy()

        self._check_node_valid(parent)
        self._check_node_valid(child)
        return (parent, child)

    def mark_in_use_by(self, node: TrieNode, seq_id: SeqId):
        """Climb up the trie starting from node, marking each node as being
        in use by this seq."""

        curr = node
        while curr != self.root:
            assert curr is not None
            # optimization: if this node is already marked as using this sequence,
            # assume that it is already marked for its parents as well
            if seq_id in curr.active_seqs:
                break
            if not curr.active_seqs:
                self.evictable_blocks -= set(curr.blocks)
            curr.active_seqs.add(seq_id)
            assert curr.parent is not None
            curr = curr.parent

    def mark_not_in_use_by(self, node: TrieNode, seq_id: SeqId):
        """Climb up the trie starting from node, marking each node as no longer
        in use by this seq. Since nodes without any users may be eligible for
        eviction, we also update its last_access_time."""

        curr = node
        while curr != self.root:
            assert curr is not None
            assert seq_id in curr.active_seqs
            curr.last_access_time = time.time()
            curr.active_seqs.remove(seq_id)
            if not curr.active_seqs:
                self.evictable_blocks.update(curr.blocks)
            assert curr.parent is not None
            curr = curr.parent

    def evict_blocks(self, desired_num_evicted: int) -> List[BlockId]:
        """Attempt to evict at most `desired_num_evicted` blocks from trie."""

        def collect_leaves() -> List[TrieNode]:
            leaves: List[TrieNode] = []
            stack: List[TrieNode] = [self.root]

            while stack:
                curr = stack.pop()
                if len(curr.children) == 0:
                    leaves.append(curr)
                else:
                    stack.extend(curr.children.values())
            return leaves

        leaves = collect_leaves()
        heapq.heapify(leaves)

        evicted_blocks: List[BlockId] = []

        while len(evicted_blocks) < desired_num_evicted and len(leaves) > 0:
            leaf = heapq.heappop(leaves)

            # don't evict the root
            if leaf == self.root:
                break
            # don't evict node if in use by any seq
            if len(leaf.active_seqs) > 0:
                continue

            remaining_blocks_to_evict = desired_num_evicted - len(
                evicted_blocks
            )
            blocks_to_evict_from_leaf = min(
                remaining_blocks_to_evict, len(leaf.blocks)
            )
            assert blocks_to_evict_from_leaf > 0

            # evict up to `left_to_evict` blocks from the leaf
            evicted_blocks.extend(leaf.blocks[-blocks_to_evict_from_leaf:])
            key = _token_to_key(leaf.tokens, self.page_size)
            leaf.tokens = leaf.tokens[
                : -(blocks_to_evict_from_leaf * self.page_size)
            ]
            leaf.blocks = leaf.blocks[:-blocks_to_evict_from_leaf]

            assert len(leaf.tokens) % self.page_size == 0
            assert len(leaf.tokens) // self.page_size == len(leaf.blocks)

            if len(leaf.tokens) == 0:
                # delete leaf node
                assert leaf.parent is not None
                del leaf.parent.children[key]

                # parent of leaf is now potentially a leaf
                if len(leaf.parent.children) == 0:
                    heapq.heappush(leaves, leaf.parent)

        self.evictable_blocks.difference_update(evicted_blocks)
        self.all_blocks.difference_update(evicted_blocks)
        if len(evicted_blocks) < desired_num_evicted:
            assert not self.evictable_blocks

        return evicted_blocks

    def get_all_blocks(self) -> set[BlockId]:
        """Returns the total number of blocks in the trie."""
        return self.all_blocks

    def get_evictable_blocks(self) -> set[BlockId]:
        """Returns the number of blocks that are eligible for eviction."""
        return self.evictable_blocks

    def pretty_format(self, print_blocks: bool = False) -> List[str]:
        """Formats the contents of the trie."""

        def helper(node: TrieNode, indent: int, lines: List[str]):
            for _, child in node.children.items():
                tokens = child.tokens
                token_list = tokens.tolist()
                if print_blocks:
                    lines.append(f"{'-' * indent}{token_list} : {child.blocks}")
                else:
                    lines.append(f"{'-' * indent}{token_list}")
                helper(child, indent + 2, lines)

        lines: List[str] = []
        helper(self.root, 0, lines)
        return lines

    def pretty_print(self, print_blocks: bool = True):
        """Prints the contents of the trie."""
        for line in self.pretty_format(print_blocks):
            print(line)
