# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import heapq
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

TokenId = Any
BlockId = Any
SeqId = int


def _token_prefix_match_len(tokens0: np.ndarray, tokens1: np.ndarray) -> int:
    """computes the length of maximum shared prefix of two tokens
    e.g: _token_prefix_match_len(["i", "like", "dogs"], ["i", "like", "cats"]) => 2
         _token_prefix_match_len(["i", "like", "dogs"], ["we", "like", "cats"]) => 0
         _token_prefix_match_len(["i", "like", "dogs"], ["i", "like", "dogs", "and", "cats"]) => 3
    """
    for i, (t0, t1) in enumerate(zip(tokens0, tokens1)):
        if t0 != t1:
            return i
    return min(len(tokens0), len(tokens1))


def _starts_with(tokens: np.ndarray, prefix: np.ndarray) -> bool:
    """checks if tokens begins with prefix
    e.g: _starts_with(["i", "like", "dogs"], ["i", "like"]) => True
         _starts_with(["i", "like", "dogs"], ["we", "like"]) => False
    """
    if len(prefix) > len(tokens):
        return False
    return (prefix == tokens[: len(prefix)]).all()


class TrieNode:
    """A TrieNode consists of a list of tokens and blocks.

    - Tokens are the ids of the tokens in the sequence.
    - Blocks are the offsets into the KVCache region that back the KV entries
      for a given token. I.e: the page index
    """

    def __init__(self) -> None:
        """Constructs a TrieNode."""
        self.children: Dict[TokenId, TrieNode] = {}
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

    def __init__(self) -> None:
        """Constructs a RadixTrie."""
        self.root = TrieNode()

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
            assert len(node.tokens) == len(node.blocks)

        for tok, child in node.children.items():
            assert len(child.tokens) > 0
            assert child.tokens[0] == tok

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

            if tokens[0] not in prev.children:
                # insert new node
                curr = TrieNode()
                curr.parent = prev
                curr.tokens = tokens
                curr.blocks = blocks
                prev.children[tokens[0]] = curr

            curr = prev.children[tokens[0]]
            prefix_len = _token_prefix_match_len(curr.tokens, tokens)

            if prefix_len == len(curr.tokens) and prefix_len == len(tokens):
                assert (curr.tokens == tokens).all()
                return curr

            unmatched_tokens = tokens[prefix_len:]
            unmatched_blocks = blocks[prefix_len:]
            if prefix_len == len(curr.tokens):
                assert _starts_with(tokens, curr.tokens)
                return insert_helper(curr, unmatched_tokens, unmatched_blocks)

            # this means that we got a partial match and must split the curr node
            #   (prev) -> (curr)
            # becomes:
            #   (prev) -> (parent) -> (child)
            (parent, _) = self._split_node(curr, prefix_len)
            assert _starts_with(tokens, parent.tokens)
            unmatched_tokens = tokens[prefix_len:]
            unmatched_blocks = blocks[prefix_len:]
            return insert_helper(parent, unmatched_tokens, unmatched_blocks)

        if len(tokens) != len(blocks):
            msg = f"Insertion failed: the number of tokens and blocks do not match. len(tokens) == {len(tokens)} but len(blocks) == {len(blocks)}."
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
            if tokens[0] not in prev.children:
                return prev

            curr = prev.children[tokens[0]]
            prefix_len = _token_prefix_match_len(curr.tokens, tokens)
            if prefix_len < len(curr.tokens):
                #   (prev) -> (curr)
                # becomes:
                #   (prev) -> (parent) -> (child)
                (parent, _) = self._split_node(curr, prefix_len)
                assert _starts_with(tokens, parent.tokens)
                blocks.extend(parent.blocks)
                return parent
            else:
                blocks.extend(curr.blocks)
                return match_prefix_helper(curr, tokens[prefix_len:], blocks)

        if len(tokens) == 0:
            msg = "Match failed: Attempted to match 0 tokens in trie. Please provide at least one token to match."
            raise ValueError(msg)

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
        parent = TrieNode()
        child = node
        parent.tokens, child.tokens = (
            child.tokens[:split_len],
            child.tokens[split_len:],
        )
        parent.blocks, child.blocks = (
            child.blocks[:split_len],
            child.blocks[split_len:],
        )

        parent.parent = child.parent
        assert parent.parent is not None
        parent.parent.children[parent.tokens[0]] = parent
        parent.children = {child.tokens[0]: child}
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
            assert seq_id in curr.active_seqs, f"{curr.active_seqs}, {seq_id}"
            curr.last_access_time = time.time()
            curr.active_seqs.remove(seq_id)
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
                remaining_blocks_to_evict, len(leaf.tokens)
            )
            # evict up to `left_to_evict` blocks from the leaf
            evicted_blocks.extend(leaf.blocks[-blocks_to_evict_from_leaf:])
            first_tok = leaf.tokens[0]
            leaf.tokens = leaf.tokens[:-blocks_to_evict_from_leaf]
            leaf.blocks = leaf.blocks[:-blocks_to_evict_from_leaf]

            if len(leaf.tokens) == 0:
                # delete leaf node
                assert leaf.parent is not None
                del leaf.parent.children[first_tok]

                # parent of leaf is now potentially a leaf
                if len(leaf.parent.children) == 0:
                    heapq.heappush(leaves, leaf.parent)

        return evicted_blocks

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
