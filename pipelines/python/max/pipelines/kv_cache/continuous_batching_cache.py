# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Continuous Batching enabled KV cache for the Transformer leveraging the mo.opaque pattern."""

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import List

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    Device as GraphDevice,
)
from max.graph import (
    Graph,
    TensorType,
    TensorValue,
    _OpaqueType,
    _OpaqueValue,
    ops,
)

from .cache_params import KVCacheParams
from .manager import KVCacheManager


class ContinuousBatchingKVCacheType(_OpaqueType):
    """Continuous Mojo KV Cache graph type."""

    def __init__(self) -> None:
        """Creates an opaque type containing a continuous batching KV Cache."""
        super().__init__("ContinuousBatchingKVCache")


class ContinuousBatchingKVCacheCollectionType(_OpaqueType):
    """The graph type for a "view" of the cache for the given sequences in the
    batch.

    This object does not own the underlying buffers in k_cache and v_cache,
    it's borrowing them from the BlockWrappers in our ContinuousKVCacheManager.
    It does own the Pointer[NDBuffer[type, 3]] and valid_lengths buffer
    """

    def __init__(self) -> None:
        """Creates an opaque type containing a continuous batching KV cache collection."""
        super().__init__("ContinuousBatchingKVCacheCollection")


class ContinuousBatchingKVCache(_OpaqueValue):
    """Continuous Mojo KV cache graph value."""


class ContinuousBatchingKVCacheCollection(_OpaqueValue):
    """The graph value for a view of the KV cache."""


class FetchContinuousBatchingKVCacheCollection:
    def __init__(self, kv_params: KVCacheParams) -> None:
        self.kv_params = kv_params

    def __call__(
        self,
        blocks: TensorValue,  # NDBuffer[type, 6, Self.blocks_shape]
        cache_lengths: TensorValue,  # NDBuffer[DType.uint32, 1],
        lookup_table: TensorValue,  # NDBuffer[DType.uint32, 1],
        is_cache_empty: TensorValue,
    ) -> ContinuousBatchingKVCacheCollection:
        """Constructs a ContinuousBatchingKVCacheCollection for use downstream."""

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

        if lookup_table.dtype != DType.uint32:
            msg = (
                "expected lookup_table to be dtype: uint32, got"
                f" {lookup_table.dtype}"
            )
            raise ValueError(msg)

        op_name = f"continuous_batching_kv_cache_collection_h{self.kv_params.n_kv_heads_per_device}_d{self.kv_params.head_dim}_bshd"
        return ContinuousBatchingKVCacheCollection(
            ops.custom(
                op_name,
                values=[
                    blocks,
                    cache_lengths,
                    lookup_table,
                    is_cache_empty,
                ],
                out_types=[ContinuousBatchingKVCacheCollectionType()],
            )[0].opaque
        )


class ContinuousBatchingKVCacheManager(KVCacheManager):
    def __init__(
        self,
        params: KVCacheParams,
        max_cache_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        devices: List[Device],
        session: InferenceSession,
    ) -> None:
        super().__init__(
            params=params,
            max_cache_batch_size=max_cache_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            devices=devices,
            session=session,
        )

        # Allocate memory for the KV cache blocks.
        self.blocks: List[Tensor] = []
        for i in range(len(self.devices)):
            self.blocks.append(
                Tensor.zeros(
                    self.block_shape(self.max_cache_batch_size),
                    self.params.dtype,
                    device=self.devices[i],
                )
            )

        self._increment_cache_lengths_graph = (
            self._create_increment_cache_lengths_graph(self.session)
        )

    @classmethod
    def estimated_memory_size(
        cls,
        params: KVCacheParams,
        max_cache_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        devices: List[Device],
    ) -> int:
        cache_size = (
            reduce(
                mul,
                cls._block_shape(
                    params, max_cache_batch_size, max_seq_len, num_layers
                ),
            )
            * params.dtype.size_in_bytes
        )
        lengths_size = max_cache_batch_size * DType.uint32.size_in_bytes
        lookup_table_size = max_cache_batch_size * DType.uint32.size_in_bytes
        size = cache_size + lengths_size + lookup_table_size
        return size * len(devices)

    def _create_increment_cache_lengths_graph(self, session: InferenceSession):
        cache_lengths_types = [
            self.input_symbols()[i][1] for i in range(len(self.devices))
        ]

        input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=GraphDevice(self.devices[0].label, self.devices[0].id),
        )

        with Graph(
            "update_cache_lengths",
            input_types=[input_row_offsets_type, *cache_lengths_types],
        ) as graph:
            inp_row_offset, *cache_lengths = graph.inputs
            # broadcast the inp_row_offset to all devices (naive)
            # get rid of this if statement after #51465 merges
            if len(self.devices) > 1:
                input_row_offsets = [
                    inp_row_offset.to(GraphDevice(d.label, d.id))  # type: ignore
                    for d in self.devices
                ]
            else:
                input_row_offsets = [inp_row_offset]
            outputs = []
            for i in range(len(self.devices)):
                right_slice = input_row_offsets[i][1:].rebind(
                    cache_lengths[i].shape  # type: ignore
                )
                left_slice = input_row_offsets[i][
                    : input_row_offsets[i].shape[0] - 1
                ].rebind(
                    cache_lengths[i].shape  # type: ignore
                )
                increment_amount = right_slice - left_slice
                outputs.append(cache_lengths[i] + increment_amount)
            graph.output(*outputs)

        return session.load(graph)

    def fetch(
        self, seq_ids: List[int]
    ) -> List[tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Fetches the KV cache state for the given sequence IDs.

        This method retrieves the current cache state for a batch of sequences, including their
        cache lengths and lookup information. It's used during token generation to access
        previously cached key/value pairs.

        Args:
            seq_ids: List of sequence IDs to fetch cache state for. Each ID must be within
                    the max_cache_batch_size and must exist in the current cache.

        Returns:
            List of tuples for each device containing:
            - blocks: Tensor containing the KV cache blocks
            - cache_lengths: Tensor of current cache lengths for each sequence
            - lookup_table: Tensor mapping sequence IDs to cache positions
            - is_cache_empty: Boolean tensor indicating if all sequences have empty caches

        Raises:
            ValueError: If any seq_id exceeds max_cache_batch_size or doesn't exist in cache
        """
        active_batch_size = len(seq_ids)

        # Lookup table and seq_ids are redundant identical tensors.
        lookup_table_tensor = Tensor.from_numpy(np.array(seq_ids, np.uint32))
        cache_lengths_np = np.zeros(active_batch_size, np.uint32)
        is_cache_empty = True
        for i, seq_id in enumerate(seq_ids):
            if seq_id > self.max_cache_batch_size:
                msg = (
                    f"seq_id: {seq_id}, beyond max_cache_batch_size, you may"
                    " want to increase `max_cache_batch_size` in the pipeline"
                    " config."
                )
                raise ValueError(msg)
            elif seq_id not in self.cache_lengths:
                raise ValueError(f"seq_id: {seq_id} not currently in cache.")

            cache_len = self.cache_lengths[seq_id]
            cache_lengths_np[i] = cache_len
            if cache_len != 0:
                is_cache_empty = False

        cache_lengths = [
            Tensor.from_numpy(cache_lengths_np).to(d) for d in self.devices
        ]
        lookup_table_tensor_list = [
            lookup_table_tensor.to(self.devices[i])
            for i in range(len(self.devices))
        ]
        is_cache_empty_buf = (
            self.true_tensor if is_cache_empty else self.false_tensor
        )

        return [
            (
                self.blocks[i],
                cache_lengths[i],
                lookup_table_tensor_list[i],
                is_cache_empty_buf,
            )
            for i in range(len(self.devices))
        ]

    def block_shape(self, n_sequences: int) -> list[int]:
        """Returns the shape of the KV cache blocks for the given number of sequences.

        Defines the 6-dimensional shape of the cache blocks used to store key and value
        tensors for transformer attention. The dimensions represent:
        [n_sequences, 2, num_layers, max_seq_len, n_kv_heads_per_device, head_dim]
        where 2 represents separate storage for keys and values.

        Args:
            n_sequences: Number of sequences that will be cached

        Returns:
            List describing the shape of the cache blocks with dimensions for:
            sequences, key/value split, layers, sequence length, attention heads, and head dimension
        """
        return self._block_shape(
            self.params,
            n_sequences,
            self.max_seq_len,
            self.num_layers,
        )

    @staticmethod
    def _block_shape(
        params: KVCacheParams,
        n_sequences: int,
        max_seq_len: int,
        num_layers: int,
    ) -> list[int]:
        return [
            n_sequences,
            2,
            num_layers,
            max_seq_len,
            params.n_kv_heads_per_device,
            params.head_dim,
        ]

    def increment_cache_lengths(
        self,
        kv_cache_inputs: List[tuple[Tensor, Tensor, Tensor, Tensor]],
        prev_model_inputs: tuple[Tensor, ...],
    ) -> List[tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Prepares cache inputs for the next token in multistep execution.

        Updates the cache lengths for the next inference step without requiring device
        synchronization or memory copies. This is crucial for maintaining performance
        during multi-token generation.

        Args:
            kv_cache_inputs: Current cache state tuples (blocks, lengths, lookup, empty flag)
            prev_model_inputs: Previous model inputs including row offsets

        Returns:
            Updated cache input tuples with incremented lengths and is_cache_empty=False
            since only the first step can be context encoding
        """
        _, input_row_offsets = prev_model_inputs
        blocks = [kv_cache_inputs[i][0] for i in range(len(self.devices))]
        cache_lengths = [
            kv_cache_inputs[i][1] for i in range(len(self.devices))
        ]
        lookup_table = [kv_cache_inputs[i][2] for i in range(len(self.devices))]

        # Update the cache_lengths of our batch by the previous sequence length
        updated_cache_lengths = self._increment_cache_lengths_graph.execute(
            input_row_offsets, *cache_lengths
        )

        # Return our updated batch. We hard-code the is_cache_empty flag to
        # False because only the first step could be context encoding.
        for i in range(len(self.devices)):
            kv_cache_inputs[i] = (
                blocks[i],
                updated_cache_lengths[i],
                lookup_table[i],
                self.false_tensor,
            )
        return kv_cache_inputs

    def input_symbols(
        self,
    ) -> List[tuple[TensorType, TensorType, TensorType, TensorType]]:
        """Returns the expected input tensor types for `fetch` on each device.

        Defines the tensor specifications needed by the cache implementation, including
        shapes and data types. This is used for graph construction and validation.

        Returns:
            List of tuples for each device containing TensorTypes for:
            - KV cache blocks: 6D tensor for storing keys and values
            - Cache lengths: 1D tensor tracking sequence lengths
            - Lookup table: 1D tensor mapping sequence IDs to cache positions
            - Cache empty flag: Scalar boolean tensor
        """
        return [
            (
                # kv_blocks
                TensorType(
                    self.params.dtype,
                    shape=[
                        "num_blocks",
                        2,
                        "num_layers",
                        "max_seq_len",
                        "num_kv_heads",
                        "head_dim",
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
                    shape=["batch_size"],
                    device=GraphDevice(
                        self.devices[i].label, self.devices[i].id
                    ),
                ),
                # is_cache_empty
                TensorType(DType.bool, shape=[1]),
            )
            for i in range(len(self.devices))
        ]
