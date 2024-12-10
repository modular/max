# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from typing import List

from max.driver import Device
from max.engine import InferenceSession

from .cache_params import KVCacheParams, KVCacheStrategy
from .continuous_batching_cache import (
    ContinuousBatchingKVCache,
    ContinuousBatchingKVCacheCollection,
    ContinuousBatchingKVCacheCollectionType,
    ContinuousBatchingKVCacheManager,
    ContinuousBatchingKVCacheType,
    FetchContinuousBatchingKVCacheCollection,
)
from .hf import ContinuousHFStaticCache
from .manager import KVCacheManager
from .naive_cache import NaiveKVCacheManager
from .radix_trie import RadixTrie


def load_kv_manager(
    params: KVCacheParams,
    max_cache_batch_size: int,
    max_seq_len: int,
    num_layers: int,
    devices: List[Device],
    session: InferenceSession,
) -> KVCacheManager:
    if params.cache_strategy == KVCacheStrategy.CONTINUOUS:
        return ContinuousBatchingKVCacheManager(
            params=params,
            max_cache_batch_size=max_cache_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            devices=devices,
            session=session,
        )
    elif params.cache_strategy == KVCacheStrategy.NAIVE:
        return NaiveKVCacheManager(
            params=params,
            max_cache_batch_size=max_cache_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            devices=devices,
            session=session,
        )
    else:
        msg = f"cache type: {params.cache_strategy} not supported."
        raise ValueError(msg)


def estimate_kv_cache_size(
    params: KVCacheParams,
    max_cache_batch_size: int,
    max_seq_len: int,
    num_layers: int,
    devices: List[Device],
) -> int:
    if params.cache_strategy == KVCacheStrategy.CONTINUOUS:
        return ContinuousBatchingKVCacheManager.estimated_memory_size(
            params=params,
            max_cache_batch_size=max_cache_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            devices=devices,
        )
    elif params.cache_strategy == KVCacheStrategy.NAIVE:
        return NaiveKVCacheManager.estimated_memory_size(
            params=params,
            max_cache_batch_size=max_cache_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            devices=devices,
        )
    else:
        msg = f"cache type: {params.cache_strategy} not supported."
        raise ValueError(msg)


__all__ = [
    "KVCacheParams",
    "KVCacheStrategy",
    "ContinuousBatchingKVCache",
    "ContinuousBatchingKVCacheCollection",
    "ContinuousBatchingKVCacheCollectionType",
    "ContinuousBatchingKVCacheManager",
    "ContinuousBatchingKVCacheType",
    "FetchContinuousBatchingKVCacheCollection",
    "KVCacheManager",
    "NaiveKVCacheManager",
    "ContinuousHFStaticCache",
    "RadixTrie",
]
