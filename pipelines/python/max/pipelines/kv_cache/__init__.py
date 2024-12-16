# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from typing import Any, Dict, List, Optional, Type

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
from .paged_cache import (
    FetchPagedKVCacheCollection,
    PagedKVCacheCollection,
    PagedKVCacheManager,
    PagedKVCacheType,
)
from .radix_trie import RadixTrie

CACHE_MANAGER_REGISTRY: dict[KVCacheStrategy, Type[KVCacheManager]] = {
    KVCacheStrategy.CONTINUOUS: ContinuousBatchingKVCacheManager,
    KVCacheStrategy.NAIVE: NaiveKVCacheManager,
    KVCacheStrategy.PAGED: PagedKVCacheManager,
}


def load_kv_manager(
    params: KVCacheParams,
    max_cache_batch_size: int,
    max_seq_len: int,
    num_layers: int,
    devices: List[Device],
    session: InferenceSession,
    available_cache_memory: Optional[int] = None,
    page_size: Optional[int] = 512,
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
    elif params.cache_strategy == KVCacheStrategy.PAGED:
        if page_size is None:
            msg = (
                "Missing required argument page_size for KVCacheStrategy.PAGED"
            )
            raise ValueError(msg)

        if available_cache_memory is None:
            msg = "Missing required argument available_cache_memory for KVCacheStrategy.PAGED"
            raise ValueError(msg)

        return PagedKVCacheManager(
            params=params,
            max_cache_batch_size=max_cache_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            devices=devices,
            session=session,
            cache_memory=available_cache_memory,
            page_size=page_size,
        )
    else:
        msg = f"cache type: {params.cache_strategy} not supported."
        raise ValueError(msg)


def estimate_kv_cache_size(
    params: KVCacheParams,
    max_cache_batch_size: int,
    max_seq_len: int,
    num_layers: int,
    available_cache_memory: int,
    devices: List[Device],
) -> int:
    if params.cache_strategy not in CACHE_MANAGER_REGISTRY:
        msg = f"cache type: {params.cache_strategy} not supported."
        raise ValueError(msg)

    return CACHE_MANAGER_REGISTRY[params.cache_strategy].estimated_memory_size(
        params=params,
        max_cache_batch_size=max_cache_batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        available_cache_memory=available_cache_memory,
        devices=devices,
    )


__all__ = [
    "KVCacheParams",
    "KVCacheStrategy",
    "ContinuousBatchingKVCache",
    "ContinuousBatchingKVCacheCollection",
    "ContinuousBatchingKVCacheCollectionType",
    "ContinuousBatchingKVCacheManager",
    "ContinuousBatchingKVCacheType",
    "FetchContinuousBatchingKVCacheCollection",
    "FetchPagedKVCacheCollection",
    "PagedKVCacheManager",
    "PagedKVCacheCollection",
    "PagedKVCacheType",
    "KVCacheManager",
    "NaiveKVCacheManager",
    "ContinuousHFStaticCache",
    "RadixTrie",
]
