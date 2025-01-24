# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from enum import Enum
from typing import Optional

from max.dtype import DType

VALID_KV_KERNELS = [
    ("bf16", 1, 16),
    ("f32", 1, 16),
    ("bf16", 3, 64),  # SmolLM
    ("f32", 3, 64),  # SmolLM
    ("bf16", 8, 128),
    ("f32", 8, 128),
    ("bf16", 8, 32),
    ("f32", 8, 32),
    ("bf16", 8, 64),
    ("f32", 8, 64),
    ("bf16", 8, 512),
    ("f32", 8, 512),
    ("bf16", 32, 128),
    ("f32", 32, 128),
    ("bf16", 8, 80),
    ("f32", 8, 80),
    ("f32", 2, 2),
]


class KVCacheStrategy(str, Enum):
    MODEL_DEFAULT = "model_default"
    NAIVE = "naive"
    CONTINUOUS = "continuous"
    PAGED = "paged"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

    def kernel_substring(self) -> str:
        """Returns the common substring that we include in the kernel name for this caching strategy."""
        if self == KVCacheStrategy.CONTINUOUS:
            return "continuous_batching"
        return str(self.value)

    def uses_opaque(self) -> bool:
        return self != KVCacheStrategy.NAIVE


class KVCacheParams:
    def __init__(
        self,
        dtype: DType,
        n_kv_heads: int,
        head_dim: int,
        enable_prefix_caching: bool = False,
        cache_strategy: KVCacheStrategy = KVCacheStrategy.CONTINUOUS,
        page_size: Optional[int] = None,
        n_devices: int = 1,
    ):
        # Initialize static attributes.
        self.dtype = dtype
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.cache_strategy = cache_strategy
        self.n_devices = n_devices
        self.n_kv_heads_per_device = n_kv_heads // n_devices
        self.page_size = page_size
        self.enable_prefix_caching = enable_prefix_caching
        # Validate inputs.
        if (
            cache_strategy == KVCacheStrategy.CONTINUOUS
            and (
                self.dtype_shorthand,
                n_kv_heads,
                head_dim,
            )
            not in VALID_KV_KERNELS
        ):
            raise ValueError(
                "Unsupported KV Cache Configuration: got dtype:"
                f" {self.dtype_shorthand}, n_kv_heads: {n_kv_heads}, head_dim:"
                f" {head_dim}"
            )
        if enable_prefix_caching and cache_strategy != KVCacheStrategy.PAGED:
            raise ValueError(
                "Prefix caching is only supported for PAGED cache strategy"
            )
        if page_size is None and cache_strategy == KVCacheStrategy.PAGED:
            raise ValueError("Page size is required for PAGED cache strategy")

    @property
    def dtype_shorthand(self) -> str:
        """The textual representation in shorthand of the dtype."""
        return "bf16" if self.dtype == DType.bfloat16 else "f32"

    @property
    def static_cache_shape(self) -> tuple[str, str, str, str, str]:
        return (
            "num_layers",
            "batch_size",
            "seq_len",
            "n_kv_heads",
            "head_dim",
        )
