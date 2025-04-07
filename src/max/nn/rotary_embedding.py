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
"""The rope embedding used within the model."""

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

from max.dtype import DType
from max.graph import Dim, TensorValue, TensorValueLike, ops

from .layer import Module


@dataclass
class RotaryEmbedding(Module):
    """
    RotaryEmbedding layer to calculate and apply the frequency tensor for complex exponentials.
    """

    dim: int
    n_heads: int
    theta: float
    """Hyperparameter used to control the frequency scaling of the sinusoidal components of the embeddings."""
    max_seq_len: int
    """The maximum sequence length for model's input."""
    _freqs_cis: Optional[TensorValueLike] = None
    interleaved: bool = True

    def __post_init__(self):
        super().__init__()

    def _compute_inv_freqs(self) -> TensorValue:
        n = self.dim // self.n_heads
        # Note: using float64 to avoid an overflow on the exponential, then converting back to float32.
        iota = ops.range(
            ops.constant(0, DType.float64),
            ops.constant(n - 1, DType.float64),
            ops.constant(2, DType.float64),
            out_dim=n // 2,
        )
        inv_freq = ops.cast(1.0 / (self.theta ** (iota / n)), DType.float32)
        return inv_freq

    def freqs_cis_base(self) -> TensorValue:
        """
        Computes the frequency tensor for complex exponentials (cis)
        for a given seq_len. Tensor is scaled with theta parameter.
        Required to apply Rotary Position Embedding (RoPE) to tensor.
        See 'Roformer: Enhanced Transformer with Rotary Embedding'
        (arxiv.org/pdf/2104.09864).

        Returns:
            The frequency tensor for complex exponentials with shape
                (max_seq_len * 2, dim//(2 * n_heads), 2)
        """
        if self._freqs_cis is None:
            inv_freqs = self._compute_inv_freqs()

            t = ops.range(
                ops.constant(0, DType.float32),
                ops.constant(self.max_seq_len * 2.0, DType.float32),
                ops.constant(1, DType.float32),
                out_dim=self.max_seq_len * 2,
            )
            freqs = ops.outer(t, inv_freqs)
            self._freqs_cis = ops.stack(
                [ops.cos(freqs), ops.sin(freqs)], axis=-1
            )
        return TensorValue(self._freqs_cis)

    @cached_property
    def freqs_cis(self) -> TensorValue:
        self._freqs_cis = self.freqs_cis_base()
        return self._freqs_cis

    def __call__(
        self,
        x: TensorValueLike,
        start_pos: Optional[TensorValue] = None,
        seq_len: Optional[Dim] = None,
    ) -> TensorValue:
        """Applies rotary positional embeddings (RoPE) to `x`.

        Args:
            x: Activation tensor with shape (batch, seq_len, n_kv_heads, head_dim).
            start_pos: starting position of input tensor, defaults to 0 if None
            seq_len: length of input tensor, defaults to x.shape[-2] if None

        Returns:
            Input activation tensor with rotary positional embeddings applied and
            the same shape as `x`.
        """
        v = TensorValue(x)

        if self.interleaved:
            complex = ops.as_interleaved_complex(v)
            x_re = complex[..., 0]
            x_im = complex[..., 1]
        else:
            head_dim = v.shape[-1]
            head_dim_val = TensorValue(head_dim)
            half_dim = head_dim // 2
            half_dim_val = TensorValue(half_dim)
            slice_re = (slice(0, half_dim_val), half_dim)
            slice_im = (slice(half_dim_val, head_dim_val), half_dim)
            x_re = v[..., slice_re]
            x_im = v[..., slice_im]

        if start_pos is None:
            start_pos = ops.constant(0, dtype=DType.int64)
        if seq_len is None:
            seq_len = v.shape[-2]

        seq_len_val = TensorValue(seq_len)
        freqs_cis_sliced = self.freqs_cis[
            (slice(start_pos, start_pos + seq_len_val), seq_len),
        ]
        # Handle optimized case that flattens freqs_cis.
        # This is needed so naive llama3 can still use Llama3RotaryEmbedding with correct freq_cis.
        if len(freqs_cis_sliced.shape) == 2:
            d0, d1 = freqs_cis_sliced.shape
            freqs_cis_sliced = freqs_cis_sliced.reshape((d0, d1 // 2, 2))

        # TODO(MSDK-1188): Ideally this cast would happen inside of the cached
        # self.freqs_cis property instead of here, but complex.dtype is not
        # known at that point.
        freqs_cis_sliced = ops.cast(freqs_cis_sliced, v.dtype)

        freqs_cis_bcast = ops.unsqueeze(ops.unsqueeze(freqs_cis_sliced, 1), 0)

        freqs_re = freqs_cis_bcast[..., 0]
        freqs_im = freqs_cis_bcast[..., 1]

        rope_re = (x_re * freqs_re) - (x_im * freqs_im)
        rope_im = (x_re * freqs_im) + (x_im * freqs_re)

        if self.interleaved:
            rope_complex = ops.stack([rope_re, rope_im], axis=-1)
        else:
            rope_complex = ops.concat((rope_re, rope_im), axis=-1)

        # Cast back to the activations dtype, which may differ from
        # freqs_cis's dtype.
        return ops.cast(ops.reshape(rope_complex, v.shape), v.dtype)


@dataclass
class OptimizedRotaryEmbedding(RotaryEmbedding):
    """
    Optimized version of RotaryEmbedding using 2D frequency tensor representation.
    """

    @cached_property
    def freqs_cis(self):
        freqs = self.freqs_cis_base()
        d1, d2, d3 = freqs.shape
        new_f_shape = [d1, d2 * d3]
        self._freqs_cis = ops.reshape(freqs, new_f_shape)
        return self._freqs_cis


@dataclass
class Llama3RopeScalingParams:
    factor: float
    """Main scaling factor for the frequency components of the rope."""
    low_freq_factor: float
    """Factor to scale the low frequency components of the rope."""
    high_freq_factor: float
    """Factor to scale the high frequency components of the rope."""
    orig_max_position: int
    """The original maximum position length supported by the model."""


@dataclass
class Llama3RotaryEmbedding(OptimizedRotaryEmbedding):
    """
    RotaryEmbedding for Llama3 that takes rope scaling into account.
    """

    scaling_params: Optional[Llama3RopeScalingParams] = None
    """Scaling parameters to enable llama to function with a longer context length."""

    def _compute_inv_freqs(self) -> TensorValue:
        inv_freqs = super()._compute_inv_freqs()
        if self.scaling_params is not None:
            low_freq_wavelen = (
                self.scaling_params.orig_max_position
                / self.scaling_params.low_freq_factor
            )
            high_freq_wavelen = (
                self.scaling_params.orig_max_position
                / self.scaling_params.high_freq_factor
            )

            wave_len = 2 * math.pi / inv_freqs
            if (
                self.scaling_params.low_freq_factor
                != self.scaling_params.high_freq_factor
            ):
                smooth = (
                    self.scaling_params.orig_max_position / wave_len
                    - self.scaling_params.low_freq_factor
                ) / (
                    self.scaling_params.high_freq_factor
                    - self.scaling_params.low_freq_factor
                )
            else:
                smooth = ops.constant(0, DType.float32)
            inv_freqs = ops.select(
                wave_len < high_freq_wavelen,
                inv_freqs,
                ops.select(
                    wave_len > low_freq_wavelen,
                    inv_freqs / self.scaling_params.factor,
                    (1 - smooth) * inv_freqs / self.scaling_params.factor
                    + smooth * inv_freqs,
                ),
            )
        return inv_freqs


@dataclass
class DeepseekYarnRopeScalingParams:
    scaling_factor: float
    """Scaling factor for frequency interpolation."""
    original_max_position_embeddings: int
    """Original maximum sequence length during training."""
    beta_fast: int
    """Fast interpolation rate."""
    beta_slow: int
    """Slow interpolation rate."""
    mscale: float
    """Scaling factor for middle frequencies."""
    mscale_all_dim: float
    """Scaling factor applied to all dimensions."""


@dataclass
class DeepseekYarnRotaryEmbedding(RotaryEmbedding):
    """YaRN (Yet another RoPE eNhancement) Rotary Position Embedding layer.

    This layer implements YaRN rotary position embeddings which extend RoPE to longer sequences.
    It computes position-dependent rotation matrices using a combination of linear interpolation
    and frequency scaling to enable extrapolation beyond the original training context length.

    Unlike the parent class, this class does not apply frequencies to the input tensor. Instead, it simply returns the frequencies which can be later applied in a kernel.
    """

    scaling_params: Optional[DeepseekYarnRopeScalingParams] = None

    def __call__(
        self,
        x: TensorValueLike,
        start_pos: Optional[TensorValue] = None,
        seq_len: Optional[Dim] = None,
    ) -> TensorValue:
        freqs_cos, freqs_sin = self._compute_yarn_freqs(TensorValue(x))
        return ops.stack([freqs_cos, freqs_sin], axis=0)

    def _compute_yarn_freqs(
        self, x: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        if self.scaling_params is None:
            raise ValueError("scaling_params must be provided")

        seq_len = x.shape[-2]

        dim = Dim(self.dim // 2)

        start = ops.constant(0, dtype=DType.float32)
        end = ops.constant(self.dim, dtype=DType.float32)
        step = ops.constant(2, dtype=DType.float32)
        range_output = ops.range(start, end, step, out_dim=dim)

        freq_base = self.theta ** (range_output / float(self.dim))
        freq_extra = 1.0 / freq_base

        freq_inter = 1.0 / (self.scaling_params.scaling_factor * freq_base)

        low, high = self._yarn_find_correction_range(
            ops.constant(self.scaling_params.beta_fast, dtype=DType.float32),
            ops.constant(self.scaling_params.beta_slow, dtype=DType.float32),
            self.dim,
            int(self.theta),  # Explicitly convert base to int
            self.scaling_params.original_max_position_embeddings,
        )

        # Ensure the mask has the correct dimension
        inv_freq_mask = 1.0 - self._yarn_linear_ramp_mask(low, high, dim).cast(
            DType.float32
        )

        # Ensure shapes match before multiplication
        inv_freq_mask = ops.broadcast_to(inv_freq_mask, freq_inter.shape)

        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

        self.inv_freq = inv_freq
        # Create range with all required parameters
        end = ops.constant(
            int(seq_len), dtype=DType.float32
        )  # Convert seq_len to int
        step = ops.constant(1, dtype=DType.float32)
        t = ops.range(start, end, step, out_dim=Dim(int(seq_len)))
        freqs = ops.outer(t, inv_freq)

        _mscale = float(
            self._yarn_get_mscale(
                self.scaling_params.scaling_factor, self.scaling_params.mscale
            )
            / self._yarn_get_mscale(
                self.scaling_params.scaling_factor,
                self.scaling_params.mscale_all_dim,
            )
        )

        emb = ops.concat((freqs, freqs), axis=-1)
        freqs_cos = ops.cos(emb) * _mscale
        freqs_sin = ops.sin(emb) * _mscale
        self._freqs_cis = ops.stack([freqs_cos, freqs_sin], axis=0)
        return freqs_cos, freqs_sin

    def _yarn_get_mscale(
        self, scale: float = 1.0, mscale: float = 1.0
    ) -> float:
        """Calculate the scaling factor for YaRN (Yet another RoPE extension) interpolation.

        Args:
            scale: The scaling factor for position embeddings. Default is 1.0.
            mscale: The multiplier for the logarithmic scaling. Default is 1.0.

        Returns:
            float: The computed scaling factor. Returns 1.0 if scale <= 1,
                otherwise returns 0.1 * mscale * log(scale) + 1.0
        """
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def _yarn_find_correction_range(
        self,
        low_rot: TensorValue,
        high_rot: TensorValue,
        dim: int,
        base: float,
        max_position_embeddings: int,
    ) -> tuple[TensorValue, TensorValue]:
        """
        Find the correction range for the rotary embeddings.

        Args:
            low_rot: Low rotation tensor
            high_rot: High rotation tensor
            dim: Dimension of the mask
            base: Base for the exponential scaling
            max_position_embeddings: Maximum position embeddings
        """
        low = ops.floor(
            self._yarn_find_correction_dim(
                low_rot, dim, base, max_position_embeddings
            )
        )
        high = ops.floor(
            self._yarn_find_correction_dim(
                high_rot, dim, base, max_position_embeddings
            )
        )
        return ops.max(low, 0), ops.min(high, dim - 1)

    # Inverse dim formula to find dim based on number of rotations
    def _yarn_find_correction_dim(
        self,
        num_rotations: TensorValue,
        dim: int,
        base: float,
        max_position_embeddings: int,
    ) -> TensorValue:
        """
        Inverse dim formula to find dim based on number of rotations.

        Args:
            num_rotations: Number of rotations tensor
            dim: Dimension of the mask
            base: Base for the exponential scaling
            max_position_embeddings: Maximum position embeddings
        """
        # Convert all inputs to TensorValues with proper types
        max_pos = ops.constant(
            float(max_position_embeddings), dtype=DType.float32
        )
        base_tensor = ops.constant(float(base), dtype=DType.float32)
        dim_tensor = ops.constant(float(dim), dtype=DType.float32)

        return (
            dim_tensor * ops.log(max_pos / (num_rotations * 2 * math.pi))
        ) / (2 * ops.log(base_tensor))

    def _yarn_linear_ramp_mask(
        self, min: TensorValue, max: TensorValue, dim: Dim
    ) -> TensorValue:
        """
        Create a linear ramp mask for interpolation.

        Args:
            min: Minimum value tensor
            max: Maximum value tensor
            dim: Dimension of the mask
        """
        if min == max:
            max += 0.001  # Prevent singularity

        start = ops.constant(0, dtype=DType.int64)
        step = ops.constant(1, dtype=DType.int64)

        linear_func = (
            ops.range(start, dim, step, out_dim=dim).cast(DType.float32) - min
        ) / (max - min)

        return ops.min(ops.max(linear_func, 0), 1)
