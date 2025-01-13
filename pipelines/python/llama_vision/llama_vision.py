# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from typing import Any, final

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Dim, Graph, Shape, TensorType, TensorValue, Type, ops
from max.graph.weights import Weights
from max.pipelines import (
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    TextAndVisionContext,
)
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheManager,
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    estimate_kv_cache_size,
    load_kv_manager,
)
from nn import Linear
from nn.layer import Layer

from .language_model import CausalLanguageModel, instantiate_language_model
from .vision_model import instantiate_vision_model


def max_seq_len(config: PipelineConfig) -> int:
    return min(
        config.max_length,
        config.huggingface_config.text_config.max_position_embeddings,
    )


class MultimodalKVCacheManager(KVCacheManager):
    """A lightweight wrapper around text and vision KV managers.

    Note on runtime and graph build time return types:
    - Currently the multi modal KV manager doesn't support multiple devices.
      So all lists that should be of length num_devices will have length 1.
    - Individual modality KV cache managers return a 4-tuple of KV cache inputs.
      Since this is a pair of KV cache managers, it returns an 8-tuple,
      where the first 4 elements are the text KV cache inputs and the remaining
      4 elements are the vision KV cache inputs.
    - This 8-tuple applies for both input symbols and return KV cache inputs.
    - TODO(bduke): We should fix both multi-device and multi-modality using an
      extensible KVCacheInput type.
    """

    text_kv_manager: KVCacheManager
    """KV cache manager for text inputs."""

    vision_kv_manager: ContinuousBatchingKVCacheManager
    """KV cache manager for image inputs."""

    def __init__(
        self,
        params: KVCacheParams,
        max_cache_batch_size: int,
        text_max_seq_len: int,
        vision_max_seq_len: int,
        text_num_layers: int,
        vision_num_layers: int,
        devices: list[Device],
        session: InferenceSession,
        available_cache_memory: int,
        page_size: int,
    ) -> None:
        self.text_kv_manager = load_kv_manager(
            params=params,
            max_cache_batch_size=max_cache_batch_size,
            max_seq_len=text_max_seq_len,
            num_layers=text_num_layers,
            devices=devices,
            available_cache_memory=available_cache_memory,
            page_size=page_size,
            session=session,
        )

        # Always use continuous batching KV cache for the vision KV projections,
        # since the number of vision tokens is fixed per batch until we support
        # multi-image, at least.
        self.vision_kv_manager = ContinuousBatchingKVCacheManager(
            params=params,
            max_cache_batch_size=max_cache_batch_size,
            max_seq_len=vision_max_seq_len,
            num_layers=vision_num_layers,
            devices=devices,
            session=session,
        )

        # Call superclass after initializing modality KV managers since the
        # superclass ctor calls methods that use the modality KV managers.
        super().__init__(
            params,
            max_cache_batch_size,
            text_max_seq_len,
            text_num_layers,
            devices,
            session,
            is_ragged=True,
        )

    @classmethod
    @final
    def estimated_memory_size(
        cls,
        params: KVCacheParams,
        max_cache_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: list[Device],
    ) -> int:
        """Returns the estimated total memory usage of the kv cache."""
        # TODO(bduke): this is incorrect. Estimated memory size should be an
        # instance method to account for different text and vision KV caches.
        return 2 * ContinuousBatchingKVCacheManager.estimated_memory_size(
            params,
            max_cache_batch_size,
            max_seq_len,
            num_layers,
            available_cache_memory,
            devices,
        )

    @final
    def _fetch(
        self, seq_ids_and_prompts: dict[int, np.ndarray], num_steps: int = 1
    ) -> list[tuple[Tensor, ...]]:
        """Returns KV cache inputs for both modalities' KV managers."""
        # Here we call into the text KV manager's fetch method to update
        # its fetch metadata.
        text_fetch_results = self.text_kv_manager.fetch(
            seq_ids_and_prompts, num_steps
        )[0]

        # For the vision KV manager, fetch metadata isn't applicable since
        # autoregressive generation is text only.
        active_batch_size = len(seq_ids_and_prompts)

        # Lookup table and seq_ids are redundant identical tensors.
        lookup_table_tensor = Tensor.from_numpy(
            np.array(list(seq_ids_and_prompts.keys()), np.uint32)
        )
        cache_lengths_np = np.zeros(active_batch_size, np.uint32)

        max_seq_length = 0
        max_cache_length = 0

        device = self.vision_kv_manager.devices[0]
        for i, seq_id in enumerate(seq_ids_and_prompts):
            # Assumption: all seq_ids with
            # `vision_kv_manager.cache_lengths[seq_id] == 0`
            # are context encoding steps and have the max image sequence length.
            # TODO(bduke): pass the vision sequence lengths in from next_token.

            # Omit validity checks on seq ids, which are done in the text fetch.
            cache_len = self.vision_kv_manager.cache_lengths[seq_id]
            if cache_len == 0:
                max_seq_length = self.vision_kv_manager.max_seq_len

            cache_lengths_np[i] = cache_len

            # Update the maximum lengths seen so far.
            max_cache_length = max(max_cache_length, cache_len)

        # Build a tensor of maximum lengths. Each step slices the first row to
        # advance to the values for the next row.
        max_lengths_np = np.empty((num_steps, 2), np.uint32)
        step_max_seq_length = max_seq_length
        step_max_cache_length = max_cache_length
        for step in range(num_steps):
            max_lengths_np[step, 0] = step_max_seq_length
            max_lengths_np[step, 1] = step_max_cache_length
            step_max_cache_length += step_max_seq_length
            step_max_seq_length = 0
        max_lengths_host = Tensor.from_numpy(max_lengths_np)

        vision_fetch_results = (
            # Block 0 for the first device (since MultimodalKVCacheManager
            # assumes only 1 device).
            self.vision_kv_manager.blocks[0],
            Tensor.from_numpy(cache_lengths_np).to(device),
            lookup_table_tensor.to(device),
            max_lengths_host,
        )

        return [text_fetch_results + vision_fetch_results]

    @final
    def input_symbols(
        self,
    ) -> Sequence[tuple[Type, ...]]:
        """Returns concatenated input symbols for text and vision KV managers.

        This has to rename input symbols that aren't necessarily the same:
        `num_layers` and `max_seq_len` differ in general between text and
        vision modalities.
        """

        def _input_symbols(
            manager: KVCacheManager, num_layers_key: str, max_seq_len_key: str
        ) -> tuple[Type, ...]:
            input_symbols = manager.input_symbols()[0]
            assert isinstance(input_symbols[0], TensorType)
            input_symbols[0].shape = Shape(
                [
                    "num_blocks",
                    2,
                    num_layers_key,
                    max_seq_len_key,
                    "num_kv_heads",
                    "head_dim",
                ]
            )
            return input_symbols

        return [
            _input_symbols(
                self.text_kv_manager, "text_num_layers", "text_max_seq_len"
            )
            + _input_symbols(
                self.vision_kv_manager,
                "vision_num_layers",
                "vision_max_seq_len",
            ),
        ]

    def step(self, seq_ids_and_new_tokens: dict[int, np.ndarray]) -> None:
        """Steps both text and vision modalities' KV managers."""
        # Step the text KV manager as usual for autoregressive text generation.
        self.text_kv_manager.step(seq_ids_and_new_tokens)

        # Keep the base class's state in sync with the text KV manager's.
        super().step(seq_ids_and_new_tokens)

        # Increment cache lengths for the vision KV manager iff this is a
        # context encoding (CE) step with an image input.
        # It's a CE step if the existing cache_lengths[seq_id] is 0.
        for seq_id in seq_ids_and_new_tokens:
            self.vision_kv_manager.cache_lengths[seq_id] += (
                self.vision_kv_manager.max_seq_len
                if self.vision_kv_manager.cache_lengths[seq_id] == 0
                else 0
            )

    def external_claim(self, seq_ids: list[int]) -> None:
        """Reserves the same sequence ids for both modalities' KV caches."""
        self.text_kv_manager.external_claim(seq_ids)
        self.vision_kv_manager.external_claim(seq_ids)

        # Keep the base class's state in sync with the text KV manager's.
        super().external_claim(seq_ids)

    def release(self, seq_id: int) -> None:
        """Marks the sequence complete for both modalities' KV caches."""
        self.text_kv_manager.release(seq_id)
        self.vision_kv_manager.release(seq_id)

    def contains(self, seq_id: int) -> bool:
        """Returns whether `seq_id` is in the KV cache."""
        text_kv_contains = self.text_kv_manager.contains(seq_id)

        # Assume that the modalities' KV caches have consistent sequence ids.
        assert text_kv_contains == self.vision_kv_manager.contains(seq_id)

        return text_kv_contains

    def num_kv_inputs(self) -> int:
        """Returns the sum of the KV input lengths for both modalities."""
        return (
            self.text_kv_manager.num_kv_inputs()
            + self.vision_kv_manager.num_kv_inputs()
        )

    def increment_cache_lengths(
        self,
        kv_cache_inputs: Sequence[tuple[Tensor, ...]],
        prev_model_inputs: Iterable[Any],
    ) -> list[tuple[Tensor, ...]]:
        """Updates the cache lengths for multistep execution.

        This increments the text and vision KV cache lengths separately using
        their respective KV cache inputs.
        """
        text_kv_inputs = kv_cache_inputs[0][
            : self.text_kv_manager.num_kv_inputs()
        ]
        vision_kv_inputs = kv_cache_inputs[0][
            self.text_kv_manager.num_kv_inputs() :
        ]
        return [
            self.text_kv_manager.increment_cache_lengths(
                [text_kv_inputs], prev_model_inputs
            )[0]
            + self.vision_kv_manager.increment_cache_lengths(
                [vision_kv_inputs], prev_model_inputs
            )[0]
        ]


class LlamaVisionModel(Layer):
    """
    The Llama 3.2 vision model.
    """

    def __init__(
        self, pipeline_config: PipelineConfig, weights: Weights
    ) -> None:
        # Set convenience attributes for the text and vision configs.
        self.vision_config = pipeline_config.huggingface_config.vision_config
        self.text_config = pipeline_config.huggingface_config.text_config

        self.vision_model = instantiate_vision_model(
            dtype=pipeline_config.dtype,
            image_size=self.vision_config.image_size,
            patch_size=self.vision_config.patch_size,
            supported_aspect_ratios=self.vision_config.supported_aspect_ratios,
            hidden_size=self.vision_config.hidden_size,
            max_num_tiles=self.vision_config.max_num_tiles,
            num_channels=self.vision_config.num_channels,
            norm_eps=self.vision_config.norm_eps,
            attention_heads=self.vision_config.attention_heads,
            num_hidden_layers=self.vision_config.num_hidden_layers,
            intermediate_size=self.vision_config.intermediate_size,
            num_global_layers=self.vision_config.num_global_layers,
            intermediate_layers_indices=self.vision_config.intermediate_layers_indices,
            weights=weights,
        )

        self.multi_modal_projector = Linear(
            weights.multi_modal_projector.weight.allocate(
                pipeline_config.dtype,
                [
                    self.text_config.hidden_size,
                    self.vision_config.vision_output_dim,
                ],
            ),
            weights.multi_modal_projector.bias.allocate(
                pipeline_config.dtype,
                [self.text_config.hidden_size],
            ),
        )

    def __call__(
        self,
        pixel_values: TensorValue,
        aspect_ratio_ids: TensorValue,
        aspect_ratio_mask: TensorValue,
    ) -> TensorValue:
        if aspect_ratio_ids is None:
            msg = (
                "`aspect_ratio_ids` must be provided if `pixel_values` is "
                "provided"
            )
            raise ValueError(msg)

        # Get vision tokens from vision model.
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
        )
        cross_attention_states = vision_outputs[0]

        num_patches = cross_attention_states.shape[-2]

        return self.multi_modal_projector(cross_attention_states).reshape(
            [
                Dim("batch_size")
                * Dim("num_concurrent_media")
                * self.vision_config.max_num_tiles
                * num_patches,
                self.text_config.hidden_size,
            ]
        )


class LlamaVisionLanguageModel(Layer):
    """The Llama 3.2 vision language model."""

    language_model: CausalLanguageModel
    """Language model composed of self and cross attention layers."""

    num_text_kv_cache_inputs: int
    """Number of KV cache inputs for self attention layers."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        weights: Weights,
        kv_params: KVCacheParams,
        num_text_kv_cache_inputs: int,
    ) -> None:
        text_config = pipeline_config.huggingface_config.text_config

        self.language_model = instantiate_language_model(
            dtype=pipeline_config.dtype,
            hidden_size=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            rope_theta=text_config.rope_theta,
            max_seq_len=max_seq_len(pipeline_config),
            num_hidden_layers=text_config.num_hidden_layers,
            cross_attention_layers=text_config.cross_attention_layers,
            vocab_size=text_config.vocab_size,
            rms_norm_eps=text_config.rms_norm_eps,
            num_key_value_heads=text_config.num_key_value_heads,
            intermediate_size=text_config.intermediate_size,
            kv_params=kv_params,
            weights=weights,
        )
        self.num_text_kv_cache_inputs = num_text_kv_cache_inputs

    def __call__(
        self,
        cross_attention_states: TensorValue,
        input_ids: TensorValue,
        hidden_input_row_offsets: TensorValue,
        hidden_max_seq_len: TensorValue,
        cross_input_row_offsets: TensorValue,
        *kv_cache_inputs: TensorValue,
    ) -> TensorValue:
        logits = self.language_model(
            text_kv_cache_inputs=kv_cache_inputs[
                : self.num_text_kv_cache_inputs
            ],
            vision_kv_cache_inputs=kv_cache_inputs[
                self.num_text_kv_cache_inputs :
            ],
            input_ids=input_ids,
            hidden_input_row_offsets=hidden_input_row_offsets,
            hidden_max_seq_len=hidden_max_seq_len,
            cross_attention_states=cross_attention_states,
            cross_input_row_offsets=cross_input_row_offsets,
        )

        # Always return float32 logits, no matter the activation type
        return ops.cast(logits, DType.float32)


class LlamaVision(PipelineModel):
    """The entire (multimodal) Llama3.2 vision model."""

    def __init__(
        self, pipeline_config: PipelineConfig, session: InferenceSession
    ) -> None:
        # Set convenience attributes for the text and vision configs.
        self.vision_config = pipeline_config.huggingface_config.vision_config
        self.text_config = pipeline_config.huggingface_config.text_config

        # These need to be set at graph instantiation time.
        self.vision_graph_input_size = -1
        self.language_graph_input_size = -1

        super().__init__(pipeline_config, session)
        self.vision_model, self.language_model = self.load_model(session)
        # Note that in a multimodal model, the language model is the last model in the
        # pipeline. Unfortunately, self.model is still being used (and exposed)
        # in the token generation code, so we still need to set it here.
        self.model = self.language_model

    def _llama3_vision_vision_graph(self) -> Graph:
        # Inserted a manual CHW -> HWC transpose here.
        pixel_values_type = TensorType(
            # This has to be of type float32 as we construct tensors from a numpy
            # array (which has no notion of some dtypes like bfloat16). Explicit
            # casting will happen inside the graph.
            DType.float32,
            shape=[
                "batch_size",
                "num_concurrent_media",
                self.vision_config.max_num_tiles,
                self.vision_config.image_size,  # height
                self.vision_config.image_size,  # width
                self.vision_config.num_channels,
            ],
        )
        aspect_ratio_ids_type = TensorType(
            DType.int64,
            shape=["batch_size", "num_concurrent_media"],
        )
        aspect_ratio_mask_type = TensorType(
            DType.int64,
            shape=[
                "batch_size",
                "num_concurrent_media",
                self.vision_config.max_num_tiles,
            ],
        )

        input_types = [
            pixel_values_type,
            aspect_ratio_ids_type,
            aspect_ratio_mask_type,
        ]
        self.vision_graph_input_size = len(input_types)
        return Graph(
            "llama3-vision-vision-model-graph",
            forward=LlamaVisionModel(
                pipeline_config=self.pipeline_config, weights=self.weights
            ),
            input_types=input_types,
        )

    def _llama3_vision_language_graph(self) -> Graph:
        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(
                self.pipeline_config.max_cache_batch_size + 1, dtype=np.uint32
            )
        ).to(self.pipeline_config.devices[0])

        input_ids_type = TensorType(DType.int64, shape=["total_seq_len"])
        # image_size = self.vision_config.image_size
        # patch_size = self.vision_config.patch_size
        cross_attention_states_type = TensorType(
            self.pipeline_config.dtype,
            shape=[
                # TODO(bduke): fix algebraic dim creation outside of graph
                # contexts.
                # Dim("batch_size")
                # * "num_concurrent_media"
                # * self.vision_config.max_num_tiles
                # * ((image_size // patch_size) ** 2 + 1),
                "num_vision_embeddings",
                self.text_config.hidden_size,
            ],
        )
        input_ids_max_seq_len_type = TensorType(DType.uint32, [1])
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"]
        )
        cross_row_offsets_type = input_row_offsets_type

        # Unpack multimodal KV inputs.
        assert isinstance(self.kv_manager, MultimodalKVCacheManager)
        num_text_kv_inputs = self.kv_manager.text_kv_manager.num_kv_inputs()
        input_symbols = self.kv_manager.input_symbols()[0]
        text_kv_input_symbols = input_symbols[:num_text_kv_inputs]
        vision_kv_input_symbols = input_symbols[num_text_kv_inputs:]

        input_types = [
            cross_attention_states_type,
            input_ids_type,
            input_row_offsets_type,
            input_ids_max_seq_len_type,
            cross_row_offsets_type,
            *text_kv_input_symbols,
            *vision_kv_input_symbols,
        ]
        self.language_graph_input_size = len(input_types)

        return Graph(
            "llama3-vision-language-model-graph",
            forward=LlamaVisionLanguageModel(
                pipeline_config=self.pipeline_config,
                weights=self.weights,
                kv_params=self._get_kv_params(),
                num_text_kv_cache_inputs=len(text_kv_input_symbols),
            ),
            input_types=input_types,
        )

    @property
    def vision_max_seq_len(self) -> int:
        """Returns the maximum number of vision tokens."""
        # Marshal out hyperparameters.
        height = self.vision_config.image_size
        width = self.vision_config.image_size
        max_num_tiles = self.vision_config.max_num_tiles
        patch_size = self.vision_config.patch_size
        # TODO(bduke): account for the actual instead of max number of tiles.
        # num_tiles * (image_dim**2 // patch_dim**2 + 1 (cls token))
        return max_num_tiles * ((height * width) // patch_size**2 + 1)

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextAndVisionContext],  # type: ignore
    ) -> tuple[Tensor, ...]:
        """Creates tensors of token and image inputs, if applicable."""
        if self.pipeline_config.cache_strategy != KVCacheStrategy.CONTINUOUS:
            msg = "Llama Vision only supports continuous batching"
            raise ValueError(msg)

        def has_image(pixel_values) -> bool:
            if isinstance(pixel_values, list):
                return len(pixel_values) > 0
            return pixel_values is not None

        # Input validation - check if the sequence of contexts in this batch
        # all have images, or none altogether.
        has_images = -1
        for context in context_batch:
            is_curr_image = has_image(context.pixel_values)
            if has_images == -1:
                has_images = is_curr_image
            elif (is_curr_image and has_images == 0) or (
                not is_curr_image and has_images == 1
            ):
                raise RuntimeError(
                    "Expected the context batch to all have images, or no images "
                    "at all. At least one context in this batch has an image and "
                    "another does not."
                )
            else:
                has_images = 0 if is_curr_image else 1

        res = []
        if has_images:
            images = []
            aspect_ratio_ids_list = []
            aspect_ratio_mask_list = []
            for context in context_batch:
                # Get first image in first batch and permute the order to (HWC).
                image = np.transpose(context.pixel_values, (0, 1, 3, 4, 2))

                # Add batch_size, num_concurrent_media, and max_num_tiles dimensions
                # [1, num_concurrent_media, max_num_tiles, H, W, C]
                image = np.expand_dims(image, axis=(0))
                images.append(image)

                if "aspect_ratio_ids" not in context.extra_model_args:
                    msg = "aspect_ratio_ids is required for image / vision model input"
                    raise ValueError(msg)

                if "aspect_ratio_mask" not in context.extra_model_args:
                    msg = "aspect_ratio_mask is required for image / vision model input"
                    raise ValueError(msg)

                aspect_ratio_ids_list.append(
                    context.extra_model_args["aspect_ratio_ids"]
                )
                aspect_ratio_mask_list.append(
                    context.extra_model_args["aspect_ratio_mask"]
                )

            # Convert the list into a single NumPy array with shape
            # (batch_size, 1, max_num_tiles, H, W, C).
            final_images = np.concatenate(images, axis=0)

            pixel_values = Tensor.from_numpy(final_images).to(
                self.pipeline_config.device
            )

            final_aspect_ratio_ids = np.concatenate(
                aspect_ratio_ids_list, axis=0
            )

            aspect_ratio_ids = Tensor.from_numpy(final_aspect_ratio_ids).to(
                self.pipeline_config.device
            )

            final_aspect_ratio_mask = np.concatenate(
                aspect_ratio_mask_list, axis=0
            )

            aspect_ratio_mask = Tensor.from_numpy(final_aspect_ratio_mask).to(
                self.pipeline_config.device
            )

            res = [
                pixel_values,
                aspect_ratio_ids,
                aspect_ratio_mask,
            ]

        # Input row offset type: ["input_row_offsets_len"], UInt32
        input_id_row_offsets = Tensor.from_numpy(
            np.cumsum(
                [0] + [ctx.seq_len for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.pipeline_config.device)

        pixel_row_offsets = Tensor.from_numpy(
            np.cumsum(
                [0]
                + [
                    # Use an input row offset of 0 to mean no image.
                    self.vision_max_seq_len
                    if ctx.pixel_values is not None
                    else 0
                    for ctx in context_batch
                ],
                dtype=np.uint32,
            )
        ).to(self.pipeline_config.device)

        # Input Ids: ["total_seq_len"], Int64
        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])
        input_id_values = Tensor.from_numpy(tokens).to(
            self.pipeline_config.device
        )
        # This lives on host / in the CPU kernel, but is later casted to a scalar on
        # device kernel side. No need for explicit .to(pipeline_config.device) call here.
        input_id_max_seq_len = Tensor.from_numpy(
            np.array(
                [max(ctx.seq_len for ctx in context_batch)], dtype=np.uint32
            )
        )

        return (
            *res,
            input_id_values,
            input_id_row_offsets,
            input_id_max_seq_len,
            pixel_row_offsets,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        # Next token inputs always go to the language model.
        # - input ids
        # - input max seq lengths
        # - hidden input row offsets
        input_id_max_seq_len: Tensor
        old_row_offsets: Tensor
        if len(prev_model_inputs) == 7:
            # If the previous inputs include the pixel values
            input_id_max_seq_len = prev_model_inputs[4]
            old_row_offsets = prev_model_inputs[6]
        else:
            # If no pixel values were included
            assert len(prev_model_inputs) == 3
            input_id_max_seq_len = prev_model_inputs[1]
            old_row_offsets = prev_model_inputs[2]
        row_offsets_size = old_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]
        next_token_inputs = (
            next_tokens,
            input_id_max_seq_len,
            next_row_offsets,
        )
        return next_token_inputs

    def execute(self, *model_inputs: Tensor) -> ModelOutputs:
        model_input_list = list(model_inputs)

        # batch_size * num_concurrent_media * max_num_tiles * num_patches
        # are set to 0 here to imitate a dummy tensor (used in text-only mode).
        cross_attention_states = Tensor.zeros(
            shape=[0, self.text_config.hidden_size],
            dtype=self.pipeline_config.dtype,
        ).to(self.pipeline_config.device)

        # Vision model has 3 more inputs.
        # pixel_values(1), aspect_ratio_ids(1), aspect_ratio_mask(1)
        if len(model_input_list) >= self.vision_graph_input_size:
            cross_attention_states = self.vision_model.execute(  # type: ignore
                *model_input_list[: self.vision_graph_input_size],
                copy_inputs_to_device=False,
            )[0]
            model_input_list = model_input_list[self.vision_graph_input_size :]

        # Insert vision model output to be fed as input to the subsequent
        # language model. This assumes cross_attention_states is the first input
        # since the list needs to be ordered.
        model_input_list.insert(0, cross_attention_states)

        # Language model has 8 inputs.
        # kv_cache_inputs (4), input_ids(1), hidden_input_row_offsets(1),
        # cross_attention_states(1), cross_input_row_offsets(1)
        if len(model_input_list) != self.language_graph_input_size:
            raise ValueError(
                "Expecting language_model inputs to have {}, got {} instead".format(
                    self.language_graph_input_size, len(model_input_list)
                )
            )

        model_outputs = self.language_model.execute(
            *model_input_list, copy_inputs_to_device=False
        )
        assert not self.pipeline_config.enable_echo
        assert isinstance(model_outputs[0], Tensor)
        return ModelOutputs(next_token_logits=model_outputs[0])

    def _get_kv_params(self) -> KVCacheParams:
        return KVCacheParams(
            dtype=self.pipeline_config.dtype,
            n_kv_heads=self.text_config.num_key_value_heads,
            head_dim=(
                self.text_config.hidden_size
                // self.text_config.num_attention_heads
            ),
            cache_strategy=self.pipeline_config.cache_strategy,
            enable_prefix_caching=self.pipeline_config.enable_prefix_caching,
        )

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        """Loads KV cache management objects for Llama vision.

        Args:
            session: Inference session to compile and init the KV cache.
            available_cache_memory: Amount of memory available to the KV cache,
                in bytes.

        Returns:
            A pair of KV managers: one for self the other for cross attention.
        """
        num_cross_attn_layers = len(self.text_config.cross_attention_layers)
        return MultimodalKVCacheManager(
            params=self._get_kv_params(),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            text_max_seq_len=max_seq_len(self.pipeline_config),
            vision_max_seq_len=self.vision_max_seq_len,
            text_num_layers=self.text_config.num_hidden_layers
            - num_cross_attn_layers,
            vision_num_layers=num_cross_attn_layers,
            devices=self.pipeline_config.devices,
            session=session,
            available_cache_memory=available_cache_memory,
            page_size=self.pipeline_config.kv_cache_page_size,
        )

    def estimate_kv_cache_size(self, available_cache_memory: int) -> int:
        return estimate_kv_cache_size(
            params=self._get_kv_params(),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=max_seq_len(self.pipeline_config),
            num_layers=self.text_config.num_hidden_layers,
            available_cache_memory=available_cache_memory,
            devices=self.pipeline_config.devices,
        )

    def load_model(
        self,
        session: InferenceSession,
    ) -> tuple[Model, Model]:
        """
        Load the Llama vision multimodal model. Since this is a multimodal model,
        we have vision and language models (graph) loaded.
        """
        self.weights = self.pipeline_config.load_weights()

        logging.info("Building vision model...")
        vision_model_graph = self._llama3_vision_vision_graph()

        logging.info("Building language model...")
        language_model_graph = self._llama3_vision_language_graph()

        logging.info("Compiling vision model...")
        before = time.perf_counter()
        vision_model = session.load(
            vision_model_graph,
            weights_registry=self.weights.allocated_weights,
        )
        after = time.perf_counter()
        logging.info(
            f"Compiling vision model took {after - before:.6f} seconds"
        )

        logging.info("Compiling language model...")
        before = time.perf_counter()
        language_model = session.load(
            language_model_graph,
            weights_registry=self.weights.allocated_weights,
        )
        after = time.perf_counter()
        logging.info(
            f"Compiling language model took {after - before:.6f} seconds"
        )
        return (vision_model, language_model)
