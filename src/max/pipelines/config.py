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

"""Standardized configuration for Pipeline Inference."""

from __future__ import annotations

import logging
import os
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Optional, get_type_hints

from max.driver import (
    DeviceSpec,
)
from max.graph.quantization import QuantizationEncoding
from max.pipelines.config_enums import PipelineEngine, RopeType
from max.pipelines.max_config import (
    KVCacheConfig,
    MAXConfig,
    MAXModelConfig,
    ProfilingConfig,
    SamplingConfig,
    repo_exists_with_retry,
)

logger = logging.getLogger("max.pipelines")


@dataclass(frozen=False)
class PipelineConfig(MAXConfig):
    """Configuration for a pipeline.

    WIP - Once a PipelineConfig is fully initialized, it should be as immutable
    as possible (frozen=True). All underlying dataclass fields should have been
    initialized to their default values, be it user specified via some CLI
    flag, config file, environment variable, or internally set to a reasonable
    default.
    """

    engine: Optional[PipelineEngine] = None
    """Engine backend to use for serving, 'max' for the max engine, or 'huggingface' as fallback option for improved model coverage."""

    serialized_model_path: Optional[str] = None
    """DEPRECATED: Serialization paths no longer supported."""

    save_to_serialized_model_path: Optional[str] = None
    """DEPRECATED: Serialization paths no longer supported."""

    max_length: Optional[int] = None
    """Maximum sequence length of the model."""

    max_new_tokens: int = -1
    """Maximum number of new tokens to generate during a single inference pass of the model."""

    max_batch_size: Optional[int] = None
    """Maximum batch size to execute with the model.
    This is set to one, to minimize memory consumption for the base case, in which a person is
    running a local server to test out MAX. For users launching in a server scenario, the expectation
    is that this value should be set higher based on server capacity."""

    max_ce_batch_size: int = 192
    """Maximum cache size to reserve for a single context encoding batch.
    The actual limit is the lesser of this and `max_batch_size`."""

    enable_chunked_prefill: bool = True
    """Enable chunked prefill to split context encoding requests into multiple chunks
    based on 'target_num_new_tokens'."""

    enable_in_flight_batching: bool = False
    """When enabled, prioritizes token generation by batching it with context
    encoding requests. Requires chunked prefill."""

    max_num_steps: int = -1
    """The number of steps to run for multi-step scheduling. -1 specifies a default value based on
    configuration and platform. Ignored for models which are not auto-regressive (e.g. embedding
    models)."""

    pad_to_multiple_of: int = 2
    """Pad input tensors to be a multiple of value provided."""

    target_num_new_tokens: Optional[int] = None
    """The target number of un-encoded tokens to include in each batch.
    If not set, this will be set to a best-guess optimal value based on model, hardware, and available memory."""

    enable_echo: bool = False
    """Whether the model should be built with echo capabilities."""

    rope_type: Optional[RopeType] = None
    """Force using a specific rope type: `none` | `normal` | `neox`. Only matters for GGUF weights."""

    pool_embeddings: bool = True
    """Whether to pool embedding outputs."""

    max_cache_batch_size: Optional[int] = None
    """DEPRECATED: The maximum cache batch size to use for the model. Use max_batch_size instead."""

    use_experimental_kernels: str = os.environ.get(
        "USE_EXPERIMENTAL_KERNELS", "false"
    )

    draft_model: Optional[str] = None
    """Draft model for use during Speculative Decoding."""

    ignore_eos: bool = False
    """Ignore EOS and continue generating tokens, even when an EOS variable is hit."""

    _model_config: MAXModelConfig = field(default_factory=MAXModelConfig)
    """The model config."""

    _sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    """The sampling config."""

    _profiling_config: ProfilingConfig = field(default_factory=ProfilingConfig)

    """The profiling config."""

    def __init__(self, **kwargs: Any) -> None:
        # Initialize all fields with their defaults first
        for curr_field in fields(self.__class__):
            if curr_field.default is not MISSING:
                setattr(self, curr_field.name, curr_field.default)
            elif curr_field.default_factory is not MISSING:
                setattr(self, curr_field.name, curr_field.default_factory())

        # Check if any kwargs are meant for other MAXConfig classes
        unmatched_kwargs: dict[str, Any] = {}
        # Then process kwargs which override defaults
        for key, value in list(kwargs.items()):
            if key in self.__dataclass_fields__:
                setattr(self, key, value)
                del kwargs[key]
            else:
                unmatched_kwargs[key] = value

        # Try to match unmatched kwargs with other config classes
        if unmatched_kwargs:
            # TODO(zheng): Make this more efficient by using MaxConfig instance
            # instead of hardcoding the config names.
            for config_name in [
                "_sampling_config",
                "_profiling_config",
                # TODO(zheng): Remove this once backward compatibility is no
                # longer needed for MAXModelConfig.
                "_model_config",
            ]:
                config_class = get_type_hints(self.__class__)[config_name]
                matched_kwargs = {}
                kv_cache_kwargs = {}

                for key, value in unmatched_kwargs.items():
                    if key in config_class.__dataclass_fields__:
                        matched_kwargs[key] = value
                    # Check if this is a KVCache config param
                    elif (
                        config_name == "_model_config"
                        and key in KVCacheConfig.__dataclass_fields__
                    ):
                        kv_cache_kwargs[key] = value

                if matched_kwargs:
                    if config_name == "_model_config" and kv_cache_kwargs:
                        # Create new model config with updated KVCache config
                        model_config = config_class(**matched_kwargs)
                        model_config._kv_cache_config = KVCacheConfig(
                            **kv_cache_kwargs
                        )
                        setattr(self, config_name, model_config)
                    else:
                        setattr(
                            self, config_name, config_class(**matched_kwargs)
                        )

                    # Remove matched kwargs
                    for key in matched_kwargs:
                        del unmatched_kwargs[key]
                    for key in kv_cache_kwargs:
                        del unmatched_kwargs[key]

        if unmatched_kwargs:
            raise ValueError(f"Unmatched kwargs: {unmatched_kwargs}")

        self.validate()

    def validate(self) -> None:
        """
        Validate the config.

        This method is called after the config is initialized, to ensure that all
        config fields have been initialized to a valid state.
        """
        self._model_config.validate()

        # Validate if a provided max_length is non-negative.
        if self.max_length is not None and self.max_length < 0:
            raise ValueError("max_length must be non-negative.")

        if self.max_cache_batch_size is not None:
            logger.warning(
                "--max-cache-batch-size is deprecated, use `--max-batch-size` instead. This setting will stop working in a future release."
            )
            self.max_batch_size = self.max_cache_batch_size

        # Set sensible defaults. These are platform-specific.
        if self.max_num_steps < 0:
            if (
                self.sampling_config.enable_structured_output
                or self._model_config.device_specs[0] == DeviceSpec.cpu()
            ):
                self.max_num_steps = 1
            else:
                self.max_num_steps = 10

        if (
            self.max_num_steps > 1
            and self.sampling_config.enable_structured_output
        ):
            raise ValueError(
                "max_num_steps > 1 not supported when enable_structured_output = True"
            )

        if self.sampling_config.enable_structured_output:
            if self._model_config.device_specs[0] == DeviceSpec.cpu():
                raise ValueError(
                    "enable_structured_output is not currently supported on CPU."
                )

        if self.draft_model:
            if not repo_exists_with_retry(self.draft_model):
                raise ValueError(
                    "draft_model provided does not exist on HuggingFace."
                    "Only public HuggingFace draft models currently supported."
                )

    def __getstate__(self) -> dict[str, Any]:
        """Override `__getstate__` to exclude the Hugging Face config."""
        state = self.__dict__.copy()
        return state

    @property
    def graph_quantization_encoding(self) -> Optional[QuantizationEncoding]:
        """Converts the CLI encoding to a MAX graph quantization encoding.

        Returns:
            The graph quantization encoding corresponding to the CLI encoding.
        """
        return self._model_config.graph_quantization_encoding

    @staticmethod
    def help() -> dict[str, str]:
        pipeline_help = {
            "engine": "Specify the engine backend to use for serving the model. Options include `max` for the MAX engine, or `huggingface` as a fallback option that provides improved model coverage.",
            "weight_path": "Provide an optional local path or path relative to the root of a Hugging Face repo to the model weights you want to use. This allows you to specify custom weights instead of using defaults. You may pass multiple, ie. `--weight-path=model-00001-of-00002.safetensors --weight-path=model-00002-of-00002.safetensors`",
            "serialized_model_path": "DEPRECATED: Serialization paths no longer supported.",
            "save_to_serialized_model_path": "DEPRECATED: Serialization paths no longer supported.",
            "max_length": "Set the maximum sequence length for input data processed by the model. This must be less than the value specified in the Hugging Face configuration file. The default is derived from the Hugging Face configuration value. Larger values may consume more memory.",
            "max_new_tokens": "Specify the maximum number of new tokens to generate during a single inference pass of the model. Default is -1, which means the model will generate until the maximum sequence length is hit, or and eos token is generated.",
            "max_batch_size": "Define the maximum cache size reserved for a single batch. This value defaults to 1. Increase this value based on server capacity when deploying in production.",
            "max_ce_batch_size": "Set the maximum cache size reserved for a single context encoding batch. The effective limit will be the lesser of this value and `max-cache-batch-size`. Default is 32.",
            "enable_chunked_prefill": "Enable chunked prefill to split context encoding requests into multiple chunks based on `target-num-new-tokens`",
            "enable_in_flight_batching": "When enabled, prioritizes token generation by batching it with context encoding requests. Requires chunked prefill.",
            "max_cache_batch_size": "DEPRECATED: Use `max_batch_size` instead.",
            "rope_type": "Force using a specific rope type, `none` | `normal' | `nexo`. Only matters for GGUF weights.",
            "max_num_steps": "Specify the number of steps to run for multi-step scheduling during inference. Default is set to 1.",
            "pad_to_multiple_of": "Pad input tensors to be a multiple of value provided. Default is set to 2.",
            "enable_echo": "Whether the model should be built with echo capabilities. This defaults to false.",
            "draft_model": "Draft model for use in speculative decoding.",
            "ignore_eos": "Ignore EOS and continue generating tokens, even when an EOS variable is hit.",
        }

        # Add help text for all MAX config classes
        # TODO(zheng): Make this more efficient by using MaxConfig instance
        # instead of hardcoding the config names.
        for config_class in [SamplingConfig, ProfilingConfig]:
            config_help = config_class.help()  # type: ignore
            for key in config_help:
                if key in pipeline_help:
                    raise ValueError(
                        f"Duplicate help key '{key}' found in {config_class.__name__}"
                    )
            pipeline_help.update(config_help)
        return pipeline_help

    @property
    def model_config(self) -> MAXModelConfig:
        return self._model_config

    @property
    def sampling_config(self) -> SamplingConfig:
        return self._sampling_config

    @property
    def profiling_config(self) -> ProfilingConfig:
        return self._profiling_config
