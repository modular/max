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
"""MAX config classes."""

from __future__ import annotations

import glob
import json
import logging
import os
import random
import struct
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Optional, Union, cast

import huggingface_hub
import torch
from huggingface_hub import constants as hf_hub_constants
from huggingface_hub import errors as hf_hub_errors
from max.driver import (
    DeviceSpec,
    devices_exist,
    scan_available_devices,
)
from max.dtype import DType
from max.engine import GPUProfilingMode
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.graph.weights import WeightsFormat
from max.pipelines.config_enums import RepoType, SupportedEncoding
from max.pipelines.kv_cache import KVCacheStrategy
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class MAXConfig:
    """Abstract base class for all MAX configs.

    There are some invariants that :obj:`MAXConfig` classes should follow:
    - All config classes should be dataclasses.
    - All config classes should have a :obj:`help()` method that returns a dictionary of config
    options and their descriptions.
    - All config classes dataclass fields should have default values, and hence
    can be trivially initialized via :obj:`cls()`.
    - All config classes should be frozen (except :obj:`KVCacheConfig` for now), to
    avoid accidental modification of config objects.
    - All config classes must have mutually exclusive dataclass fields among
    themselves.
    """

    @abstractmethod
    def help(self) -> dict[str, str]:
        """Documentation for this config class. Return a dictionary of config
        options and their descriptions."""
        ...


# frozen is False (for now) because of _available_cache_memory being set by
# internal code.
@dataclass(frozen=False)
class KVCacheConfig(MAXConfig):
    cache_strategy: KVCacheStrategy = KVCacheStrategy.MODEL_DEFAULT
    """The cache strategy to use. This defaults to :obj:`model_default`, which will set the cache
    strategy based on the default strategy for the architecture requested.

    You can also force the engine to use a specific caching strategy: :obj:`naive` | :obj:`continuous` | :obj:`paged`.
    """

    kv_cache_page_size: int = 128
    """The number of tokens in a single page in the paged KVCache."""

    enable_prefix_caching: bool = False
    """Whether to enable prefix caching for the paged attention KVCache."""

    device_memory_utilization: float = 0.9
    """The fraction of available device memory that the process should consume.

    This is used to inform the size of the KVCache workspace. The calculation is:

    .. math::

        kv\\_cache\\_workspace = (total\\_free\\_memory \\times device\\_memory\\_utilization) - model\\_weights\\_size
    """

    _available_cache_memory: Optional[int] = None
    """The amount of available cache memory in bytes. This should only be set by internal code."""

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "cache_strategy": "Force a specific cache strategy: 'naive' or 'continuous'. If not provided, the optimal caching strategy for the model requested will be selected.",
            "kv_cache_page_size": "The number of tokens in a single page in the paged KVCache. Default is set to 512.",
            "enable_prefix_caching": "Whether to enable prefix caching for the paged attention KVCache. This defaults to false.",
            "device_memory_utilization": "The fraction of available device memory that the process should consume. This is used to inform the size of the KVCache workspace: kv_cache_workspace = (total_free_memory * device_memory_utilization) - model_weights_size. Default is set to 0.9.",
        }


@dataclass
class MAXModelConfigBase(MAXConfig):
    """Abstract base class for all (required) MAX model configs.

    This base class is used to configure a model to use for a pipeline, but also
    handy to sidestep the need to pass in optional fields when subclassing
    MAXModelConfig.
    """

    @staticmethod
    def help() -> dict[str, str]:
        return {}


@dataclass
class MAXModelConfig(MAXModelConfigBase):
    """Abstract base class for all MAX model configs.

    This class is used to configure a model to use for a pipeline.
    """

    # NOTE: model_path is made a str of "" by default, to avoid having
    # it be Optional to check for None and then littering the codebase with
    # asserts just to keep mypy happy.
    model_path: str = ""
    """:obj:`repo_id` of a Hugging Face model repository to use."""

    huggingface_repo_id: str = ""
    """DEPRECATED: :obj:`repo_id` of a Hugging Face model repository to use. Use :obj:`model_path` instead."""

    weight_path: list[Path] = field(default_factory=list)
    """Optional path or url of the model weights to use."""

    # TODO(zheng): Move this under QuantizationConfig.
    quantization_encoding: Optional[SupportedEncoding] = None
    """Weight encoding type."""

    # Tuck "huggingface_revision" and "trust_remote_code" under a separate
    # HuggingFaceConfig class.
    huggingface_revision: str = hf_hub_constants.DEFAULT_REVISION
    """Branch or Git revision of Hugging Face model repository to use."""

    trust_remote_code: bool = False
    """Whether or not to allow for custom modelling files on Hugging Face."""

    device_specs: list[DeviceSpec] = field(
        default_factory=scan_available_devices
    )
    """Devices to run inference upon. This option is not documented in :obj:`help()` as it shouldn't be used directly via the CLI entrypoint."""

    force_download: bool = False
    """Whether to force download a given file if it's already present in the local cache."""

    _weights_repo_id: Optional[str] = None
    """Hugging Face repo id to load weights from only. This should only be set by internal code."""

    # TODO(zheng): Refactor QuantizationConfig to be a MAXConfig subclass that
    # also autopopulates default values.
    _quant_config: Optional[QuantizationConfig] = None
    """Optional config for specifying quantization parameters. This should only be set by internal code."""

    _kv_cache_config: KVCacheConfig = field(default_factory=KVCacheConfig)
    """The KVCache config."""

    # TODO(zheng): This can't just be a __post_init__ method, because we need to
    # it also sets and updates other fields which may not be determined /
    # initialized in the default factory.
    # Realistically, this shouldn't become a problem in the long term once we
    # instantiate these MAXConfigs with probably DAG depedency flows in our
    # larger config refactor.
    def validate(self):
        """Validates the config.

        This method is called after the model config is initialized, to ensure that all
        config fields have been initialized to a valid state. It will also set
        and update other fields which may not be determined / initialized in the
        default factory.
        """
        # Validate that the device_specs provided are available
        if not devices_exist(self.device_specs):
            available_devices = scan_available_devices()
            msg = f"device specs provided ({self.device_specs}) do not exist."
            msg += f"\navailable devices: {available_devices}"
            raise ValueError(msg)

        if self.huggingface_repo_id != "":
            logger.warning(
                "--huggingface-repo-id is deprecated, use `--model-path` instead. This setting will stop working in a future release."
            )
            self.model_path = self.huggingface_repo_id

        # Replit model_paths are kinda broken due to transformers
        # version mismatch. We manually update trust_remote_code to True
        # because the modularai version does not have the custom Python code needed
        # Without this, we get:
        #     ValueError: `attn_type` has to be either `multihead_attention` or
        #     `multiquery_attention`. Received: grouped_query_attention
        # Another reason why we override this flag here is because at PipelineConfig
        # instantiation below, we'll call AutoConfig.from_pretrained, which will
        # trigger the error above if not set to True.
        if "replit" in self.model_path.lower():
            self.trust_remote_code = True

        if (
            "llama" not in self.model_path.lower()
            and len(self.device_specs) > 1
            and self.device_specs[0].device_type == "gpu"
        ):
            raise ValueError(
                "Multiple GPU inference is currently not supported for non-Llama models."
            )

        # Validate that if weight_paths are passed as strings, they are converted to Path.
        if isinstance(self.weight_path, tuple):
            self.weight_path = list(self.weight_path)
        elif not isinstance(self.weight_path, list):
            self.weight_path = [self.weight_path]
        weight_paths = []
        # Validate that if weight_paths are passed as strings, they are converted to Path.
        for path in self.weight_path:
            if isinstance(path, str):
                path = Path(path)
            elif not isinstance(path, Path):
                raise ValueError(
                    "weight_path provided must either be string or Path:"
                    f" '{path}'"
                )
            elif path.is_file():
                # If we already exist on the OS. Dont parse the path, just continue.
                weight_paths.append(path)
                continue

            # If the path, looks like it may start with a Hugging Face repo id,
            # check if the repo_id is the same as the one provided.
            # If it is the same, set the weight_path to just be the file_name post repo_id
            # If it is different, set the _weights_repo_id to be that repo_id
            # and set the path to be the file_name without the repo_id.
            if path_pieces := str(path).split("/"):
                if len(path_pieces) >= 3:
                    repo_id = f"{path_pieces[0]}/{path_pieces[1]}"
                    file_name = "/".join(path_pieces[2:])
                    if self.model_path != "" and repo_id == self.model_path:
                        path = Path(file_name)
                    elif huggingface_hub.file_exists(repo_id, file_name):
                        self._weights_repo_id = repo_id
                        path = Path(file_name)
                elif self.model_path == "":
                    raise ValueError(
                        "Unable to derive model_path from weight_path, "
                        "please provide a valid Hugging Face repository id."
                    )

            weight_paths.append(path)

        self.weight_path = weight_paths

        # If we cannot infer the weight path, we lean on the model_path
        # to provide it.
        if len(self.weight_path) == 0:
            if self.model_path == "":
                raise ValueError(
                    "model_path must be provided and must be a valid Hugging Face repository"
                )
            elif (not os.path.exists(self.model_path)) and (
                not repo_exists_with_retry(self.model_path)
            ):
                raise ValueError(
                    f"{self.model_path} is not a valid Hugging Face repository"
                )
        elif self.model_path == "" and self._weights_repo_id is not None:
            # weight_path is used and we should derive the repo_id from it.
            # At this point, we should have a resolved weight path - be it local or remote HF.
            # weight_path should not be used directly anymore.
            self.model_path = self._weights_repo_id

    @property
    def kv_cache_config(self) -> KVCacheConfig:
        return self._kv_cache_config

    @property
    def graph_quantization_encoding(self) -> Optional[QuantizationEncoding]:
        """Converts the CLI encoding to a MAX Graph quantization encoding.

        Returns:
            The graph quantization encoding corresponding to the CLI encoding.

        Raises:
            ValueError: If no CLI encoding was specified.
        """
        if self.quantization_encoding is None:
            raise ValueError(
                "can't convert `None` CLI encoding to graph quantization encoding"
            )

        return self.quantization_encoding.quantization_encoding

    def finalize_encoding_config(self):
        if self.quantization_encoding == SupportedEncoding.gptq:
            hf_config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                revision=self.huggingface_revision,
            )
            hf_quant_config = hf_config.quantization_config

            if hf_config.torch_dtype is not torch.float16:
                raise ValueError(
                    "bfloat16 scales are not supported for GPTQ-quantized models."
                )

            self._quant_config = QuantizationConfig(
                quant_method=hf_quant_config["quant_method"],
                bits=hf_quant_config["bits"],
                group_size=hf_quant_config["group_size"],
                desc_act=hf_quant_config["desc_act"],
                sym=hf_quant_config["sym"],
            )

    def weights_size(self) -> int:
        size = 0
        hf_repo = HuggingFaceRepo(
            (
                self._weights_repo_id
                if self._weights_repo_id
                else self.model_path
            ),
            trust_remote_code=self.trust_remote_code,
        )
        for file_path in self.weight_path:
            if os.path.exists(file_path):
                size += os.path.getsize(file_path)
                continue

            next_size = hf_repo.size_of(str(file_path))

            if next_size is None:
                raise ValueError(
                    f"Failed to get size of weight file {file_path}"
                )
            size += next_size

        return size

    def huggingface_weights_repo(self) -> HuggingFaceRepo:
        return HuggingFaceRepo(
            (
                self._weights_repo_id
                if self._weights_repo_id
                else self.model_path
            ),
            trust_remote_code=self.trust_remote_code,
        )

    @staticmethod
    def help() -> dict[str, str]:
        max_model_help = {
            "model_path": "Specify the repository ID of a Hugging Face model repository to use. This is used to load both Tokenizers, architectures and model weights.",
            "huggingface_repo_id": "DEPRECATED: Use `model_path` instead.",
            "weight_path": "Provide an optional local path or path relative to the root of a Hugging Face repo to the model weights you want to use. This allows you to specify custom weights instead of using defaults. You may pass multiple, ie. `--weight-path=model-00001-of-00002.safetensors --weight-path=model-00002-of-00002.safetensors`",
            "quantization_encoding": "Define the weight encoding type for quantization. This can help optimize performance and memory usage during inference. ie. q4_k, bfloat16 etc.",
            "huggingface_revision": "Branch or Git revision of Hugging Face model repository to use.",
            "trust_remote_code": "Indicate whether to allow custom modelling files from Hugging Face repositories. Set this to true with caution, as it may introduce security risks.",
            "force_download": "Specify whether to forcefully download a file even if it already exists in local cache. Set this to true if you want to ensure you have the latest version.",
        }

        config_help = KVCacheConfig.help()
        for key in config_help:
            if key in max_model_help:
                raise ValueError(
                    f"Duplicate help key '{key}' found in {KVCacheConfig.__name__}"
                )
        max_model_help.update(config_help)
        return max_model_help


@dataclass
class SamplingConfig(MAXConfig):
    top_k: int = 1
    """Limits the sampling to the K most probable tokens. This defaults to 1, which enables greedy sampling."""

    in_dtype: DType = DType.float32
    """The data type of the input tokens."""

    out_dtype: DType = DType.float32
    """The data type of the output logits."""

    enable_structured_output: bool = False
    """Enable structured generation/guided decoding for the server. This allows the user to pass a json
    schema in the response_format field, which the LLM will adhere to."""

    enable_variable_logits: bool = False
    """Enable the sampling graph to accept a ragged tensor of different sequences as inputs, along with
    their associated logit_offsets. This is needed to produce additional logits for echo and speculative
    decoding purposes."""

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "top_k": "Limit sampling to the top K most probable tokens during generation. This can help control randomness and improve output quality. This defaults to 1, which defaults to greedy sampling.",
            "enable_structured_output": "Whether to enable constrained decoding in the text generation pipeline. This defaults to false.",
        }


@dataclass
class ProfilingConfig(MAXConfig):
    gpu_profiling: GPUProfilingMode = GPUProfilingMode.OFF
    """Whether to enable GPU profiling of the model."""

    def __post_init__(self):
        gpu_profiling_env = os.environ.get("MODULAR_ENABLE_PROFILING", "off")

        if self.gpu_profiling == GPUProfilingMode.OFF:
            try:
                self.gpu_profiling = GPUProfilingMode(gpu_profiling_env)
            except ValueError:
                valid_values = [mode.value for mode in GPUProfilingMode]
                raise ValueError(
                    "gpu_profiling must be one of: " + ", ".join(valid_values)
                )

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "gpu_profiling": "Whether to turn on GPU profiling for the model. This defaults to 'off'.",
        }


def repo_exists_with_retry(repo_id: str) -> bool:
    """
    Wrapper around huggingface_hub.repo_exists with retry logic.
    Uses exponential backoff with 25% jitter, starting at 1s and doubling each retry.

    See huggingface_hub.repo_exists for details
    """
    max_attempts = 5
    base_delays = [2**i for i in range(max_attempts)]
    retry_delays_in_seconds = [
        d * (1 + random.uniform(-0.25, 0.25)) for d in base_delays
    ]

    for attempt, delay_in_seconds in enumerate(retry_delays_in_seconds):
        try:
            return huggingface_hub.repo_exists(repo_id)
        except (
            hf_hub_errors.RepositoryNotFoundError,
            hf_hub_errors.GatedRepoError,
            hf_hub_errors.RevisionNotFoundError,
            hf_hub_errors.EntryNotFoundError,
        ) as e:
            # Forward these specific errors to the user
            logger.error(f"Hugging Face repository error: {str(e)}")
            raise
        except (hf_hub_errors.HfHubHTTPError, RequestsConnectionError) as e:
            if attempt == max_attempts - 1:
                logger.error(
                    f"Failed to connect to Hugging Face Hub after {max_attempts} attempts: {str(e)}"
                )
                raise

            logger.warning(
                f"Transient Hugging Face Hub connection error (attempt {attempt + 1}/{max_attempts}): {str(e)}"
            )
            logger.warning(
                f"Retrying Hugging Face connection in {delay_in_seconds} seconds..."
            )
            time.sleep(delay_in_seconds)

    assert False, (
        "This should never be reached due to the raise in the last attempt"
    )


@dataclass
class HuggingFaceRepo:
    repo_id: str
    trust_remote_code: bool = False
    repo_type: Optional[RepoType] = None

    def __post_init__(self) -> None:
        # Get repo type.
        if not self.repo_type:
            if os.path.exists(self.repo_id):
                self.repo_type = RepoType.local
            else:
                self.repo_type = RepoType.online

        if self.repo_type == RepoType.online and not repo_exists_with_retry(
            self.repo_id
        ):
            raise ValueError(f"model_path: {self.repo_id} does not exist")

    def __str__(self) -> str:
        return self.repo_id

    def __repr__(self) -> str:
        return self.repo_id

    @cached_property
    def info(self) -> huggingface_hub.ModelInfo:
        if self.repo_type == RepoType.local:
            raise ValueError(
                "using model info, on local repos is not supported."
            )
        elif self.repo_type == RepoType.online:
            return huggingface_hub.model_info(
                self.repo_id, files_metadata=False
            )
        else:
            raise ValueError(f"Unsupported repo type: {self.repo_type}")

    @cached_property
    def weight_files(self) -> dict[WeightsFormat, list[str]]:
        safetensor_search_pattern = "*.safetensors"
        gguf_search_pattern = "*.gguf"
        pytorch_search_pattern = "*.bin"

        weight_files = {}
        if self.repo_type == RepoType.local:
            safetensor_paths = glob.glob(
                os.path.join(self.repo_id, safetensor_search_pattern)
            )
            gguf_paths = glob.glob(
                os.path.join(self.repo_id, gguf_search_pattern)
            )
            pytorch_paths = glob.glob(
                os.path.join(self.repo_id, pytorch_search_pattern)
            )
        elif self.repo_type == RepoType.online:
            fs = huggingface_hub.HfFileSystem()
            safetensor_paths = cast(
                list[str],
                fs.glob(f"{self.repo_id}/{safetensor_search_pattern}"),
            )
            gguf_paths = cast(
                list[str],
                fs.glob(f"{self.repo_id}/{gguf_search_pattern}"),
            )
            pytorch_paths = cast(
                list[str],
                fs.glob(f"{self.repo_id}/{pytorch_search_pattern}"),
            )
        else:
            raise ValueError(f"Unsupported repo type: {self.repo_type}")

        if safetensor_paths:
            if len(safetensor_paths) == 1:
                # If there is only one weight allow any name.
                weight_files[WeightsFormat.safetensors] = [
                    safetensor_paths[0].replace(f"{self.repo_id}/", "")
                ]
            else:
                # If there is more than one weight, ignore consolidated tensors.
                weight_files[WeightsFormat.safetensors] = [
                    f.replace(f"{self.repo_id}/", "")
                    for f in safetensor_paths
                    if "consolidated" not in f
                ]

        if gguf_paths:
            weight_files[WeightsFormat.gguf] = [
                f.replace(f"{self.repo_id}/", "") for f in gguf_paths
            ]

        if pytorch_paths:
            weight_files[WeightsFormat.pytorch] = [
                f.replace(f"{self.repo_id}/", "") for f in pytorch_paths
            ]

        return weight_files

    def size_of(self, filename: str) -> Union[int, None]:
        if self.repo_type == RepoType.online:
            url = huggingface_hub.hf_hub_url(self.repo_id, filename)
            metadata = huggingface_hub.get_hf_file_metadata(url)
            return metadata.size
        raise NotImplementedError("not implemented for non-online repos.")

    @cached_property
    def supported_encodings(self) -> list[SupportedEncoding]:
        # TODO(AITLIB-128): Detection of supported encodings in weights can be cleaned up
        supported_encodings = set([])

        # Parse gguf file names.
        for gguf_path in self.weight_files.get(WeightsFormat.gguf, []):
            encoding = SupportedEncoding.parse_from_file_name(gguf_path)
            if encoding:
                supported_encodings.add(encoding)

        # Get Safetensor Metadata.
        if WeightsFormat.safetensors in self.weight_files:
            if self.repo_type == RepoType.local:
                # Safetensor repos are assumed to only have one encoding in them.
                with open(
                    os.path.join(
                        self.repo_id,
                        self.weight_files[WeightsFormat.safetensors][0],
                    ),
                    "rb",
                ) as file:
                    # Read the first 8 bytes of the file
                    length_bytes = file.read(8)
                    # Interpret the bytes as a little-endian unsigned 64-bit integer
                    length_of_header = struct.unpack("<Q", length_bytes)[0]
                    # Read length_of_header bytes
                    header_bytes = file.read(length_of_header)
                    # Interpret the bytes as a JSON object
                    header = json.loads(header_bytes)

                    encoding = None
                    for weight_value in header.values():
                        if weight_dtype := weight_value.get("dtype", None):
                            if weight_dtype == "F32":
                                supported_encodings.add(
                                    SupportedEncoding.float32
                                )
                            elif weight_dtype == "BF16":
                                supported_encodings.add(
                                    SupportedEncoding.bfloat16
                                )
                            else:
                                logger.warning(
                                    f"unknown dtype found in safetensors file: {weight_dtype}"
                                )

            elif self.repo_type == RepoType.online:
                if safetensors_info := self.info.safetensors:
                    for params in safetensors_info.parameters:
                        if "BF16" in params:
                            supported_encodings.add(SupportedEncoding.bfloat16)
                        elif "F32" in params:
                            supported_encodings.add(SupportedEncoding.float32)
                if safetensors_config := self.info.config:
                    if quant_config := safetensors_config.get(
                        "quantization_config"
                    ):
                        if quant_config["quant_method"] == "gptq":
                            supported_encodings.add(SupportedEncoding.gptq)
            else:
                raise ValueError(f"Unsupported repo_type: {self.repo_type}")

        # Get torch dtype for pytorch files.
        if WeightsFormat.pytorch in self.formats_available:
            cfg = AutoConfig.from_pretrained(
                self.repo_id, trust_remote_code=self.trust_remote_code
            )

            if torch_dtype := getattr(cfg, "torch_dtype", None):
                if torch_dtype == torch.float32:
                    supported_encodings.add(SupportedEncoding.float32)
                elif torch_dtype == torch.bfloat16:
                    supported_encodings.add(SupportedEncoding.bfloat16)
            else:
                logger.warning(
                    "torch_dtype not available, cant infer encoding from config.json"
                )

        return list(supported_encodings)

    def _get_gguf_files_for_encoding(
        self, encoding: SupportedEncoding
    ) -> dict[WeightsFormat, list[Path]]:
        files = []
        for gguf_file in self.weight_files.get(WeightsFormat.gguf, []):
            file_encoding = SupportedEncoding.parse_from_file_name(gguf_file)
            if file_encoding == encoding:
                files.append(Path(gguf_file))

        if files:
            return {WeightsFormat.gguf: files}
        else:
            return {}

    def _get_safetensor_files_for_encoding(
        self, encoding: SupportedEncoding
    ) -> dict[WeightsFormat, list[Path]]:
        if (
            WeightsFormat.safetensors in self.weight_files
            and encoding == self.supported_encodings[0]
        ):
            return {
                WeightsFormat.safetensors: [
                    Path(f)
                    for f in self.weight_files[WeightsFormat.safetensors]
                ]
            }

        return {}

    def _get_pytorch_files_for_encoding(
        self, encoding: SupportedEncoding
    ) -> dict[WeightsFormat, list[Path]]:
        if (
            WeightsFormat.pytorch in self.weight_files
            and encoding == self.supported_encodings[0]
        ):
            return {
                WeightsFormat.pytorch: [
                    Path(f) for f in self.weight_files[WeightsFormat.pytorch]
                ]
            }

        return {}

    def files_for_encoding(
        self,
        encoding: SupportedEncoding,
        weights_format: Optional[WeightsFormat] = None,
        alternate_encoding: Optional[SupportedEncoding] = None,
    ) -> dict[WeightsFormat, list[Path]]:
        if weights_format == WeightsFormat.pytorch:
            logger.warning(
                "cannot infer encoding from .bin files, returning all bin files"
            )
            return self._get_pytorch_files_for_encoding(encoding)

        if weights_format is WeightsFormat.gguf:
            return self._get_gguf_files_for_encoding(encoding)
        elif weights_format == WeightsFormat.safetensors:
            return self._get_safetensor_files_for_encoding(encoding)

        gguf_files = self._get_gguf_files_for_encoding(encoding)

        safetensor_files = self._get_safetensor_files_for_encoding(encoding)
        gguf_files.update(safetensor_files)

        pytorch_files = self._get_pytorch_files_for_encoding(encoding)
        gguf_files.update(pytorch_files)

        if not gguf_files and alternate_encoding:
            logger.warning(
                "Could not find checkpoint with %s encoding, searching for %s files instead.",
                encoding,
                alternate_encoding,
            )
            return self.files_for_encoding(alternate_encoding, weights_format)
        return gguf_files

    def file_exists(self, filename: str) -> bool:
        return huggingface_hub.file_exists(self.repo_id, filename)

    def download(self, filename: str, force_download: bool = False) -> Path:
        return Path(
            huggingface_hub.hf_hub_download(
                self.repo_id, filename, force_download=force_download
            )
        )

    @property
    def formats_available(self) -> list[WeightsFormat]:
        return list(self.weight_files.keys())

    def encoding_for_file(self, file: Union[str, Path]) -> SupportedEncoding:
        if str(file).endswith(".safetensors"):
            # If this file is safetensors, return the first encoding, as Safetensor repos can only have one.
            return self.supported_encodings[0]
        elif str(file).endswith(".gguf"):
            encoding = SupportedEncoding.parse_from_file_name(str(file))
            if encoding:
                return encoding

            raise ValueError(
                f"gguf file, but encoding not found in file name: {file}"
            )
        elif str(file).endswith(".bin"):
            # If this file is pytorch, return the first encoding, as Pytorch repos only likely have one.
            return self.supported_encodings[0]
        else:
            raise ValueError(
                f"weight path: {file} not gguf or safetensors, cannot infer encoding from file."
            )
