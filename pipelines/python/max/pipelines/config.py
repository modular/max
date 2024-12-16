# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Standardized config for Pipeline Inference."""

import datetime
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Union

from huggingface_hub import (
    file_exists,
    get_hf_file_metadata,
    get_safetensors_metadata,
    hf_hub_download,
    hf_hub_url,
    list_repo_files,
    model_info,
    repo_exists,
)
from huggingface_hub.hf_api import ModelInfo
from huggingface_hub.utils import SafetensorsRepoMetadata
from max.driver import CPU, CUDA, Device, DeviceSpec, accelerator_count
from max.dtype import DType
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import (
    GGUFWeights,
    SafetensorWeights,
    Weights,
    WeightsConverter,
)
from max.pipelines.kv_cache import KVCacheStrategy
from transformers import AutoConfig

logger = logging.getLogger(__name__)


class PipelineEngine(str, Enum):
    MAX = "max"
    HUGGINGFACE = "huggingface"


class SupportedEncoding(str, Enum):
    """All possible encodings which may be supported by a particular model."""

    float32 = "float32"
    bfloat16 = "bfloat16"
    q4_k = "q4_k"
    q4_0 = "q4_0"
    q6_k = "q6_k"

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    @classmethod
    def parse_from_file_name(cls, name: str):
        if "f32" in name or "float32" in name:
            return SupportedEncoding.float32
        elif "bf16" in name or "bfloat16" in name:
            return SupportedEncoding.bfloat16
        elif "q4_k_m" in name:
            return SupportedEncoding.q4_k
        elif "q4_0" in name:
            return SupportedEncoding.q4_0
        elif "q6_k" in name:
            return SupportedEncoding.q6_k
        else:
            return None

    @property
    def quantization_encoding(self) -> Optional[QuantizationEncoding]:
        if self in [SupportedEncoding.float32, SupportedEncoding.bfloat16]:
            return None
        elif self == SupportedEncoding.q4_k:
            return QuantizationEncoding.Q4_K
        elif self == SupportedEncoding.q4_0:
            return QuantizationEncoding.Q4_0
        elif self == SupportedEncoding.q6_k:
            return QuantizationEncoding.Q6_K
        else:
            raise ValueError(
                "SupportedEncoding does not have corresponding"
                " QuantizationEncoding."
            )

    @property
    def dtype(self) -> DType:
        """The underlying model dtype associated with a quantization_encoding."""
        if self == SupportedEncoding.bfloat16:
            return DType.bfloat16
        elif self in [
            SupportedEncoding.q4_k,
            SupportedEncoding.q4_0,
            SupportedEncoding.q6_k,
        ]:
            return DType.uint8

        return DType.float32


class WeightsFormat(str, Enum):
    gguf = "gguf"
    safetensors = "safetensors"
    pytorch = "pytorch"


@dataclass
class HuggingFaceRepo:
    repo_id: str
    _info: Optional[ModelInfo] = None
    _formats_available: list[WeightsFormat] = field(default_factory=list)
    _supported_encodings: list[SupportedEncoding] = field(default_factory=list)
    _gguf_architecture: Optional[str] = None
    _files: Iterable[str] = field(default_factory=list)
    _safetensors_metadata: Optional[SafetensorsRepoMetadata] = None

    def __post_init__(self) -> None:
        # Check if repo exists.
        if not repo_exists(self.repo_id):
            msg = f"huggingface_repo_id: {self.repo_id} does not exist"
            raise ValueError(msg)

    def __str__(self) -> str:
        return self.repo_id

    def __repr__(self) -> str:
        return self.repo_id

    @property
    def info(self) -> ModelInfo:
        if not self._info:
            self._info = model_info(self.repo_id, files_metadata=False)

        return self._info

    @property
    def files(self) -> Iterable[str]:
        if not self._files:
            self._files = list_repo_files(self.repo_id)

        return self._files

    def size_of(self, filename: str) -> Union[int, None]:
        url = hf_hub_url(self.repo_id, filename)
        metadata = get_hf_file_metadata(url)
        return metadata.size

    @property
    def safetensors_metadata(self) -> Optional[SafetensorsRepoMetadata]:
        if not self._safetensors_metadata:
            if WeightsFormat.safetensors in self.formats_available:
                self._safetensors_metadata = get_safetensors_metadata(
                    repo_id=self.repo_id
                )

        return self._safetensors_metadata

    @property
    def supported_encodings(self) -> list[SupportedEncoding]:
        if not self._supported_encodings:
            if WeightsFormat.gguf in self.formats_available:
                for file_name in self.files:
                    encoding = SupportedEncoding.parse_from_file_name(file_name)
                    if encoding:
                        self._supported_encodings.append(encoding)

            if WeightsFormat.safetensors in self.formats_available:
                if safetensors_info := self.info.safetensors:
                    for params in safetensors_info.parameters:
                        if "BF16" in params:
                            self._supported_encodings.append(
                                SupportedEncoding.bfloat16
                            )
                        elif "F32" in params:
                            self._supported_encodings.append(
                                SupportedEncoding.float32
                            )

        return self._supported_encodings

    def _get_gguf_files_for_encoding(
        self, encoding: SupportedEncoding
    ) -> dict[WeightsFormat, list[Path]]:
        if WeightsFormat.gguf not in self.formats_available:
            return {}

        files = []
        for file_name in self.files:
            file_encoding = SupportedEncoding.parse_from_file_name(file_name)
            if file_encoding == encoding:
                files.append(Path(file_name))

        if files:
            return {WeightsFormat.gguf: files}
        else:
            return {}

    def _get_safetensor_files_for_encoding(
        self, encoding: SupportedEncoding
    ) -> dict[WeightsFormat, list[Path]]:
        if WeightsFormat.safetensors not in self.formats_available:
            return {}

        if encoding not in self.supported_encodings:
            return {}

        if paths := [
            Path(file) for file in self.files if file.endswith(".safetensors")
        ]:
            return {WeightsFormat.safetensors: paths}
        else:
            return {}

    def file_exists(self, filename: str) -> bool:
        return file_exists(self.repo_id, filename)

    def download(self, filename: str, force_download: bool = False) -> Path:
        return Path(
            hf_hub_download(
                self.repo_id, filename, force_download=force_download
            )
        )

    def files_for_encoding(
        self,
        encoding: SupportedEncoding,
        weights_format: Optional[WeightsFormat] = None,
    ) -> dict[WeightsFormat, list[Path]]:
        if weights_format is WeightsFormat.gguf:
            return self._get_gguf_files_for_encoding(encoding)
        elif weights_format == WeightsFormat.safetensors:
            return self._get_safetensor_files_for_encoding(encoding)

        gguf_files = self._get_gguf_files_for_encoding(encoding)
        safetensor_files = self._get_safetensor_files_for_encoding(encoding)

        gguf_files.update(safetensor_files)
        return gguf_files

    @property
    def gguf_architecture(self) -> Optional[str]:
        if not self._gguf_architecture:
            if hasattr(self.info, "gguf"):
                gguf_info = getattr(self.info, "gguf")
                self._gguf_architecture = gguf_info["architecture"]

        return self._gguf_architecture

    @property
    def formats_available(self) -> list[WeightsFormat]:
        if not self._formats_available:
            # Retrieve formats.
            if hasattr(self.info, "gguf"):
                self._formats_available.append(WeightsFormat.gguf)

            if getattr(self.info, "safetensors", None):
                self._formats_available.append(WeightsFormat.safetensors)

        return self._formats_available

    def encoding_for_file(self, file: Union[str, Path]) -> SupportedEncoding:
        if str(file).endswith(".safetensors"):
            # If this file is safetensors, return the first encoding, as Safetensor repos can only have one.
            return self.supported_encodings[0]
        elif str(file).endswith(".gguf"):
            encoding = SupportedEncoding.parse_from_file_name(str(file))
            if encoding:
                return encoding

            msg = f"gguf file, but encoding not found in file name: {file}"
            raise ValueError(msg)
        else:
            msg = f"weight path: {file} not gguf or safetensors"
            raise ValueError(msg)


@dataclass(frozen=False)
class PipelineConfig:
    huggingface_repo_id: str
    """repo_id of a huggingface model repository to use."""

    engine: Optional[PipelineEngine] = None
    """Engine backend to use for serving, 'max' for the max engine, or 'huggingface' as fallback option for improved model coverage."""

    architecture: Optional[str] = None
    """Model architecture to run."""

    version: Optional[str] = None
    """Name of the model version to run."""

    weight_path: list[Path] = field(default_factory=list)
    """Optional path or url of the model weights to use."""

    device_specs: list[DeviceSpec] = field(
        default_factory=lambda: [DeviceSpec.cpu()]
    )
    """Devices to run inference upon."""

    quantization_encoding: Optional[SupportedEncoding] = None
    """Weight encoding type."""

    serialized_model_path: Optional[str] = None
    """If specified, tries to load a serialized model from this path."""

    save_to_serialized_model_path: Optional[str] = None
    """If specified, tries to save a serialized model to this path."""

    max_length: int = 512
    """Maximum sequence length of the model."""

    max_new_tokens: int = -1
    """Maximum number of new tokens to generate during a single inference pass of the model."""

    max_cache_batch_size: int = 1
    """Maximum cache size to reserve for a single batch.
    This is set to one, to minimize memory consumption for the base case, in which a person is
    running a local server to test out MAX. For users launching in a server scenario, the expectation
    is that this value should be set higher based on server capacity."""

    max_ce_batch_size: int = 32
    """Maximum cache size to reserve for a single context encoding batch.
    The actual limit is the lesser of this and max_cache_batch_size."""

    cache_strategy: KVCacheStrategy = KVCacheStrategy.CONTINUOUS
    """Force using a specific cache strategy, 'naive' or 'continuous'."""

    max_num_steps: int = 1
    """The number of steps to run for multi-step scheduling."""

    pad_to_multiple_of: int = 2
    """Pad input tensors to be a multiple of value provided."""

    kv_cache_page_size: int = 512
    """The number of tokens in a single page in the KVCache."""

    gpu_memory_utilization: float = 0.9
    """The fraction of available device memory that the our process should consume.

    This is used to inform the size of the KVCache workspace:
        kv_cache_workspace = (total_free_memory * gpu_memory_utilization) - model_weights_size
    """

    top_k: Optional[int] = None
    """Limits the sampling to the K most probable tokens. If None, will default to greedy sampling."""

    trust_remote_code: bool = False
    """Whether or not to allow for custom modelling files on Huggingface."""

    force_download: bool = False
    """Whether to force download a given file if it’s not already present in the local cache."""

    _huggingface_config: Optional[AutoConfig] = None
    """The huggingface config associated with the huggingface repo id."""

    _devices: list[Device] = field(default_factory=list)
    """The underlying initialized devices, created by the specific device_specs."""

    _weights_converter: Optional[type[WeightsConverter]] = None
    """Weight converter for the provided `weight_path`."""

    enable_echo: bool = False
    """Whether the model should be built with echo capabilities."""

    def __post_init__(self) -> None:
        # Default if weight_path is passed as None
        if self.weight_path is None:
            msg = (
                "weight_path cannot be None, if no weight_paths are provided,"
                " pass an empty list."
            )
            raise ValueError(msg)

        # Validate that if weight_paths are passed as strings, they are converted to Path.
        if isinstance(self.weight_path, tuple):
            self.weight_path = list(self.weight_path)

        elif not isinstance(self.weight_path, list):
            self.weight_path = [self.weight_path]

        weight_paths = []
        for path in self.weight_path:
            if isinstance(path, Path):
                weight_paths.append(path)
            elif isinstance(path, str):
                weight_paths.append(Path(path))
            else:
                msg = (
                    "weight_path provided must either be string or Path:"
                    f" '{path}'"
                )
                raise ValueError(msg)

        self.weight_path = weight_paths

        # Validate that the repo exists.
        self.huggingface_repo = HuggingFaceRepo(self.huggingface_repo_id)

    @property
    def huggingface_config(self) -> AutoConfig:
        """Given the huggingface_repo_id, return the Huggingface Config."""

        if self._huggingface_config is None:
            self._huggingface_config = AutoConfig.from_pretrained(
                self.huggingface_repo_id,
                trust_remote_code=self.trust_remote_code,
            )
            # Update config for defaults.
            # Not all Huggingface Configs have max_seq_len,
            # if its provided, we update if needed,
            # if its not provided, we set it based on the max_length.
            assert self._huggingface_config is not None
            if hasattr(self.huggingface_config, "max_seq_len"):
                if self.max_length < self._huggingface_config.max_seq_len:
                    self._huggingface_config.max_seq_len = self.max_length
            elif hasattr(self.huggingface_config, "max_position_embeddings"):
                if (
                    self.max_length
                    < self._huggingface_config.max_position_embeddings
                ):
                    self._huggingface_config.max_seq_len = self.max_length
                else:
                    self._huggingface_config.max_seq_len = (
                        self._huggingface_config.max_position_embeddings
                    )
            else:
                self._huggingface_config.max_seq_len = self.max_length

        return self._huggingface_config

    @property
    def dtype(self) -> DType:
        if self.quantization_encoding is None:
            msg = "quantization_encoding must be provided to infer dtype."
            raise ValueError(msg)

        return self.quantization_encoding.dtype

    @property
    def devices(self) -> list[Device]:
        """Initialize and return a list of devices, given a list of device specs."""
        if self._devices:
            return self._devices
        num_devices_available = accelerator_count()
        for device_spec in self.device_specs:
            assert device_spec.id < num_devices_available
            self._devices.append(
                CPU(device_spec.id)
                if device_spec.device_type == "cpu"
                else CUDA(device_spec.id)
            )
        return self._devices

    @property
    def device(self) -> Device:
        """Initialize and return a singular device, given a singular device spec."""
        return self.devices[0]

    @property
    def weights_format(self) -> WeightsFormat:
        """Identify which format our weights are expected in."""

        if not self.weight_path:
            msg = "no weight_path provided cannot infer weights format."
            raise ValueError(msg)

        # Get all weight paths.
        if all(
            [weight_path.suffix == ".gguf" for weight_path in self.weight_path]
        ):
            return WeightsFormat.gguf
        elif all(
            [
                weight_path.suffix == ".safetensors"
                for weight_path in self.weight_path
            ]
        ):
            return WeightsFormat.safetensors
        else:
            msg = f"weights type cannot be inferred from {self.weight_path}"
            raise ValueError(msg)

    def weights_size(self) -> Optional[int]:
        size = 0
        for file_path in self.weight_path:
            if os.path.exists(file_path):
                size += os.path.getsize(file_path)
                continue

            next_size = self.huggingface_repo.size_of(str(file_path))

            if next_size is None:
                return None
            size += next_size

        return size

    def download_weights(self) -> None:
        # Try to load locally.
        if all([os.path.exists(file_path) for file_path in self.weight_path]):
            logger.info("All files exist locally, skipping download.")
            return

        start_time = datetime.datetime.now()
        logger.info(f"Starting download of model: {self.huggingface_repo_id}")
        for i, file_path in enumerate(self.weight_path):
            self.weight_path[i] = self.huggingface_repo.download(
                str(file_path),
                force_download=self.force_download,
            )

        logger.info(
            f"Finished download of model: {self.huggingface_repo_id} in {(datetime.datetime.now() - start_time).total_seconds()} seconds."
        )

    def load_weights(self) -> Weights:
        self.download_weights()

        if self._weights_converter:
            return self._weights_converter.load_weights(
                self.weight_path, config=self
            )

        if self.weights_format == WeightsFormat.gguf:
            if len(self.weight_path) > 1:
                raise ValueError("loading multiple gguf files is not supported")
            return GGUFWeights(self.weight_path[0])

        elif self.weights_format == WeightsFormat.safetensors:
            return SafetensorWeights(self.weight_path)

        else:
            msg = (
                f"loading weights format '{self.weights_format}' not supported"
            )
            raise ValueError(msg)

    @property
    def short_name(self) -> str:
        """Returns a short name for the model defined by this PipelineConfig."""
        # TODO: Deprecate use of short_name.
        return self.huggingface_repo_id

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "huggingface_repo_id": "Specify the repository ID of a Huggingface model repository to use. This is used to load both Tokenizers, architectures and model weights.",
            "engine": "Specify the engine backend to use for serving the model. Options include 'max' for the MAX engine, or 'huggingface' as a fallback option that provides improved model coverage.",
            "weight_path": "Provide an optional local path or path relative to the root of a Huggingface repo to the model weights you want to use. This allows you to specify custom weights instead of using defaults. You may pass multiple, ie. `--weight-path=model-00001-of-00002.safetensors --weight-path=model-00002-of-00002.safetensors`",
            "quantization_encoding": "Define the weight encoding type for quantization. This can help optimize performance and memory usage during inference. ie. q4_k, bfloat16 etc.",
            "maximum_length": "Set the maximum sequence length for input data processed by the model. The default is 512 tokens. This can be raised up until the maximum of the model provided. As this value increases memory consumed by the engine increases accordingly.",
            "max_new_tokens": "Specify the maximum number of new tokens to generate during a single inference pass of the model. Default is -1, which means the model will generate until the maximum sequence length is hit, or and eos token is generated.",
            "max_cache_batch_size": "Define the maximum cache size reserved for a single batch. This value defaults to 1. Increase this value based on server capacity when deploying in production.",
            "max_ce_batch_size": "Set the maximum cache size reserved for a single context encoding batch. The effective limit will be the lesser of this value and max-cache-batch-size. Default is 32.",
            "cache_strategy": "Force a specific cache strategy: 'naive' or 'continuous'. If not provided, the optimal caching strategy for the model requested will be selected.",
            "max_num_steps": "Specify the number of steps to run for multi-step scheduling during inference. Default is set to 1.",
            "top_k": "Limit sampling to the top K most probable tokens during generation. This can help control randomness and improve output quality. This defaults to 0, which defaults to greedy sampling.",
            "trust_remote_code": "Indicate whether to allow custom modelling files from Huggingface repositories. Set this to true with caution, as it may introduce security risks.",
            "force_download": "Specify whether to forcefully download a file even if it already exists in local cache. Set this to true if you want to ensure you have the latest version.",
            "architecture": "Deprecated - Please set `huggingface-repo-id` instead. Define the model architecture to run. This should match one of the supported architectures for your selected engine.",
            "version": "Deprecated - Please set `huggingface-repo-id` instead. Indicate the specific version of the model to run. If not specified, the default version will be used.",
            "serialized_model_path": "If specified, this flag attempts to load a serialized MEF model from the given path. This is useful for reusing previously saved models.",
            "save_to_serialized_model_path": "If specified, this flag attempts to save the current model state to a serialized format at the given path for later use.",
        }
