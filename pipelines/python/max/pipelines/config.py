# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Standardized config for Pipeline Inference."""

from __future__ import annotations

import datetime
import glob
import json
import logging
import os
import struct
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Union, cast

import torch
from huggingface_hub import (
    HfFileSystem,
    file_exists,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
    model_info,
    repo_exists,
)
from huggingface_hub.hf_api import ModelInfo
from max.driver import CPU, Accelerator, Device, DeviceSpec, accelerator_count
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


class RepoType(str, Enum):
    online = "online"
    local = "local"


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

        if self.repo_type == RepoType.online and not repo_exists(self.repo_id):
            msg = f"huggingface_repo_id: {self.repo_id} does not exist"
            raise ValueError(msg)

    def __str__(self) -> str:
        return self.repo_id

    def __repr__(self) -> str:
        return self.repo_id

    @cached_property
    def info(self) -> ModelInfo:
        if self.repo_type == RepoType.local:
            msg = "using model info, on local repos is not supported."
            raise ValueError(msg)
        elif self.repo_type == RepoType.online:
            return model_info(self.repo_id, files_metadata=False)
        else:
            msg = f"Unsupported repo type: {self.repo_type}"
            raise ValueError(msg)

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
            fs = HfFileSystem()
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
            msg = f"Unsupported repo type: {self.repo_type}"
            raise ValueError(msg)

        if safetensor_paths:
            weight_files[WeightsFormat.safetensors] = [
                f.replace(f"{self.repo_id}/", "") for f in safetensor_paths
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
            url = hf_hub_url(self.repo_id, filename)
            metadata = get_hf_file_metadata(url)
            return metadata.size
        raise NotImplementedError("not implemented for non-online repos.")

    @cached_property
    def supported_encodings(self) -> list[SupportedEncoding]:
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
                                msg = f"unknown dtype found in safetensors file: {weight_dtype}"
                                logging.warning(msg)

            elif self.repo_type == RepoType.online:
                if safetensors_info := self.info.safetensors:
                    for params in safetensors_info.parameters:
                        if "BF16" in params:
                            supported_encodings.add(SupportedEncoding.bfloat16)
                        elif "F32" in params:
                            supported_encodings.add(SupportedEncoding.float32)
            else:
                msg = f"Unsupported repo_type: {self.repo_type}"
                raise ValueError(msg)

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
                msg = "torch_dtype not available, cant infer encoding from config.json"
                logging.warning(msg)

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
            msg = (
                "cannot infer encoding from .bin files, returning all bin files"
            )
            logging.warning(msg)
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
            logging.warning(
                "Could not find checkpoint with %s encoding, searching for %s files instead.",
                encoding,
                alternate_encoding,
            )
            return self.files_for_encoding(alternate_encoding, weights_format)
        return gguf_files

    def file_exists(self, filename: str) -> bool:
        return file_exists(self.repo_id, filename)

    def download(self, filename: str, force_download: bool = False) -> Path:
        return Path(
            hf_hub_download(
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

            msg = f"gguf file, but encoding not found in file name: {file}"
            raise ValueError(msg)
        elif str(file).endswith(".bin"):
            # If this file is pytorch, return the first encoding, as Pytorch repos only likely have one.
            return self.supported_encodings[0]
        else:
            msg = f"weight path: {file} not gguf or safetensors, cannot infer encoding from file."
            raise ValueError(msg)


@dataclass(frozen=False)
class PipelineConfig:
    # When adding a new config parameter here, please remember to add a
    # description to the `help()` method below

    huggingface_repo_id: str
    """repo_id of a huggingface model repository to use."""

    engine: Optional[PipelineEngine] = None
    """Engine backend to use for serving, 'max' for the max engine, or 'huggingface' as fallback option for improved model coverage."""

    architecture: Optional[str] = None
    """Model architecture to run."""

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
    """The number of tokens in a single page in the paged KVCache."""

    enable_prefix_caching: bool = False
    """Whether to enable prefix caching for the paged attention KVCache."""

    gpu_memory_utilization: float = 0.9
    """The fraction of available device memory that the process should consume.

    This is used to inform the size of the KVCache workspace:
        kv_cache_workspace = (total_free_memory * gpu_memory_utilization) - model_weights_size
    """

    top_k: Optional[int] = None
    """Limits the sampling to the K most probable tokens. If None, will default to greedy sampling."""

    trust_remote_code: bool = False
    """Whether or not to allow for custom modelling files on Huggingface."""

    force_download: bool = False
    """Whether to force download a given file if it’s not already present in the local cache."""

    enable_echo: bool = False
    """Whether the model should be built with echo capabilities."""

    _huggingface_config: Optional[AutoConfig] = None
    """The HuggingFace config associated with the huggingface repo id."""

    _devices: list[Device] = field(default_factory=list)
    """The underlying initialized devices, created by the specific device_specs."""

    _weights_converter: Optional[type[WeightsConverter]] = None
    """Weight converter for the provided `weight_path`."""

    _weights_repo_id: Optional[str] = None
    """Huggingface Repo id to load weights from only. This should only be set by internal code."""

    def __post_init__(self) -> None:
        if not self.huggingface_repo_id:
            msg = "huggingface_repo_id must be provided and must be a valid HuggingFace repo or local directory"
            raise ValueError(msg)

        if (not os.path.exists(self.huggingface_repo_id)) and (
            not repo_exists(self.huggingface_repo_id)
        ):
            msg = f"{self.huggingface_repo_id} is not a valid HuggingFace repo, or local directory"
            raise ValueError(msg)

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
            if isinstance(path, str):
                path = Path(path)

            if not isinstance(path, Path):
                msg = (
                    "weight_path provided must either be string or Path:"
                    f" '{path}'"
                )
                raise ValueError(msg)

            # If we already exist on the OS. Dont parse the path, just continue.
            if path.is_file():
                weight_paths.append(path)
                continue

            # If the path, looks like it may start with a HuggingFace repo id,
            # check if the repo_id is the same as the one provided.
            # If it is the same, set the weight_path to just be the file_name post repo_id
            # If it is different, set the _weights_repo_id to be that repo_id
            # and set the path to be the file_name without the repo_id.
            if path_pieces := str(path).split("/"):
                if len(path_pieces) >= 3:
                    repo_id = f"{path_pieces[0]}/{path_pieces[1]}"
                    file_name = "/".join(path_pieces[2:])
                    if repo_id == self.huggingface_repo_id:
                        path = Path(file_name)
                    elif file_exists(repo_id, file_name):
                        self._weights_repo_id = repo_id
                        path = Path(file_name)

            weight_paths.append(path)

        self.weight_path = weight_paths

    def __getstate__(self) -> dict[str, Any]:
        """Override `__getstate__` to exclude the HuggingFace config."""
        state = self.__dict__.copy()
        state.pop("_huggingface_config")
        return state

    def update_architecture(self) -> None:
        if self.architecture is None:
            # Retrieve architecture from huggingface_repo_id.
            # This is done without using the huggingface config, to reduce the
            # memory stored in this object, before it reaches the model worker.
            hf_config = AutoConfig.from_pretrained(
                self.huggingface_repo_id,
                trust_remote_code=self.trust_remote_code,
            )

            # If we cannot get an architecture from the huggingface_repo_id,
            # we cannot map the model to an internal architecture, and cannot
            # be run using the MAX engine.

            architectures = getattr(hf_config, "architectures", [])
            if len(architectures) > 1:
                msg = (
                    "more than one architecture listed in HuggingFace config,"
                    " using the first one."
                )
                logging.warning(msg)

            if architectures:
                self.architecture = architectures[0]
            else:
                msg = "architectures not listed in HuggingFace config, trying with general `huggingface` engine"
                logging.warning(msg)

                self.engine = PipelineEngine.HUGGINGFACE

    @property
    def huggingface_config(self) -> AutoConfig:
        """Given the huggingface_repo_id, return the HuggingFace Config."""

        if self._huggingface_config is None:
            # Lazy initialize the HuggingFace config field.
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
                else Accelerator(device_spec.id)
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
        elif all(
            [weight_path.suffix == ".bin" for weight_path in self.weight_path]
        ):
            return WeightsFormat.pytorch
        else:
            msg = f"weights type cannot be inferred from {self.weight_path}"
            raise ValueError(msg)

    def weights_size(self) -> Optional[int]:
        size = 0
        hf_repo = HuggingFaceRepo(
            self._weights_repo_id
            if self._weights_repo_id
            else self.huggingface_repo_id,
            trust_remote_code=self.trust_remote_code,
        )
        for file_path in self.weight_path:
            if os.path.exists(file_path):
                size += os.path.getsize(file_path)
                continue

            next_size = hf_repo.size_of(str(file_path))

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
        weights_repo_id = (
            self._weights_repo_id
            if self._weights_repo_id
            else self.huggingface_repo_id
        )
        logger.info(f"Starting download of model: {weights_repo_id}")
        for i, filename in enumerate(self.weight_path):
            self.weight_path[i] = Path(
                hf_hub_download(
                    weights_repo_id,
                    str(filename),
                    force_download=self.force_download,
                )
            )

        logger.info(
            f"Finished download of model: {weights_repo_id} in {(datetime.datetime.now() - start_time).total_seconds()} seconds."
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
            "architecture": "Deprecated - Please set `huggingface-repo-id` instead. Define the model architecture to run. This should match one of the supported architectures for your selected engine.",
            "weight_path": "Provide an optional local path or path relative to the root of a Huggingface repo to the model weights you want to use. This allows you to specify custom weights instead of using defaults. You may pass multiple, ie. `--weight-path=model-00001-of-00002.safetensors --weight-path=model-00002-of-00002.safetensors`",
            "device_specs": "Devices to run inference upon. Default is set to CPU.",
            "quantization_encoding": "Define the weight encoding type for quantization. This can help optimize performance and memory usage during inference. ie. q4_k, bfloat16 etc.",
            "serialized_model_path": "If specified, this flag attempts to load a serialized MEF model from the given path. This is useful for reusing previously saved models.",
            "save_to_serialized_model_path": "If specified, this flag attempts to save the current model state to a serialized format at the given path for later use.",
            "max_length": "Set the maximum sequence length for input data processed by the model. The default is 512 tokens. This can be raised up until the maximum of the model provided. As this value increases memory consumed by the engine increases accordingly.",
            "max_new_tokens": "Specify the maximum number of new tokens to generate during a single inference pass of the model. Default is -1, which means the model will generate until the maximum sequence length is hit, or and eos token is generated.",
            "max_cache_batch_size": "Define the maximum cache size reserved for a single batch. This value defaults to 1. Increase this value based on server capacity when deploying in production.",
            "max_ce_batch_size": "Set the maximum cache size reserved for a single context encoding batch. The effective limit will be the lesser of this value and max-cache-batch-size. Default is 32.",
            "cache_strategy": "Force a specific cache strategy: 'naive' or 'continuous'. If not provided, the optimal caching strategy for the model requested will be selected.",
            "max_num_steps": "Specify the number of steps to run for multi-step scheduling during inference. Default is set to 1.",
            "pad_to_multiple_of": "Pad input tensors to be a multiple of value provided. Default is set to 2.",
            "kv_cache_page_size": "The number of tokens in a single page in the paged KVCache. Default is set to 512.",
            "enable_prefix_caching": "Whether to enable prefix caching for the paged attention KVCache. This defaults to false.",
            "gpu_memory_utilization": "The fraction of available device memory that the process should consume. This is used to inform the size of the KVCache workspace: kv_cache_workspace = (total_free_memory * gpu_memory_utilization) - model_weights_size. Default is set to 0.9.",
            "top_k": "Limit sampling to the top K most probable tokens during generation. This can help control randomness and improve output quality. This defaults to 0, which defaults to greedy sampling.",
            "trust_remote_code": "Indicate whether to allow custom modelling files from Huggingface repositories. Set this to true with caution, as it may introduce security risks.",
            "force_download": "Specify whether to forcefully download a file even if it already exists in local cache. Set this to true if you want to ensure you have the latest version.",
            "enable_echo": "Whether the model should be built with echo capabilities. This defaults to false.",
        }

    def huggingface_weights_repo(self) -> HuggingFaceRepo:
        return HuggingFaceRepo(
            self._weights_repo_id
            if self._weights_repo_id
            else self.huggingface_repo_id,
            trust_remote_code=self.trust_remote_code,
        )
