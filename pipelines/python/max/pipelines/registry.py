# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Model registry, for tracking various model variants."""

from __future__ import annotations

import functools
import json
import logging
import os
from pathlib import Path
from typing import Callable, Type, Union, cast

import torch
from huggingface_hub import HfFileSystem
from max.graph.weights import WeightsConverter
from transformers import AutoConfig
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from .config import (
    PipelineConfig,
    PipelineEngine,
    SupportedEncoding,
    WeightsFormat,
)
from .hf_pipeline import (
    HFTextGenerationPipeline,
)
from .hf_utils import HuggingFaceFile
from .interfaces import PipelineTokenizer, TokenGenerator
from .kv_cache import KVCacheStrategy
from .pipeline import (
    PipelineModel,
    TextGenerationPipeline,
)
from .tokenizer import TextAndVisionTokenizer, TextTokenizer


class SupportedVersion:
    def __init__(
        self,
        name: str,
        encodings: dict[
            SupportedEncoding,
            tuple[list[HuggingFaceFile], list[KVCacheStrategy]],
        ],
        default_encoding: SupportedEncoding,
    ):
        self.name = name
        self.encodings = encodings
        self.default_encoding = default_encoding

    def huggingface_files(
        self, encoding: SupportedEncoding
    ) -> list[HuggingFaceFile]:
        if encoding not in self.encodings:
            raise ValueError(f"No weights file found for encoding '{encoding}'")

        return self.encodings[encoding][0]

    def is_supported_cache_strategy(
        self, encoding: SupportedEncoding, cache_strategy: KVCacheStrategy
    ) -> bool:
        """Identify if an encoding supports a specific cache_strategy."""
        if encoding not in self.encodings:
            raise ValueError(f"encoding '{encoding}' not supported")
        return cache_strategy in self.encodings[encoding][1]

    def default_cache_strategy(
        self,
        encoding: SupportedEncoding,
    ) -> KVCacheStrategy:
        """Get the default cache strategy for an encoding."""
        if encoding not in self.encodings:
            raise ValueError(f"encoding '{encoding}' not supported")

        return self.encodings[encoding][1][0]


class SupportedArchitecture:
    def __init__(
        self,
        name: str,
        default_encoding: SupportedEncoding,
        supported_encodings: dict[SupportedEncoding, list[KVCacheStrategy]],
        versions: list[SupportedVersion],
        default_version: str,
        pipeline_model: Type[PipelineModel],
        tokenizer: Type[Union[TextTokenizer, TextAndVisionTokenizer]],
        default_weights_format: WeightsFormat,
        weight_converters: dict[WeightsFormat, Type[WeightsConverter]]
        | None = None,
    ):
        """Initializes a model architecture supported by MAX pipelines.

        New architectures should be registered into the `PipelineRegistry`.

        args:
            name: Architecture name.
            versions: List of supported versions (each with a list of supported
                encodings).
            default_version: Name of default version to use if no version is
                specified.
            pipeline_model: PipelineModel class that defines the model graph
                and execution.
            tokenizer: Tokenizer used to preprocess model inputs.
            default_weights_format: The weights format used in `pipeline_model`.
            weight_converters: A dictionary of weight loaders to use if the
                input checkpoint has a different format than the default.
        """
        self.name = name
        self.default_encoding = default_encoding
        self.supported_encodings = supported_encodings
        self.versions = {version.name: version for version in versions}

        if default_version not in self.versions:
            raise ValueError(
                f"default version: {default_version} not provided in supported"
                " versions."
            )

        self.default_version = self.versions[default_version]
        self.pipeline_model = pipeline_model
        self.tokenizer = tokenizer
        self.default_weights_format = default_weights_format
        self.weight_converters = weight_converters or {}


class PipelineRegistry:
    def __init__(self, architectures: list[SupportedArchitecture]):
        self.architectures = {arch.name: arch for arch in architectures}

    def register(self, architecture: SupportedArchitecture):
        """Add new architecture to registry."""
        if architecture.name in self.architectures:
            msg = (
                "Refusing to override existing architecture for"
                f" '{architecture.name}'"
            )
            raise ValueError(msg)

        self.architectures[architecture.name] = architecture

    def validate_pipeline_config(
        self, pipeline_config: PipelineConfig
    ) -> PipelineConfig:
        """Update pipeline config with appropriate values if not provided.
        If invalid config is provided, error out with detailed reason."""

        # Validate architecture.
        if (
            pipeline_config.architecture is None
            and pipeline_config.huggingface_repo_id is None
        ):
            msg = "architecture or huggingface_repo_id must be provided."
            raise ValueError(msg)

        # Get architecture from Huggingface Repo id, if not provided.
        if (
            pipeline_config.architecture is None
            and pipeline_config.huggingface_repo_id is not None
        ):
            # Retrieve architecture from huggingface_repo_id.
            hf_config = AutoConfig.from_pretrained(
                pipeline_config.huggingface_repo_id,
                trust_remote_code=pipeline_config.trust_remote_code,
            )

            # If we do not get an architecture from the huggingface_repo_id,
            # we cannot map the model to an internal architecture.
            architectures = getattr(hf_config, "architectures", [])

            if len(architectures) > 1:
                msg = (
                    "more than one architecture listed in HuggingFace config,"
                    " using the first one."
                )
                logging.warning(msg)

            if architectures:
                pipeline_config.architecture = architectures[0]
            else:
                msg = (
                    "architectures not listed in HuggingFace config, unable to"
                    " load model."
                )
                raise ValueError(msg)

        if not pipeline_config.engine:
            if pipeline_config.architecture in self.architectures:
                msg = (
                    "optimized architecture found for"
                    f" '{pipeline_config.architecture}' running MAX engine."
                )
                logging.info(msg)
                pipeline_config.engine = PipelineEngine.MAX
            else:
                msg = (
                    "optimized architecture not available for"
                    f" '{pipeline_config.architecture}' falling back to"
                    " HuggingFace."
                )
                logging.info(msg)
                pipeline_config.engine = PipelineEngine.HUGGINGFACE
                return pipeline_config
        elif pipeline_config.engine != PipelineEngine.MAX:
            return pipeline_config

        # TODO(KERN-1104) remove this constraint after we support passing context length to MHA kernel.
        if pipeline_config.max_length > 16384:
            msg = f"MAX engine currently has a max_length of 16384, got {pipeline_config.max_length}"
            raise ValueError(msg)

        assert pipeline_config.architecture
        if pipeline_config.architecture not in self.architectures:
            msg = (
                f"architecture '{pipeline_config.architecture}' not supported"
                " by the MAX engine."
            )
            raise ValueError(msg)

        arch = self.architectures[pipeline_config.architecture]

        # Check version.
        if pipeline_config.version is None:
            pipeline_config.version = arch.default_version.name
        elif pipeline_config.version not in arch.versions:
            raise ValueError(
                f"version '{pipeline_config.version}' not supported for"
                f" '{pipeline_config.architecture}' architecture"
            )

        version = arch.versions[pipeline_config.version]

        # Check encoding.
        if pipeline_config.quantization_encoding is None:
            pipeline_config.quantization_encoding = version.default_encoding
        elif pipeline_config.quantization_encoding not in version.encodings:
            raise ValueError(
                "version"
                f" '{pipeline_config.architecture}/{pipeline_config.version}'"
                f" does not supported '{pipeline_config.quantization_encoding}'"
            )

        # Ensure quantization_encoding is correct.
        if not version.is_supported_cache_strategy(
            pipeline_config.quantization_encoding,
            pipeline_config.cache_strategy,
        ):
            # This just sets the cache_strategy to the first available.
            pipeline_config.cache_strategy = version.default_cache_strategy(
                pipeline_config.quantization_encoding
            )

        # Assume architecture is valid, thus a default huggingface_repo is available.
        default_files = version.huggingface_files(
            pipeline_config.quantization_encoding
        )
        if pipeline_config.huggingface_repo_id is None:
            # Get the default huggingface_id.
            pipeline_config.huggingface_repo_id = default_files[0].repo_id
        elif (
            pipeline_config.huggingface_repo_id != default_files[0].repo_id
            and not pipeline_config.weight_path
        ):
            # If huggingface repo is provided (and not the modular repo), and
            # weights are not set, then use the weights from the provided repo.
            pipeline_config.weight_path = _weights_from_huggingface(
                pipeline_config.huggingface_repo_id,
                arch.default_weights_format,
                arch.weight_converters,
                arch.name,
            )

        # Assume at this point, an architecture, must be valid.
        # If weight_paths are not available, get the defaults.
        assert (
            pipeline_config.huggingface_repo_id
        ), "huggingface_repo_id must be provided."
        if not pipeline_config.weight_path:
            pipeline_config.weight_path = [
                Path(file.filename) for file in default_files
            ]

        # Assume at this point, an architecture,
        # a huggingface_repo_id and weight_paths are available.
        assert pipeline_config.weight_path, "weight_path must be provided."
        for path in pipeline_config.weight_path:
            # Check if file exists locally.
            if not os.path.exists(path):
                # If does not exist locally, try and retrieve from Huggingface.
                hf_file = HuggingFaceFile(
                    repo_id=pipeline_config.huggingface_repo_id,
                    filename=str(path),
                )

                if not hf_file.exists():
                    msg = (
                        f"weight_path: '{path}' does not exist locally, and"
                        f" '{pipeline_config.huggingface_repo_id}/{path}' does"
                        " not exist on HuggingFace."
                    )
                    raise ValueError(msg)

        # Check if the weight converter should be used.
        if pipeline_config.weights_format != arch.default_weights_format:
            weights_converter = arch.weight_converters.get(
                pipeline_config.weights_format
            )
            if weights_converter is None:
                expected_format = arch.default_weights_format.name
                actual_format = pipeline_config.weights_format.name
                raise ValueError(
                    f"{pipeline_config.short_name} expects {expected_format}, "
                    f"but the weights file is in the {actual_format} format "
                    f"({pipeline_config.weight_path}). Support for converting "
                    "between these has not been added."
                )
            pipeline_config._weights_converter = weights_converter
        return pipeline_config

    def _load_logging_message(
        self,
        pipeline_config: PipelineConfig,
        tokenizer_type: Type[PipelineTokenizer],
        pipeline_name: str,
        pipeline_model: str,
        factory: bool,
    ):
        weight_path = ",\n        ".join(
            [
                f"                               {path}"
                for path in pipeline_config.weight_path
            ]
        )
        factory_str = "factory" if factory else ""
        message = f"""

        Loading {tokenizer_type.__name__} and {pipeline_name}({pipeline_model}) {factory_str} for:
            engine:                 {pipeline_config.engine}
            architecture:           {pipeline_config.architecture}
            huggingface_repo_id:    {pipeline_config.huggingface_repo_id}
            quantization_encoding:  {pipeline_config.quantization_encoding}
            weight_path:            [
        {weight_path}
                                    ]
        """

        return message

    def retrieve_factory(
        self, pipeline_config: PipelineConfig
    ) -> tuple[
        PipelineTokenizer,
        Callable[[], TokenGenerator],
    ]:
        tokenizer: PipelineTokenizer
        pipeline_factory: Callable[[], TokenGenerator]

        # Validate pipeline_config, and update missing values.
        pipeline_config = self.validate_pipeline_config(pipeline_config)

        if pipeline_config.engine == PipelineEngine.MAX:
            # MAX pipeline
            pipeline_config = self.validate_pipeline_config(pipeline_config)
            if pipeline_config.architecture is None:
                msg = "architecture must be provided to load pipeline."
                raise ValueError(msg)

            arch = self.architectures[pipeline_config.architecture]
            logging.info(
                self._load_logging_message(
                    pipeline_config=pipeline_config,
                    tokenizer_type=arch.tokenizer,
                    pipeline_model=arch.pipeline_model.__name__,
                    pipeline_name="TextGenerationPipeline",
                    factory=True,
                )
            )

            # Old Mistral model like Mistral-7B-Instruct-v0.3 uses LlamaTokenizer
            # and suffers from the whitespace decoding bug. So, we enable the fix
            # for only MistralModel in order to avoid any issues with performance
            # for rest of the models. This can be applied more generically once
            # we have more time verifying this for all the models.
            # More information:
            # https://linear.app/modularml/issue/AIPIPE-197/add-support-for-mistral-7b-instruct-v03
            # TODO: remove this pipeline_model.__name__ check
            if (
                arch.pipeline_model.__name__ == "MistralModel"
                and arch.tokenizer is TextTokenizer
            ):
                text_tokenizer = cast(Type[TextTokenizer], arch.tokenizer)
                tokenizer = text_tokenizer(
                    pipeline_config, enable_llama_whitespace_fix=True
                )
            else:
                tokenizer = arch.tokenizer(pipeline_config)

            pipeline_factory = functools.partial(
                TextGenerationPipeline,
                pipeline_config=pipeline_config,
                pipeline_model=arch.pipeline_model,
                eos_token_id=tokenizer.eos,
            )
        else:
            torch_device_type = str(pipeline_config.device_specs[0].device_type)
            if pipeline_config.device_specs[0].device_type == "gpu":
                torch_device_type = "cuda"
                torch.multiprocessing.set_start_method("spawn", force=True)

            # Generalized pipeline
            tokenizer = TextTokenizer(
                config=pipeline_config, enable_llama_whitespace_fix=True
            )
            logging.info(
                self._load_logging_message(
                    pipeline_config=pipeline_config,
                    tokenizer_type=TextTokenizer,
                    pipeline_model="",
                    pipeline_name="HFTextGenerationPipeline",
                    factory=True,
                )
            )
            pipeline_factory = functools.partial(
                HFTextGenerationPipeline,
                pipeline_config=pipeline_config,
                torch_device_type=torch_device_type,
            )

        if tokenizer.eos is None:
            msg = (
                "tokenizer.eos value is None, tokenizer configuration is"
                " incomplete."
            )
            raise ValueError(msg)

        return tokenizer, pipeline_factory

    def retrieve(
        self,
        pipeline_config: PipelineConfig,
    ) -> tuple[PipelineTokenizer, TokenGenerator]:
        tokenizer, pipeline_factory = self.retrieve_factory(pipeline_config)
        return tokenizer, pipeline_factory()

    def reset(self) -> None:
        self.architectures.clear()


PIPELINE_REGISTRY = PipelineRegistry([])


def _weights_from_huggingface(
    repo_id: str,
    default_weights_format: WeightsFormat,
    weight_converters: dict[WeightsFormat, Type[WeightsConverter]],
    arch_name: str,
) -> list[Path]:
    fs = HfFileSystem()
    ggufs = cast(list[str], fs.glob(f"{repo_id}/*.gguf"))
    safetensors = cast(list[str], fs.glob(f"{repo_id}/model*.safetensors*"))
    pytorch_bin = cast(list[str], fs.glob(f"{repo_id}/*.bin"))

    matching_exts = bool(ggufs) + bool(safetensors) + bool(pytorch_bin)
    if not matching_exts:
        raise ValueError(
            f"Checkpoint could not be found in HuggingFace repo {repo_id}"
        )
    elif matching_exts > 1:
        raise ValueError(
            f"Found multiple checkpoint types in HuggingFace repo {repo_id}"
        )

    # First check to see if there are weights available in the default format
    # expected by the model architecture.
    if default_weights_format == WeightsFormat.gguf and ggufs:
        return _load_gguf(repo_id, ggufs)
    elif default_weights_format == WeightsFormat.safetensors and safetensors:
        return _load_safetensors(repo_id, safetensors)
    elif default_weights_format == WeightsFormat.pytorch and pytorch_bin:
        return _load_pytorch(repo_id, pytorch_bin)

    # If the repo doesn't contain the weights with default format, look for
    # any available weights and check if there is a corresponding converter.
    if ggufs and WeightsFormat.gguf in weight_converters:
        return _load_gguf(repo_id, ggufs)
    elif safetensors and WeightsFormat.safetensors in weight_converters:
        return _load_safetensors(repo_id, safetensors)
    elif pytorch_bin and WeightsFormat.pytorch in weight_converters:
        return _load_pytorch(repo_id, pytorch_bin)

    # No weights were found, raise an error.
    expected_format = default_weights_format.name
    available_formats = []
    if ggufs:
        available_formats.append("gguf")
    if safetensors:
        available_formats.append("safetensors")
    if pytorch_bin:
        available_formats.append("pytorch")

    if available_formats:
        available = ",".join(available_formats)
        raise ValueError(
            f"{arch_name} expects weights in the {expected_format} format, but "
            f"only found {available}. Support for converting between these to "
            f"{expected_format} has not been added."
        )
    else:
        raise ValueError(
            f"Could not find checkpoint files from HuggingFace repo {repo_id}."
        )


def _load_gguf(repo_id: str, files: list[str]) -> list[Path]:
    trim_prefix = len(repo_id) + 1  # Trim "{user}/{repo_name}/" from the path.
    if len(files) > 1:
        raise ValueError(f"Found multiple ggufs in HuggingFace repo {repo_id}")
    return [Path(files[0][trim_prefix:])]


def _load_safetensors(repo_id: str, files: list[str]) -> list[Path]:
    trim_prefix = len(repo_id) + 1  # Trim "{user}/{repo_name}/" from the path.
    trimmed_files = [f[trim_prefix:] for f in files]
    sharded = SAFE_WEIGHTS_INDEX_NAME in trimmed_files
    if sharded:
        index_file = HuggingFaceFile(
            repo_id, SAFE_WEIGHTS_INDEX_NAME
        ).download()
        return [Path(path) for path in _safetensor_paths_from_index(index_file)]
    elif SAFE_WEIGHTS_NAME in trimmed_files:
        return [Path(SAFE_WEIGHTS_NAME)]
    raise ValueError(
        f"Could not find {SAFE_WEIGHTS_INDEX_NAME} or {SAFE_WEIGHTS_NAME} "
        f"in HuggingFace repo {repo_id}"
    )


def _load_pytorch(repo_id: str, files: list[str]) -> list[Path]:
    trim_prefix = len(repo_id) + 1  # Trim "{user}/{repo_name}/" from the path.
    if len(files) > 1:
        raise ValueError(
            "Found multiple pytorch checkpoints in HuggingFace repo"
            f" {repo_id}"
        )
    return [Path(repo_id, files[0][trim_prefix:])]


def _safetensor_paths_from_index(path: Path) -> list[str]:
    index = json.loads(path.read_text())
    return list({weight_file for weight_file in index["weight_map"].values()})
