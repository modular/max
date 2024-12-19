# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Model registry, for tracking various model variants."""

from __future__ import annotations

import functools
import logging
import os
from typing import Callable, Optional, Type, Union, cast

import torch
from max.graph.weights import WeightsConverter

from .config import (
    PipelineConfig,
    PipelineEngine,
    SupportedEncoding,
    WeightsFormat,
)
from .hf_pipeline import (
    HFTextGenerationPipeline,
)
from .interfaces import PipelineTokenizer, TokenGenerator
from .kv_cache import KVCacheStrategy
from .pipeline import (
    PipelineModel,
    TextGenerationPipeline,
)
from .tokenizer import TextAndVisionTokenizer, TextTokenizer


class SupportedArchitecture:
    def __init__(
        self,
        name: str,
        example_repo_ids: list[str],
        default_encoding: SupportedEncoding,
        supported_encodings: dict[SupportedEncoding, list[KVCacheStrategy]],
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
            example_repo_ids: HuggingFace repo_id which runs this architecture.
            default_encoding: Default encoding for the model.
            supported_encodings: Alternate encodings supported.
            pipeline_model: PipelineModel class that defines the model graph
                and execution.
            tokenizer: Tokenizer used to preprocess model inputs.
            default_weights_format: The weights format used in `pipeline_model`.
            weight_converters: A dictionary of weight loaders to use if the
                input checkpoint has a different format than the default.
        """
        self.name = name
        self.example_repo_ids = example_repo_ids
        self.default_encoding = default_encoding
        self.supported_encodings = supported_encodings
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

    def architecture_details(
        self, pipeline_config: PipelineConfig
    ) -> Optional[SupportedArchitecture]:
        """Return architecture details for pipeline_config if available, None if not found."""

        # If no architecture is provided in the pipeline_config, we have nothing to retrieve.
        if not pipeline_config.architecture:
            return None

        # If the engine is not provided or MAX, we should retrieve the architecture and validate it.
        if (
            not pipeline_config.engine
            or pipeline_config.engine == PipelineEngine.MAX
        ):
            if pipeline_config.architecture in self.architectures:
                msg = (
                    "optimized architecture found for"
                    f" '{pipeline_config.architecture}' validating architecture for MAX engine."
                )
                logging.info(msg)
                return self.architectures[pipeline_config.architecture]
            else:
                return None
        else:
            return None

    def validate_pipeline_config(
        self, pipeline_config: PipelineConfig
    ) -> PipelineConfig:
        """Update pipeline config with appropriate values if not provided.
        If invalid config is provided, error out with detailed reason."""

        # This will update the architecture, and engine if no architecture is available.
        pipeline_config.update_architecture()

        # This will retrieve the architecture, if we support it.
        arch = self.architecture_details(pipeline_config)

        # If nothing is provided, we should not update any more params.
        # Instead, fall back to the HuggingFace engine.
        if not arch and pipeline_config.engine == PipelineEngine.MAX:
            msg = (
                "optimized architecture not available for"
                f" '{pipeline_config.architecture}', failing as engine is provided as 'MAX'"
            )
            raise ValueError(msg)

        elif not arch:
            msg = (
                "optimized architecture not available for"
                f" '{pipeline_config.architecture}' falling back to"
                " HuggingFace."
            )
            logging.info(msg)
            pipeline_config.engine = PipelineEngine.HUGGINGFACE
            return pipeline_config

        # TODO(KERN-1104) remove this constraint after we support passing context length to MHA kernel.
        if pipeline_config.max_length > 16384:
            msg = f"MAX engine currently has a max_length of 16384, got {pipeline_config.max_length}"
            raise ValueError(msg)

        # The remainder of this function, assumes we have both a valid huggingface_repo_id,
        # and a SupportedArchitecture. We should then validate the details of the existing architecture
        # and fallback to HuggingFace if needed.

        # If weight_path and quantization_encoding are provided, verify that they are consistent.
        if (
            pipeline_config.weight_path
            and pipeline_config.quantization_encoding
            # Cannot validate quantization_encoding for pytorch.
            and pipeline_config.weights_format != WeightsFormat.pytorch
        ):
            # Get the encoding of the first weight path file.
            if os.path.exists(pipeline_config.weight_path[0]):
                file_encoding = SupportedEncoding.parse_from_file_name(
                    str(pipeline_config.weight_path[0])
                )
            else:
                file_encoding = (
                    pipeline_config.huggingface_repo.encoding_for_file(
                        pipeline_config.weight_path[0]
                    )
                )

            if file_encoding:
                if file_encoding != pipeline_config.quantization_encoding:
                    msg = f"weight_path provided '{pipeline_config.weight_path[0]}' has an inconsistent encoding '{file_encoding}' than quantization_encoding provided '{pipeline_config.quantization_encoding}'. Please update one."
                    raise ValueError(msg)
        # If weight path is not None, infer the quantization_encoding from the weight_path.
        elif (
            pipeline_config.weight_path
            and not pipeline_config.quantization_encoding
            and pipeline_config.weights_format != WeightsFormat.pytorch
        ):
            if os.path.exists(pipeline_config.weight_path[0]):
                # Not currently supported. Infer encoding from local path.
                if pipeline_config.weight_path[0].suffix == ".safetensors":
                    msg = "If a local safetensors file is provided, please provide a quantization_encoding."
                    raise ValueError(msg)

                if encoding := SupportedEncoding.parse_from_file_name(
                    str(pipeline_config.weight_path[0])
                ):
                    msg = f"encoding inferred from weights file: {encoding}"
                    logging.info(msg)
                    pipeline_config.quantization_encoding = encoding

            else:
                if (
                    encoding
                    := pipeline_config.huggingface_repo.encoding_for_file(
                        pipeline_config.weight_path[0]
                    )
                ):
                    msg = f"encoding inferred from weights file: {encoding}"
                    logging.info(msg)
                    pipeline_config.quantization_encoding = encoding
                else:
                    msg = f"encoding cannot be inferred from weights file: {pipeline_config.weight_path[0]}, please pass a quantization_encoding explictly."
                    raise ValueError(msg)
        elif not pipeline_config.quantization_encoding:
            msg = f"encoding not provided, using default encoding of {arch.default_encoding}"
            logging.info(msg)
            pipeline_config.quantization_encoding = arch.default_encoding

        # by this point, the quantization_encoding must be provided. verify it is supported.
        if (
            pipeline_config.quantization_encoding
            not in arch.supported_encodings
        ):
            if pipeline_config.engine == PipelineEngine.MAX:
                msg = f"quantization_encoding of '{pipeline_config.quantization_encoding}' not supported by MAX engine, unable to run with engine = 'max'."
                raise ValueError(msg)

            else:
                msg = f"quantization_encoding of '{pipeline_config.quantization_encoding}' not supported by MAX engine, falling back to HuggingFace."
                pipeline_config.engine = PipelineEngine.HUGGINGFACE
                return pipeline_config

        # We should now have a valid quantization_encoding, and possibly a weight_path.
        # If no weight_path is provided, we should grab the default.
        if not pipeline_config.weight_path:
            # Retrieve the default files for each weights format.
            weight_files = pipeline_config.huggingface_repo.files_for_encoding(
                encoding=pipeline_config.quantization_encoding
            )

            if default_weight_files := weight_files.get(
                arch.default_weights_format, []
            ):
                pipeline_config.weight_path = default_weight_files
            else:
                for (
                    converter_format,
                    converter,
                ) in arch.weight_converters.items():
                    if converter_format_files := weight_files.get(
                        converter_format, []
                    ):
                        pipeline_config.weight_path = converter_format_files
                        pipeline_config._weights_converter = converter
                        break

        if not pipeline_config.weight_path:
            msg = f"compatabile weights cannot be found for '{pipeline_config.quantization_encoding}' in '{arch.default_weights_format}' format, or any convertible format: '{', '.join([converter for converter in arch.weight_converters.keys()])}'."
            raise ValueError(msg)

        # Check supported_cache_strategy
        supported_cache_strategies = arch.supported_encodings.get(
            pipeline_config.quantization_encoding, []
        )
        if pipeline_config.cache_strategy not in supported_cache_strategies:
            supported_strategy = supported_cache_strategies[0]

            msg = f"cache_strategy, not supported for '{pipeline_config.quantization_encoding}', using '{supported_strategy}' cache strategy."
            logging.info(msg)

            pipeline_config.cache_strategy = supported_strategy

        # Assume at this point, an architecture,
        # a huggingface_repo_id and weight_paths are available.
        assert pipeline_config.weight_path, "weight_path must be provided."
        for path in pipeline_config.weight_path:
            # Check if file exists locally.
            if not os.path.exists(path):
                # If does not exist locally, verify that it exists on Huggingface.
                if not pipeline_config.huggingface_repo.file_exists(str(path)):
                    msg = (
                        f"weight_path: '{path}' does not exist locally, and"
                        f" '{pipeline_config.huggingface_repo_id}/{path}' does"
                        " not exist on HuggingFace."
                    )
                    raise ValueError(msg)

        # If we pass validation ensure, the engine is set as MAX.
        pipeline_config.engine = PipelineEngine.MAX
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
            # Keep MyPy happy.
            assert pipeline_config.architecture is not None

            # MAX pipeline
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
