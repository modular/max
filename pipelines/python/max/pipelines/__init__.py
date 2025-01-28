# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Types to interface with ML pipelines such as text/token generation."""

from typing import Callable as _Callable
from typing import Union as _Union

from .config import (
    PipelineConfig,
    PipelineEngine,
    RopeType,
    SupportedEncoding,
    WeightsFormat,
)
from .context import InputContext, TextAndVisionContext, TextContext
from .embeddings_pipeline import EmbeddingsPipeline
from .hf_utils import HuggingFaceFile
from .interfaces import (
    PipelineTask,
    PipelineTokenizer,
    TokenGenerator,
    TokenGeneratorContext,
    TokenGeneratorRequest,
    TokenGeneratorRequestFunction,
    TokenGeneratorRequestMessage,
    TokenGeneratorRequestTool,
)
from .pipeline import (
    ModelInputs,
    ModelOutputs,
    PipelineModel,
    TextGenerationPipeline,
    upper_bounded_default,
)
from .registry import PIPELINE_REGISTRY, SupportedArchitecture
from .response import EmbeddingsResponse, LogProbabilities, TextResponse
from .tokenizer import (
    IdentityPipelineTokenizer,
    PreTrainedPipelineTokenizer,
    TextAndVisionTokenizer,
    TextTokenizer,
)

PipelinesFactory = _Callable[[], _Union[TokenGenerator, EmbeddingsPipeline]]


__all__ = [
    "HuggingFaceFile",
    "PipelineConfig",
    "PipelineEngine",
    "PipelineTask",
    "HuggingFaceFile",
    "PIPELINE_REGISTRY",
    "SupportedArchitecture",
    "SupportedEncoding",
    "TokenGenerator",
    "TokenGeneratorContext",
    "TokenGeneratorRequest",
    "TokenGeneratorRequestMessage",
    "TokenGeneratorRequestTool",
    "TokenGeneratorRequestFunction",
    "IdentityPipelineTokenizer",
    "InputContext",
    "TextContext",
    "TextAndVisionContext",
    "PipelineTokenizer",
    "PreTrainedPipelineTokenizer",
    "SupportedEncoding",
    "TextTokenizer",
    "TextAndVisionTokenizer",
    "TextGenerationPipeline",
    "WeightsFormat",
    "RopeType",
    "PipelineModel",
    "ModelInputs",
    "ModelOutputs",
    "TextResponse",
    "LogProbabilities",
    "EmbeddingsPipeline",
    "EmbeddingsResponse",
    "upper_bounded_default",
]
