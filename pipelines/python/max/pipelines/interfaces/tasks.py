# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Pipeline Tasks."""

from enum import Enum


class PipelineTask(str, Enum):
    TEXT_GENERATION = "text_generation"
    EMBEDDINGS_GENERATION = "embeddings_generation"
