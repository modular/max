# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for interacting with HuggingFace Files/Repos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import (
    file_exists,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
)


@dataclass(frozen=True)
class HuggingFaceFile:
    """A simple object for tracking huggingface model metadata.
    The repo_id will frequently be used to load a tokenizer,
    whereas the filename is used to download model weights."""

    repo_id: str
    filename: str

    def download(self, force_download: bool = False) -> Path:
        """Download the file and return the file path where the data is saved locally."""
        return Path(
            hf_hub_download(
                self.repo_id, self.filename, force_download=force_download
            )
        )

    def size(self) -> int | None:
        url = hf_hub_url(self.repo_id, self.filename)
        metadata = get_hf_file_metadata(url)
        return metadata.size

    def exists(self) -> bool:
        return file_exists(self.repo_id, self.filename)
