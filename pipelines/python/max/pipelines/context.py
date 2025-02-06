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

"""Standardized context object for Pipeline Inference."""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, Union, runtime_checkable

import numpy as np

CHUNK_SIZE = 128


@runtime_checkable
class InputContext(Protocol):
    """A base class for model contexts, represent model inputs for TokenGenerators."""

    @property
    def cache_seq_id(self) -> int: ...

    @property
    def current_length(self) -> int:
        """The current length of the sequence, including completed and active tokens."""
        ...

    @property
    def max_length(self) -> int | None:
        """The maximum length of this sequence."""
        ...

    @property
    def log_probabilities(self) -> int:
        """When > 0, returns the log probabilities for the top N tokens for each
        element token in the sequence."""
        ...

    @property
    def log_probabilities_echo(self) -> bool:
        """When True, the input tokens are added to the returned logprobs."""
        ...

    @property
    def seq_len(self) -> int:
        """Current sequence length: num tokens input this iteration.

        This will be the prompt size for context encoding, and simply 1 for
        token generation.
        """
        ...

    @property
    def next_tokens(self) -> np.ndarray:
        """The next prompt tokens to be input during this iteration.

        This should be a 1D array of tokens of length seq_len.
        """
        ...

    def update(
        self,
        new_token: int,
    ) -> None:
        """Updates the next_tokens and extends existing tokens to include all generated tokens."""
        ...

    def trim_prompt(self, trim_len: int) -> None:
        """Trims the current prompt by the given number of tokens."""
        ...

    @property
    def matcher(self) -> Optional["xgr.GrammarMatcher"]:  # type: ignore
        """An optional xgr Grammar Matcher provided when using structured output."""
        ...

    @property
    def json_schema(self) -> str | None:
        """A json schema to use during constrained decoding."""
        ...

    def set_matcher(self, matcher: "xgr.GrammarMatcher") -> None:  # type: ignore
        """Set a grammar matcher for use during constrained decoding."""
        ...

    def reset(self) -> None:
        """Resets the context's state by combining all tokens into a new prompt.
        This method is used when a request is evicted, meaning that the context
        needed to be re-encoded in the following CE iteration."""
        ...


class TextContext:
    """A base class for model context, specifically for Text model variants."""

    def __init__(
        self,
        cache_seq_id: int,
        prompt: Union[str, Sequence[int]],
        max_length: int | None,
        tokens: np.ndarray,
        log_probabilities: int = 0,
        log_probabilities_echo: bool = False,
        json_schema: str | None = None,
    ) -> None:
        self.cache_seq_id = cache_seq_id
        self.prompt = prompt
        self.max_length = max_length

        if tokens.ndim != 1:
            msg = f"tokens must be one dimensional array: got shape '{tokens.shape}'"
            raise ValueError(msg)

        self.size = int(np.ceil(len(tokens) / CHUNK_SIZE) * CHUNK_SIZE)

        # Create a fresh array since the input tokens may be a view or share memory with
        # another array in the caller, which prevents us from resizing it directly.
        # The extra space is initialized to zero and will be filled with generated tokens.
        assert len(tokens) <= self.size
        self.tokens = np.zeros(self.size, dtype=tokens.dtype)
        self.tokens[: len(tokens)] = tokens

        self.current_length = len(tokens)
        self.active_length = self.current_length
        self.active_idx = self.current_length
        self.start_idx = 0

        self.log_probabilities = log_probabilities
        self.log_probabilities_echo = log_probabilities_echo

        self.matcher = None
        self.json_schema = json_schema

    def set_matcher(self, matcher: "xgr.GrammarMatcher") -> None:  # type: ignore
        self.matcher = matcher

    @property
    def seq_len(self) -> int:
        """Current sequence length: num tokens input this iteration.

        This will be the prompt size for context encoding, and simply 1 for
        token generation.
        """
        return self.active_length

    @property
    def next_tokens(self) -> np.ndarray:
        return self.tokens[self.start_idx : self.active_idx]

    def update(
        self,
        new_token: int,
    ) -> None:
        """Updates the next_tokens and extends existing tokens to include all generated tokens."""

        if self.active_idx >= self.size:
            self.size += CHUNK_SIZE
            if self.tokens.flags.owndata:
                self.tokens.resize(self.size)
            else:
                self.tokens = np.resize(self.tokens, self.size)

        self.tokens[self.active_idx] = new_token
        self.start_idx = self.active_idx
        self.active_idx += 1
        self.current_length += 1
        self.active_length = 1

        # Accept the token, and move the FSM for constrained decoding forward.
        if self.matcher:
            assert self.matcher.accept_token(new_token)

    def trim_prompt(self, trim_len: int) -> None:
        """Trims the current prompt by the given number of tokens."""
        if trim_len == 0:
            return

        assert trim_len < (self.active_idx - self.start_idx)
        self.start_idx += trim_len
        self.active_length = self.active_idx - self.start_idx

    def reset(self) -> None:
        """Resets the context's state by combining all tokens into a new prompt."""
        tokens_in_new_prompt = self.active_idx
        self.start_idx = 0
        self.active_idx = tokens_in_new_prompt
        self.current_length = tokens_in_new_prompt
        self.active_length = tokens_in_new_prompt


class TextAndVisionContext:
    """A base class for model context, specifically for Vision model variants."""

    def __init__(
        self,
        cache_seq_id: int,
        prompt: Union[str, Sequence[int]],
        max_length: int | None,
        tokens: np.ndarray,
        pixel_values: Union[np.ndarray, list[np.ndarray]],
        extra_model_args: dict[str, Any],
        log_probabilities: int = 0,
        log_probabilities_echo: bool = False,
        json_schema: str | None = None,
    ) -> None:
        self.cache_seq_id = cache_seq_id
        self.prompt = prompt
        self.max_length = max_length

        if tokens.ndim != 1:
            msg = f"tokens must be one dimensional array: got shape '{tokens.shape}'"
            raise ValueError(msg)

        self.size = int(np.ceil(len(tokens) / CHUNK_SIZE) * CHUNK_SIZE)

        # Create a fresh array since the input tokens may be a view or share memory with
        # another array in the caller, which prevents us from resizing it directly.
        # The extra space is initialized to zero and will be filled with generated tokens.
        assert len(tokens) <= self.size
        self.tokens = np.zeros(self.size, dtype=tokens.dtype)
        self.tokens[: len(tokens)] = tokens

        self.current_length = len(tokens)
        self.active_length = self.current_length
        self.active_idx = self.current_length
        self.start_idx = 0

        self.pixel_values = pixel_values
        self.extra_model_args = extra_model_args

        self.log_probabilities = log_probabilities
        self.log_probabilities_echo = log_probabilities_echo

        self.matcher = None
        self.json_schema = json_schema

    def set_matcher(self, matcher: "xgr.GrammarMatcher") -> None:  # type: ignore
        self.matcher = matcher

    @property
    def next_tokens(self) -> np.ndarray:
        return self.tokens[self.start_idx : self.active_idx]

    @property
    def seq_len(self) -> int:
        """Current sequence length: num tokens input this iteration.

        This will be the prompt size for context encoding, and simply 1 for
        token generation.
        """
        return self.active_length

    def update(
        self,
        new_token: int,
    ) -> None:
        """Updates the next_tokens attribute, and extends current_length if needed, based on the provided num_steps."""
        if self.active_idx >= self.size:
            self.size += CHUNK_SIZE
            if self.tokens.flags.owndata:
                self.tokens.resize(self.size)
            else:
                self.tokens = np.resize(self.tokens, self.size)

        self.tokens[self.active_idx] = new_token
        self.start_idx = self.active_idx
        self.active_idx += 1
        self.current_length += 1
        self.active_length = 1

        # Update context not to re-encode the same image in next steps. There are no image tokens
        # expected after context encoding.
        self.pixel_values = []

        # Accept the token, and move the FSM for constrained decoding forward.
        if self.matcher:
            assert self.matcher.accept_token(new_token)

    def trim_prompt(self, trim_len: int) -> None:
        """Trims the current prompt by the given number of tokens."""
        if trim_len == 0:
            return

        assert trim_len < (self.active_idx - self.start_idx)
        self.start_idx += trim_len
        self.active_length = self.active_idx - self.start_idx

    def reset(self) -> None:
        """Resets the context's state by combining all tokens into a new prompt."""
        tokens_in_new_prompt = self.active_idx
        self.start_idx = 0
        self.active_idx = tokens_in_new_prompt
        self.current_length = tokens_in_new_prompt
        self.active_length = tokens_in_new_prompt
