# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Standardized context object for Pipeline Inference."""

from typing import Any, Protocol, Sequence, Union, runtime_checkable

import numpy as np


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
    def max_length(self) -> int:
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
        num_steps: int = 1,
    ) -> None:
        """Updates the next_tokens and extends existing tokens to include all generated tokens."""
        ...

    def trim_prompt(self, trim_len: int) -> None:
        """Trims the current prompt by the given number of tokens."""
        ...

    # TODO: AIPIPE248 - Remove is_done interface and implementations
    def is_done(self, eos: int) -> bool:
        """Returns true if token gen for this context completed, else false."""
        ...


class TextContext:
    """A base class for model context, specifically for Text model variants."""

    def __init__(
        self,
        cache_seq_id: int,
        prompt: Union[str, Sequence[int]],
        max_length: int,
        next_tokens: np.ndarray,
        log_probabilities: int = 0,
        log_probabilities_echo: bool = False,
    ) -> None:
        self.cache_seq_id = cache_seq_id
        self.prompt = prompt
        self.max_length = max_length

        if next_tokens.ndim != 1:
            msg = f"next_tokens must be one dimensional array: got shape '{next_tokens.shape}'"
            raise ValueError(msg)

        self._next_tokens: Union[np.ndarray, int] = next_tokens
        self.current_length = self._next_tokens.shape[-1]
        self.active_length = self.current_length

        self.log_probabilities = log_probabilities
        self.log_probabilities_echo = log_probabilities_echo

    @property
    def seq_len(self) -> int:
        """Current sequence length: num tokens input this iteration.

        This will be the prompt size for context encoding, and simply 1 for
        token generation.
        """
        return self.active_length

    @property
    def next_tokens(self) -> np.ndarray:
        if isinstance(self._next_tokens, int):
            return np.array([self._next_tokens]).reshape(-1)

        return self._next_tokens

    def update(
        self,
        new_token: int,
        num_steps: int = 1,
    ) -> None:
        """Updates the next_tokens and extends existing tokens to include all generated tokens."""
        self._next_tokens = new_token
        self.current_length += num_steps

        self.active_length = 1

    def trim_prompt(self, trim_len: int) -> None:
        """Trims the current prompt by the given number of tokens."""
        if trim_len == 0:
            return

        assert trim_len < len(self.next_tokens)
        next_tokens = self.next_tokens
        new_prompt = next_tokens[trim_len:]
        self._next_tokens = new_prompt
        self.active_length = len(new_prompt)
        assert self.active_length > 0

    def is_done(self, eos: int) -> bool:
        """Returns true if token gen for this context completed, else false."""
        test_token = (
            self._next_tokens[-1]
            if isinstance(self._next_tokens, np.ndarray)
            else self._next_tokens
        )
        return test_token == eos or self.current_length > self.max_length


class TextAndVisionContext:
    """A base class for model context, specifically for Vision model variants."""

    def __init__(
        self,
        cache_seq_id: int,
        prompt: Union[str, Sequence[int]],
        max_length: int,
        next_tokens: np.ndarray,
        pixel_values: Union[np.ndarray, list[np.ndarray]],
        extra_model_args: dict[str, Any],
        log_probabilities: int = 0,
        log_probabilities_echo: bool = False,
    ) -> None:
        self.cache_seq_id = cache_seq_id
        self.prompt = prompt
        self.max_length = max_length

        self._next_tokens: Union[np.ndarray, int] = next_tokens
        self.pixel_values = pixel_values
        self.current_length = self._next_tokens.shape[-1]
        self.active_length = self.current_length
        self.extra_model_args = extra_model_args

        self.log_probabilities = log_probabilities
        self.log_probabilities_echo = log_probabilities_echo

    @property
    def next_tokens(self) -> np.ndarray:
        if isinstance(self._next_tokens, int):
            return np.array([self._next_tokens])

        return self._next_tokens

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
        num_steps: int = 1,
    ) -> None:
        """Updates the next_tokens attribute, and extends current_length if needed, based on the provided num_steps."""
        self._next_tokens = new_token
        self.current_length += num_steps

        self.active_length = 1

    def trim_prompt(self, trim_len: int) -> None:
        """Trims the current prompt by the given number of tokens."""
        if trim_len == 0:
            return

        assert trim_len < len(self.next_tokens)
        next_tokens = self.next_tokens
        new_prompt = next_tokens[trim_len:]
        self._next_tokens = new_prompt
        self.active_length = len(new_prompt)
        assert self.active_length > 0

    def is_done(self, eos: int) -> bool:
        """Returns true if token gen for this context completed, else false."""
        test_token = (
            self._next_tokens[-1]
            if isinstance(self._next_tokens, np.ndarray)
            else self._next_tokens
        )
        return test_token == eos or self.current_length > self.max_length
