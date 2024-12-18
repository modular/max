# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implementations of provided tokenizers."""

import asyncio
import io
import logging
from typing import Any, Sequence, Union, cast

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CodeLlamaTokenizer,
    CodeLlamaTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from .config import PipelineConfig
from .context import TextAndVisionContext, TextContext
from .interfaces import (
    PipelineTokenizer,
    TokenGeneratorContext,
    TokenGeneratorRequest,
    TokenGeneratorRequestMessage,
)


class IdentityPipelineTokenizer(
    PipelineTokenizer[TokenGeneratorContext, str],
):
    @property
    def eos(self) -> int:
        return 0

    @property
    def expects_content_wrapping(self) -> bool:
        return False

    async def encode(self, prompt: str) -> str:
        return prompt

    async def decode(
        self,
        context: TokenGeneratorContext,
        encoded: Any,
    ) -> str:
        if isinstance(encoded, str):
            return encoded
        return ""


class PreTrainedPipelineTokenizer(
    PipelineTokenizer[TokenGeneratorContext, np.ndarray]
):
    def __init__(
        self,
        delegate: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        assert isinstance(
            delegate, (PreTrainedTokenizer, PreTrainedTokenizerFast)
        )
        self.delegate = delegate

    def apply_chat_template(
        self, messages: list[TokenGeneratorRequestMessage]
    ) -> str:
        try:
            templated_message = self.delegate.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return cast(str, templated_message)
        except Exception:
            msg = (
                "apply_chat_template failed for"
                " PreTrainedTokenGeneratorTokenizer"
            )
            logging.warning(msg)
            return "\n".join([str(message["content"]) for message in messages])

    @property
    def eos(self) -> int:
        return self.delegate.eos_token_id

    @property
    def expects_content_wrapping(self) -> bool:
        return False

    async def encode(self, prompt: str) -> np.ndarray:
        return np.array(self.delegate.encode(prompt))

    async def decode(
        self,
        context: TokenGeneratorContext,
        encoded: np.ndarray,
    ) -> str:
        return self.delegate.decode(encoded)


def max_tokens_to_generate(
    prompt_size: int,
    max_length: int,
    max_new_tokens: int = -1,
) -> int:
    """Returns the max number of new tokens to generate."""
    _difference_between_max_and_prompt = max(max_length - prompt_size, 0)
    if max_new_tokens < 0:
        return _difference_between_max_and_prompt
    return min(max_new_tokens, _difference_between_max_and_prompt)


async def run_with_default_executor(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)


class TextTokenizer(PipelineTokenizer[TextContext, np.ndarray]):
    """Encapsulates creation of TextContext and specific token encode/decode logic."""

    def __init__(
        self, config: PipelineConfig, enable_llama_whitespace_fix: bool = False
    ):
        self.config = config

        if config.huggingface_repo_id is None:
            msg = (
                "a huggingface_repo_id must be provided to load the tokenizer."
            )
            raise ValueError(msg)

        self.delegate = AutoTokenizer.from_pretrained(
            config.huggingface_repo_id,
            trust_remote_code=config.trust_remote_code,
        )

        # configure Llama whitespace fix if needed
        self._enable_llama_whitespace_fix = (
            enable_llama_whitespace_fix and self._is_llama_tokenizer
        )
        (
            self._llama_whitespace_fix_dummy_token_id,
            self._llama_whitespace_fix_dummy_token_len,
        ) = self._llama_whitespace_fix_dummy_token

    def apply_chat_template(
        self, messages: list[TokenGeneratorRequestMessage]
    ) -> str:
        try:
            templated_message = self.delegate.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return cast(str, templated_message)
        except Exception:
            msg = (
                "apply_chat_template failed for"
                f" TextTokenizer({self.config.huggingface_repo_id})"
            )
            logging.warning(msg)
            return "\n".join([str(message["content"]) for message in messages])

    @property
    def eos(self) -> int:
        return self.delegate.eos_token_id

    @property
    def expects_content_wrapping(self) -> bool:
        return False

    async def encode(self, prompt: Union[str, Sequence[int]]) -> np.ndarray:
        """Transform the provided prompt into a token array."""

        encoded_prompt: np.ndarray
        if isinstance(prompt, str):
            # Note: the underlying tokenizer may not be thread safe in some cases, see https://github.com/huggingface/tokenizers/issues/537
            # Add a standard (non-async) lock in the executor thread if needed.
            encoded_prompt = await run_with_default_executor(
                self.delegate.encode, prompt
            )
        else:
            encoded_prompt = np.array(list(prompt))

        if len(encoded_prompt) >= self.config.max_length:
            msg = (
                f"Prompt length of {len(encoded_prompt)} is greater than the"
                " configured max model context length of"
                f" {self.config.max_length}"
            )
            raise ValueError(msg)

        return encoded_prompt

    async def decode(
        self, context: TextContext, encoded: np.ndarray, **kwargs
    ) -> str:
        """Transformer a provided encoded token array, back into readable text."""
        # Sometimes, encoded comes in as an int so, make it np array
        if isinstance(encoded, int):
            encoded = np.array(encoded)

        # There is an issue where Llama tokenizer strips leading spaces
        # if a single token is decoded at a time. This is a temporary
        # fix until the issue resolved on the Tokenizers side.
        # More information:
        # https://github.com/huggingface/transformers/issues/31643
        # https://github.com/Lightning-AI/litgpt/pull/1559
        if self._enable_llama_whitespace_fix and encoded.size == 1:
            return self._decode_with_llama_whitespace_fix(encoded, **kwargs)

        return self.delegate.decode(encoded, **kwargs)

    async def new_context(self, request: TokenGeneratorRequest) -> TextContext:
        """Create a new TextContext object, leveraging necessary information like
        cache_seq_id and prompt from TokenGeneratorRequest."""

        prompt: Union[str, Sequence[int]]
        if request.prompt is not None:
            prompt = request.prompt
        elif request.messages is not None:
            prompt = self.apply_chat_template(request.messages)
        else:
            raise ValueError(f"{request} does not provide messages or prompt.")
        encoded_prompt = await self.encode(prompt)

        max_gen_tokens = max_tokens_to_generate(
            len(encoded_prompt),
            self.config.max_length,
            request.max_new_tokens
            if request.max_new_tokens is not None
            else self.config.max_new_tokens,
        )
        context = TextContext(
            prompt=prompt,
            cache_seq_id=request.index,
            max_length=len(encoded_prompt) + max_gen_tokens,
            next_tokens=np.array(encoded_prompt),
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
        )
        return context

    @property
    def _is_llama_tokenizer(self) -> bool:
        tokenizers = (
            LlamaTokenizer,
            LlamaTokenizerFast,
            CodeLlamaTokenizer,
            CodeLlamaTokenizerFast,
        )
        return isinstance(self.delegate, tokenizers)

    @property
    def _llama_whitespace_fix_dummy_token(self) -> tuple[int, int]:
        dummy_token_id = 33  # \x1e
        dummy_token_decoded = self.delegate.decode([dummy_token_id])
        return dummy_token_id, len(dummy_token_decoded)

    def _decode_with_llama_whitespace_fix(
        self, encoded: np.ndarray, **kwargs
    ) -> str:
        decoded = self.delegate.decode(
            np.insert(encoded, 0, self._llama_whitespace_fix_dummy_token_id),
            **kwargs,
        )
        return decoded[self._llama_whitespace_fix_dummy_token_len :]


class TextAndVisionTokenizer(
    PipelineTokenizer[TextAndVisionContext, np.ndarray]
):
    """Encapsulates creation of TextContext and specific token encode/decode logic."""

    def __init__(self, config: PipelineConfig):
        self.config = config

        if config.huggingface_repo_id is None:
            msg = "a huggingface_repo_id must be provided to load tokenizer"
            raise ValueError(msg)

        self.delegate = AutoTokenizer.from_pretrained(
            config.huggingface_repo_id,
            trust_remote_code=config.trust_remote_code,
        )
        self.processor = AutoProcessor.from_pretrained(
            config.huggingface_repo_id,
            trust_remote_code=config.trust_remote_code,
        )

    def apply_chat_template(
        self, messages: list[TokenGeneratorRequestMessage]
    ) -> str:
        try:
            templated_message = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return cast(str, templated_message)
        except Exception as e:
            msg = "apply_chat_template failed for TextAndVisionTokenizer"
            logging.warning(msg)
            logging.warning(str(e))
            prompt = []
            for message in messages:
                if isinstance(message["content"], str):
                    prompt.append(message["content"])
                elif isinstance(message["content"], list):
                    for content in message["content"]:
                        if content["type"] == "text":
                            if "text" in content:
                                prompt.append(content["text"])
                            else:
                                prompt.append(content["content"])
            return "\n".join(prompt)

    @property
    def eos(self) -> int:
        return self.delegate.eos_token_id

    @property
    def expects_content_wrapping(self) -> bool:
        return True

    async def encode(self, prompt: Union[str, Sequence[int]]) -> np.ndarray:
        """Transform the provided prompt into a token array."""

        encoded_prompt: np.ndarray
        if isinstance(prompt, str):
            # Note: the underlying tokenizer may not be thread safe in some cases, see https://github.com/huggingface/tokenizers/issues/537
            # Add a standard (non-async) lock in the executor thread if needed.
            encoded_prompt = await run_with_default_executor(
                self.delegate.encode, prompt
            )
        else:
            encoded_prompt = np.array(list(prompt))

        if len(encoded_prompt) >= self.config.max_length:
            msg = (
                f"Prompt length of {len(encoded_prompt)} is greater than the"
                " configured max model context length of"
                f" {self.config.max_length}"
            )
            raise ValueError(msg)

        return encoded_prompt

    async def decode(
        self, context: TextAndVisionContext, encoded: np.ndarray, **kwargs
    ) -> str:
        """Transformer a provided encoded token array, back into readable text."""
        return self.delegate.decode(encoded, **kwargs)

    async def new_context(
        self, request: TokenGeneratorRequest
    ) -> TextAndVisionContext:
        """Create a new TextAndVisionContext object, leveraging necessary information like
        cache_seq_id and prompt from TokenGeneratorRequest."""

        prompt: Union[str, Sequence[int]]
        if request.prompt is not None:
            prompt = request.prompt
        elif request.messages is not None:
            prompt = self.apply_chat_template(request.messages)
        else:
            msg = f"{request} does not provide messages or prompt."
            raise ValueError(msg)

        # Load images.
        images = (
            [
                Image.open(io.BytesIO(image_data))
                for image_data in request.images
            ]
            if request.images
            else None
        )
        # PixtralProcessor returns a list of torch tensors.
        # LlamaVision returns a np Array.
        inputs = self.processor(
            text=prompt,
            images=images,
        )
        if "input_ids" not in inputs:
            msg = "input_ids not provided in AutoProcessor output, please ensure you are using the correct processor for multi-modal inputs."
            raise ValueError(msg)
        encoded_prompt = np.array(inputs["input_ids"][0])

        max_gen_tokens = max_tokens_to_generate(
            encoded_prompt.shape[0],
            self.config.max_length,
            request.max_new_tokens
            if request.max_new_tokens is not None
            else self.config.max_new_tokens,
        )

        if images is not None:
            if "pixel_values" not in inputs:
                msg = "pixel_values not provided in AutoProcessor output, please ensure you are using the correct processor for multi-modal inputs."
                raise ValueError(msg)
            pixel_values = inputs["pixel_values"][0]
            if isinstance(pixel_values, list):
                pixel_values = [
                    tensor.numpy() if torch.is_tensor(tensor) else tensor
                    for tensor in pixel_values
                ]
        else:
            pixel_values = []

        context = TextAndVisionContext(
            prompt=prompt,
            pixel_values=pixel_values,
            cache_seq_id=request.index,
            next_tokens=encoded_prompt,
            max_length=encoded_prompt.shape[0] + max_gen_tokens,
        )
        return context
