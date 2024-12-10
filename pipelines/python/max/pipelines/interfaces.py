# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Interfaces for different pipeline behaviors."""

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
)


class TokenGeneratorRequestMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Union[str, list[dict[str, Any]]]
    """ Content can be simple string or a list of message parts of different
    modalities. For e.g.
    {
      "role": "user",
      "content": "What'\''s the weather like in Boston today?"
    }
    or
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What'\''s in this image?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                }
            }
        ]
    }
    """


@dataclass(frozen=True)
class TokenGeneratorRequest:
    id: str
    index: int
    model_name: str
    prompt: Optional[str] = None
    """Prompt here is to support legacy /completion APIs"""
    messages: Optional[list[TokenGeneratorRequestMessage]] = None
    """Chat completion APIs work off messages."""
    images: Optional[list[bytes]] = None
    max_new_tokens: Optional[int] = None
    req_recv_time_ns: int = 0
    request_path: str = "/"
    logprobs: int = 0
    echo: bool = False

    def __str__(self) -> str:
        txt = f"Id: {self.id}"
        if self.max_new_tokens:
            txt += f", MaxNewTokens: {self.max_new_tokens}"
        return txt


TokenGeneratorContext = TypeVar("TokenGeneratorContext")
TokenGeneratorBatchKey = TypeVar("TokenGeneratorBatchKey")

TokenizerEncoded = TypeVar("TokenizerEncoded")


@runtime_checkable
class PipelineTokenizer(
    Generic[TokenGeneratorContext, TokenizerEncoded], Protocol
):
    """Interface for LLM tokenizers."""

    @property
    def eos(self) -> int:
        """The end of sequence token for this tokenizer."""
        ...

    @property
    def expects_content_wrapping(self) -> bool:
        """If true, this tokenizer expects messages to have a 'content' property.
        Text messages are formatted as
        { "type" : "text", "content" : "text content"}
        instead of, the OpenAI spec.
        { "type" : "text", "text": "text content" }.
        NOTE: Multimodal messages omit the content property.
        Both "image_urls" and "image" content parts are converted to simply
        { "type" : "image" }
        Their content is provided as byte arrays and by the top level property on
        the request object, i.e. "TokenGeneratorRequest.images".
        """
        ...

    async def new_context(
        self, request: TokenGeneratorRequest
    ) -> TokenGeneratorContext:
        """Creates a new context from a request object. This is sent to the
        worker process once and then cached locally.

        Args:
            request (TokenGeneratorRequest): Incoming request.

        Returns:
            TokenGeneratorContext: Initialized context.
        """
        ...

    async def encode(self, prompt: str) -> TokenizerEncoded:
        """Encodes text prompts as tokens.

        Args:
            prompt (str): Un-encoded prompt text.

        Raises:
            ValueError: If the prompt exceeds the configured maximum length.

        Returns:
            TokenizerEncoded: Encoded prompt tokens.
        """
        ...

    async def decode(
        self,
        context: TokenGeneratorContext,
        encoded: TokenizerEncoded,
    ) -> str:
        """Decodes response tokens to text.

        Args:
            context (TokenGeneratorContext): Current generation context.
            encoded (TokenizerEncoded): Encoded response tokens.

        Returns:
            str: Un-encoded response text.
        """
        ...


@runtime_checkable
class TokenGenerator(Generic[TokenGeneratorContext], Protocol):
    """Interface for LLM token-generator models."""

    def next_token(
        self, batch: dict[str, TokenGeneratorContext], num_steps: int = 1
    ) -> list[dict[str, Any]]:
        """Computes the next token response for a single batch.

        Args:
            batch (dict[str, TokenGeneratorContext]): Batch of contexts.
            num_steps (int, optional): Number of forward steps. Defaults to 1.

        Returns:
            list[dict[str, Any]]: List of encoded responses (indexed by req. ID)
        """
        ...

    def release(self, context: TokenGeneratorContext) -> None:
        """Releases resources associated with this context.

        Args:
            context (TokenGeneratorContext): Finished context.
        """
        ...


TokenGeneratorFactory = Callable[[], TokenGenerator]
