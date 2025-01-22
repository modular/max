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

from __future__ import annotations

import asyncio
import uuid

from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
)
from max.pipelines.interfaces import (
    PipelineTokenizer,
    TokenGenerator,
    TokenGeneratorRequest,
)
from tqdm import tqdm

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_BATCH_SIZE = 32


async def generate_responses(
    tokenizer: PipelineTokenizer,
    pipeline: TokenGenerator,
    prompts: list[str],
    max_new_tokens: int | None = None,
) -> list[str]:
    """Generates a batch of responses by running the pipeline on each prompt.

    This function is directly using low level context management APIs.
    These APIs are the basis for serving and all pipeline flows.
    """

    if len(prompts) > MAX_BATCH_SIZE:
        msg = f"Number of prompts to process ({len(prompts)}) is larger than the max batch size ({MAX_BATCH_SIZE})"
        raise ValueError(msg)

    # Generate a context to track each prompt.
    requests = {}
    for i, prompt in enumerate(prompts):
        # Use a uuid as the request id to ensure every context has a unique tracker.
        req_id = str(uuid.uuid4())
        context = await tokenizer.new_context(
            TokenGeneratorRequest(
                id=req_id,
                index=i,
                prompt=prompt,
                model_name=MODEL,
                max_new_tokens=max_new_tokens,
            )
        )
        requests[req_id] = context

    # Generate response tokens until all prompts are completed.
    responses_encoded = [[] for _ in prompts]  # type: list[list[int]]
    progress = tqdm(desc="Generating tokens", total=max_new_tokens)
    while True:
        (next_tokens,) = pipeline.next_token(requests, 1)
        if len(next_tokens) == 0:
            # All prompts have reached the end of stream.
            break

        for req_id, context in requests.items():
            if req_id not in next_tokens:
                # This prompt has reached the end of stream.
                continue

            # Accumulate new tokens for each response
            i = context.cache_seq_id
            responses_encoded[i].append(next_tokens[req_id].next_token)
        progress.update()
    progress.close()

    # Decode responses and append the text to the original prompts.
    responses = prompts
    for context in requests.values():
        i = context.cache_seq_id
        encoded_text = responses_encoded[i]
        responses[i] += await tokenizer.decode(context, encoded_text)

    # Free up contexts from the pipeline.
    for context in requests.values():
        pipeline.release(context)

    return responses


def main():
    pipeline_config = PipelineConfig(
        huggingface_repo_id=MODEL,
        max_cache_batch_size=MAX_BATCH_SIZE,
    )
    tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(pipeline_config)
    if not isinstance(pipeline, TokenGenerator):
        print("Pipeline not supported for text generation.")
        exit()

    prompts = [
        "In the beginning, there was",
        "I believe the meaning of life is",
        "The fastest way to learn python is",
    ]

    responses = asyncio.run(
        generate_responses(tokenizer, pipeline, prompts, max_new_tokens=20)
    )

    for i, response in enumerate(responses):
        print(f"========== Response {i} ==========")
        print(response)
        print()


if __name__ == "__main__":
    main()
