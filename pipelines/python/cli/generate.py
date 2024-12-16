# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""Utilities for generating text in the cli."""

import asyncio
import logging
import uuid
from typing import Optional

from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
)
from max.pipelines.interfaces import (
    PipelineTokenizer,
    TokenGenerator,
    TokenGeneratorRequest,
)

from .metrics import TextGenerationMetrics

logger = logging.getLogger(__name__)

MODEL_NAME = "model"


async def stream_text_to_console(
    pipeline: TokenGenerator,
    tokenizer: PipelineTokenizer,
    prompt: str,
    metrics: Optional[TextGenerationMetrics] = None,
    print_tokens: bool = True,
):
    # Length of request_id_context_dict should be == batch_size.
    request_id_context = {}

    # create a dict of request_id: contexts
    decoded_responses = {}

    req_id = str(uuid.uuid4())
    context = await tokenizer.new_context(
        TokenGeneratorRequest(
            id=req_id, index=0, prompt=prompt, model_name=MODEL_NAME
        )
    )
    decoded_responses[req_id] = [prompt]
    request_id_context[req_id] = context
    prompt_size = context.current_length

    if metrics:
        metrics.prompt_size = prompt_size
        metrics.signpost("begin_generation")

    if print_tokens:
        print(prompt, end="", flush=True)

    first_token = True
    while True:
        responses = pipeline.next_token(request_id_context)[0]
        if not responses:
            break

        for req_id, context in request_id_context.items():
            if req_id not in responses or context.is_done(tokenizer.eos):
                del request_id_context[req_id]
                continue

            encoded_text = responses[req_id].next_token
            response_text = await tokenizer.decode(context, encoded_text)
            if metrics:
                if first_token:
                    first_token = False
                    metrics.signpost("first_token")
                metrics.new_token()
            if print_tokens:
                print(response_text, end="", flush=True)
            else:
                decoded_responses[req_id].append(response_text)

        if not request_id_context:
            break

    if metrics:
        metrics.signpost("end_generation")

    for context in request_id_context.values():
        pipeline.release(context)

    if print_tokens:
        print()


def generate_text_for_pipeline(
    pipeline_config: PipelineConfig, prompt: str, num_warmups: int = 0
):
    # Run timed run & print results.
    with TextGenerationMetrics(print_report=True) as metrics:
        # Load tokenizer and Pipeline.
        tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(pipeline_config)

        # Run warmups if requested.
        if num_warmups > 0:
            logger.info("Running warmup...")
            for _ in range(num_warmups):
                asyncio.run(
                    stream_text_to_console(
                        pipeline,
                        tokenizer,
                        prompt,
                        metrics=None,
                        print_tokens=False,
                    )
                )

        # Run and print results.
        logger.info("Beginning text generation...")
        asyncio.run(
            stream_text_to_console(
                pipeline,
                tokenizer,
                prompt,
                metrics=metrics,
                print_tokens=True,
            )
        )
