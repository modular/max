# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""A high level interface for interacting with LLMs built from MAX pipelines"""

from __future__ import annotations

import asyncio
import queue
from queue import Queue
from threading import Event, Thread
from typing import Optional, Sequence

import tqdm
from max.pipelines.config import PipelineConfig
from max.pipelines.registry import PIPELINE_REGISTRY
from max.serve.pipelines.llm import (
    TokenGeneratorPipeline,
    TokenGeneratorRequest,
    batch_config_from_pipeline_config,
)
from max.serve.pipelines.model_worker import start_model_worker

RequestQueue = Queue[tuple[Sequence[str], Optional[int], bool]]
ResponseQueue = Queue[list[str]]


# For now, the LLM class only supports the direct token generation use case.
# Long term, there are multiple other potential use cases to support.
# This class loosely mirrors vllm.LLM for offline inference: https://docs.vllm.ai/en/stable/dev/offline_inference/llm.html
class LLM:
    """A high level interface for interacting with LLMs."""

    _async_runner: Thread
    _shutdown: Event
    _request_queue: RequestQueue
    _response_queue: ResponseQueue

    def __init__(self, pipeline_config: PipelineConfig):
        self._shutdown = Event()
        self._request_queue = Queue()
        self._response_queue = Queue()
        model_ready = Event()
        self._async_runner = Thread(
            target=_run_async_worker,
            args=(
                pipeline_config,
                model_ready,
                self._shutdown,
                self._request_queue,
                self._response_queue,
            ),
        )
        self._async_runner.start()
        model_ready.wait()

    def __del__(self):
        self._shutdown.set()
        self._async_runner.join()

    def generate(
        self,
        prompts: str | Sequence[str],
        max_new_tokens: int | None = None,
        use_tqdm: bool = True,
    ) -> list[str]:
        if isinstance(prompts, str):
            # Handle the degenerate case where the users just passes in a single string
            return self.generate((prompts,), max_new_tokens, use_tqdm)

        self._request_queue.put((prompts, max_new_tokens, use_tqdm))
        return self._response_queue.get()


def _run_async_worker(
    pipeline_config: PipelineConfig,
    model_ready: Event,
    shutdown: Event,
    request_queue: RequestQueue,
    response_queue: ResponseQueue,
):
    asyncio.run(
        _async_worker(
            pipeline_config,
            model_ready,
            shutdown,
            request_queue,
            response_queue,
        )
    )


async def _async_worker(
    pipeline_config: PipelineConfig,
    model_ready: Event,
    shutdown: Event,
    request_queue: RequestQueue,
    response_queue: ResponseQueue,
):
    tokenizer, model_factory = PIPELINE_REGISTRY.retrieve_factory(
        pipeline_config
    )
    batch_config = batch_config_from_pipeline_config(pipeline_config)
    model_name = pipeline_config.huggingface_repo_id

    async with (
        # Start the model worker process.
        start_model_worker(
            model_factory=model_factory,
            batch_config=batch_config,
        ) as engine_queue,
        # Create dynamic and continuous batching workers and associated queues
        # to feed the model worker process.
        TokenGeneratorPipeline(
            model_name=model_name,
            tokenizer=tokenizer,
            engine_queue=engine_queue,
        ) as pipeline,
    ):
        model_ready.set()
        while True:
            if shutdown.is_set():
                break

            try:
                (prompts, max_new_tokens, use_tqdm) = request_queue.get(
                    timeout=0.3
                )

                if use_tqdm:
                    pbar = tqdm.tqdm(total=len(prompts))

                # Lambda to do a full text generation for a request.
                async def all_tokens(
                    i: int,
                    prompt: str,
                ) -> tuple[int, str]:
                    request = TokenGeneratorRequest(
                        id=str(i),
                        index=0,
                        model_name=model_name,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                    )

                    # Generate this request until complete
                    tokens = await pipeline.all_tokens(request)
                    if use_tqdm:
                        pbar.update(1)
                    return (i, "".join(t.decoded_token for t in tokens))

                all_tokens_tasks = [
                    all_tokens(i, prompt) for i, prompt in enumerate(prompts)
                ]
                responses = [""] * len(prompts)
                for i, response in await asyncio.gather(*all_tokens_tasks):
                    responses[i] = response

                if use_tqdm:
                    pbar.close()

                response_queue.put(responses)

            except queue.Empty:
                pass
