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

import functools
import logging
import os

import click
from architectures import register_all_models
from cli import (
    generate_text_for_pipeline,
    list_pipelines_to_console,
    pipeline_config_options,
    pipeline_encode,
    serve_pipeline,
)
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig

logger = logging.getLogger(__name__)
try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass


class ModelGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        supported = ", ".join(self.list_commands(ctx))
        ctx.fail(
            f"Command not supported: {cmd_name}\nSupported commands:"
            f" {supported}"
        )


@click.command(cls=ModelGroup)
def main():
    register_all_models()


def common_server_options(func):
    @click.option(
        "--profile-serve",
        is_flag=True,
        show_default=True,
        default=False,
        help=(
            "Whether to enable pyinstrument profiling on the serving endpoint."
        ),
    )
    @click.option(
        "--performance-fake",
        type=click.Choice(["none", "no-op", "speed-of-light", "vllm"]),
        default="none",
        help="Fake the engine performance (for benchmarking)",
    )
    @click.option(
        "--batch-timeout",
        type=float,
        default=0.0,
        help="Custom timeout for any particular batch.",
    )
    @click.option(
        "--model-name",
        type=str,
        help="Deprecated, please use `huggingface_repo_id` instead. Optional model alias for serving the model.",
    )
    @click.option(
        "--sim-failure",
        type=int,
        default=0,
        help="Simulate fake-perf with failure percentage",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@main.command(name="serve")
@pipeline_config_options
@common_server_options
def cli_serve(
    profile_serve,
    performance_fake,
    batch_timeout,
    model_name,
    sim_failure,
    **config_kwargs,
):
    # Initialize config, and serve.
    pipeline_config = PipelineConfig(**config_kwargs)
    failure_percentage = None
    if sim_failure > 0:
        failure_percentage = sim_failure
    serve_pipeline(
        pipeline_config=pipeline_config,
        profile=profile_serve,
        performance_fake=performance_fake,
        batch_timeout=batch_timeout,
        model_name=model_name,
        failure_percentage=failure_percentage,
    )


@main.command(name="generate")
@pipeline_config_options
@click.option(
    "--prompt",
    type=str,
    default="I believe the meaning of life is",
    help="The text prompt to use for further generation.",
)
@click.option(
    "--image_url",
    type=str,
    multiple=True,
    default=[],
    help=(
        "Images to include along with prompt, specified as URLs."
        " The images are ignored if the model does not support"
        " image inputs."
    ),
)
@click.option(
    "--num-warmups",
    type=int,
    default=0,
    show_default=True,
    help="# of warmup iterations to run before the final timed run.",
)
def cli_pipeline(prompt, image_url, num_warmups, **config_kwargs):
    # Replit huggingface_repo_ids are kinda broken due to transformers
    # version mismatch. We manually update trust_remote_code to True
    # because the modularai version does not have the custom Python code needed
    # Without this, we get:
    #     ValueError: `attn_type` has to be either `multihead_attention` or
    #     `multiquery_attention`. Received: grouped_query_attention
    # Another reason why we override this flag here is because at PipelineConfig
    # instantiation below, we'll call AutoConfig.from_pretrained, which will
    # trigger the error above if not set to True.
    if "replit" in config_kwargs["huggingface_repo_id"]:
        config_kwargs["trust_remote_code"] = True

    # Load tokenizer & pipeline.
    pipeline_config = PipelineConfig(**config_kwargs)
    generate_text_for_pipeline(
        pipeline_config,
        prompt=prompt,
        image_urls=image_url,
        num_warmups=num_warmups,
    )


@main.command(name="encode")
@pipeline_config_options
@click.option(
    "--prompt",
    type=str,
    default="I believe the meaning of life is",
    help="The text prompt to use for further generation.",
)
@click.option(
    "--num-warmups",
    type=int,
    default=0,
    show_default=True,
    help="# of warmup iterations to run before the final timed run.",
)
def encode(prompt, num_warmups, **config_kwargs):
    # Load tokenizer & pipeline.
    pipeline_config = PipelineConfig(**config_kwargs)
    pipeline_encode(
        pipeline_config,
        prompt=prompt,
        num_warmups=num_warmups,
    )


@main.command(name="warm-cache")
@pipeline_config_options
def cli_warm_cache(**config_kwargs) -> None:
    """Load the model and do nothing with it, warming the cache in the process.

    This command is particularly useful in combination with
    --save-to-serialized-model-path.  Providing that option to this command
    will result in a compiled model being stored to that path.  Subsequent
    invocations of other commands can then use --serialized-model-path to reuse
    the previously-compiled model.

    Even without --save-to-serialized-model-path, this command will as a side
    effect warm the HuggingFace cache and in some cases, MAX compilation
    caches.
    """
    pipeline_config = PipelineConfig(**config_kwargs)
    _ = PIPELINE_REGISTRY.retrieve(pipeline_config)


@main.command(name="list")
def cli_list():
    list_pipelines_to_console()


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    main()
