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
"""Utilities for working with Config objects in Click."""

import functools
import inspect
import pathlib
from dataclasses import MISSING, Field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import click
from max.driver import DeviceSpec
from max.pipelines import PipelineConfig, SupportedEncoding

from .device_options import DevicesOptionType

VALID_CONFIG_TYPES = [str, bool, Enum, Path, DeviceSpec, int, float]


def get_interior_type(type_hint: Union[type, str, Any]):
    interior_args = set(get_args(type_hint)) - set([type(None)])
    if len(interior_args) > 1:
        msg = (
            "Parsing does not currently supported Union type, with more than"
            " one non-None type: {type_hint}"
        )
        raise ValueError(msg)

    return get_args(type_hint)[0]


def is_optional(type_hint: Union[type, str, Any]):
    return get_origin(type_hint) is Union and type(None) in type_hint.__args__  # type: ignore


def is_flag(dataclass_field: Field):
    return dataclass_field.type is bool


def validate_field(dataclass_field: Field):
    if is_optional(dataclass_field.type):
        test_type = get_args(dataclass_field.type)[0]
    elif get_origin(dataclass_field.type) is list:
        test_type = get_interior_type(dataclass_field.type)
    else:
        test_type = dataclass_field.type

    for valid_type in VALID_CONFIG_TYPES:
        if valid_type == test_type:
            return True

        if get_origin(valid_type) is None and inspect.isclass(test_type):
            if issubclass(test_type, valid_type):
                return True

    msg = f"type '{test_type}' not supported in config."
    raise ValueError(msg)


def get_field_type(dataclass_field: Field):
    validate_field(dataclass_field)

    # Get underlying core field type, is Optional or list.
    field_type = dataclass_field.type
    if is_optional(dataclass_field.type):
        field_type = get_interior_type(dataclass_field.type)
    elif get_origin(dataclass_field.type) is list:
        field_type = get_interior_type(dataclass_field.type)

    # Update the field_type to be format specific.
    if field_type == Path:
        field_type = click.Path(path_type=pathlib.Path)
    elif inspect.isclass(field_type):
        if issubclass(field_type, Enum):
            field_type = click.Choice(field_type)  # type: ignore

    return field_type


def get_default(dataclass_field: Field):
    if dataclass_field.default_factory != MISSING:
        default = dataclass_field.default_factory()
    elif dataclass_field.default != MISSING:
        default = dataclass_field.default
    else:
        default = None

    return default


def is_multiple(dataclass_field: Field):
    return get_origin(dataclass_field.type) is list


def create_click_option(
    help_for_fields: dict[str, str],
    dataclass_field: Field,
) -> click.option:  # type: ignore
    # Get name.
    normalized_name = dataclass_field.name.lower().replace("_", "-")

    # Get Help text.
    help_text = help_for_fields.get(dataclass_field.name, None)

    # Get help field.
    return click.option(
        f"--{normalized_name}",
        show_default=True,
        help=help_text,
        is_flag=is_flag(dataclass_field),
        default=get_default(dataclass_field),
        multiple=is_multiple(dataclass_field),
        type=get_field_type(dataclass_field),
    )


def config_to_flag(cls):
    options = []
    if hasattr(cls, "help"):
        help_text = cls.help()
    else:
        help_text = {}
    for _field in fields(cls):
        if _field.name.startswith("_"):
            # Skip private config fields.
            continue

        new_option = create_click_option(
            help_text,
            _field,
        )
        options.append(new_option)

    def apply_flags(func):
        for option in reversed(options):
            func = option(func)  # type: ignore
        return func

    return apply_flags


def pipeline_config_options(func):
    @config_to_flag(PipelineConfig)
    @click.option(
        "--use-gpu",
        is_flag=False,
        type=DevicesOptionType(),
        show_default=False,
        default="",
        flag_value="0",
        help=(
            "Whether to run the model on the available GPU. An ID value can be"
            " provided optionally to indicate the device ID to target."
        ),
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs["use_gpu"]:
            kwargs["device_specs"] = [
                DeviceSpec.accelerator(id=gpu_id)
                for gpu_id in kwargs["use_gpu"]
            ]
            # If the user is passing in a specific, quantization_encoding don't overwrite it.
            # If it is empty, set it to default to bfloat16 on gpu.
            if kwargs["quantization_encoding"] is None:
                kwargs["quantization_encoding"] = SupportedEncoding.bfloat16
        else:
            kwargs["device_specs"] = [DeviceSpec.cpu()]

        del kwargs["use_gpu"]

        return func(*args, **kwargs)

    return wrapper
