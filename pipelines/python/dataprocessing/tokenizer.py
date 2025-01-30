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

"""Utilities for creating a HuggingFace tokenizer for a pipeline model."""

import os
from typing import Any, Optional, Union

import gguf
from gguf import GGUFReader, Keys
from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast

from . import gguf_utils


def tokenizer_from_gguf(
    gguf_or_path: Union[GGUFReader, os.PathLike],
) -> Tokenizer:
    reader = gguf_or_path
    if not isinstance(gguf_or_path, GGUFReader):
        reader = GGUFReader(gguf_or_path)

    architecture = gguf_utils.read_string(reader, gguf.KEY_GENERAL_ARCHITECTURE)
    if architecture != "llama":
        raise NotImplementedError(
            f"Unsupported GGUF architecture: {architecture}"
        )

    model = gguf_utils.read_string(reader, Keys.Tokenizer.MODEL)
    if model != "gpt2":
        raise NotImplementedError(f"Unsupported tokenizer model: {model}")

    vocab_list = gguf_utils.read_string_array(reader, Keys.Tokenizer.LIST)
    if vocab_list is None:
        raise ValueError("Unable to find vocab list in GGUF weight file.")
    bpe_vocab = {token: n for n, token in enumerate(vocab_list)}

    merges: Optional[list[Any]] = gguf_utils.read_string_array(
        reader, Keys.Tokenizer.MERGES
    )
    if merges:
        merges = [tuple(s.split(" ")) for s in merges]

    tokenizer = Tokenizer(
        BPE(
            bpe_vocab,
            merges,
            fuse_unk=False,
            byte_fallback=False,
        )
    )

    # Mark special tokens.
    special_tokens = []
    bos_token = None
    bos_token_id = gguf_utils.read_number(reader, Keys.Tokenizer.BOS_ID)
    if bos_token_id is not None:
        bos_token = vocab_list[bos_token_id]
        special_tokens.append(bos_token)

    eos_token = None
    eos_token_id = gguf_utils.read_number(reader, Keys.Tokenizer.EOS_ID)
    if eos_token_id is not None:
        eos_token = vocab_list[eos_token_id]
        special_tokens.append(vocab_list[eos_token_id])
    token_type = gguf_utils.read_array(reader, Keys.Tokenizer.TOKEN_TYPE)
    if token_type:
        for i, type in enumerate(token_type):
            if type == LlamaTokenType.LLAMA_TOKEN_TYPE_CONTROL:
                special_tokens.append(vocab_list[i])
    chat_template = gguf_utils.read_string(reader, Keys.Tokenizer.CHAT_TEMPLATE)

    # Note: special tokens do not increase the size of the vocabulary, since
    # they are already in the vocab list
    tokenizer.add_special_tokens(special_tokens)
    pretrained_tokenizer_kwargs = {
        "bos_token": bos_token,
        "eos_token": eos_token,
        "add_prefix_space": True,
        "legacy": True,
        "clean_up_tokenization_spaces": True,
        "chat_template": chat_template,
    }

    # Set up different processors.
    # These settings are not defined in the GGUF file, and so these are manually
    # pulled from:
    # https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/raw/main/tokenizer.json
    tokenizer.normalizers = None  # type: ignore
    pattern = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(  # type: ignore
        [
            pre_tokenizers.Split(
                Regex(pattern), behavior="isolated", invert=False
            ),
            pre_tokenizers.ByteLevel(  # type: ignore
                add_prefix_space=False, trim_offsets=True, use_regex=False
            ),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel(  # type: ignore
        add_prefix_space=True, trim_offsets=True, use_regex=True
    )
    bos = bos_token
    add_bos_token = True
    eos = eos_token
    add_eos_token = False
    single = f"{(bos + ':0 ') if add_bos_token else ''}$A:0{(' ' + eos + ':0') if add_eos_token else ''}"  # type: ignore
    pair = f"{single}{(' ' + bos + ':1') if add_bos_token else ''} $B:1{(' ' + eos + ':1') if add_eos_token else ''}"  # type: ignore
    special_tokens = [(bos_token, bos_token_id)]
    tokenizer.post_processor = processors.TemplateProcessing(  # type: ignore
        single=single, pair=pair, special_tokens=special_tokens
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, **pretrained_tokenizer_kwargs
    )


# Llama token types from
# https://github.com/ggerganov/llama.cpp/blob/master/include/llama.h
class LlamaTokenType:
    LLAMA_TOKEN_TYPE_UNDEFINED = 0
    LLAMA_TOKEN_TYPE_NORMAL = 1
    LLAMA_TOKEN_TYPE_UNKNOWN = 2
    LLAMA_TOKEN_TYPE_CONTROL = 3
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4
    LLAMA_TOKEN_TYPE_UNUSED = 5
    LLAMA_TOKEN_TYPE_BYTE = 6
