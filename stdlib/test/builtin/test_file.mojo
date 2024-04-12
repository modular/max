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
# RUN: %mojo -D CURRENT_DIR=%S -D TEMP_FILE_DIR=%T -debug-level full %s


from pathlib import Path
from sys import os_is_windows, env_get_string

from testing import assert_equal, assert_true

alias CURRENT_DIR = env_get_string["CURRENT_DIR"]()
alias TEMP_FILE_DIR = env_get_string["TEMP_FILE_DIR"]()


def test_file_read():
    var f = open(
        Path(CURRENT_DIR) / "test_file_dummy_input.txt",
        "r",
    )
    assert_true(
        f.read().startswith(
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        )
    )
    f.close()


def test_file_read_multi():
    var f = open(
        (Path(CURRENT_DIR) / "test_file_dummy_input.txt"),
        "r",
    )

    assert_equal(f.read(12), "Lorem ipsum ")
    assert_equal(f.read(6), "dolor ")
    assert_true(f.read().startswith("sit amet, consectetur adipiscing elit."))

    f.close()


def test_file_read_bytes_multi():
    var f = open(
        Path(CURRENT_DIR) / "test_file_dummy_input.txt",
        "r",
    )

    var bytes1 = f.read_bytes(12)
    assert_equal(bytes1, "Lorem ipsum")

    var bytes2 = f.read_bytes(6)
    assert_equal(String(bytes2).strip(), "dolor")

    # Read where N is greater than the number of bytes in the file.
    var s: String = f.read(1e9)

    assert_equal(len(s), 936)

    assert_true(s.startswith("sit amet, consectetur adipiscing elit."))

    f.close()


def test_file_read_path():
    var file_path = Path(CURRENT_DIR) / "test_file_dummy_input.txt"

    var f = open(file_path, "r")
    assert_true(
        f.read().startswith(
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        )
    )
    f.close()


def test_file_path_direct_read():
    var file_path = Path(CURRENT_DIR) / "test_file_dummy_input.txt"
    assert_true(
        file_path.read_text().startswith(
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        )
    )


def test_file_read_context():
    with open(
        Path(CURRENT_DIR) / "test_file_dummy_input.txt",
        "r",
    ) as f:
        assert_true(
            f.read().startswith(
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
            )
        )


def test_file_seek():
    with open(
        Path(CURRENT_DIR) / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var pos = f.seek(6)
        assert_equal(pos, 6)

        alias expected_msg1 = "ipsum dolor sit amet, consectetur adipiscing elit."
        assert_equal(f.read(len(expected_msg1)), expected_msg1)

        try:
            f.seek(-12)
        except e:
            alias expected_msg = "seek error"
            assert_equal(str(e)[: len(expected_msg)], expected_msg)


def test_file_open_nodir():
    var f = open(Path("test_file_open_nodir"), "w")
    f.close()


def test_file_write():
    var content = "The quick brown fox jumps over the lazy dog"
    var TEMP_FILE = Path(TEMP_FILE_DIR) / "test_file_write"
    var f = open(TEMP_FILE, "w")
    f.write(content)
    f.close()

    var read_file = open(TEMP_FILE, "r")
    assert_equal(read_file.read(), content)
    read_file.close()


def test_file_write_again():
    var unexpected_content = "foo bar baz"
    var expected_content = "foo bar"
    var TEMP_FILE = Path(TEMP_FILE_DIR) / "test_file_write_again"
    with open(TEMP_FILE, "w") as f:
        f.write(unexpected_content)

    with open(TEMP_FILE, "w") as f:
        f.write(expected_content)

    var read_file = open(TEMP_FILE, "r")
    assert_equal(read_file.read(), expected_content)
    read_file.close()


def main():
    test_file_read()
    test_file_read_multi()
    test_file_read_bytes_multi()
    test_file_read_path()
    test_file_path_direct_read()
    test_file_read_context()
    test_file_seek()
    test_file_open_nodir()
    test_file_write()
    test_file_write_again()
