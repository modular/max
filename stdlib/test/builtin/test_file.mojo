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
    assert_equal(len(bytes1), 12, "12 bytes")
    # we add the null terminator
    bytes1.append(0)
    var string1 = String(bytes1)
    assert_equal(len(string1), 12, "12 chars")
    assert_equal(string1, String("Lorem ipsum "))

    var bytes2 = f.read_bytes(6)
    assert_equal(len(bytes2), 6, "6 bytes")
    # we add the null terminator
    bytes2.append(0)
    var string2 = String(bytes2)
    assert_equal(len(string2), 6, "6 chars")
    assert_equal(string2, "dolor ")

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
    import os

    with open(
        Path(CURRENT_DIR) / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var pos = f.seek(6)
        assert_equal(pos, 6)

        alias expected_msg1 = "ipsum dolor sit amet, consectetur adipiscing elit."
        assert_equal(f.read(len(expected_msg1)), expected_msg1)

        # Seek from the end of the file
        pos = f.seek(-16, os.SEEK_END)
        assert_equal(pos, 938)

        print(f.read(6))

        # Seek from current possition, skip the space
        pos = f.seek(1, os.SEEK_CUR)
        assert_equal(pos, 945)
        assert_equal(f.read(7), "rhoncus")

        try:
            _ = f.seek(-12)
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


@value
@register_passable
struct Word:
    var first_letter: Int8
    var second_letter: Int8
    var third_letter: Int8
    var fourth_letter: Int8
    var fith_letter: Int8

    fn __str__(self) -> String:
        var word = List[Int8](capacity=6)
        word.append(self.first_letter)
        word.append(self.second_letter)
        word.append(self.third_letter)
        word.append(self.fourth_letter)
        word.append(self.fith_letter)
        word.append(0)
        return word


# CHECK-LABEL: test_file_read_to_dtype_pointer
def test_file_read_to_dtype_pointer():
    print("== test_file_read_to_dtype_pointer")

    var f = open(Path(CURRENT_DIR) / "test_file_dummy_input.txt", "r")

    var ptr = DTypePointer[DType.int8].alloc(8)
    var data = f.read(ptr, 8)
    assert_equal(ptr.load[width=8](0), "[76, 111, 114, 101, 109, 32, 105, 112]")

    var ptr2 = DTypePointer[DType.int8].alloc(8)
    var data2 = f.read(ptr2, 8)
    assert_equal(
        ptr2.load[width=8](0), "[115, 117, 109, 32, 100, 111, 108, 111]"
    )


def test_file_iter():
    var TMP_FILE = Path(TEMP_FILE_DIR) / "dummy_newline_iterator.txt"
    alias textitems = List(
        "Lorem",
        "ipsum",
        "dolor",
        "sit",
        "namet,",
        "consectetur",
        "adipiscing elit.",
    )
    alias text2 = "Lorem\nipsum\ndolor\nsit\namet,\nconsectetur\nadipiscing elit."
    alias text1 = "Lorem\nipsum\ndolor\nsit\namet,\nconsectetur\nadipiscing elit.\n"
    for text in List(text1, text2):
        with open(TMP_FILE, "w") as f:
            f.write(text1)
        var f_r = open(TMP_FILE, "r")
        var counter = 0
        for line in f_r.__iter__().__iter__():
            assert_equal(line[], textitems[counter])
            counter += 1
        f_r.close()
        assert_equal(counter, len(textitems))


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
    test_file_read_to_dtype_pointer()
    test_file_iter()
