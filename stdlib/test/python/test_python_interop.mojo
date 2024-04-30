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
# XFAIL: asan && !system-darwin
# RUN: %mojo -D TEST_DIR=%S %s

from sys import env_get_string

from python._cpython import CPython, PyObjectPtr
from python.python import Python, _get_global_python_itf, PythonObject

from testing import assert_equal

alias TEST_DIR = env_get_string["TEST_DIR"]()


fn test_execute_python_string(inout python: Python) -> String:
    try:
        _ = Python.evaluate("print('evaluated by PyRunString')")
        return Python.evaluate("'a' + 'b'")
    except e:
        return e


fn test_local_import(inout python: Python) -> String:
    try:
        Python.add_to_path(TEST_DIR)
        var my_module: PythonObject = Python.import_module("my_module")
        if my_module:
            var foo = my_module.Foo("apple")
            foo.bar = "orange"
            return foo.bar
        return "no module, no fruit"
    except e:
        return e


fn test_call(inout python: Python) -> String:
    try:
        Python.add_to_path(TEST_DIR)
        var my_module: PythonObject = Python.import_module("my_module")
        return str(
            my_module.eat_it_all(
                "carrot",
                "bread",
                "rice",
                fruit="pear",
                protein="fish",
                cake="yes",
            )
        )
    except e:
        return e


def main():
    var python = Python()
    assert_equal(test_local_import(python), "orange")

    assert_equal(
        test_call(python),
        (
            "carrot ('bread', 'rice') fruit=pear {'protein': 'fish', 'cake':"
            " 'yes'}"
        ),
    )

    var obj: PythonObject = [1, 2.4, True, "False"]
    assert_equal(str(obj), "[1, 2.4, True, 'False']")

    obj = (1, 2.4, True, "False")
    assert_equal(str(obj), "(1, 2.4, True, 'False')")

    obj = None
    assert_equal(str(obj), "None")

    assert_equal(test_execute_python_string(python), "ab")
