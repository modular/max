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
# XFAIL: asan && !system-darwin
# RUN: %mojo %s

from python import Python, PythonObject
from python._cpython import PyObjectPtr
from memory import UnsafePointer
from testing import assert_equal, assert_false, assert_raises, assert_true


def test_PyObject_HasAttrString(mut python: Python):
    var Cpython_env = python.impl._cpython

    var the_object = PythonObject(0)
    var result = Cpython_env[].PyObject_HasAttrString(
        the_object.py_object, "__contains__"
    )
    assert_equal(0, result)

    the_object = Python.list(1, 2, 3)
    result = Cpython_env[].PyObject_HasAttrString(
        the_object.py_object, "__contains__"
    )
    assert_equal(1, result)
    _ = the_object


fn destructor(capsule: PyObjectPtr) -> None:
    pass


def test_PyCapsule(mut python: Python):
    var Cpython_env = python.impl._cpython

    # Not a PyCapsule, a NULL pointer is expected.
    var the_object = PythonObject(0)
    var result = Cpython_env[].PyCapsule_GetPointer(
        the_object.py_object, "some_name"
    )
    var expected = UnsafePointer[NoneType]()
    assert_equal(expected, result)

    # Build a capsule.
    var capsule_impl = UnsafePointer[UInt64].alloc(1)
    var capsule = Cpython_env[].PyCapsule_New(
        capsule_impl.bitcast[NoneType](), "some_name", destructor
    )
    var capsule_pointer = Cpython_env[].PyCapsule_GetPointer(
        capsule, "some_name"
    )
    assert_equal(capsule_impl.bitcast[NoneType](), capsule_pointer)


def main():
    # initializing Python instance calls init_python
    var python = Python()
    test_PyObject_HasAttrString(python)
    test_PyCapsule(python)
