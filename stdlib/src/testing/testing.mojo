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
"""Implements various testing utils.

You can import these APIs from the `testing` package. For example:

```mojo
from testing import assert_true
```
"""
from collections import Optional

# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


@always_inline
fn _abs(x: SIMD) -> __type_of(x):
    return (x > 0).select(x, -x)


@always_inline
fn _isclose(
    a: SIMD, b: __type_of(a), *, atol: Scalar[a.type], rtol: Scalar[a.type]
) -> SIMD[DType.bool, a.size]:
    @parameter
    if a.type.is_bool() or a.type.is_integral():
        return a == b

    var atol_vec = SIMD[a.type, a.size](atol)
    var rtol_vec = SIMD[a.type, a.size](rtol)
    return _abs(a - b) <= (atol_vec.max(rtol_vec * _abs(a).max(_abs(b))))


# ===----------------------------------------------------------------------=== #
# Assertions
# ===----------------------------------------------------------------------=== #


@always_inline
fn assert_true[
    T: Boolable
](val: T, msg: String = "condition was unexpectedly False") raises:
    """Asserts that the input value is True. If it is not then an
    Error is raised.

    Parameters:
        T: A Boolable type.

    Args:
        val: The value to assert to be True.
        msg: The message to be printed if the assertion fails.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if not val:
        raise Error("AssertionError: " + msg)


@always_inline
fn assert_false[
    T: Boolable
](val: T, msg: String = "condition was unexpectedly True") raises:
    """Asserts that the input value is False. If it is not then an Error is
    raised.

    Parameters:
        T: A Boolable type.

    Args:
        val: The value to assert to be False.
        msg: The message to be printed if the assertion fails.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if val:
        raise Error("AssertionError: " + msg)


trait Testable(EqualityComparable, Stringable):
    """A trait that a struct should conform to if we do equality testing on it.
    """

    pass


@always_inline
fn assert_equal[T: Testable](lhs: T, rhs: T, msg: String = "") raises:
    """Asserts that the input values are equal. If it is not then an Error
    is raised.

    Args:
        lhs: The lhs of the equality.
        rhs: The rhs of the equality.
        msg: The message to be printed if the assertion fails.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if lhs != rhs:
        raise _assert_equal_error(str(lhs), str(rhs), msg=msg)


# TODO: Remove the String and SIMD overloads once we have more powerful traits.
@always_inline
fn assert_equal(lhs: String, rhs: String, msg: String = "") raises:
    """Asserts that the input values are equal. If it is not then an Error
    is raised.

    Args:
        lhs: The lhs of the equality.
        rhs: The rhs of the equality.
        msg: The message to be printed if the assertion fails.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if lhs != rhs:
        raise _assert_equal_error(lhs, rhs, msg=msg)


@always_inline
fn assert_equal[
    type: DType, size: Int
](lhs: SIMD[type, size], rhs: SIMD[type, size], msg: String = "") raises:
    """Asserts that the input values are equal. If it is not then an
    Error is raised.

    Parameters:
        type: The dtype of the left- and right-hand-side SIMD vectors.
        size: The width of the left- and right-hand-side SIMD vectors.

    Args:
        lhs: The lhs of the equality.
        rhs: The rhs of the equality.
        msg: The message to be printed if the assertion fails.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if lhs != rhs:
        raise _assert_equal_error(str(lhs), str(rhs), msg=msg)


@always_inline
fn assert_not_equal[T: Testable](lhs: T, rhs: T, msg: String = "") raises:
    """Asserts that the input values are not equal. If it is not then an
    Error is raised.

    Args:
        lhs: The lhs of the inequality.
        rhs: The rhs of the inequality.
        msg: The message to be printed if the assertion fails.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if lhs == rhs:
        raise _assert_not_equal_error(str(lhs), str(rhs), msg=msg)


@always_inline
fn assert_not_equal(lhs: String, rhs: String, msg: String = "") raises:
    """Asserts that the input values are not equal. If it is not then an
    an Error is raised.

    Args:
        lhs: The lhs of the inequality.
        rhs: The rhs of the inequality.
        msg: The message to be printed if the assertion fails.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if lhs == rhs:
        raise _assert_not_equal_error(str(lhs), str(rhs), msg=msg)


@always_inline
fn assert_not_equal[
    type: DType, size: Int
](lhs: SIMD[type, size], rhs: SIMD[type, size], msg: String = "") raises:
    """Asserts that the input values are not equal. If it is not then an
    Error is raised.

    Parameters:
        type: The dtype of the left- and right-hand-side SIMD vectors.
        size: The width of the left- and right-hand-side SIMD vectors.

    Args:
        lhs: The lhs of the inequality.
        rhs: The rhs of the inequality.
        msg: The message to be printed if the assertion fails.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if lhs == rhs:
        raise _assert_not_equal_error(str(lhs), str(rhs), msg=msg)


@always_inline
fn assert_almost_equal[
    type: DType, size: Int
](
    lhs: SIMD[type, size],
    rhs: SIMD[type, size],
    /,
    *,
    msg: String = "",
    atol: Scalar[type] = 1e-08,
    rtol: Scalar[type] = 1e-05,
) raises:
    """Asserts that the input values are equal up to a tolerance. If it is
    not then an Error is raised.

    Parameters:
        type: The dtype of the left- and right-hand-side SIMD vectors.
        size: The width of the left- and right-hand-side SIMD vectors.

    Args:
        lhs: The lhs of the equality.
        rhs: The rhs of the equality.
        msg: The message to print.
        atol: The _absolute tolerance.
        rtol: The relative tolerance.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    var almost_equal = _isclose(lhs, rhs, atol=atol, rtol=rtol)
    if not almost_equal:
        var err = "AssertionError: " + str(lhs) + " is not close to " + str(
            rhs
        ) + " with a diff of " + _abs(lhs - rhs)
        if msg:
            err += " (" + msg + ")"
        raise err


fn _assert_equal_error(
    lhs: String, rhs: String, msg: String = ""
) raises -> Error:
    var err = "AssertionError: `left == right` comparison failed:\n   left: " + lhs + "\n  right: " + rhs
    if msg:
        err += "\n  reason: " + msg
    raise err


fn _assert_not_equal_error(
    lhs: String, rhs: String, msg: String = ""
) raises -> Error:
    var err = "AssertionError: `left != right` comparison failed:\n   left: " + lhs + "\n  right: " + rhs
    if msg:
        err += "\n  reason: " + msg
    raise err


struct assert_raises:
    """Context manager that asserts that the block raises an exception.

    You can use this to test expected error cases, and to test that the correct
    errors are raised. For instance:

    ```mojo
    from testing import assert_raises

    # Good! Caught the raised error, test passes
    with assert_raises():
        raise "SomeError"

    # Also good!
    with assert_raises(contains="Some"):
        raise "SomeError"

    # This will assert, we didn't raise
    with assert_raises():
        pass

    # This will let the underlying error propagate, failing the test
    with assert_raises(contains="Some"):
        raise "OtherError"
    ```
    """

    var message_contains: Optional[String]
    """If present, check that the error message contains this literal string."""

    fn __init__(inout self):
        """Construct a context manager with no message pattern."""
        self.message_contains = None

    fn __init__(inout self, *, contains: String):
        """Construct a context manager matching specific errors.

        Args:
            contains: The test will only pass if the error message
                includes the literal text passed.
        """
        self.message_contains = contains

    fn __enter__(self):
        """Enter the context manager."""
        pass

    fn __exit__(self) raises:
        """Exit the context manager with no error.

        Raises:
            AssertionError: Always. The block must raise to pass the test.
        """
        raise Error("AssertionError: Didn't raise")

    fn __exit__(self, error: Error) raises -> Bool:
        """Exit the context manager with an error.

        Raises:
            Error: If the error raised doesn't match the expected error to raise.
        """
        if self.message_contains:
            return self.message_contains.value() in str(error)
        return True
