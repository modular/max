"""Public API accessors to reduce the number of load statements needed in BUILD.bazel files."""

load("@modular_test_deps//:requirements.bzl", _requirement = "requirement")
load("@rules_mojo//mojo:mojo_binary.bzl", _mojo_binary = "mojo_binary")
load("@rules_mojo//mojo:mojo_library.bzl", _mojo_library = "mojo_library")
load("@rules_mojo//mojo:mojo_test.bzl", _mojo_test = "mojo_test")
load("@rules_pkg//pkg:mappings.bzl", _strip_prefix = "strip_prefix")
load("@rules_python//python:py_library.bzl", "py_library")
load("//bazel/internal:binary_test.bzl", "binary_test")

modular_py_library = py_library
mojo_test = _mojo_test
requirement = _requirement
strip_prefix = _strip_prefix

def _has_internal_reference(deps):
    return any([dep.startswith(("//GenericML/", "//Kernels/", "//SDK/")) for dep in deps])

# buildifier: disable=function-docstring
def mojo_library(
        validate_missing_docs = False,  # buildifier: disable=unused-variable
        build_docs = False,  # buildifier: disable=unused-variable
        deps = [],
        **kwargs):
    if _has_internal_reference(deps):
        return

    _mojo_library(
        deps = deps,
        **kwargs
    )

def mojo_binary(
        data = [],
        deps = [],
        **kwargs):
    if _has_internal_reference(deps) or _has_internal_reference(data):
        return
    _mojo_binary(
        data = data,
        deps = deps,
        **kwargs
    )

def modular_run_binary_test(name, **kwargs):
    if name.endswith(".example-test"):
        return  # TODO: Fix custom_ops python examples
    binary_test(
        name = name,
        **kwargs
    )

def _noop(**_kwargs):
    pass

lit_tests = _noop
modular_py_binary = _noop
mojo_doc = _noop
mojo_kgen_lib = _noop
pkg_attributes = _noop
pkg_filegroup = _noop
pkg_files = _noop
