"""A test rule that executes a mojo_binary, passing its output to FileCheck."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_mojo//mojo:mojo_binary.bzl", "mojo_binary")

def mojo_filecheck_test(name, srcs, copts = [], deps = [], enable_assertions = True, expect_crash = False, size = None, **kwargs):
    if len(srcs) != 1:
        fail("Only a single source file may be passed")

    mojo_binary(
        name = name + ".binary",
        copts = copts,
        srcs = srcs,
        deps = deps,
        testonly = True,
        enable_assertions = enable_assertions,
        **kwargs
    )

    native.sh_test(
        name = name,
        srcs = ["//bazel/internal:mojo-filecheck-test"],
        args = [paths.join(native.package_name(), src) for src in srcs],
        size = size,
        data = srcs + [
            name + ".binary",
            "@llvm-project//llvm:FileCheck",
            "@llvm-project//llvm:not",
        ],
        env = {
            "BINARY": "$(location :{}.binary)".format(name),
            "EXPECT_CRASH": "1" if expect_crash else "0",
            "FILECHECK": "$(location @llvm-project//llvm:FileCheck)",
            "NOT": "$(location @llvm-project//llvm:not)",
        },
        **kwargs
    )
