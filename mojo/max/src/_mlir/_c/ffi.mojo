# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.string import StaticString
from os import abort, getenv
from pathlib import Path
from sys.ffi import (
    RTLD,
    _get_dylib_function,
    _Global,
    _OwnedDLHandle,
    external_call,
)
from sys.param_env import env_get_string, is_defined

from memory import UnsafePointer


# Init fns inspired by gpu.host._utils
fn _init_dylib() -> _OwnedDLHandle:
    alias mlirc_dylib = env_get_string["MLIRC_DYLIB", ".graph_lib"]()

    # TODO: Move KGEN_CompilerRT_getMAXConfigValue to a helper somewhere.
    var mof_lib_path_str_ptr = external_call[
        "KGEN_CompilerRT_getMAXConfigValue", UnsafePointer[UInt8]
    ](mlirc_dylib.unsafe_ptr(), mlirc_dylib.byte_length())

    if not mof_lib_path_str_ptr:
        abort("cannot get graph library location from modular.cfg")

    var mof_lib_path = String(unsafe_from_utf8_ptr=mof_lib_path_str_ptr)

    # KGEN_CompilerRT_getMAXConfigValue returns an allocated pointer.
    mof_lib_path_str_ptr.free()

    if not Path(mof_lib_path).exists():
        abort("cannot load graph library from " + mof_lib_path)

    return _OwnedDLHandle(mof_lib_path, RTLD.NOW | RTLD.GLOBAL)


alias MOF_LIB = _Global["MOF_LIB", _OwnedDLHandle, _init_dylib]


@always_inline
fn MLIR_func[
    name: StaticString, T: AnyTrivialRegType, *Args: AnyType
](*args: *Args) -> T:
    var loaded_args_pack = args.get_loaded_kgen_pack()

    @parameter
    if not is_defined["MLIRCAPI_LINKED"]():
        var f = _get_dylib_function[
            MOF_LIB(),
            name,
            fn (__type_of(loaded_args_pack)) -> T,
        ]()
        var ptr = UnsafePointer(to=f).bitcast[UnsafePointer[NoneType]]()[]
        if not ptr:
            abort(String("cannot load ", name, " from graph library"))
        return f(loaded_args_pack)
    else:
        return external_call[name, T](loaded_args_pack)
