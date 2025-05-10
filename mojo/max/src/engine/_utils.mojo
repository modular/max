# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle


@value
@register_passable("trivial")
struct CString:
    """Represents `const char*` in C. Useful for binding with C APIs."""

    var ptr: DTypePointer[DType.int8]

    fn get_as_string_ref(self) -> StringRef:
        """
        Get the `CString` as `StringRef`. Lifetime is tied to C API.
        For owning version use `__str__()`.
        """
        return StringRef(self.ptr)

    fn __str__(self) -> String:
        """
        Get `CString` as a owning `String`.
        """
        return String(self.ptr)


@always_inline("nodebug")
fn exchange[T: AnyRegType](inout old_var: T, owned new_value: T) -> T:
    """
    Assign `new_value` to `old_var` and returns the value previously
    contained in `old_var`.
    """
    let old = old_var
    old_var = new_value
    return old


# ======================================================================#
#                                                                       #
# Utility structs and functions to interact with dylibs.                #
#                                                                       #
# ======================================================================#


@value
@register_passable("trivial")
struct SingleArgCallable[ResultTy: AnyRegType, ArgTy: AnyRegType]:
    var func: fn (ArgTy) -> ResultTy

    @always_inline("nodebug")
    fn __call__(self, arg: ArgTy) -> ResultTy:
        return self.func(arg)


@value
@register_passable("trivial")
struct TwoArgCallable[
    ResultTy: AnyRegType, Arg1Ty: AnyRegType, Arg2Ty: AnyRegType
]:
    var func: fn (Arg1Ty, Arg2Ty) -> ResultTy

    @always_inline("nodebug")
    fn __call__(self, arg1: Arg1Ty, arg2: Arg2Ty) -> ResultTy:
        return self.func(arg1, arg2)


@value
@register_passable("trivial")
struct ThreeArgCallable[
    ResultTy: AnyRegType,
    Arg1Ty: AnyRegType,
    Arg2Ty: AnyRegType,
    Arg3Ty: AnyRegType,
]:
    var func: fn (Arg1Ty, Arg2Ty, Arg3Ty) -> ResultTy

    @always_inline("nodebug")
    fn __call__(self, arg1: Arg1Ty, arg2: Arg2Ty, arg3: Arg3Ty) -> ResultTy:
        return self.func(arg1, arg2, arg3)


@value
@register_passable("trivial")
struct FourArgCallable[
    ResultTy: AnyRegType,
    Arg1Ty: AnyRegType,
    Arg2Ty: AnyRegType,
    Arg3Ty: AnyRegType,
    Arg4Ty: AnyRegType,
]:
    var func: fn (Arg1Ty, Arg2Ty, Arg3Ty, Arg4Ty) -> ResultTy

    @always_inline("nodebug")
    fn __call__(
        self, arg1: Arg1Ty, arg2: Arg2Ty, arg3: Arg3Ty, arg4: Arg4Ty
    ) -> ResultTy:
        return self.func(arg1, arg2, arg3, arg4)


@always_inline("nodebug")
fn call_dylib_func[
    ResultTy: AnyRegType
](borrowed lib: DLHandle, name: StringRef) -> ResultTy:
    """Call function `name` in dylib with one result and no arguments."""
    return lib.get_function[SingleArgCallable[ResultTy, NoneType]](name)(None)


@always_inline("nodebug")
fn call_dylib_func[
    ArgTy: AnyRegType
](borrowed lib: DLHandle, name: StringRef, arg: ArgTy) -> None:
    """Call function `name` in dylib with no result and one argument."""
    lib.get_function[SingleArgCallable[NoneType, ArgTy]](name)(arg)


@always_inline("nodebug")
fn call_dylib_func[
    ResultTy: AnyRegType, ArgTy: AnyRegType
](borrowed lib: DLHandle, name: StringRef, arg: ArgTy) -> ResultTy:
    """Call function `name` in dylib with one result and one argument."""
    return lib.get_function[SingleArgCallable[ResultTy, ArgTy]](name)(arg)


@always_inline("nodebug")
fn call_dylib_func[
    Arg1Ty: AnyRegType, Arg2Ty: AnyRegType
](borrowed lib: DLHandle, name: StringRef, arg1: Arg1Ty, arg2: Arg2Ty):
    """Call function `name` in dylib with no result and two arguments."""
    lib.get_function[TwoArgCallable[NoneType, Arg1Ty, Arg2Ty]](name)(arg1, arg2)


@always_inline("nodebug")
fn call_dylib_func[
    ResultTy: AnyRegType, Arg1Ty: AnyRegType, Arg2Ty: AnyRegType
](
    borrowed lib: DLHandle, name: StringRef, arg1: Arg1Ty, arg2: Arg2Ty
) -> ResultTy:
    """Call function `name` in dylib with one result and two arguments."""
    return lib.get_function[TwoArgCallable[ResultTy, Arg1Ty, Arg2Ty]](name)(
        arg1, arg2
    )


@always_inline("nodebug")
fn call_dylib_func[
    Arg1Ty: AnyRegType, Arg2Ty: AnyRegType, Arg3Ty: AnyRegType
](
    borrowed lib: DLHandle,
    name: StringRef,
    arg1: Arg1Ty,
    arg2: Arg2Ty,
    arg3: Arg3Ty,
):
    """Call function `name` in dylib with no result and three arguments."""
    lib.get_function[ThreeArgCallable[NoneType, Arg1Ty, Arg2Ty, Arg3Ty]](name)(
        arg1, arg2, arg3
    )


@always_inline("nodebug")
fn call_dylib_func[
    ResultTy: AnyRegType,
    Arg1Ty: AnyRegType,
    Arg2Ty: AnyRegType,
    Arg3Ty: AnyRegType,
](
    borrowed lib: DLHandle,
    name: StringRef,
    arg1: Arg1Ty,
    arg2: Arg2Ty,
    arg3: Arg3Ty,
) -> ResultTy:
    """Call function `name` in dylib with one result and three arguments."""
    return lib.get_function[ThreeArgCallable[ResultTy, Arg1Ty, Arg2Ty, Arg3Ty]](
        name
    )(arg1, arg2, arg3)


@always_inline("nodebug")
fn call_dylib_func[
    Arg1Ty: AnyRegType,
    Arg2Ty: AnyRegType,
    Arg3Ty: AnyRegType,
    Arg4Ty: AnyRegType,
](
    borrowed lib: DLHandle,
    name: StringRef,
    arg1: Arg1Ty,
    arg2: Arg2Ty,
    arg3: Arg3Ty,
    arg4: Arg4Ty,
):
    """Call function `name` in dylib with no result and four arguments."""
    return lib.get_function[
        FourArgCallable[NoneType, Arg1Ty, Arg2Ty, Arg3Ty, Arg4Ty]
    ](name)(arg1, arg2, arg3, arg4)


@always_inline("nodebug")
fn call_dylib_func[
    ResultTy: AnyRegType,
    Arg1Ty: AnyRegType,
    Arg2Ty: AnyRegType,
    Arg3Ty: AnyRegType,
    Arg4Ty: AnyRegType,
](
    borrowed lib: DLHandle,
    name: StringRef,
    arg1: Arg1Ty,
    arg2: Arg2Ty,
    arg3: Arg3Ty,
    arg4: Arg4Ty,
) -> ResultTy:
    """Call function `name` in dylib with one result and four arguments."""
    return lib.get_function[
        FourArgCallable[ResultTy, Arg1Ty, Arg2Ty, Arg3Ty, Arg4Ty]
    ](name)(arg1, arg2, arg3, arg4)


struct OwningVector[T: Movable](Sized):
    var ptr: AnyPointer[T]
    var size: Int

    alias initial_capacity = 5
    var capacity: Int

    fn __init__(inout self):
        let ptr = AnyPointer[T].alloc(Self.initial_capacity)
        self.ptr = ptr
        self.size = 0
        self.capacity = Self.initial_capacity

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = existing.ptr
        self.size = existing.size
        self.capacity = existing.capacity

    fn emplace_back(inout self, owned value: T):
        if self.size < self.capacity:
            (self.ptr + self.size).emplace_value(value ^)
            self.size += 1
            return

        self.capacity = self.capacity * 2
        let new_ptr = AnyPointer[T].alloc(self.capacity)
        for i in range(self.size):
            (new_ptr + i).emplace_value((self.ptr + i).take_value())
        self.ptr.free()
        self.ptr = new_ptr
        self.emplace_back(value ^)

    fn get(self, idx: Int) raises -> AnyPointer[T]:
        if idx >= self.size:
            raise "requested index(" + String(
                idx
            ) + ") exceeds size of vector(" + self.size + ")"
        return self.ptr + idx

    fn __len__(self) -> Int:
        return self.size

    fn __del__(owned self):
        for i in range(self.size):
            _ = (self.ptr + i).take_value()
        self.ptr.free()
