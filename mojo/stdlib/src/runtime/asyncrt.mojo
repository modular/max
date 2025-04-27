# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module implements the low level concurrency library."""

from os import PathLike, abort
from os.atomic import Atomic
from sys import external_call
from sys.ffi import _get_global, _get_global_or_null
from sys.info import num_physical_cores
from sys.param_env import is_defined

from builtin.coroutine import AnyCoroutine, _coro_resume_fn, _suspend_async
from gpu.host import Context as CudaContext
from gpu.host import CudaInstance, DeviceContext, KernelProfilingInfo, Stream
from memory import UnsafePointer

from utils import StringRef

from .tracing import TraceLevel, is_mojo_profiling_disabled

# ===----------------------------------------------------------------------===#
# AsyncContext
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct Chain(Boolable):
    """A proxy for the C++ runtime's AsyncValueRef<Chain> type."""

    # Actually an AsyncValueRef<Chain>, which is just an AsyncValue*
    var storage: UnsafePointer[Int]

    fn __init__(inout self):
        self.storage = UnsafePointer[Int]()

    fn __bool__(self) -> Bool:
        return self.storage != UnsafePointer[Int]()


@register_passable("trivial")
struct AsyncContext:
    """This struct models the coroutine context contained in every coroutine
    instance. The struct consists of a unary callback function that accepts a
    pointer argument. It is invoked with the second struct field, which is an
    opaque pointer. This struct is essentially a completion callback closure
    that is invoked by a coroutine when it completes and its results are made
    available.

    In async execution, a task's completion callback is to set its async token
    to available.
    """

    alias callback_fn_type = fn (Chain) -> None

    var callback: Self.callback_fn_type
    var chain: Chain

    @staticmethod
    fn get_chain(ctx: UnsafePointer[AsyncContext]) -> UnsafePointer[Chain]:
        return UnsafePointer.address_of(ctx[].chain)

    @staticmethod
    fn complete(ch: Chain):
        var tmp = ch
        _async_complete(UnsafePointer[Chain].address_of(tmp))
        _ = tmp


# ===----------------------------------------------------------------------===#
# AsyncRT C Shims
# ===----------------------------------------------------------------------===#


fn _init_asyncrt_chain(chain: UnsafePointer[Chain]):
    external_call["KGEN_CompilerRT_AsyncRT_InitializeChain", NoneType](
        _get_current_runtime(), chain.address
    )


fn _del_asyncrt_chain(chain: UnsafePointer[Chain]):
    external_call["KGEN_CompilerRT_AsyncRT_DestroyChain", NoneType](
        chain.address
    )


fn _async_and_then(hdl: AnyCoroutine, chain: UnsafePointer[Chain]):
    external_call["KGEN_CompilerRT_AsyncRT_AndThen", NoneType](
        _coro_resume_fn, chain.address, hdl
    )


fn _async_execute[type: AnyType](handle: AnyCoroutine, desired_worker_id: Int):
    external_call["KGEN_CompilerRT_AsyncRT_Execute", NoneType](
        _coro_resume_fn, handle, _get_current_runtime(), desired_worker_id
    )


fn _async_wait(chain: UnsafePointer[Chain]):
    external_call["KGEN_CompilerRT_AsyncRT_Wait", NoneType](chain.address)


fn _async_complete(chain: UnsafePointer[Chain]):
    external_call["KGEN_CompilerRT_AsyncRT_Complete", NoneType](chain.address)


fn _async_wait_timeout(chain: UnsafePointer[Chain], timeout: Int) -> Bool:
    return external_call["KGEN_CompilerRT_AsyncRT_Wait_Timeout", Bool](
        chain.address, timeout
    )


struct ChainPromise:
    var chain: Chain

    fn __init__(inout self):
        self.chain = Chain()
        _init_asyncrt_chain(UnsafePointer.address_of(self.chain))

    fn __init__(inout self, owned chain: Chain):
        self.chain = chain

    fn __del__(owned self):
        if self.chain:
            _del_asyncrt_chain(UnsafePointer.address_of(self.chain))

    @always_inline
    fn __await__(self):
        @always_inline
        @parameter
        fn await_body(cur_hdl: AnyCoroutine):
            _async_and_then(cur_hdl, UnsafePointer.address_of(self.chain))

        _suspend_async[await_body]()

    fn wait(self):
        _async_wait(UnsafePointer.address_of(self.chain))


# ===----------------------------------------------------------------------===#
# Global Runtime
# ===----------------------------------------------------------------------===#


@always_inline
fn _get_current_runtime() -> UnsafePointer[NoneType]:
    """Returns the current runtime. The runtime is either created by the
    surrounding mojo tool (mojo-repl, mojo-jit, ...) or by the entry main
    function.
    """
    return external_call[
        "KGEN_CompilerRT_AsyncRT_GetCurrentRuntime", UnsafePointer[NoneType]
    ]()


@always_inline
fn parallelism_level() -> Int:
    """Gets the parallelism level of the Runtime."""
    return int(
        external_call[
            "KGEN_CompilerRT_AsyncRT_ParallelismLevel",
            Int32,
        ](_get_current_runtime())
    )


fn create_task(
    owned handle: Coroutine[*_],
) -> Task[handle.type, handle.lifetimes] as task:
    """Run the coroutine as a task on the AsyncRT Runtime."""
    var ctx = handle._get_ctx[AsyncContext]()
    _init_asyncrt_chain(AsyncContext.get_chain(ctx))
    ctx[].callback = AsyncContext.complete
    task.__init__(handle^)
    _async_execute[handle.type](task._handle._handle, desired_worker_id=-1)


@always_inline
fn run(owned handle: Coroutine[*_]) -> handle.type as out:
    var ctx = handle._get_ctx[AsyncContext]()
    _init_asyncrt_chain(AsyncContext.get_chain(ctx))
    ctx[].callback = AsyncContext.complete
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(out))
    handle._set_result_slot(UnsafePointer.address_of(out))
    _async_execute[handle.type](handle._handle, -1)
    _async_wait(AsyncContext.get_chain(ctx))
    _del_asyncrt_chain(AsyncContext.get_chain(ctx))
    _ = handle^


@always_inline
fn run(owned handle: RaisingCoroutine[*_]) raises -> handle.type as out:
    var ctx = handle._get_ctx[AsyncContext]()
    _init_asyncrt_chain(AsyncContext.get_chain(ctx))
    ctx[].callback = AsyncContext.complete
    handle._set_result_slot(
        __mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(out)),
        __mlir_op.`lit.ref.to_pointer`(
            __get_mvalue_as_litref(__get_nearest_error_slot())
        ),
    )
    _async_execute[handle.type](handle._handle, -1)
    _async_wait(AsyncContext.get_chain(ctx))
    _del_asyncrt_chain(AsyncContext.get_chain(ctx))
    if __mlir_op.`co.get_results`[_type = __mlir_type.i1](handle._handle):
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(__get_nearest_error_slot())
        )
        __mlir_op.`lit.raise`()
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(out))
    _ = handle^


# ===----------------------------------------------------------------------===#
# Task
# ===----------------------------------------------------------------------===#


struct Task[type: AnyType, lifetimes: LifetimeSet]:
    var _handle: Coroutine[type, lifetimes]
    var _result: type

    fn __init__(inout self, owned handle: Coroutine[type, lifetimes]):
        self._handle = handle^
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._result)
        )
        self._handle._set_result_slot(UnsafePointer.address_of(self._result))

    fn get(self) -> ref [__lifetime_of(self._result)] type:
        """Get the task's result value. Calling this on an incomplete task is
        undefined behaviour."""
        return self._result

    fn __del__(owned self):
        """Destroy the memory associated with a task. This must be manually
        called when a task goes out of scope.
        """
        var ctx = self._handle._get_ctx[AsyncContext]()
        var chainPtr: UnsafePointer[Chain] = AsyncContext.get_chain(ctx)
        _del_asyncrt_chain(chainPtr)
        _ = self._handle^

    @always_inline
    fn __await__(self) -> ref [__lifetime_of(self.get())] type:
        """Suspend the current async function until the task completes and its
        result becomes available. This function must be force inlined into the
        calling async function.
        """

        @always_inline
        @parameter
        fn await_body(cur_hdl: AnyCoroutine):
            _async_and_then(
                cur_hdl,
                AsyncContext.get_chain(self._handle._get_ctx[AsyncContext]()),
            )

        _suspend_async[await_body]()
        return self.get()

    fn wait(self) -> ref [__lifetime_of(self.get())] type:
        """Block the current thread until the future value becomes available."""
        _async_wait(
            AsyncContext.get_chain(self._handle._get_ctx[AsyncContext]())
        )
        return self.get()


# ===----------------------------------------------------------------------===#
# TaskGroup
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct TaskGroupContext[lifetimes: LifetimeSet]:
    alias tg_callback_fn_type = fn (inout TaskGroup[lifetimes]) -> None

    var callback: Self.tg_callback_fn_type
    var task_group: UnsafePointer[TaskGroup[lifetimes]]


@register_passable
struct _TaskGroupBox(CollectionElement):
    """This struct is a type-erased owning box for an opaque coroutine."""

    var handle: AnyCoroutine

    fn __init__[type: AnyType](inout self, owned coro: Coroutine[type]):
        var handle = coro._handle
        __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(coro))
        self.handle = handle

    fn __init__(inout self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    fn __del__(owned self):
        __mlir_op.`co.destroy`(self.handle)

    # FIXME(MSTDL-573): `List` requires copyability. Just crash here because it
    # should never get called.
    fn __copyinit__(inout self, existing: Self):
        abort("_TaskGroupBox.__copyinit__ should never get called")
        while True:
            pass


struct TaskGroup[lifetimes: LifetimeSet]:
    var counter: Atomic[DType.index]
    var chain: Chain
    var tasks: List[_TaskGroupBox]

    fn __init__(inout self):
        var chain = Chain()
        _init_asyncrt_chain(UnsafePointer[Chain].address_of(chain))
        self.counter = 1
        self.chain = chain
        self.tasks = List[_TaskGroupBox](capacity=16)

    fn __del__(owned self):
        _del_asyncrt_chain(UnsafePointer[Chain].address_of(self.chain))

    @always_inline
    fn _counter_decr(inout self) -> Int:
        var prev: Int = self.counter.fetch_sub(1).value
        return prev - 1

    @staticmethod
    fn _task_complete_callback(inout tg: TaskGroup[lifetimes]):
        tg._task_complete()

    fn _task_complete(inout self):
        if self._counter_decr() == 0:
            _async_complete(UnsafePointer[Chain].address_of(self.chain))

    fn create_task(
        inout self,
        # FIXME(MSTDL-722): Avoid accessing ._mlir_type here, use `NoneType`.
        owned task: Coroutine[NoneType._mlir_type],
    ):
        self._create_task(task^, desired_worker_id=-1)

    # Deprecated, use create_task() instead
    # Only sync_parallelize() uses this to pass desired_worker_id
    fn _create_task(
        inout self,
        # FIXME(MSTDL-722): Avoid accessing ._mlir_type here, use `NoneType`.
        owned task: Coroutine[NoneType._mlir_type],
        desired_worker_id: Int = -1,
    ):
        # TODO(MOCO-771): Enforce that `task.lifetimes` is a subset of
        # `Self.lifetimes`.
        self.counter += 1
        task._get_ctx[TaskGroupContext[lifetimes]]()[] = TaskGroupContext[
            lifetimes
        ] {
            callback: Self._task_complete_callback,
            task_group: UnsafePointer[Self].address_of(self),
        }
        _async_execute[task.type](task._handle, desired_worker_id)
        self.tasks.append(_TaskGroupBox(task^))

    @staticmethod
    fn await_body_impl(hdl: AnyCoroutine, inout task_group: Self):
        _async_and_then(hdl, UnsafePointer[Chain].address_of(task_group.chain))
        task_group._task_complete()

    @always_inline
    fn __await__(inout self):
        @always_inline
        @parameter
        fn await_body(cur_hdl: AnyCoroutine):
            Self.await_body_impl(cur_hdl, self)

        _suspend_async[await_body]()

    fn wait(inout self):
        self._task_complete()
        _async_wait(UnsafePointer[Chain].address_of(self.chain))


# ===----------------------------------------------------------------------===#
# MojoCallContext
# ===----------------------------------------------------------------------===#


@register_passable
struct MojoCallContextPtr:
    """A pointer to a C++ MojoCallContext struct, which is used by the Modular
    C++ runtime to coordinate execution with Mojo kernels.
    """

    # Actually a MojoCallContext*
    alias ptr_type = UnsafePointer[NoneType]
    var ptr: Self.ptr_type

    @always_inline
    fn __init__(inout self):
        self.ptr = UnsafePointer[NoneType]()

    @always_inline
    fn __init__(inout self, ptr: Self.ptr_type):
        """Casts a raw pointer to our MojoCallContextPtr."""
        self.ptr = ptr

    @always_inline
    fn complete(self):
        """Indicates to the C++ runtime that the async kernel has finished."""
        external_call[
            "KGEN_CompilerRT_AsyncRT_MojoCallContext_Complete", NoneType
        ](
            self.ptr,
        )

    @always_inline
    fn set_stream(self, stream: Stream):
        """Set the cuda stream."""
        external_call[
            "KGEN_CompilerRT_AsyncRT_MojoCallContext_SetCUStream",
            NoneType._mlir_type,
        ](self.ptr, stream.stream.handle)

    @always_inline
    fn set_context(self, context: CudaContext):
        """Get the cuda stream."""
        external_call[
            "KGEN_CompilerRT_AsyncRT_MojoCallContext_SetCUContext",
            NoneType._mlir_type,
        ](self.ptr, context.ctx.handle)

    @always_inline
    fn get_device_context(self) -> ref [ImmutableStaticLifetime] DeviceContext:
        """Get the device context held by the MojoCallContext.

        Note: it is safe to use ImmutableStaticLifetime here because get_device_context()
        is only used within kernels and the DeviceContext lifetime is managed by
        the graph compiler.
        """
        var ctx_ptr = external_call[
            "KGEN_CompilerRT_AsyncRT_MojoCallContext_GetDeviceContext",
            UnsafePointer[DeviceContext],
        ](
            self.ptr,
        )
        return ctx_ptr[]

    @always_inline
    fn set_to_error(self, err: Error):
        """Indicates to the C++ runtime that the kernel has failed."""
        var str = err.__str__()
        var strref = str._strref_dangerous()
        external_call[
            "KGEN_CompilerRT_AsyncRT_MojoCallContext_SetToError", NoneType
        ](self.ptr, strref.data, strref.length)
        str._strref_keepalive()

    fn alloc(self, byte_size: Int, alignment: Int) -> UnsafePointer[NoneType]:
        return external_call[
            "KGEN_CompilerRT_AsyncRT_MojoCallContext_Allocate",
            UnsafePointer[NoneType],
        ](self.ptr, byte_size, alignment)
