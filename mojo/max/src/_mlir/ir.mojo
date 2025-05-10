# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from collections.optional import Optional

from utils.variant import Variant

import ._c
import ._c.IR
from ._c.ffi import MLIR_func
from .diagnostics import (
    Diagnostic,
    DiagnosticHandler,
    DiagnosticHandlerID,
    DiagnosticSeverity,
    ErrorCapturingDiagnosticHandler,
)

# Ownership:
#
#   See https://mlir.llvm.org/docs/Bindings/Python/#ownership-in-the-core-ir
#   for full ownership semantics. We'll attempt to follow the same patterns
#   for consistency.
#
# - Context owns most things, we don't need to memory manage them
# - Most things are value-semantic and are actually implicitly mutable references
#   with the same lifetime as the owning Context
# - Context ownership is therefore the main thing people might need to consider
# - Exceptions:
#   - When objects are created without a context, they're often "owned" objects
#     _until_ they've been added to another context-owned object.


trait DialectType:
    fn to_mlir(self) -> Type:
        ...

    @staticmethod
    fn from_mlir(type: Type) raises -> Self:
        ...


trait DialectAttribute:
    fn to_mlir(self) -> Attribute:
        ...

    @staticmethod
    fn from_mlir(attr: Attribute) raises -> Self:
        ...


struct DialectRegistry:
    alias cType = _c.IR.MlirDialectRegistry
    var c: Self.cType

    fn __init__(inout self):
        self.c = _c.IR.mlirDialectRegistryCreate()

    fn __del__(owned self):
        # We only want to do this for objects which are not added to a context
        pass  # _c.IR.mlirDialectRegistryDestroy(self.c)

    fn insert(inout self, handle: DialectHandle):
        _c.IR.mlirDialectHandleInsertDialect(handle.c, self.c)

    fn load_modular_dialects(self):
        MLIR_func["MAXG_loadModularDialects", fn (Self.cType) -> NoneType]()(
            self.c
        )


@value
@register_passable("trivial")
struct Dialect(CollectionElement):
    alias cType = _c.IR.MlirDialect
    var c: Self.cType

    fn context(self) -> Context:
        return _c.IR.mlirDialectGetContext(self.c)

    fn __eq__(self, other: Self) -> Bool:
        return _c.IR.mlirDialectEqual(self.c, other.c)

    fn namespace(self) -> StringRef:
        return _c.IR.mlirDialectGetNamespace(self.c)


@value
@register_passable("trivial")
struct DialectHandle(CollectionElement):
    alias cType = _c.IR.MlirDialectHandle
    var c: Self.cType

    fn namespace(self) -> StringRef:
        return _c.IR.mlirDialectHandleGetNamespace(self.c)


@value
@register_passable("trivial")
struct Context:
    alias cType = _c.IR.MlirContext
    var c: Self.cType

    fn __init__(inout self):
        self.c = _c.IR.mlirContextCreateWithThreading(False)

    fn __init__(inout self, threading_enabled: Bool):
        self.c = _c.IR.mlirContextCreateWithThreading(threading_enabled)

    fn __init__(
        inout self, owned registry: DialectRegistry, threading_enabled: Bool
    ):
        self.c = _c.IR.mlirContextCreateWithRegistry(
            registry.c, threading_enabled
        )

    fn __enter__(inout self) -> Self:
        return self

    fn __exit__(inout self):
        _c.IR.mlirContextDestroy(self.c)

    fn __exit__(self, err: Error) -> Bool:
        _c.IR.mlirContextDestroy(self.c)
        return False

    fn __eq__(self, other: Self) -> Bool:
        return _c.IR.mlirContextEqual(self.c, other.c)

    fn append(inout self, owned registry: DialectRegistry):
        return _c.IR.mlirContextAppendDialectRegistry(self.c, registry.c)

    fn register(inout self, handle: DialectHandle):
        _c.IR.mlirDialectHandleRegisterDialect(handle.c, self.c)

    fn load(self, handle: DialectHandle) -> Dialect:
        return _c.IR.mlirDialectHandleLoadDialect(handle.c, self.c)

    fn load_modular_dialects(inout self):
        var registry = DialectRegistry()
        registry.load_modular_dialects()
        self.append(registry^)

    fn allow_unregistered_dialects(inout self, allow: Bool = True):
        _c.IR.mlirContextSetAllowUnregisteredDialects(self.c, allow)

    fn allows_unregistered_dialects(self) -> Bool:
        return _c.IR.mlirContextGetAllowUnregisteredDialects(self.c)

    fn num_registered_dialects(self) -> Int:
        return _c.IR.mlirContextGetNumRegisteredDialects(self.c)

    fn num_loaded_dialects(self) -> Int:
        return _c.IR.mlirContextGetNumLoadedDialects(self.c)

    fn get_or_load_dialect(self, dialect_name: String) -> Optional[Dialect]:
        var result = _c.IR.mlirContextGetOrLoadDialect(
            self.c, dialect_name._strref_dangerous()
        )
        dialect_name._strref_keepalive()
        return Optional(Dialect(result)) if result.ptr else None

    fn enable_multithreading(inout self, enable: Bool = True):
        _c.IR.mlirContextEnableMultithreading(self.c, enable)

    fn load_all_available_dialects(inout self):
        _c.IR.mlirContextLoadAllAvailableDialects(self.c)

    fn is_registered_operation(self, opname: String) -> Bool:
        var result = _c.IR.mlirContextIsRegisteredOperation(
            self.c, opname._strref_dangerous()
        )
        opname._strref_keepalive()
        return result

    fn print_on_diagnostic(self):
        fn print_diagnostic(diagnostic: Diagnostic) -> Bool:
            print(diagnostic)
            return False

        # For now no way to detach
        _ = DiagnosticHandler[print_diagnostic].attach(self)

    fn diagnostic_error(self) -> ErrorCapturingDiagnosticHandler:
        """Uses a DiagnosticHandler to capture errors from MLIR.
        If the handler catches an error, it will re-raise with the error
        message provided by MLIR.
        This will drop any information in the original error message.
        """
        return ErrorCapturingDiagnosticHandler(self)

    # TODO: mlirContextSetThreadPool


@value
@register_passable("trivial")
struct Location(CollectionElement, Stringable):
    alias cType = _c.IR.MlirLocation
    var c: Self.cType

    fn __init__(
        inout self, ctx: Context, filename: String, line: Int, col: Int
    ):
        self.c = _c.IR.mlirLocationFileLineColGet(
            ctx.c, filename._strref_dangerous(), line, col
        )
        filename._strref_keepalive()

    @staticmethod
    fn from_attribute(attr: Attribute) -> Self:
        return Self(_c.IR.mlirLocationFromAttribute(attr.c))

    @staticmethod
    fn call_site(callee: Self, caller: Self) -> Self:
        return Self(_c.IR.mlirLocationCallSiteGet(callee.c, caller.c))

    # TODO: locationFusedGet, locationNameGet

    @staticmethod
    fn unknown(ctx: Context) -> Self:
        return Self(_c.IR.mlirLocationUnknownGet(ctx.c))

    fn attribute(self) -> Attribute:
        return Attribute(_c.IR.mlirLocationGetAttribute(self.c))

    fn context(self) -> Context:
        return _c.IR.mlirLocationGetContext(self.c)

    fn __eq__(self, other: Self) -> Bool:
        return _c.IR.mlirLocationEqual(self.c, other.c)

    fn __str__(self) -> String:
        return _to_string[Self.cType, _c.IR.mlirLocationPrint](self.c)


@value
@register_passable("trivial")
struct Module(Stringable):
    alias cType = _c.IR.MlirModule
    var c: Self.cType

    fn __init__(inout self, location: Location):
        self.c = _c.IR.mlirModuleCreateEmpty(location.c)

    # TODO: The lifetime of module appears to be iffy in the current codebase.
    # For now, this is manually called when known to be safe to prevent ASAN
    # from complaining for certain tests.
    fn destroy(owned self):
        _c.IR.mlirModuleDestroy(self.c)

    @staticmethod
    fn parse(ctx: Context, module: String) -> Self:
        # TODO: how can this fail?
        var c = _c.IR.mlirModuleCreateParse(ctx.c, module._strref_dangerous())
        module._strref_keepalive()
        return Self(c)

    @staticmethod
    fn from_op(module_op: Operation) raises -> Self:
        var module = _c.IR.mlirModuleFromOperation(module_op.c)
        if not module.ptr:
            raise "Op must be a ModuleOp"
        return module

    fn context(self) -> Context:
        return _c.IR.mlirModuleGetContext(self.c)

    fn body(self) -> Block:
        return Block(_c.IR.mlirModuleGetBody(self.c))

    fn as_op(self) -> Operation:
        return Operation(_c.IR.mlirModuleGetOperation(self.c))

    fn __str__(self) -> String:
        return str(self.as_op())


# Helper class with a bunch of implicit conversions for things that go on
# Operations.
struct _OpBuilderList[T: CollectionElement]:
    var elements: List[T]

    fn __init__(inout self):
        self.elements = List[T]()

    fn __init__(inout self, empty: ListLiteral[]):
        self.elements = List[T]()

    fn __init__(inout self, owned elements: List[T]):
        self.elements = elements^

    fn __init__(inout self, element: T):
        self.elements = List[T]()
        self.elements.append(element)

    fn __bool__(self) -> Bool:
        return len(self.elements).__bool__()


@value
struct NamedAttribute(CollectionElement):
    alias cType = _c.IR.MlirNamedAttribute
    var name: Identifier
    var attr: Attribute

    fn __init__(inout self, attr: Self.cType):
        self.name = Identifier(attr.name)
        self.attr = Attribute(attr.attribute)

    fn c(self) -> Self.cType:
        return Self.cType {name: self.name.c, attribute: self.attr.c}

    # TODO: tuple init so we can write these a bit less verbosely.


@value
struct _WriteState:
    var handle: UnsafePointer[FileHandle]
    var errors: List[String]


# TODO: how to correctly destroy "owned" Operations?
@value
@register_passable("trivial")
struct Operation(CollectionElement, Stringable):
    alias cType = _c.IR.MlirOperation
    var c: Self.cType

    fn __init__(inout self, op: Self.cType):
        self.c = op

    fn __init__(
        inout self,
        name: String,
        location: Location,
        *,
        attributes: _OpBuilderList[NamedAttribute] = [],
        operands: _OpBuilderList[Value] = [],
        results: _OpBuilderList[Type] = [],
        regions: _OpBuilderList[Region] = [],
        successors: _OpBuilderList[Block] = [],
    ):
        var state = _c.IR.mlirOperationStateGet(
            name._strref_dangerous(), location.c
        )
        Self._init_op_state(
            state,
            attributes.elements,
            operands.elements,
            results.elements,
            regions.elements,
            successors.elements,
        )
        name._strref_keepalive()
        self.c = _c.IR.mlirOperationCreate(Pointer.address_of(state))

    fn __init__(
        inout self,
        name: String,
        location: Location,
        *,
        enable_result_type_inference: Bool,
        attributes: _OpBuilderList[NamedAttribute] = [],
        operands: _OpBuilderList[Value] = [],
        results: _OpBuilderList[Type] = [],
        regions: _OpBuilderList[Region] = [],
        successors: _OpBuilderList[Block] = [],
    ) raises:
        var state = _c.IR.mlirOperationStateGet(
            name._strref_dangerous(), location.c
        )
        Self._init_op_state(
            state,
            attributes.elements,
            operands.elements,
            results.elements,
            regions.elements,
            successors.elements,
        )
        if enable_result_type_inference:
            _c.IR.mlirOperationStateEnableResultTypeInference(
                Pointer.address_of(state)
            )

        var result: Self.cType
        with location.context().diagnostic_error():
            result = _c.IR.mlirOperationCreate(Pointer.address_of(state))
            if not result.ptr:
                raise "operation create failed"

        self.c = result

        name._strref_keepalive()

    @staticmethod
    fn _init_op_state(
        ref [_]state: _c.IR.MlirOperationState,
        attributes: List[NamedAttribute],
        operands: List[Value],
        results: List[Type],
        regions: List[Region],
        successors: List[Block],
    ):
        if attributes:
            _c.IR.mlirOperationStateAddAttributes(
                UnsafePointer.address_of(state),
                len(attributes),
                # This technically works as long as `Attribute` is only `MlirAttribute`.
                attributes.data.bitcast[NamedAttribute.cType](),
            )
        if operands:
            _c.IR.mlirOperationStateAddOperands(
                UnsafePointer.address_of(state),
                len(operands),
                operands.data.bitcast[Value.cType](),
            )
        if results:
            _c.IR.mlirOperationStateAddResults(
                UnsafePointer.address_of(state),
                len(results),
                results.data.bitcast[Type.cType](),
            )
        # TODO: how to express to the caller that we're taking ownership
        #       over Regions.
        if regions:
            _c.IR.mlirOperationStateAddOwnedRegions(
                UnsafePointer.address_of(state),
                len(regions),
                regions.data.bitcast[Region.cType](),
            )
        if successors:
            _c.IR.mlirOperationStateAddSuccessors(
                UnsafePointer.address_of(state),
                len(successors),
                successors.data.bitcast[Block.cType](),
            )

    @staticmethod
    fn parse(ctx: Context, source: String, source_name: String) raises -> Self:
        var result = _c.IR.mlirOperationCreateParse(
            ctx.c, source._strref_dangerous(), source_name._strref_dangerous()
        )
        source._strref_keepalive()
        source_name._strref_keepalive()
        if not result.ptr:
            raise "Operation.parse failed"
        return Self(result)

    fn destroy(owned self):
        _c.IR.mlirOperationDestroy(self.c)

    fn context(self) -> Context:
        return _c.IR.mlirOperationGetContext(self.c)

    fn verify(self) -> Bool:
        return _c.IR.mlirOperationVerify(self.c)

    fn write(self, file: FileHandle, version: Optional[Int64] = None) raises:
        var config = _c.IR.mlirBytecodeWriterConfigCreate()
        if version:
            _c.IR.mlirBytecodeWriterConfigDesiredEmitVersion(
                config, version.value()
            )

        var write_state = _WriteState(
            handle=UnsafePointer.address_of(file),
            errors=List[String](),
        )

        fn callback(buf: StringRef, _data: UnsafePointer[NoneType]):
            var state = _data.bitcast[_WriteState]()[]

            try:
                state.handle[].write(buf)
            except e:
                state.errors.append(str(e))

        var result = _c.IR.mlirOperationWriteBytecodeWithConfig(
            self.c,
            config=config,
            callback=callback,
            user_data=UnsafePointer.address_of(write_state).bitcast[NoneType](),
        )
        _c.IR.mlirBytecodeWriterConfigDestroy(config)

        if result.value == 0:
            raise "Writing op bytecode to file failed"

        if len(write_state.errors):
            # Only report the first error
            var msg = write_state.errors[0]
            raise "One or more errors while writing op to bytecode: " + msg

    fn bytecode(self, version: Optional[Int64] = None) raises -> List[Int8]:
        var config = _c.IR.mlirBytecodeWriterConfigCreate()
        if version:
            _c.IR.mlirBytecodeWriterConfigDesiredEmitVersion(
                config, version.value()
            )

        var data = List[Int8]()

        fn callback(buf: StringRef, _data: UnsafePointer[NoneType]):
            for i in range(len(buf)):
                _data.bitcast[List[UInt8]]()[].append(buf.data[i])

        var result = _c.IR.mlirOperationWriteBytecodeWithConfig(
            self.c,
            config=config,
            callback=callback,
            user_data=UnsafePointer.address_of(data).bitcast[NoneType](),
        )
        _c.IR.mlirBytecodeWriterConfigDestroy(config)

        if result.value == 0:
            raise "Bytecode conversion failed"

        # Add a trailing 0 for default-conversion to a string
        data.append(0)

        return data

    fn name(self) -> Identifier:
        return _c.IR.mlirOperationGetName(self.c)

    fn block(self) -> Block:
        return _c.IR.mlirOperationGetBlock(self.c)

    fn parent(self) -> Self:
        return _c.IR.mlirOperationGetParentOperation(self.c)

    fn successor(self, successor_idx: Int) raises -> Block:
        var block = _c.IR.mlirOperationGetSuccessor(self.c, successor_idx)
        if not block.ptr:
            raise "IndexError"
        return block

    fn region(self, region_idx: Int) raises -> Region:
        var region = _c.IR.mlirOperationGetRegion(self.c, region_idx)
        if not region.ptr:
            raise "IndexError"
        return region

    fn num_results(self) -> Int:
        return _c.IR.mlirOperationGetNumResults(self.c)

    fn result(self, idx: Int) -> Value:
        return _c.IR.mlirOperationGetResult(self.c, idx)

    fn num_operands(self) -> Int:
        return _c.IR.mlirOperationGetNumOperands(self.c)

    fn operand(self, idx: Int) -> Value:
        return _c.IR.mlirOperationGetOperand(self.c, idx)

    fn set_inherent_attr(inout self, name: String, attr: Attribute):
        _c.IR.mlirOperationSetInherentAttributeByName(
            self.c, name._strref_dangerous(), attr.c
        )
        name._strref_keepalive()

    fn get_inherent_attr(self, name: String) -> Attribute:
        var result = _c.IR.mlirOperationGetInherentAttributeByName(
            self.c, name._strref_dangerous()
        )
        name._strref_keepalive()
        return result

    fn set_discardable_attr(inout self, name: String, attr: Attribute):
        _c.IR.mlirOperationSetDiscardableAttributeByName(
            self.c, name._strref_dangerous(), attr.c
        )
        name._strref_keepalive()

    fn get_discardable_attr(self, name: String) -> Attribute:
        var result = _c.IR.mlirOperationGetDiscardableAttributeByName(
            self.c, name._strref_dangerous()
        )
        name._strref_keepalive()
        return result

    fn __str__(self) -> String:
        return _to_string[Self.cType, _c.IR.mlirOperationPrint](self.c)


@value
@register_passable("trivial")
struct Identifier(CollectionElement, Stringable):
    alias cType = _c.IR.MlirIdentifier
    var c: Self.cType

    fn __init__(inout self, ctx: Context, identifier: String):
        self.c = _c.IR.mlirIdentifierGet(ctx.c, identifier._strref_dangerous())
        identifier._strref_keepalive()

    fn __str__(self) -> String:
        return _c.IR.mlirIdentifierStr(self.c)


@value
@register_passable("trivial")
struct Type(CollectionElement, Stringable):
    alias cType = _c.IR.MlirType
    var c: Self.cType

    fn __init__[T: DialectType](inout self, type: T):
        self = type.to_mlir()

    @staticmethod
    fn parse(ctx: Context, s: String) raises -> Self:
        var result = _c.IR.mlirTypeParseGet(ctx.c, s._strref_dangerous())
        if not result.ptr:
            raise "Failed to parse type: " + s
        s._strref_keepalive()
        return result

    fn context(self) -> Context:
        return _c.IR.mlirTypeGetContext(self.c)

    fn __str__(self) -> String:
        return _to_string[Self.cType, _c.IR.mlirTypePrint](self.c)


@value
@register_passable("trivial")
struct Value(CollectionElement, Stringable):
    alias cType = _c.IR.MlirValue
    var c: Self.cType

    fn type(self) -> Type:
        return _c.IR.mlirValueGetType(self.c)

    fn context(self) -> Context:
        return self.type().context()

    fn parent(self) -> Variant[Block, Operation]:
        if self.is_block_argument():
            return self._block()
        else:
            debug_assert(self.is_op_result(), "Invalid Value state")
            return self._defining_op()

    fn is_block_argument(self) -> Bool:
        return _c.IR.mlirValueIsABlockArgument(self.c)

    fn is_op_result(self) -> Bool:
        return _c.IR.mlirValueIsAOpResult(self.c)

    fn set_type(inout self, type: Type):
        return _c.IR.mlirValueSetType(self.c, type.c)

    fn _block(self) -> Block:
        return _c.IR.mlirBlockArgumentGetOwner(self.c)

    fn _defining_op(self) -> Operation:
        return _c.IR.mlirOpResultGetOwner(self.c)

    fn replace_all_uses_with(self, other: Self):
        _c.IR.mlirValueReplaceAllUsesOfWith(of=self.c, `with`=other.c)

    fn __str__(self) -> String:
        return _to_string[Self.cType, _c.IR.mlirValuePrint](self.c)


@value
@register_passable("trivial")
struct Attribute(CollectionElement, Stringable):
    alias cType = _c.IR.MlirAttribute
    var c: Self.cType

    fn __init__[T: DialectAttribute](inout self, attr: T):
        self = attr.to_mlir()

    fn context(self) -> Context:
        return _c.IR.mlirAttributeGetContext(self.c)

    @staticmethod
    fn parse(ctx: Context, attr: String) raises -> Self:
        var result = _c.IR.mlirAttributeParseGet(
            ctx.c, attr._strref_dangerous()
        )
        if not result.ptr:
            raise "Failed to parse attribute:" + attr
        attr._strref_keepalive()
        return result

    fn __str__(self) -> String:
        return _to_string[Self.cType, _c.IR.mlirAttributePrint](self.c)


@value
@register_passable("trivial")
struct Block(CollectionElement, Stringable):
    alias cType = _c.IR.MlirBlock
    var c: Self.cType

    fn __init__(inout self, args: List[Type]):
        var locations = List[Location]()
        for i in range(len(args)):
            var ctx = args[i].context()
            locations.append(Location.unknown(ctx))
        self.__init__(args, locations)

    fn __init__(
        inout self,
        args: List[Type],
        locations: List[Location],
    ):
        debug_assert(
            len(args) == len(locations), "Each arg must have a location"
        )
        self.c = _c.IR.mlirBlockCreate(
            len(args),
            Pointer[Type.cType](address=int(args.data)),
            Pointer[Location.cType](address=int(locations.data)),
        )
        _ = args
        _ = locations

    fn region(self) -> Region:
        return _c.IR.mlirBlockGetParentRegion(self.c)

    fn parent(self) -> Operation:
        return _c.IR.mlirBlockGetParentOperation(self.c)

    fn num_arguments(self) -> Int:
        return _c.IR.mlirBlockGetNumArguments(self.c)

    fn argument(self, idx: Int) -> Value:
        return _c.IR.mlirBlockGetArgument(self.c, idx)

    fn append(self, op: Operation):
        return _c.IR.mlirBlockAppendOwnedOperation(self.c, op.c)

    fn insert_before(self, reference: Operation, op: Operation):
        return _c.IR.mlirBlockInsertOwnedOperationBefore(
            self.c, reference.c, op.c
        )

    fn insert_after(self, reference: Operation, op: Operation):
        return _c.IR.mlirBlockInsertOwnedOperationAfter(
            self.c, reference.c, op.c
        )

    fn terminator(self) -> Optional[Operation]:
        var op = _c.IR.mlirBlockGetTerminator(self.c)
        return Optional(Operation(op)) if op.ptr else None

    fn __str__(self) -> String:
        return _to_string[Self.cType, _c.IR.mlirBlockPrint](self.c)


@value
@register_passable("trivial")
struct Region(CollectionElement):
    alias cType = _c.IR.MlirRegion
    var c: Self.cType

    fn __init__(inout self):
        self.c = _c.IR.mlirRegionCreate()

    fn append(self, block: Block):
        _c.IR.mlirRegionAppendOwnedBlock(self.c, block.c)

    fn first_block(self) raises -> Block:
        var block = _c.IR.mlirRegionGetFirstBlock(self.c)
        if not block.ptr:
            raise "Region has no block"
        return block


alias _ToStringCallback = fn (StringRef, UnsafePointer[String]) -> NoneType


fn _to_string_callback(chunk: StringRef, data: UnsafePointer[NoneType]):
    var data_ = data.bitcast[String]()
    for i in range(chunk.length):
        data_[]._buffer.append(chunk.unsafe_ptr()[i])


@always_inline
fn _to_string[
    T: AnyTrivialRegType,
    call: fn (
        T, _c.Support.MlirStringCallback, UnsafePointer[NoneType]
    ) -> NoneType,
](t: T) -> String:
    var string = String()
    call(
        t,
        _to_string_callback,
        UnsafePointer.address_of(string).bitcast[NoneType](),
    )
    string._buffer.append(0)  # null terminate
    return string
