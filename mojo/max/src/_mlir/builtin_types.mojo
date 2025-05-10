# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import List

from ._c.ffi import MLIR_func
from .ir import Context, DialectType, Type


@value
struct FunctionType(DialectType):
    var ctx: Context
    var inputs: List[Type]
    var results: List[Type]

    fn __init__(inout self, ctx: Context):
        self.__init__(ctx, List[Type](), List[Type]())

    fn __init__(inout self, inputs: List[Type], results: List[Type]):
        debug_assert(
            len(inputs).__bool__() or len(results).__bool__(),
            "nullary functions must use the context constructor",
        )
        var ctx = (inputs if len(inputs) else results)[0].context()
        self.__init__(ctx, inputs, results)

    fn to_mlir(self) -> Type:
        return _c.BuiltinTypes.mlirFunctionTypeGet(
            self.ctx.c,
            len(self.inputs),
            Pointer[_c.IR.MlirType](address=int(self.inputs.data)),
            len(self.results),
            Pointer[_c.IR.MlirType](address=int(self.results.data)),
        )

    @staticmethod
    fn from_mlir(type: Type) raises -> Self:
        if not _c.BuiltinTypes.mlirTypeIsAFunction(type.c):
            raise "Type is not a Function"
        var inputs = List[Type]()
        var results = List[Type]()
        for i in range(_c.BuiltinTypes.mlirFunctionTypeGetNumInputs(type.c)):
            inputs.append(_c.BuiltinTypes.mlirFunctionTypeGetInput(type.c, i))
        for i in range(_c.BuiltinTypes.mlirFunctionTypeGetNumResults(type.c)):
            results.append(_c.BuiltinTypes.mlirFunctionTypeGetResult(type.c, i))
        return Self(type.context(), inputs, results)
