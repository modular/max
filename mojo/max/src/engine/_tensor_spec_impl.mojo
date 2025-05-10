# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from ._utils import *
from ._status import *
from tensor import TensorSpec
from ._dtypes import EngineDType
from collections.vector import DynamicVector
from .session import InferenceSession


@value
@register_passable("trivial")
struct CTensorSpec:
    """Mojo representation of Engine's TensorSpec pointer.
    This doesn't free the memory on destruction.
    """

    var ptr: DTypePointer[DType.invalid]

    alias FreeTensorSpecFnName = "M_freeTensorSpec"
    alias GetDimAtFnName = "M_getDimAt"
    alias GetRankFnName = "M_getRank"
    alias GetNameFnName = "M_getName"
    alias GetDTypeFnName = "M_getDtype"
    alias IsDynamicallyRankedFnName = "M_isDynamicRanked"

    fn get_dim_at(self, idx: Int, lib: DLHandle) -> Int:
        return call_dylib_func[Int](lib, Self.GetDimAtFnName, self, idx)

    fn get_rank(self, lib: DLHandle) -> Int:
        return call_dylib_func[Int](lib, Self.GetRankFnName, self)

    fn get_name(self, lib: DLHandle) -> String:
        let name = call_dylib_func[CString](lib, Self.GetNameFnName, self)
        return name.__str__()

    fn get_dtype(self, lib: DLHandle) -> EngineDType:
        return call_dylib_func[EngineDType](lib, Self.GetDTypeFnName, self)

    fn is_dynamically_ranked(self, lib: DLHandle) -> Bool:
        let is_dynamic = call_dylib_func[Int](
            lib, Self.IsDynamicallyRankedFnName, self
        )
        return is_dynamic == 1

    fn free(self, borrowed lib: DLHandle):
        call_dylib_func(lib, Self.FreeTensorSpecFnName, self)
