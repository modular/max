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
# RUN: %mojo %s -t

from random import *

from benchmark import Bench, BenchConfig, Bencher, BenchId, Unit, keep, run
from memory.memory import sizeof
from bit import bit_ceil
from stdlib.collections.dict import Dict, DictEntry


# ===----------------------------------------------------------------------===#
# Benchmark Data
# ===----------------------------------------------------------------------===#
fn make_dict[size: Int]() -> Dict[Int, Int]:
    var d = Dict[Int, Int]()
    for i in range(0, size):
        d[i] = random.random_si64(0, size).value
    return d


# ===----------------------------------------------------------------------===#
# Benchmark Dict init
# ===----------------------------------------------------------------------===#
@parameter
fn bench_dict_init(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn():
        for _ in range(1000):
            var _d: Dict[Int, Int] = Dict[Int, Int]()
            keep(_d._entries.data)
            keep(_d._index.data)

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark Dict Insert
# ===----------------------------------------------------------------------===#
@parameter
fn bench_dict_insert[size: Int](inout b: Bencher) raises:
    var items = make_dict[size]()

    @always_inline
    @parameter
    fn call_fn() raises:
        for key in range(size, (3 * size) // 2):
            items[key] = random.random_si64(0, size).value

    b.iter[call_fn]()
    keep(bool(items))


# ===----------------------------------------------------------------------===#
# Benchmark Dict Lookup
# ===----------------------------------------------------------------------===#
@parameter
fn bench_dict_lookup[size: Int](inout b: Bencher) raises:
    var items = make_dict[size]()

    @always_inline
    @parameter
    fn call_fn() raises:
        for key in range(0, size // 4):
            var res = items[key]
            keep(res)

    b.iter[call_fn]()
    keep(bool(items))


# ===----------------------------------------------------------------------===#
# Benchmark Dict Memory Footprint
# ===----------------------------------------------------------------------===#


fn total_bytes_used(items: Dict[Int, Int]) -> Int:
    # the allocated memory by entries:
    var entry_size = sizeof[Optional[DictEntry[Int, Int]]]()
    var amnt_bytes = items._entries.capacity * entry_size
    amnt_bytes += sizeof[Dict[Int, Int]]() - entry_size

    # the allocated memory by index table:
    var reserved = bit_ceil(len(items))
    if (len(items) * 3) > (reserved * 2):  # it resizes up if overloaded
        reserved *= 2

    if reserved <= 128:
        amnt_bytes += sizeof[Scalar[DType.int8]]() * reserved
    elif reserved <= 2**16 - 2:
        amnt_bytes += sizeof[Scalar[DType.int16]]() * reserved
    elif reserved <= 2**32 - 2:
        amnt_bytes += sizeof[Scalar[DType.int32]]() * reserved
    else:
        amnt_bytes += sizeof[Scalar[DType.int64]]() * reserved

    return amnt_bytes


# ===----------------------------------------------------------------------===#
# Benchmark Main
# ===----------------------------------------------------------------------===#
def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=1, warmup_iters=100))
    m.bench_function[bench_dict_init](BenchId("bench_dict_init"))
    alias sizes = (
        10,
        50,
        100,
        500,
        1000,
        5000,
        10_000,
        50_000,
        100_000,
        300_000,
        400_000,
        500_000,
        600_000,
        700_000,
        800_000,
        900_000,
        1_000_000,
    )

    @parameter
    for i in range(len(sizes)):
        alias size = sizes.get[i, Int]()
        m.bench_function[bench_dict_insert[size]](
            BenchId("bench_dict_insert[" + str(size) + "]")
        )
        m.bench_function[bench_dict_lookup[size]](
            BenchId("bench_dict_lookup[" + str(size) + "]")
        )

    m.dump_report()

    @parameter
    for i in range(len(sizes)):
        alias size = sizes.get[i, Int]()
        var mem_s = total_bytes_used(make_dict[size]())
        print(
            '"bench_dict_memory_size[' + str(size) + ']",' + str(mem_s) + ",0"
        )
