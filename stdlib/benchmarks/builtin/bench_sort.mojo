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

from benchmark import Bench, Bencher, BenchId, keep, BenchConfig, Unit, run
from random import *
from stdlib.builtin.sort import sort, _small_sort, _insertion_sort

# ===----------------------------------------------------------------------===#
# Benchmark Utils
# ===----------------------------------------------------------------------===#

alias dtypes = List(
    DType.uint8,
    DType.int8,
    DType.uint16,
    DType.int16,
    DType.float16,
    DType.uint32,
    DType.int32,
    DType.float32,
    DType.uint64,
    DType.int64,
    DType.float64,
)


fn random_scalar_list[
    dt: DType
](size: Int, max: Scalar[dt] = Scalar[dt].MAX) -> List[Scalar[dt]]:
    var result = List[Scalar[dt]](capacity=size)
    for _ in range(size):

        @parameter
        if dt.is_integral() and dt.is_signed():
            result.append(random_si64(0, max.cast[DType.int64]()).cast[dt]())
        elif dt.is_integral() and dt.is_unsigned():
            result.append(random_ui64(0, max.cast[DType.uint64]()).cast[dt]())
        else:
            var res = random_float64(0, max.cast[DType.float64]())
            # GCC doesn't support cast from float64 to float16
            result.append(res.cast[DType.float32]().cast[dt]())
    return result


# ===----------------------------------------------------------------------===#
# Benchmark sort functions with a tiny list size
# ===----------------------------------------------------------------------===#


@parameter
fn bench_tiny_list_sort(inout m: Bench) raises:
    alias counts = List(2, 3, 4, 5)

    @parameter
    for type_index in range(len(dtypes)):
        alias dt = dtypes[type_index]

        @parameter
        for count_index in range(len(counts)):
            alias count = counts[count_index]
            var list = random_scalar_list[dt](count)

            @parameter
            fn bench_sort_list(inout b: Bencher) raises:
                @always_inline
                @parameter
                fn call_fn():
                    var l1 = list
                    sort(l1)

                b.iter[call_fn]()

            @parameter
            fn bench_small_sort(inout b: Bencher) raises:
                @always_inline
                @parameter
                fn call_fn():
                    var l1 = list
                    var small_p = rebind[Pointer[Scalar[dt]]](l1.data)

                    @parameter
                    fn _less_than_equal[
                        ty: AnyTrivialRegType
                    ](lhs: ty, rhs: ty) -> Bool:
                        return rebind[Scalar[dt]](lhs) <= rebind[Scalar[dt]](
                            rhs
                        )

                    _small_sort[4, Scalar[dt], _less_than_equal](small_p)
                    _ = l1

                b.iter[call_fn]()

            @parameter
            fn bench_insertion_sort(inout b: Bencher) raises:
                @always_inline
                @parameter
                fn call_fn():
                    var l1 = list
                    var small_p = rebind[Pointer[Scalar[dt]]](l1.data)

                    @parameter
                    fn _less_than_equal[
                        ty: AnyTrivialRegType
                    ](lhs: ty, rhs: ty) -> Bool:
                        return rebind[Scalar[dt]](lhs) <= rebind[Scalar[dt]](
                            rhs
                        )

                    _insertion_sort[Scalar[dt], _less_than_equal](
                        small_p, 0, count
                    )
                    _ = l1

                b.iter[call_fn]()

            m.bench_function[bench_sort_list](
                BenchId(
                    "bench_std_sort_random_list_"
                    + str(count)
                    + "_type_"
                    + str(dt)
                )
            )
            m.bench_function[bench_small_sort](
                BenchId(
                    "bench_sml_sort_random_list_"
                    + str(count)
                    + "_type_"
                    + str(dt)
                )
            )
            m.bench_function[bench_insertion_sort](
                BenchId(
                    "bench_ins_sort_random_list_"
                    + str(count)
                    + "_type_"
                    + str(dt)
                )
            )
            _ = list^


# ===----------------------------------------------------------------------===#
# Benchmark sort functions with a small list size
# ===----------------------------------------------------------------------===#


@parameter
fn bench_small_list_sort(inout m: Bench) raises:
    alias counts = List(10, 20, 32, 64, 100)

    @parameter
    for type_index in range(len(dtypes)):
        alias dt = dtypes[type_index]

        @parameter
        for count_index in range(len(counts)):
            alias count = counts[count_index]
            var list = random_scalar_list[dt](count)

            @parameter
            fn bench_sort_list(inout b: Bencher) raises:
                @always_inline
                @parameter
                fn call_fn():
                    var l1 = list
                    sort(l1)

                b.iter[call_fn]()

            @parameter
            fn bench_insertion_sort(inout b: Bencher) raises:
                @always_inline
                @parameter
                fn call_fn():
                    var l1 = list
                    var small_p = rebind[Pointer[Scalar[dt]]](l1.data)

                    @parameter
                    fn _less_than_equal[
                        ty: AnyTrivialRegType
                    ](lhs: ty, rhs: ty) -> Bool:
                        return rebind[Scalar[dt]](lhs) <= rebind[Scalar[dt]](
                            rhs
                        )

                    _insertion_sort[Scalar[dt], _less_than_equal](
                        small_p, 0, count
                    )
                    _ = l1

                b.iter[call_fn]()

            m.bench_function[bench_sort_list](
                BenchId(
                    "bench_std_sort_random_list_"
                    + str(count)
                    + "_type_"
                    + str(dt)
                )
            )
            m.bench_function[bench_insertion_sort](
                BenchId(
                    "bench_ins_sort_random_list_"
                    + str(count)
                    + "_type_"
                    + str(dt)
                )
            )
            _ = list^


# ===----------------------------------------------------------------------===#
# Benchmark Main
# ===----------------------------------------------------------------------===#


def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=3, warmup_iters=100))

    bench_tiny_list_sort(m)
    bench_small_list_sort(m)

    m.dump_report()
