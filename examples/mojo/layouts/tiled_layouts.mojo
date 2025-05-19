# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from layout import IntTuple, Layout, print_layout
from layout.layout import blocked_product, make_ordered_layout, tile_to_shape


fn use_layout_constructor():
    print("layout constructor")
    var tiled_layout = Layout(
        IntTuple(IntTuple(3, 2), IntTuple(2, 5)),  # shape
        IntTuple(IntTuple(1, 6), IntTuple(3, 12)),  # strides
    )
    print_layout(tiled_layout)
    print()


fn use_tile_to_shape():
    print("tile to shape")
    var tts = tile_to_shape(Layout.col_major(3, 2), IntTuple(6, 10))
    print_layout(tts)
    print()


fn use_blocked_product():
    print("blocked product")
    # Define 2x3 tile
    var tile = Layout.col_major(3, 2)
    # Define a 2x5 tiler
    var tiler = Layout.col_major(2, 5)
    var blocked = blocked_product(tile, tiler)

    print("Tile:")
    print_layout(tile)
    print("\nTiler:")
    print_layout(tiler)
    print("\nTiled layout:")
    print(blocked)
    print()


fn use_make_ordered_layout():
    print("make ordered layout")
    var ordered = make_ordered_layout(
        IntTuple(IntTuple(3, 2), IntTuple(2, 5)),  # shape
        IntTuple(IntTuple(0, 2), IntTuple(1, 3)),  # order
    )
    print(ordered)


def main():
    use_layout_constructor()
    use_tile_to_shape()
    use_blocked_product()
    use_make_ordered_layout()
