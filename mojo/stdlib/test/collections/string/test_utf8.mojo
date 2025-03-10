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
# RUN: %mojo %s

from testing import assert_equal, assert_false, assert_raises, assert_true
from collections.string._utf8 import _is_valid_utf8
from collections.string.string_slice import (
    StringSlice,
    _count_utf8_continuation_bytes,
    _utf8_byte_type,
    _utf8_first_byte_sequence_length,
)

from memory import Span

# ===----------------------------------------------------------------------=== #
# Reusable testing data
# ===----------------------------------------------------------------------=== #

alias GOOD_SEQUENCES = List[String](
    "a",
    "\xc3\xb1",
    "\xe2\x82\xa1",
    "\xf0\x90\x8c\xbc",
    "안녕하세요, 세상",
    "\xc2\x80",
    "\xf0\x90\x80\x80",
    "\xee\x80\x80",
    "very very very long string 🔥🔥🔥",
)


# TODO: later on, don't use String because
# it will likely refuse non-utf8 data.
alias BAD_SEQUENCES = List[String](
    "\xc3\x28",  # continuation bytes does not start with 10xx
    "\xa0\xa1",  # first byte is continuation byte
    "\xe2\x28\xa1",  # second byte should be continuation byte
    "\xe2\x82\x28",  # third byte should be continuation byte
    "\xf0\x28\x8c\xbc",  # second byte should be continuation byte
    "\xf0\x90\x28\xbc",  # third byte should be continuation byte
    "\xf0\x28\x8c\x28",  # fourth byte should be continuation byte
    "\xc0\x9f",  # overlong, could be just one byte
    "\xf5\xff\xff\xff",  # missing continuation bytes
    "\xed\xa0\x81",  # UTF-16 surrogate pair
    "\xf8\x90\x80\x80\x80",  # 5 bytes is too long
    "123456789012345\xed",  # Continuation bytes are missing
    "123456789012345\xf1",  # Continuation bytes are missing
    "123456789012345\xc2",  # Continuation bytes are missing
    "\xC2\x7F",  # second byte is not continuation byte
    "\xce",  # Continuation byte missing
    "\xce\xba\xe1",  # two continuation bytes missing
    "\xce\xba\xe1\xbd",  # One continuation byte missing
    "\xce\xba\xe1\xbd\xb9\xcf",  # fifth byte should be continuation byte
    "\xce\xba\xe1\xbd\xb9\xcf\x83\xce",  # missing continuation byte
    "\xce\xba\xe1\xbd\xb9\xcf\x83\xce\xbc\xce",  # missing continuation byte
    "\xdf",  # missing continuation byte
    "\xef\xbf",  # missing continuation byte
)

# ===----------------------------------------------------------------------=== #
# Tests
# ===----------------------------------------------------------------------=== #


fn test_utf8_validation() raises:
    var text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam
    varius tellus quis tincidunt dictum. Donec eros orci, ultricies ac metus non
    , rutrum faucibus neque. Nunc ultricies turpis ut lacus consequat dapibus.
    Nulla nec risus a purus volutpat blandit. Donec sit amet massa velit. Aenean
    fermentum libero eu pharetra placerat. Sed id molestie tellus. Fusce
    sollicitudin a purus ac placerat.
    Lorem Ipsum，也称乱数假文或者哑元文本， 是印刷及排版领域所常用的虚拟文字
    由于曾经一台匿名的打印机刻意打乱了一盒印刷字体从而造出一本字体样品书，Lorem
    Ipsum从西元15世纪起就被作为此领域的标准文本使用。它不仅延续了五个世纪，
    还通过了电子排版的挑战，其雏形却依然保存至今。在1960年代，”Leatraset”公司发布了印刷着
    Lorem Ipsum段落的纸张，从而广泛普及了它的使用。最近，计算机桌面出版软件
    למה אנו משתמשים בזה?
    זוהי עובדה מבוססת שדעתו של הקורא תהיה מוסחת על ידי טקטס קריא כאשר הוא יביט בפריסתו. המטרה בשימוש
     ב- Lorem Ipsum הוא שיש לו פחות או יותר תפוצה של אותיות, בניגוד למלל ' יסוי
    יסוי  יסוי', ונותן חזות קריאה יותר.הרבה הוצאות מחשבים ועורכי דפי אינטרנט משתמשים כיום ב-
    Lorem Ipsum כטקסט ברירת המחדל שלהם, וחיפוש של 'lorem ipsum' יחשוף אתרים רבים בראשית
    דרכם.גרסאות רבות נוצרו במהלך השנים, לעתים בשגגה
    Lorem Ipsum е едноставен модел на текст кој се користел во печатарската
    индустрија.
    Lorem Ipsum - це текст-"риба", що використовується в друкарстві та дизайні.
    Lorem Ipsum คือ เนื้อหาจำลองแบบเรียบๆ ที่ใช้กันในธุรกิจงานพิมพ์หรืองานเรียงพิมพ์
    มันได้กลายมาเป็นเนื้อหาจำลองมาตรฐานของธุรกิจดังกล่าวมาตั้งแต่ศตวรรษที่
    Lorem ipsum" في أي محرك بحث ستظهر العديد
     من المواقع الحديثة العهد في نتائج البحث. على مدى السنين
     ظهرت نسخ جديدة ومختلفة من نص لوريم إيبسوم، أحياناً عن طريق
     الصدفة، وأحياناً عن عمد كإدخال بعض العبارات الفكاهية إليها.
    """
    assert_true(_is_valid_utf8(text.as_bytes()))
    assert_true(_is_valid_utf8(text.as_bytes()))

    var positive = List[List[UInt8]](
        List[UInt8](0x0),
        List[UInt8](0x00),
        List[UInt8](0x66),
        List[UInt8](0x7F),
        List[UInt8](0x00, 0x7F),
        List[UInt8](0x7F, 0x00),
        List[UInt8](0xC2, 0x80),
        List[UInt8](0xDF, 0xBF),
        List[UInt8](0xE0, 0xA0, 0x80),
        List[UInt8](0xE0, 0xA0, 0xBF),
        List[UInt8](0xED, 0x9F, 0x80),
        List[UInt8](0xEF, 0x80, 0xBF),
        List[UInt8](0xF0, 0x90, 0xBF, 0x80),
        List[UInt8](0xF2, 0x81, 0xBE, 0x99),
        List[UInt8](0xF4, 0x8F, 0x88, 0xAA),
    )
    for item in positive:
        assert_true(_is_valid_utf8(Span(item[])))
        assert_true(_is_valid_utf8(Span(item[])))
    var negative = List[List[UInt8]](
        List[UInt8](0x80),
        List[UInt8](0xBF),
        List[UInt8](0xC0, 0x80),
        List[UInt8](0xC1, 0x00),
        List[UInt8](0xC2, 0x7F),
        List[UInt8](0xDF, 0xC0),
        List[UInt8](0xE0, 0x9F, 0x80),
        List[UInt8](0xE0, 0xC2, 0x80),
        List[UInt8](0xED, 0xA0, 0x80),
        List[UInt8](0xED, 0x7F, 0x80),
        List[UInt8](0xEF, 0x80, 0x00),
        List[UInt8](0xF0, 0x8F, 0x80, 0x80),
        List[UInt8](0xF0, 0xEE, 0x80, 0x80),
        List[UInt8](0xF2, 0x90, 0x91, 0x7F),
        List[UInt8](0xF4, 0x90, 0x88, 0xAA),
        List[UInt8](0xF4, 0x00, 0xBF, 0xBF),
        List[UInt8](
            0xC2, 0x80, 0x00, 0x00, 0xE1, 0x80, 0x80, 0x00, 0xC2, 0xC2, 0x80
        ),
        List[UInt8](0x00, 0xC2, 0xC2, 0x80, 0x00, 0x00, 0xE1, 0x80, 0x80),
        List[UInt8](0x00, 0x00, 0x00, 0xF1, 0x80, 0x00),
        List[UInt8](0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF1),
        List[UInt8](0x00, 0x00, 0x00, 0x00, 0xF1, 0x00, 0x80, 0x80),
        List[UInt8](0x00, 0x00, 0xF1, 0x80, 0xC2, 0x80, 0x00),
        List[UInt8](0x00, 0x00, 0xF0, 0x80, 0x80, 0x80),
    )
    for item in negative:
        assert_false(_is_valid_utf8(Span(item[])))
        assert_false(_is_valid_utf8(Span(item[])))


fn validate_utf8(slice: StringSlice) -> Bool:
    return _is_valid_utf8(slice.as_bytes())


def test_good_utf8_sequences():
    for sequence in GOOD_SEQUENCES:
        assert_true(validate_utf8(sequence[]))


def test_bad_utf8_sequences():
    for sequence in BAD_SEQUENCES:
        assert_false(validate_utf8(sequence[]))


def test_stringslice_from_utf8():
    for sequence in GOOD_SEQUENCES:
        var bytes = sequence[].as_bytes()
        _ = StringSlice.from_utf8(bytes)

    for sequence in BAD_SEQUENCES:
        with assert_raises(contains="buffer is not valid UTF-8"):
            var bytes = sequence[].as_bytes()
            _ = StringSlice.from_utf8(bytes)


def test_combination_good_utf8_sequences():
    # any combination of good sequences should be good
    for i in range(0, len(GOOD_SEQUENCES)):
        for j in range(i, len(GOOD_SEQUENCES)):
            var sequence = GOOD_SEQUENCES[i] + GOOD_SEQUENCES[j]
            assert_true(validate_utf8(sequence))


def test_combination_bad_utf8_sequences():
    # any combination of bad sequences should be bad
    for i in range(0, len(BAD_SEQUENCES)):
        for j in range(i, len(BAD_SEQUENCES)):
            var sequence = BAD_SEQUENCES[i] + BAD_SEQUENCES[j]
            assert_false(validate_utf8(sequence))


def test_combination_good_bad_utf8_sequences():
    # any combination of good and bad sequences should be bad
    for i in range(0, len(GOOD_SEQUENCES)):
        for j in range(0, len(BAD_SEQUENCES)):
            var sequence = GOOD_SEQUENCES[i] + BAD_SEQUENCES[j]
            assert_false(validate_utf8(sequence))


def test_combination_10_good_utf8_sequences():
    # any 10 combination of good sequences should be good
    for i in range(0, len(GOOD_SEQUENCES)):
        for j in range(i, len(GOOD_SEQUENCES)):
            var sequence = GOOD_SEQUENCES[i] * 10 + GOOD_SEQUENCES[j] * 10
            assert_true(validate_utf8(sequence))


def test_combination_10_good_10_bad_utf8_sequences():
    # any 10 combination of good and bad sequences should be bad
    for i in range(0, len(GOOD_SEQUENCES)):
        for j in range(0, len(BAD_SEQUENCES)):
            var sequence = GOOD_SEQUENCES[i] * 10 + BAD_SEQUENCES[j] * 10
            assert_false(validate_utf8(sequence))


def test_count_utf8_continuation_bytes():
    alias c = UInt8(0b1000_0000)
    alias b1 = UInt8(0b0100_0000)
    alias b2 = UInt8(0b1100_0000)
    alias b3 = UInt8(0b1110_0000)
    alias b4 = UInt8(0b1111_0000)

    def _test(amnt: Int, items: List[UInt8]):
        var p = items.unsafe_ptr()
        var span = Span[Byte, StaticConstantOrigin](ptr=p, length=len(items))
        var str_slice = StringSlice(unsafe_from_utf8=span)
        assert_equal(amnt, _count_utf8_continuation_bytes(str_slice))

    _test(5, List[UInt8](c, c, c, c, c))
    _test(2, List[UInt8](b2, c, b2, c, b1))
    _test(2, List[UInt8](b2, c, b1, b2, c))
    _test(2, List[UInt8](b2, c, b2, c, b1))
    _test(2, List[UInt8](b2, c, b1, b2, c))
    _test(2, List[UInt8](b1, b2, c, b2, c))
    _test(2, List[UInt8](b3, c, c, b1, b1))
    _test(2, List[UInt8](b1, b1, b3, c, c))
    _test(2, List[UInt8](b1, b3, c, c, b1))
    _test(3, List[UInt8](b1, b4, c, c, c))
    _test(3, List[UInt8](b4, c, c, c, b1))
    _test(3, List[UInt8](b3, c, c, b2, c))
    _test(3, List[UInt8](b2, c, b3, c, c))


def test_utf8_byte_type():
    alias c = UInt8(0b1000_0000)
    alias b1 = UInt8(0b0100_0000)
    alias b2 = UInt8(0b1100_0000)
    alias b3 = UInt8(0b1110_0000)
    alias b4 = UInt8(0b1111_0000)
    alias r1 = _utf8_byte_type(b1)
    assert_equal(r1, 0)
    assert_equal(_utf8_byte_type(b1), 0)
    alias r2 = _utf8_byte_type(c)
    assert_equal(r2, 1)
    assert_equal(_utf8_byte_type(c), 1)
    alias r3 = _utf8_byte_type(b2)
    assert_equal(r3, 2)
    assert_equal(_utf8_byte_type(b2), 2)
    alias r4 = _utf8_byte_type(b3)
    assert_equal(r4, 3)
    assert_equal(_utf8_byte_type(b3), 3)
    alias r5 = _utf8_byte_type(b4)
    assert_equal(r5, 4)
    assert_equal(_utf8_byte_type(b4), 4)


def test_utf8_first_byte_sequence_length():
    alias b1 = UInt8(0b0100_0000)
    alias b2 = UInt8(0b1100_0000)
    alias b3 = UInt8(0b1110_0000)
    alias b4 = UInt8(0b1111_0000)
    alias r1 = _utf8_first_byte_sequence_length(b1)
    assert_equal(r1, 1)
    assert_equal(_utf8_first_byte_sequence_length(b1), 1)
    alias r2 = _utf8_first_byte_sequence_length(b2)
    assert_equal(r2, 2)
    assert_equal(_utf8_first_byte_sequence_length(b2), 2)
    alias r3 = _utf8_first_byte_sequence_length(b3)
    assert_equal(r3, 3)
    assert_equal(_utf8_first_byte_sequence_length(b3), 3)
    alias r4 = _utf8_first_byte_sequence_length(b4)
    assert_equal(r4, 4)
    assert_equal(_utf8_first_byte_sequence_length(b4), 4)


def main():
    test_utf8_validation()
    test_good_utf8_sequences()
    test_bad_utf8_sequences()
    test_stringslice_from_utf8()
    test_combination_good_utf8_sequences()
    test_combination_bad_utf8_sequences()
    test_combination_good_bad_utf8_sequences()
    test_combination_10_good_utf8_sequences()
    test_combination_10_good_10_bad_utf8_sequences()
    test_count_utf8_continuation_bytes()
    test_utf8_byte_type()
    test_utf8_first_byte_sequence_length()
