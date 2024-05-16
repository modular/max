# Mojo unreleased changelog

This is a list of UNRELEASED changes for the Mojo language and tools.

When we cut a release, these notes move to `changelog-released.md` and that's
what we publish.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ⭐️ New
[//]: ### 🦋 Changed
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### 🔥 Legendary

### ⭐️ New

- Mojo now supports adding a `@deprecated` decorator on structs, functions,
  traits, aliases, and global variables. The decorator marks the attached decl
  as deprecated and causes a warning to be emitted when the deprecated decl is
  referenced in user code. The decorator requires a deprecation message to be
  specified as a string literal.

  ```mojo
  @deprecated("Foo is deprecated, use Bar instead")
  struct Foo:
      pass

  fn outdated_api(x: Foo): # warning: Foo is deprecated, use Bar instead
      pass

  @deprecated("use another function!")
  fn bar():
      pass

  fn techdebt():
      bar() # warning: use another function!
  ```

- `int()` can now take a string and a specified base to parse an integer from a
  string: `int("ff", 16)` returns `255`. Additionally, if a base of zero is
  specified, the string will be parsed as if it was an integer literal, with the
  base determined by whether the string contains the prefix `"0x"`, `"0o"`, or
  `"0b"`. ([PR #2273](https://github.com/modularml/mojo/pull/2273) by
  [@artemiogr97](https://github.com/artemiogr97), fixes
  [#2274](https://github.com/modularml/mojo/issues/2274))

- Mojo now supports types to opt in to use the `abs` and `round` functions by
  implementing the `__abs__` and `__round__` methods (i.e. by conforming to the
  new `Absable` and `Roundable` traits), respectively, e.g.:

  ```mojo
  from math import sqrt

  @value
  struct Complex(Absable, Roundable):
      var re: Float64
      var im: Float64

      fn __abs__(self) -> Self:
          return Self(sqrt(self.re * self.re + self.im * self.im), 0.0)

      fn __round__(self) -> Self:
          return Self(round(self.re), round(self.im))
  ```

- The `abs`, `round`, `min`, `max`, and `divmod` functions have moved from
  `math` to `builtin`, so you no longer need to do
  `from math import abs, round, min, max, divmod`.

- Mojo now allows types to opt in to use the `floor()`, `ceil()`, and `trunc()`
  functions in the `math` module by implementing the `__floor__()`,
  `__ceil__()`, and `__trunc__()` methods (and so conforming to the new
  `math.Floorable`, `math.Ceilable`, and `math.Truncable` traits, respectively).
  For example:

  ```mojo
    from math import Ceilable, Floorable, Truncable, ceil, floor, trunc

    @value
    struct Complex(Ceilable, Floorable, Truncable):
      var re: Float64
      var im: Float64

      fn __ceil__(self) -> Self:
          return Self(ceil(re), ceil(im))

      fn __floor__(self) -> Self:
          return Self(floor(re), floor(im))

      fn __trunc__(self) -> Self:
          return Self(trunc(re), trunc(im))
  ```

- Add builtin `any()` and `all()` functions to check for truthy elements in a collection.
  This also works to get the truthy value of a SIMD vector.
  ([PR #2600](https://github.com/modularml/mojo/pull/2600) by [@helehex](https://github.com/helehex))
  For example:

  ```mojo
    fn truthy_simd():
        var vec = SIMD[DType.int32, 4](0, 1, 2, 3)
        if any(vec):
            print("any elements are truthy")
        if all(vec):
            print("all elements are truthy")
  ```

- Add an `InlinedArray` type that works on memory-only types.
  Compare with the existing `StaticTuple` type, which is conceptually an array
  type, but only worked on `AnyRegType`.
    ([PR #2294](https://github.com/modularml/mojo/pull/2294) by [@lsh](https://github.com/lsh))

- Base64 decoding support has been added.
    ([PR #2364](https://github.com/modularml/mojo/pull/2364) by [@mikowals](https://github.com/mikowals))

- Add `repr()` function and `Representable` trait.
    ([PR #2361](https://github.com/modularml/mojo/pull/2361) by [@gabrieldemarmiesse](https://github.com/gabrieldemarmiesse))

- Add `SIMD.shuffle()` with `StaticIntTuple` mask.
    ([PR #2315](https://github.com/modularml/mojo/pull/2315) by [@mikowals](https://github.com/mikowals))

- Invoking `mojo package my-package -o my-dir` on the command line, where
  `my-package` is a Mojo package source directory, and `my-dir` is an existing
  directory, now outputs a Mojo package to `my-dir/my-package.mojopkg`.
  Previously, this had to be spelled out, as in `-o my-dir/my-package.mojopkg`.

- The Mojo Language Server now reports a warning when a local variable is unused.

- The `math` module now has `CeilDivable` and `CeilDivableRaising` traits that
  allow users to opt into the `math.ceildiv` function.

- Mojo now allows methods to declare `self` as a `Reference` directly, which
  can be useful for advanced cases of parametric mutabilty and custom lifetime
  processing.  Previously it required the use of an internal MLIR type to
  achieve this.

- `object` now implements all the bitwise operators.
    ([PR #2324](https://github.com/modularml/mojo/pull/2324) by [@LJ-9801](https://github.com/LJ-9801))

- A new `--validate-doc-strings` option has been added to `mojo` to emit errors
  on invalid doc strings instead of warnings.

- Several `mojo` subcommands now support a `--diagnostic-format` option that
  changes the format with which errors, warnings, and other diagnostics are
  printed. By specifying `--diagnostic-format json` on the command line, errors
  and other diagnostics will be output in a structured
  [JSON Lines](https://jsonlines.org) format that is easier for machines to
  parse.

  The full list of subcommands that support `--diagnostic-format` is as follows:
  `mojo build`, `mojo doc`, `mojo run`, `mojo package`, and `mojo test`.
  Further, the `mojo test --json` option has been subsumed into this new option;
  for the same behavior, run `mojo test --diagnostic-format json`.

  Note that the format of the JSON output may change; we don't currently
  guarantee its stability across releases of Mojo.

- A new decorator, `@doc_private`, was added that can be used to hide a decl
  from being generated in the output of `mojo doc`. It also removes the
  requirement that the decl has documentation (e.g. when used with
  --diagnose-missing-doc-strings).

- Added a new `Span` type for taking slices of contiguous collections.
  ([PR #2595](https://github.com/modularml/mojo/pull/2595) by [lsh](https://github.com/lsh))

- Added new `as_bytes_slice()` methods to `String` and `StringLiteral`, which
  returns a `Span` of the bytes owned by the string.

- Add new `ImmStaticLifetime` and `MutStaticLifetime` helpers

- `Dict` now implements `get(key)` and `get(key, default)` functions.
    ([PR #2519](https://github.com/modularml/mojo/pull/2519) by [@martinvuyk](https://github.com/martinvuyk))

- Debugger users can now set breakpoints on function calls in O0 builds even if
  the call has been inlined by the compiler.

- The `os` module now provides functionalty for adding and removing directories
  using `mkdir` and `rmdir`.
    ([PR #2430](https://github.com/modularml/mojo/pull/2430) by [@artemiogr97](https://github.com/artemiogr97))

- `Dict.__get_ref(key)`, allowing to get references to dictionary values.

- `String.strip()`, `lstrip()` and `rstrip()` can now remove custom characters
  other than whitespace.  In addition, there are now several useful aliases for
  whitespace, ASCII lower/uppercase, and so on.
    ([PR #2555](https://github.com/modularml/mojo/pull/2555) by [@toiletsandpaper](https://github.com/toiletsandpaper))

- `Dict()` now supports `reversed` for `dict.items()` and `dict.values()`.
    ([PR #2340](https://github.com/modularml/mojo/pull/2340) by [@jayzhan211](https://github.com/jayzhan211))

- `List` now has an `index` method that allows one to find the (first) location
  of an element in a `List` of `EqualityComparable` types. For example:

  ```mojo
  var my_list = List[Int](2, 3, 5, 7, 3)
  print(my_list.index(3))  # prints 1
  ```

### 🦋 Changed

- The `let` keyword has been completely removed from the language. We previously
  removed `let` declarations but still provided an error message to users. Now,
  it is completely gone from the grammar. Long live `var`!

- The `abs` and `round` functions have moved from `math` to `builtin`, so you no
  longer need to do `from math import abs, round`.

- Many functions returning a pointer type have been unified to have a public
  API function of `unsafe_ptr()`.

- The `--warn-missing-doc-strings` flag for `mojo` has been renamed to
  `--diagnose-missing-doc-strings`.

- The `take` function in `Variant` and `Optional` has been renamed to
  `unsafe_take`.

- The `get` function in `Variant` has been replaced by `__refitem__`. That is,
  `v.get[T]()` should be replaced with `v[T]`.

- Various functions in the `algorithm` module are now moved to be
  builtin-functions.  This includes `sort`, `swap`, and `partition`.
  `swap` and `partition` will likely shuffle around as we're reworking
  our builtnin `sort` function and optimizing it.

- `ListLiteral` and `Tuple` now only requires that element types be `Copyable`.
  Consequently, `ListLiteral` and `Tuple` are themselves no longer `Copyable`.

- The `math.bit` module has been moved to a new top-level `bit` module. The
  following functions in this module have been renamed:
  - `ctlz` -> `countl_zero`
  - `cttz` -> `countr_zero`
  - `bit_length` -> `bit_width`
  - `ctpop` -> `pop_count`
  - `bswap` -> `byte_reverse`
  - `bitreverse` -> `bit_reverse`

- The `math.rotate_bits_left` and `math.rotate_bits_right` functions have been
  moved to the `bit` module.

- The implementation of the following functions have been moved from the `math`
  module to the new `utils.numerics` module: `isfinite`, `isinf`, `isnan`,
  `nan`, `nextafter`, and `ulp`. The functions continue to be exposed in the
  `math` module.

### ❌ Removed

- The method `object.print()` has been removed. Since now, `object` has the
  `Stringable` trait, you can use `print(my_object)` instead.

- The `math.clamp` function has been removed in favor of a new `SIMD.clamp`
  method.

- The `math.round_half_down` and `math.round_half_up` functions are removed.
  These can be trivially implemented using the `ceil` and `floor` functions.

- The `add`, `sub`, `mul`, `div`, and `mod` functions have been removed from the
  `math` module. Instead, users should rely on the `+`, `-`, `*`, `/`, and `%`
  operators, respectively.

- The `math.roundeven` function has been removed from the `math` module. The new
  `SIMD.roundeven` method now provides the identical functionality.

- The `math.div_ceil` function has been removed in favor of the `math.ceildiv`
  function.

- The `math.bit.select` and `math.bit.bit_and` functions have been removed. The
  same functionality is available in the builtin `SIMD.select` and
  `SIMD.__and__` methods, respectively.

- The `math.rotate_left` and `math.rotate_right` functions have been removed.
  The same functionality is available in the builtin `SIMD.rotate_{left,right}`
  methods for `SIMD` types, and the `bit.rotate_bits_{left,right}` methods for
  `Int`.

- The `math.limit` module has been removed. The same functionality is available
  as follows:
  - `math.limit.inf`: use `utils.numerics.max_or_inf`
  - `math.limit.neginf`: use `utils.numerics.min_or_neg_inf`
  - `math.limit.max_finite`: use `utils.numerics.max_finite`
  - `math.limit.min_finite`: use `utils.numerics.min_finite`

### 🛠️ Fixed

- [#2363](https://github.com/modularml/mojo/issues/2363) Fix LSP crashing on
  simple trait definitions.
- [#1787](https://github.com/modularml/mojo/issues/1787) Fix error when using
  `//` on `FloatLiteral` in alias expression.
- Made several improvements to dictionary performance. Dicts with integer keys
  are most heavily affected, but large dicts and dicts with large values
  will also see large improvements.
