
This is a running list of significant UNRELEASED changes for the Mojo language
and tools. Please add any significant user-visible changes here.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ⭐️ New
[//]: ### 🦋 Changed
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### 🔥 Legendary

### ⭐️ New

- Heterogenous variadic pack arguments now work reliably even with memory types,
  and have a more convenient API to use, as defined on the `VariadicPack` type.
  For example, a simplified version of `print` can be implemented as:

  ```mojo
  fn print[T: Stringable, *Ts: Stringable](first: T, *rest: *Ts):
      print_string(str(first))

      @parameter
      fn print_elt[T: Stringable](a: T):
          print_string(" ")
          print_string(a)
      rest.each[print_elt]()
  ```

- The `sys` module now contains an `exit` function that would exit a Mojo
  program with the specified error code.

- The constructors for `tensor.Tensor` have been changed to be more consistent.
  As a result, one has to pass in the shape as first argument (instead of the
  second) when constructing a tensor with pointer data.

- The constructor for `tensor.Tensor` will now splat a scalar if its passed in.
  For example, `Tensor[DType.float32](TensorShape(2,2), 0)` will construct a
  `2x2` tensor which is initialized with all zeros. This provides an easy way
  to fill the data of a tensor.

- The `mojo build` and `mojo run` commands now support a `-g` option. This
  shorter alias is equivalent to writing `--debug-level full`. This option is
  also available in the `mojo debug` command, but is already the default.

- `PythonObject` now conforms to the `KeyElement` trait, meaning that it can be
  used as key type for `Dict`. This allows on to easily build and interact with
  Python dictionaries in mojo:

  ```mojo
  def main():
      d = PythonObject(Dict[PythonObject, PythonObject]())
      d["foo"] = 12
      d[7] = "bar"
      d["foo"] = [1, 2, "something else"]
      print(d)  # prints `{'foo': [1, 2, 'something else'], 7: 'bar'}`
  ```

- `List` now has a `pop(index)` API for removing an element
  at a particular index.  By default, `List.pop()` removes the last element
  in the list.

- `Dict` now has a `update()` method to update keys/values from another `Dict`.

- `Tuple.get` now supports a form that just takes an element index but does not
  require you to specify the result type.  Instead of `tup.get[1, Int]()` you
  can now just use `tup.get[1]()`.

- `Reference` interoperates with unsafe code better: `AnyPointer` now has a
  constructor that forms it from `Reference` directly (inferring element type
  and address space). `AnyPointer` can convert to an immortal mutable
  `Reference` with `yourptr[]`.  Both `Reference` and `AnyPointer` now have an
  `address_space_cast` method like `Pointer`.

### 🦋 Changed

- The behavior of `mojo build` when invoked without an output `-o` argument has
  changed slightly: `mojo build ./test-dir/program.mojo` now outputs an
  executable to the path `./program`, whereas before it would output to the path
  `./test-dir/program`.
- The REPL no longer allows type level variable declarations to be
  uninitialized, e.g. it will reject `var s: String`.  This is because it does
  not do proper lifetime tracking (yet!) across cells, and so such code would
  lead to a crash.  You can work around this by initializing to a dummy value
  and overwriting later.  This limitation only applies to top level variables,
  variables in functions work as they always have.

### ❌ Removed

- Support for "register only" variadic packs has been removed. Instead of
  `AnyRegType`, please upgrade your code to `AnyType` in examples like this:

  ```mojo
  fn your_function[*Types: AnyRegType](*args: *Ts): ...
  ```

  This move gives you access to nicer API and has the benefit of being memory
  safe and correct for non-trivial types.  If you need specific APIs on the
  types, please use the correct trait bound instead of `AnyType`.

- `List.pop_back()` has been removed.  Use `List.pop()` instead which defaults
  to popping the last element in the list.

### 🛠️ Fixed

- [#1987](https://github.com/modularml/mojo/issues/1987) Defining `main`
  in a Mojo package is an error, for now. This is not intended to work yet,
  erroring for now will help to prevent accidental undefined behavior.

- [#1215](https://github.com/modularml/mojo/issues/1215) and
  [#1949](https://github.com/modularml/mojo/issues/1949) The Mojo LSP server no
  longer cuts off hover previews for functions with functional arguments,
  parameters, or results.

- [#1901](https://github.com/modularml/mojo/issues/1901) Fixed Mojo LSP and
  documentation generation handling of inout arguments.

- [#1913](https://github.com/modularml/mojo/issues/1913) - `0__` no longer
  crashes the Mojo parser.

- [#1924](https://github.com/modularml/mojo/issues/1924) JIT debugging on Mac
  has been fixed.

- [#1963](https://github.com/modularml/mojo/issues/1963) `a!=0` is now parsed
  and formatted correctly by `mojo format`.
