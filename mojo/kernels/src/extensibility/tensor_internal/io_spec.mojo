# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct IO:
    var value: Int

    alias Output = IO(0)
    alias Input = IO(1)
    alias Unknown = IO(2)


@value
@register_passable("trivial")
struct IOSpec[mut: Bool, input: IO]:
    """
    Parameter used to encode whether a particular tensor argument to a DPS kernel
    is an output, input, or mutable input.

    ```mojo
    Input == IOSpec[False, IO.Input]()
    Output == IOSpec[True, IO.Output]()
    MutableInput == IOSpec[True, IO.Input]()
    ```
    """

    ...


alias IOUnknown = IOSpec[False, IO.Unknown]()

alias Input = IOSpec[False, IO.Input]()
alias Output = IOSpec[True, IO.Output]()
alias MutableInput = IOSpec[True, IO.Input]()
