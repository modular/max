# Duration Module

Brian Grenier, February 21st, 2025

## Motivation

Representing time durations in programming can be a general hassle at best,
and a source of bugs at worst. Different functions interpret the same plain
integer value in different ways, and developers must be careful that
they calculate the intended time interval properly. Consider the following
example, where you wish to create a recurring function call that is
executed every 15 minutes.

```mojo
alias TIMEOUT_IN_MILLISECONDS = 15 * (60 * 1000)
struct Job:
    fn set_timeout(t: Int):
        """
        Args:
            t: The timeout duration in milliseconds
        """
        ...

def main():
    var j = Job()
    j.set_timeout(TIMEOUT_IN_MILLISECONDS)
```

Now most programmers have done such calculations frequently enough that we do
not consider this to be such a terrible burden, but we _can_ do better,
removing the risk entirely. One piece of the C++ standard library that
was actually done quite well in my opinion is
[`std::chrono::duration`](https://en.cppreference.com/w/cpp/chrono/duration).
By encoding the meaning of the numerical value in the type system, we
can write much safer,
and easy to understand APIs.

## Proposal

I propose we follow a similar pattern to `std::chrono::duration`, and
define a `Duration` struct that takes a `Ratio` parameter to denote its
relation to the base value of one second.

```mojo
@value
@register_passable("trivial")
struct Ratio[N: UInt, D: UInt = 1]:
    alias Milli = Ratio[1, 1000]
    pass

alias Seconds = Duration[Ratio[1](), 's']
alias Milliseconds = Duration[Milli, 'ms']
alias Minutes = Duration[Ratio[60](), 'm']

@value
@register_passable("trivial")
struct Duration[R: Ratio, postfix: StringLiteral='']:
    var _value: Int

    fn cast[R: Ratio](self) -> Duration[R]:
        return Duration[R](0)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.count(), postfix)

    ...
```

With these pieces we can now convert the above example into something much
more ergonomic.

```mojo
struct Job:
    fn set_timeout(t: Milliseconds):
        """
        Args:
            t: The timeout duration in milliseconds
        """
        ...

def main():
    var j = Job()
    j.set_timeout(Minutes(15).cast[Ratio.Milli]())

    sleep(Seconds(10))
```

## Limitations

I've currently identified two limitations when implementing something
like this, but I believe they can both be easily remedied when the appropriate
language features are available.

### Inter-ratio arithmetic

In C++ you can do arithmetic on durations of differing ratios
because the compiler can determine the higher precision timeframe, which will
be used as the return type of the operation, and the other value will be cast
down.

```c++
#include <chrono>

int main()
{

    using namespace std::chrono;

    seconds s(10);
    milliseconds m(100);

    std::cout << m + s; // prints 10100ms
}
```

Here `s` will be implicitly cast to `milliseconds`. This is facilitated by
[`std::common_type`](
    https://en.cppreference.com/w/cpp/chrono/duration/common_type),
which as far as I am aware, Mojo currently cannot express. So for now we will
have to stick to doing arithmetic with matching types.

### Generic representation type

`std::chrono::duration` also makes the representation type of the value
generic, but since Mojo doesn't currently have something like a `Numeric`
trait, we would either have to stick to using `Int`, or be generic over
`Scalar`
