# Writing custom CPU or GPU graph operations using Mojo

> [!NOTE]
> This is a preview of an interface for writing custom operations in Mojo,
> and may be subject to change before the next stable release.

Graphs in MAX can be extended to use custom operations written in Mojo. Two
examples of this are shown here:

- Adding 1 to every element of an input tensor.
- Calculating the Mandelbrot set.

Custom kernels have been written in Mojo to carry out these calculations. For
each example, a simple graph containing a single operation is constructed
in Python. This graph is compiled and dispatched onto a supported GPU if one is
available, or the CPU if not. Input tensors, if there are any, are moved from
the host to the device on which the graph is running. The graph then runs and
the results are copied back to the host for display.

One thing to note is that this same Mojo code runs on CPU as well as GPU. In
the construction of the graph, it runs on a supported accelerator if one is
available or falls back to the CPU if not. No code changes for either path.

The `kernels/` directory contains the custom kernel implementations, and the
graph construction occurs in `addition.py` or `mandelbrot.py`. These examples
are designed to stand on their own, so that they can be used as templates for
experimentation.

A single Magic command runs each of the examples:

```sh
magic run addition
magic run mandelbrot
```

The execution has two phases: first a `kernels.mojopkg` is compiled from the
custom Mojo kernel, and then the graph is constructed and run in Python. The
inference session is pointed to the `kernels.mojopkg` in order to load the
custom operations.
