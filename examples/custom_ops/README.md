### Writing custom CPU or GPU graph operations using Mojo

> [!NOTE]
> This is a preview of an interface for writing custom operations in Mojo,
> and may be subject to change before the next stable release.

Graphs in MAX can be extended to use custom operations written in Mojo. An
example of this is shown here, where a Mojo kernel has been written that adds
1 to every element of an input tensor. A simple graph is then constructed in
Python that takes in an input tensor, runs it through this addition operation,
and then returns the resulting tensor.

One thing to note is that this same Mojo code runs on CPU as well as GPU. In
the construction of the graph, it runs on an accelerator if one is available
or falls back to the CPU if not. No code changes for either path.

The `kernels/` directory contains the custom kernel implementation, and the
graph construction occurs in `addition.py`.

A single Magic command runs the entire example:

```sh
magic run addition
```

The execution has two phases: first a `kernels.mojopkg` is compiled from the
custom Mojo kernel, and then the graph is constructed and run in Python. The
inference session is pointed to the `kernels.mojopkg` in order to load the
custom operations.
