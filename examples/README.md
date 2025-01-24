# MAX examples

These examples demonstrate the power and flexibility of
[MAX](https://docs.modular.com/max/). They include:

## [Custom GPU and CPU operations in Mojo](custom_ops/)

The [MAX Graph API](https://docs.modular.com/max/graph/) provides a powerful
framework for staging computational graphs to be run on GPUs, CPUs, and more.
Each operation in one of these graphs is defined in
[Mojo](https://docs.modular.com/mojo/), an easy-to-use language for writing
high-performance code.

The examples here illustrate how to construct custom graph operations in Mojo
that run on GPUs and CPUs, as well as how to build computational graphs that
contain and run them on different hardware architectures.

## [PyTorch and ONNX inference on MAX](inference/)

MAX has the power to accelerate existing PyTorch and ONNX models directly, and
provides Python, Mojo, and C APIs for this. These examples showcase common
models from these frameworks and how to run them even faster via MAX.

## [Jupyter notebooks](notebooks/)

Jupyter notebooks that showcase PyTorch and ONNX models being accelerated
through MAX.
