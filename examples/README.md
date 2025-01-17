# MAX examples

These examples demonstrate the power and flexibility of
[MAX](https://docs.modular.com/max/). They include:

## [MAX Pipelines](../pipelines/python)

End-to-end demonstrations of common AI workloads and more, built using
[Mojo](https://docs.modular.com/mojo/) and the
[MAX Graph API](https://docs.modular.com/max/graph/).

## [PyTorch and ONNX inference on MAX](inference/)

MAX has the power to accelerate existing PyTorch and ONNX models directly, and
provides Python, Mojo, and C APIs for this. These examples showcase common
models from these frameworks and how to run them even faster via MAX.

## MAX Engine extensibility

We removed the extensibility API in v24.5 (it was added in v24.3) and we're
replacing it with a better version soon. Because MAX is still a preview, we
don't want to leave APIs in the platform that we have no intention to support.
Stay tuned for an improved extensibility API that works on both CPUs and GPUs.

## [Jupyter notebooks](notebooks/)

Jupyter notebooks that showcase PyTorch and ONNX models being accelerated
through MAX.
