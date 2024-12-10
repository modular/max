# Benchmark MAX Serve

This repository contains tools to benchmark
[MAX Serve](https://docs.modular.com/max/serve/) performance. You can also use
these scripts to compare different LLM serving backends such as
[vLLM](https://github.com/vllm-project/vllm) and
[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) against MAX. The
benchmarking tools measure throughput, latency, and resource utilization
metrics.

> The `benchmark_serving.py` script is adapted from
> [vLLM](https://github.com/vllm-project/vllm/blob/main/benchmarks),
> licensed under Apache 2.0. We forked this script to ensure consistency with
> vLLM's measurement methodology and extended it with features we found helpful,
> such as client-side GPU metric collection via `nvitop`.

## Table of contents

- [Setup](#setup)
- [Benchmarking scripts](#benchmarking-scripts)
- [Output](#output)
- [Reference](#reference)
- [Recommended arguments](#recommended-arguments)
- [Troubleshooting](#troubleshooting)

## Setup

### Prerequisites

To benchmark model performance with the provided scripts, be sure to have the
following:

- Python 3.9.0 - 3.12.0
- Docker with credentials for pulling container images
- A local or cloud environment with access to NVIDIA A100 GPUs (the benchmarking
 scripts are also compatible with A10, L4, and L40 GPUs)
- A Hugging Face account

### Install requirements

Clone the repository and navigate to the `benchmarks` directory.

```bash
git clone -b nightly https://github.com/modularml/max.git
cd pipelines/benchmarking
```

Create a virtual environment and install the provided benchmarking requirements.

```bash
python3 -m venv serving-benchmarking
source serving-benchmarking/bin/activate
pip install -r requirements.txt
```

### Prepare benchmarking dataset

We recommend using the
[ShareGPT](
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
dataset for benchmarking model performance. You can download the dataset with
the following command:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered\
/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

You can optionally use your own datasets for custom evaluations.

### Verify model access through Hugging Face

To download and use
[Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
from Hugging Face, you must have a Hugging Face account, a Hugging Face user
access token, and access to Meta's Llama 3.1 Hugging Face gated repository.

To create a Hugging Face user access token, see
[Access Tokens](https://huggingface.co/settings/tokens). Within your local
environment, save your access token as an environment variable.

```bash
export HF_TOKEN="your_huggingface_token"
```

Use your user access token to log into Hugging Face.

```bash
huggingface-cli login
```

Verify that you have access to the
[Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
model. For more information, see
[Access gated models as a user](
  https://huggingface.co/docs/hub/en/models-gated).

## Benchmarking scripts

This repository provides the following script to benchmark MAX Serve:

### HTTP endpoint benchmarking with `benchmark_serving.py`

This approach allows benchmarking any HTTP endpoint that implements
OpenAI-compatible APIs:

```bash
python benchmark_serving.py \
    --base-url https://company_url.xyz \
    --endpoint /v1/completions \
    --backend modular \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500
```

Key features:

- Tests any OpenAI-compatible HTTP endpoint
- Supports both chat and completion APIs
- Measures detailed latency metrics
- Works with hosted services

## Output

Results are saved in JSON format under the `results/` directory with the
following naming convention:

```bash
{backend}-{request_rate}qps-{model_name}-{timestamp}.json
```

The output should look similar to the following:

```bash
============ Serving Benchmark Result ============
Successful requests:                     500
Failed requests:                         0
Benchmark duration (s):                  46.27
Total input tokens:                      100895
Total generated tokens:                  106511
Request throughput (req/s):              10.81
Input token throughput (tok/s):          2180.51
Output token throughput (tok/s):         2301.89
---------------Time to First Token----------------
Mean TTFT (ms):                          15539.31
Median TTFT (ms):                        15068.37
P99 TTFT (ms):                           33034.17
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          34.23
Median TPOT (ms):                        28.47
P99 TPOT (ms):                           138.55
---------------Inter-token Latency----------------
Mean ITL (ms):                           26.76
Median ITL (ms):                         5.42
P99 ITL (ms):                            228.45
-------------------Token Stats--------------------
Max input tokens:                        933
Max output tokens:                       806
Max total tokens:                        1570
--------------------GPU Stats---------------------
GPU Utilization (%):                     94.74
Peak GPU Memory Used (MiB):              37228.12
GPU Memory Available (MiB):              3216.25
==================================================
```

### Key metrics explained

- **Request throughput**: Number of complete requests processed per second
- **Input token throughput**: Number of input tokens processed per second
- **Output token throughput**: Number of tokens generated per second
- **TTFT**: Time to first token (TTFT), the time from request start to first
token generation
- **TPOT**: Time per output token (TPOT), the average time taken to generate
each output token
- **ITL**: Inter-token latency (ITL), the average time between consecutive token
or token-chunk generations
- **GPU utilization**: Percentage of time during which at least one GPU kernel
is being executed
- **Peak GPU memory used**: Peak memory usage during benchmark run

## Reference

### Command line arguments for `benchmark_serving.py`

- Backend configuration:
  - `--backend`: Choose from `modular` (MAX Serve), `vllm` (vLLM), or`trt-llm`
  (TensorRT-LLM)
  - `--model`: Hugging Face model ID or local path
  - `--batch-size`: Maximum batch size for inference
  - `--max-token-length`: Maximum combined length of input and output tokens
  - `--version`: Container version tag (optional)
- Load generation:
  - `--num-prompts`: Number of prompts to process (default: `500`)
  - `--request-rate`: Request rate in requests/second (default: `inf`)
- Docker options:
  - `-rungroup`: Label for grouping related containers (default: `unset`)
- Serving options
  - `--base-url`: Base URL of the API service
  - `--endpoint`: Specific API endpoint (`/v1/completions` or
  `/v1/chat/completions`)
  - `--tokenizer`: Hugging Face tokenizer to use (can be different from model)
  - `--dataset-name`: (default:`sharegpt`) Real-world conversation data in the
  form of variable length prompts and responses. ShareGPT is automatically
  downloaded if not already present.

## Recommended arguments

### Throughput testing

Recommended arguments to measure maximum request processing capacity:

- Batch size: 250 (modular), 512 (vLLM), 2048 (TRT-LLM)
- Request rate: infinite
- Number of prompts: 500

### Latency testing

Recommended arguments to measure response times with controlled request rates:

- Batch size: 1
- Request rate: 10
- Number of prompts: 100

### Resource utilization

Recommended arguments to monitor GPU memory and usage during extended
operations:

- Batch size: 80
- Number of prompts: 500

## Troubleshooting

### Container start failures

Check the container logs with the following command:

```bash
docker logs <container_id>
```

### Memory issues

- Reduce batch size
- Check GPU memory availability: `nvidia-smi`

### Network issues

Test the container network with the following command:

```bash
docker network inspect bridge
```

### Permission issues

- Verify `HF_TOKEN` is set correctly
- Check Docker permissions
- Ensure model access on Hugging Face
