# TRL (Transformer Reinforcement Learning) Test Cases

This directory contains test cases for distributed training with [Hugging Face TRL](https://huggingface.co/docs/trl), a library for post-training LLMs using reinforcement learning techniques such as GRPO, PPO, DPO, and SFT.

## Tested Configurations

| Instance | GPUs | Status | Notes |
|----------|------|--------|-------|
| p5en.48xlarge | 8 x H200 80 GB | Untested | Expected to work |
| p5.48xlarge | 8 x H100 80 GB | Untested | Expected to work |
| p4de.24xlarge | 8 x A100 80 GB | Untested | Expected to work |
| g5.12xlarge | 4 x A10G 24 GB | Untested | May need offloading for large models; see GRPO sub-cases |

> See the [Instance Compatibility Guide](../../../docs/instance-compatibility.md)
> for parameter adjustments needed across instance types.

## Base Docker Image

All test cases share a common base Docker image defined in [`Dockerfile`](Dockerfile). It includes Python 3.12, PyTorch 2.6.0, TRL with vLLM backend, Flash Attention, FlashInfer, and common training dependencies.

Build the shared base image:

```bash
cd 3.test_cases/pytorch/trl
docker build -t trl-base:latest .
```

## Test Cases

| Test Case | Description | Platform | Model |
|-----------|-------------|----------|-------|
| [grpo-math-reasoning](grpo-math-reasoning/) | Multi-node GRPO training for math reasoning | Slurm | Qwen2.5-72B + NuminaMath |
| [gpt-oss-lora-grpo](gpt-oss-lora-grpo/) | SFT LoRA + GRPO for multilingual reasoning | HyperPod EKS | GPT-OSS 20B |

## Prerequisites

- GPU cluster with EFA networking (see [`1.architectures/`](../../../1.architectures/) for cluster setup)
- Shared filesystem (e.g., Amazon FSx for Lustre) accessible from all nodes
- [Enroot](https://github.com/NVIDIA/enroot) and [Pyxis](https://github.com/NVIDIA/pyxis) for Slurm container execution
- Hugging Face model access tokens configured via `HF_HOME`
