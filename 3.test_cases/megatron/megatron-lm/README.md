# MegatronLM Test Case

## Tested Configurations

| Instance | GPUs | Status | Notes |
|----------|------|--------|-------|
| p5en.48xlarge | 8 x H200 80 GB | Untested | Expected to work |
| p5.48xlarge | 8 x H100 80 GB | Untested | Expected to work |
| p4de.24xlarge | 8 x A100 80 GB | Untested | Expected to work |
| g5.12xlarge | 4 x A10G 24 GB | Untested | May need adjusted TP/PP for smaller VRAM |

> See the [Instance Compatibility Guide](../../../docs/instance-compatibility.md)
> for parameter adjustments needed across instance types.

[MegatronLM](https://github.com/NVIDIA/Megatron-LM) is a framework from Nvidia designed for training large language models (LLMs). We recommend reading the following papers to understand the various tuning options available:

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [Reducing Activation Recomputatio in Large Transformer Models](https://arxiv.org/pdf/2205.05198)

To run a test case, follow these steps:

1. Prepare your environment.
2. Build a container, download, and preprocess the data.
3. Train the model.

We provide guidance for both Slurm and Kubernetes users. For detailed instructions, refer to the [slurm](./slurm) or [kubernetes](./kubernetes) subdirectories.