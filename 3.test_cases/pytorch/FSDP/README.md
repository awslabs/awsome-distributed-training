# Get Started Training Llama 2, Mixtral 8x7B, and Mistral Mathstral with PyTorch FSDP in 5 Minutes

This content provides a quickstart with multinode PyTorch [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) training on Slurm and Kubernetes.
It is designed to be simple with no data preparation or tokenizer to download, and uses Python virtual environment.

## Tested Configurations

| Instance | GPUs | Models | Nodes | Status |
|----------|------|--------|-------|--------|
| p5en.48xlarge | 8 x H200 141 GB | Llama 2/3, Mixtral 8x7B | Various | Tested (CI) |
| p5.48xlarge | 8 x H100 80 GB | Llama 2/3, Mixtral 8x7B | Various | Tested (CI) |
| p4de.24xlarge | 8 x A100 80 GB | Llama 2/3 | Various | Tested |
| g5.12xlarge | 4 x A10G 24 GB | Various | Various | Tested |
| g5.xlarge | 1 x A10G 24 GB | Various | 1 | Tested |
| g4dn | Various | Various | Various | Tested |

> See the [Instance Compatibility Guide](../../../docs/instance-compatibility.md)
> for parameter adjustments needed across instance types.
> For untested instance types, check the guide for parameter adjustment recommendations.

## Prerequisites

To run FSDP training, you will need to create a training cluster based on Slurm or Kubermetes with an [Amazon FSx for Lustre](https://docs.aws.amazon.com/fsx/latest/LustreGuide/what-is.html)
You can find instruction how to create a Amazon SageMaker Hyperpod cluster with [Slurm](https://catalog.workshops.aws/sagemaker-hyperpod/en-US), [Kubernetes](https://catalog.workshops.aws/sagemaker-hyperpod-eks/en-US) or with in [Amazon EKS](../../1.architectures).

## FSDP Training

This fold provides examples on how to train with PyTorch FSDP with Slurm or Kubernetes.
You will find instructions for [Slurm](slurm) or [Kubernetes](kubernetes) in the subdirectories.