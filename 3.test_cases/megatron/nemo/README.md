# NVIDIA NeMo 2.0 Distributed Training

This test case contains examples and configurations for running distributed training with NVIDIA NeMo 2.0.

## Tested Configurations

| Instance | GPUs | Status | Notes |
|----------|------|--------|-------|
| p5en.48xlarge | 8 x H200 80 GB | Tested | Primary target; see PERFORMANCE.md in slurm/ |
| p5.48xlarge | 8 x H100 80 GB | Tested | |
| p4de.24xlarge | 8 x A100 80 GB | Untested | Expected to work |
| g5.12xlarge | 4 x A10G 24 GB | Untested | May need smaller model configs |
| g6e.12xlarge | 4 x L40S 48 GB | Untested | |

> See the [Instance Compatibility Guide](../../../docs/instance-compatibility.md)
> for parameter adjustments needed across instance types.

## Overview

[NVIDIA NeMo](https://developer.nvidia.com/nemo-framework) is a cloud-native framework for training and deploying generative AI models, optimized for architectures ranging from billions to trillions of parameters. NeMo 2.0 introduces a Python-based configuration system, providing enhanced flexibility, better IDE integration, and streamlined customization for large language model training.


- **Comprehensive development tools** for data preparation, model training, and deployment.
- **Advanced customization** for fine-tuning models to specific use cases.
- **Optimized infrastructure** with multi-GPU and multi-node support.
- **Enterprise-grade features** such as parallelism techniques, memory optimization, and deployment pipelines.

NeMo 2.0 introduces a Python-based configuration system, providing enhanced flexibility, better IDE integration, and streamlined customization.

## Slurm-based Deployment

The [slurm](./slurm/) directory provides implementation examples for running NeMo 2.0 using Slurm as the workload manager. This approach leverages AWS's purpose-built infrastructure for large-scale AI training. See the [README in the slurm directory](./slurm/README.md) for detailed setup and usage instructions.

## Kubernetes based Deployment

The [kubernetes](./kubernetes/) directory provides implementation examples for running NeMo 2.0 using Kubernetes as the orhestrator. This approach leverages AWS EKS for large-scale AI training. See the [README in the kubernetes directory](./kubernetes/README.md) for detailed setup and usage instructions.
