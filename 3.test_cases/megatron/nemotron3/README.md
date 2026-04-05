# NVIDIA Nemotron 3 — Training on AWS

This directory contains training recipes for the [NVIDIA Nemotron 3](https://developer.nvidia.com/nemotron) model family on AWS GPU instances. The Nemotron 3 family uses a hybrid **Mamba-2 + MoE + Attention** architecture for efficient, high-accuracy language modeling with up to 1M token context.

## Model Family

| Model | Total Params | Active Params | Architecture | Experts | Directory |
|-------|-------------|---------------|-------------|---------|-----------|
| **Nemotron 3 Nano** | 30B | 3.5B | Mamba-2 + MoE + Attention | 128+1 shared, 5 active | [nano/](nano/) |
| **Nemotron 3 Super** | 120B | 12B | LatentMoE + Mamba-2 + Attention + MTP | 512+shared, 22 active | [super/](super/) |

## Training Capabilities

| Capability | Framework | Nano | Super |
|-----------|-----------|------|-------|
| LoRA SFT | NeMo Megatron Bridge | [nano/slurm/](nano/slurm/) | [super/slurm/](super/slurm/) |
| Full Fine-Tuning | NeMo Megatron Bridge | [nano/slurm/](nano/slurm/) | [super/slurm/](super/slurm/) |
| GRPO/DAPO RL | NeMo RL + NeMo Gym | [nano/slurm/](nano/slurm/) | [super/slurm/](super/slurm/) |
| Kubernetes | NeMo-Run + SkyPilot | [nano/kubernetes/](nano/kubernetes/) | [super/kubernetes/](super/kubernetes/) |

## Instance Compatibility

| Instance | GPUs | VRAM | Nano LoRA | Super LoRA | Nano GRPO | Super GRPO |
|----------|------|------|-----------|------------|-----------|------------|
| p6-B200 | 8x B200 | 1440GB | 1 node | 1 node | 1 node | 1-2 nodes |
| p5en.48xlarge | 8x H200 | 1128GB | 1 node | 1 node | 1 node | 2+ nodes |
| p5.48xlarge | 8x H100 | 640GB | 1 node | 1 node (FP8) | 1-2 nodes | 2+ nodes |
| p4de.24xlarge | 8x A100 | 640GB | 1 node | Marginal | 2+ nodes | Multi-node |

## Container Strategy

Each model variant uses **two separate containers** — one for LoRA/SFT (NeMo base) and one for GRPO/RL (NeMo RL base):

| Container | Nano Base Image | Super Base Image |
|-----------|----------------|-----------------|
| LoRA/SFT | `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano` | `nvcr.io/nvidia/nemo:25.07.00` |
| GRPO/RL | `nvcr.io/nvidia/nemo-rl:v0.5.0` | `nvcr.io/nvidia/nemo-rl:v0.5.0` |

Both containers include the AWS EFA networking stack (GDRCopy, EFA Installer, NCCL) for optimal multi-node performance.

## Quick Start

### Nano LoRA SFT (simplest starting point)

```bash
cd nano/slurm

# Build container
docker build --progress=plain -t aws-nemotron3-nano:25.11 -f ../Dockerfile ..
enroot import -o ~/aws-nemotron3-nano.sqsh dockerd://aws-nemotron3-nano:25.11

# Launch training
python run_lora_sft.py \
    --container_image ~/aws-nemotron3-nano.sqsh \
    --nodes 1 --partition dev
```

### Super LoRA SFT

```bash
cd super/slurm

# Build container
docker build --progress=plain -t aws-nemotron3-super:25.07 -f ../Dockerfile ..
enroot import -o ~/aws-nemotron3-super.sqsh dockerd://aws-nemotron3-super:25.07

# Launch training (FP8 by default)
python run_lora_sft.py \
    --container_image ~/aws-nemotron3-super.sqsh \
    --nodes 1 --partition dev
```

## Dataset Abstraction

All training scripts support pluggable datasets:

- **Default datasets**: SQuAD (Nano) / Text2SQL (Super) — matching NVIDIA's official cookbooks
- **Custom HuggingFace datasets**: Pass `--dataset <hf_dataset_id>`
- **Local data**: Pass `--dataset /path/to/data.jsonl`

## Why NeMo RL for GRPO (not veRL)?

The Nemotron 3 architecture (Mamba-2 + LatentMoE hybrid) is **not supported in veRL** due to:
- No Mamba-2 model initializer in veRL's Megatron backend
- No LatentMoE support in any veRL backend
- Broken sequence parallelism for hybrid attention/SSM models ([veRL Issue #5552](https://github.com/verl-project/verl/issues/5552))

**NeMo RL** is NVIDIA's own framework that was used to train both Nemotron 3 Nano and Super with GRPO/DAPO. It has first-class support for the Nemotron-H architecture.

## References

- [NVIDIA Nemotron Developer Page](https://developer.nvidia.com/nemotron)
- [Nemotron 3 Super HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8)
- [Nemotron 3 Nano HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [NVIDIA Nemotron Cookbooks](https://github.com/NVIDIA-NeMo/Nemotron)
- [NeMo Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/)
- [NeMo RL](https://github.com/NVIDIA-NeMo/RL)
- [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)
