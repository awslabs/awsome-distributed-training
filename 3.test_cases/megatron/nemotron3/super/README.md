# Nemotron 3 Super (120B / 12B Active)

NVIDIA Nemotron 3 Super is a hybrid LatentMoE + Mamba-2 + Attention + MTP (Multi-Token Prediction) model with 120B total parameters and 12B active parameters per token. Released March 11, 2026, it is designed for high-quality reasoning, code generation, and agentic AI with a 1M token context window.

## Architecture Highlights

- **LatentMoE**: 512 experts + 1 shared expert, 22 active per token (latent routing in compressed space)
- **Mamba-2 layers**: Linear-time sequence modeling for long-context efficiency
- **Attention layers**: Grouped-query attention for retrieval-heavy operations
- **MTP (Multi-Token Prediction)**: Speculative decoding-aware training for faster inference

## Training Capabilities

| Capability | Framework | Status |
|-----------|-----------|--------|
| LoRA SFT | NeMo Megatron Bridge | Supported |
| Full Fine-Tuning | NeMo Megatron Bridge | Supported |
| GRPO/DAPO RL | NeMo RL + NeMo Gym | Supported |
| Pretraining | Megatron-LM | Supported |

## Quick Start

- **LoRA SFT on Slurm**: See [slurm/README.md](slurm/README.md)
- **LoRA SFT on Kubernetes**: See [kubernetes/README.md](kubernetes/README.md)
- **GRPO Reinforcement Learning**: See [slurm/README.md](slurm/README.md) (GRPO section)

## Containers

| Container | Base Image | Purpose |
|-----------|-----------|---------|
| `Dockerfile` | `nvcr.io/nvidia/nemo:26.02.nemotron_3_super` | LoRA SFT and full fine-tuning |
| `Dockerfile.grpo` | `nvcr.io/nvidia/nemo-rl:v0.5.0` | GRPO/DAPO reinforcement learning |

> **Note**: The LoRA/SFT container uses the dedicated Nemotron 3 Super NeMo image
> which includes all Mamba-2, LatentMoE, and MTP dependencies pre-installed.

## Instance Compatibility

| Instance | GPUs | VRAM | LoRA SFT | Full FT | GRPO |
|----------|------|------|----------|---------|------|
| p6-B200 (8x B200) | 8 | 1440GB | 1 node | 1 node | 1-2 nodes |
| p5en.48xlarge (8x H200) | 8 | 1128GB | 1 node | 1-2 nodes | 2+ nodes |
| p5.48xlarge (8x H100) | 8 | 640GB | 1 node (FP8) | 2+ nodes | 2+ nodes |
| p4de.24xlarge (8x A100) | 8 | 640GB | Marginal | Not recommended | Not recommended |

> **Note**: The Super model (120B total) is significantly larger than Nano (30B total).
> FP8 precision is strongly recommended on Hopper (H100/H200) to fit within memory.
> A100 support is marginal and may require aggressive memory optimization.

## Model Details

- **Architecture**: LatentMoE + Mamba-2 + Attention + MTP hybrid
- **Total Parameters**: 120B
- **Active Parameters**: 12B per token
- **Experts**: 512 + 1 shared, 22 active per token
- **Context**: Up to 1M tokens
- **Languages**: English, French, German, Italian, Japanese, Spanish, Chinese
- **License**: [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/)
- **HuggingFace (FP8)**: [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8)
- **HuggingFace (BF16)**: [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16)
- **NeMo Container**: `nvcr.io/nvidia/nemo:26.02.nemotron_3_super`
