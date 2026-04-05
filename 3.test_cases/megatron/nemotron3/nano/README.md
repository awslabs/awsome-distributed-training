# Nemotron 3 Nano (30B / 3.5B Active)

NVIDIA Nemotron 3 Nano is a hybrid Mamba-2 + MoE + Attention model with 30B total parameters and 3.5B active parameters per token. It is designed for efficient agentic AI, reasoning, and code generation with a 1M token context window.

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
| `Dockerfile` | `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano` | LoRA SFT and full fine-tuning |
| `Dockerfile.grpo` | `nvcr.io/nvidia/nemo-rl:v0.5.0` | GRPO/DAPO reinforcement learning |

## Instance Compatibility

| Instance | GPUs | VRAM | LoRA SFT | Full FT | GRPO |
|----------|------|------|----------|---------|------|
| p6-B200 (8x B200) | 8 | 1440GB | 1 node | 1 node | 1 node |
| p5en.48xlarge (8x H200) | 8 | 1128GB | 1 node | 1 node | 1 node |
| p5.48xlarge (8x H100) | 8 | 640GB | 1 node | 1-2 nodes | 1-2 nodes |
| p4de.24xlarge (8x A100) | 8 | 640GB | 1 node | 2 nodes | 2+ nodes |

## Model Details

- **Architecture**: Mamba-2 + MoE + Attention hybrid
- **Experts**: 128 + 1 shared, 5 active per token
- **Context**: Up to 1M tokens
- **Languages**: English, French, German, Italian, Japanese, Spanish, Chinese
- **License**: [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/)
- **HuggingFace**: [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- **NGC Container**: `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano`
