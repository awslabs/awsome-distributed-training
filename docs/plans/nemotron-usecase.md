# Nemotron 3 Use Case — Implementation Plan

> Created: 2026-03-12 | Status: Building
> Branch: `feature/nemotron-usecase`
> Worktree: `/Users/nchkumar/Code/smml-work/adt-nemotron`
> Base: `main` (commit f20676d5)

## Overview

Add training use cases for the NVIDIA Nemotron 3 model family to `3.test_cases/megatron/nemotron3/`.
Two models: **Nano** (30B total / 3.5B active) and **Super** (120B total / 12B active).
Two training modes: **LoRA SFT** and **GRPO/DAPO reinforcement learning**.

---

## Model Specs

| | Nemotron 3 Nano | Nemotron 3 Super |
|---|---|---|
| Total params | 30B | 120B |
| Active params | 3.5B | 12B |
| Architecture | Mamba-2 + MoE + Attention hybrid | LatentMoE + Mamba-2 + Attention + MTP |
| Experts | 128 + 1 shared, 5 active/token | 512 + shared, 22 active/token |
| Context | 1M tokens | 1M tokens |
| NGC Container | `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano` | `nvcr.io/nvidia/nemo:26.02.nemotron_3_super` |
| Megatron Bridge | Fully supported (pretrain, FFT, LoRA) | LoRA SFT via cookbook; nemotronh recipe |
| HuggingFace | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` |
| Released | December 2025 | March 11, 2026 |

---

## Directory Structure

```
3.test_cases/megatron/nemotron3/
├── README.md                          # Family overview
├── nano/
│   ├── README.md
│   ├── Dockerfile                     # LoRA/SFT container
│   ├── Dockerfile.grpo                # GRPO/RL container (separate)
│   ├── PERFORMANCE.md
│   ├── slurm/
│   │   ├── README.md
│   │   ├── run_lora_sft.py
│   │   ├── run_grpo.py
│   │   ├── venv.sh
│   │   └── env_vars.json
│   └── kubernetes/
│       ├── README.md
│       ├── Dockerfile
│       ├── Dockerfile.grpo
│       ├── build.sh / push.sh
│       ├── lora_sft.py
│       ├── grpo_training.py
│       ├── venv.sh
│       └── env_vars.json
└── super/
    ├── README.md
    ├── Dockerfile
    ├── Dockerfile.grpo
    ├── PERFORMANCE.md
    ├── slurm/
    │   ├── README.md
    │   ├── run_lora_sft.py
    │   ├── run_grpo.py
    │   ├── venv.sh
    │   └── env_vars.json
    └── kubernetes/
        ├── README.md
        ├── Dockerfile
        ├── Dockerfile.grpo
        ├── build.sh / push.sh
        ├── lora_sft.py
        ├── grpo_training.py
        ├── venv.sh
        └── env_vars.json
```

---

## Training Capabilities

### 1. LoRA SFT (Supervised Fine-Tuning)

- **Framework**: NeMo Megatron Bridge
- **Container**: NeMo base container + AWS networking
- **Default dataset**: SQuAD (Nano, matching Bridge defaults) / Text2SQL (Super, matching NVIDIA cookbook)
- **Custom datasets**: Abstracted via configurable `--dataset` flag or HF dataset ID
- **Checkpoint flow**: Import HF → Megatron → LoRA fine-tune → Merge → Export HF

### 2. GRPO/DAPO Reinforcement Learning

- **Framework**: NeMo RL + NeMo Gym (NOT veRL)
- **Container**: Separate NeMo RL container (`nvcr.io/nvidia/nemo-rl:v0.5.0`) + AWS networking
- **Why not veRL**: Nemotron-H architecture (Mamba-2 + LatentMoE) is not supported in veRL.
  veRL has no Mamba-2 model initializer, no LatentMoE support, broken SP for hybrid models (Issue #5552).
  NeMo RL is NVIDIA's own framework that was used to train Nemotron 3 Nano and Super.
- **Default environment**: Math reasoning via NeMo Gym
- **Custom environments**: Abstracted to allow custom reward functions

---

## Instance Compatibility Matrix

| Instance | GPUs | GPU | VRAM/GPU | Nano LoRA | Nano GRPO | Super LoRA | Super GRPO |
|----------|------|-----|----------|-----------|-----------|------------|------------|
| p6-B200 | 8 | B200 | 180GB | 1 node | 1 node | 1 node | 1-2 nodes |
| p5en.48xlarge | 8 | H200 | 141GB | 1 node | 1 node | 1 node | 2+ nodes |
| p5.48xlarge | 8 | H100 | 80GB | 1 node | 1-2 nodes | 1 node (FP8) | 2+ nodes |
| p4de.24xlarge | 8 | A100 | 80GB | 1 node | 2+ nodes | Marginal | Multi-node |
| g6e.48xlarge | 8 | L40S | 48GB | Possible | Unlikely | No | No |
| g5.48xlarge | 8 | A10G | 24GB | No | No | No | No |

---

## Parallelism Configurations

### Nano (30B total, 3.5B active)

| Config | Instance | TP | EP | PP | CP | GBS | MBS |
|--------|----------|----|----|----|----|-----|-----|
| LoRA SFT | p5 (1 node) | 1 | 8 | 1 | 1 | 128 | 1 |
| LoRA SFT | p4de (1 node) | 1 | 8 | 1 | 1 | 64 | 1 |
| FFT | p5 (2 nodes) | 1 | 8 | 1 | 1 | 128 | 1 |
| Pretrain | p5 (4 nodes) | 4 | 8 | 1 | 1 | 3072 | - |

### Super (120B total, 12B active)

| Config | Instance | TP | EP | PP | CP | GBS | MBS |
|--------|----------|----|----|----|----|-----|-----|
| LoRA SFT | p5en (1 node) | 2 | 4 | 1 | 1 | 64 | 1 |
| LoRA SFT | p5 H100 (1 node) | 2 | 4 | 2 | 1 | 64 | 1 |
| GRPO | p5en (2 nodes) | 2 | 4 | 1 | 1 | 32 | 1 |

---

## Performance Optimization Pointers

1. **FP8 precision** on Hopper (H100/H200) — significant memory savings
2. **NVFP4** on Blackwell (B200) — 4x faster than FP8 on Hopper
3. **Expert Parallelism (EP)** — critical for MoE; Nano default EP=8
4. **Activation checkpointing** — trade compute for memory on large models
5. **Async checkpoint saving** — via NeMo-Run fault tolerance plugins
6. **Sequence packing** — pack shorter sequences to maximize GPU utilization
7. **EFA tuning**: `FI_PROVIDER=efa`, `FI_EFA_USE_HUGE_PAGE=0`, `NCCL_DEBUG=INFO`
8. **TransformerEngine**: `NVTE_DP_AMAX_REDUCE_INTERVAL=0`, `NVTE_ASYNC_AMAX_REDUCTION=1`

---

## Container Strategy

- **LoRA/SFT containers**: Based on NeMo base images + AWS EFA/NCCL stack
  - Nano: `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano`
  - Super: `nvcr.io/nvidia/nemo:26.02.nemotron_3_super` (dedicated container with Mamba-2, LatentMoE, MTP)
- **GRPO/RL containers**: Separate builds based on NeMo RL image
  - Both: `nvcr.io/nvidia/nemo-rl:v0.5.0` + AWS EFA/NCCL stack

---

## Implementation Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Nano directory structure + Dockerfile + LoRA SFT (Slurm) | Building |
| 2 | Nano LoRA SFT (Kubernetes) | Pending |
| 3 | Nano GRPO (Slurm) with NeMo RL | Pending |
| 4 | Super Dockerfile + LoRA SFT (Slurm + Kubernetes) | Building |
| 5 | Super GRPO (Slurm + Kubernetes) with NeMo RL | Building |
| 6 | Super validation on p5en | In Progress |
| 7 | Multi-instance validation + PERFORMANCE.md | Pending (needs capacity) |
| 8 | README documentation polish | Pending |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-12 | Created branch `feature/nemotron-usecase` | Isolated git worktree |
| 2026-03-12 | Directory: `nemotron3/` with `super/` and `nano/` subdirs | Future-proof for Ultra |
| 2026-03-12 | RL framework: NeMo RL (not veRL) | veRL lacks Mamba-2/LatentMoE support |
| 2026-03-12 | Separate containers for LoRA and GRPO | Different base images (NeMo vs NeMo RL) |
| 2026-03-12 | Start with Nano, then Super | Nano is cheaper to validate, has dedicated NGC container |
| 2026-03-12 | Dataset abstraction: defaults + custom interface | Match NVIDIA cookbooks but allow user datasets |
| 2026-03-12 | Super base image: nemo:25.07.00 | Latest stable; may update post-GTC if dedicated container released |
| 2026-04-04 | Super base image updated: nemo:26.02.nemotron_3_super | Dedicated Super container found on NGC with Mamba-2/LatentMoE/MTP pre-installed |
| 2026-04-04 | Super EP=4 default | SGLang and vLLM reference configs from model card use EP=4 for 512 experts |
| 2026-04-04 | Super K8s scripts created | lora_sft.py and grpo_training.py added for EKS deployment |
| 2026-04-04 | Priority shift: Super first on p5en | p5en nodes available; Super is the higher-value target |
| 2026-04-04 | LatentMoE + EP research completed | All EP backends handle LatentMoE transparently; 4x smaller all-to-all payloads; no special adaptation needed |

---

## LatentMoE + Expert Parallelism Analysis

> Completed: 2026-04-04

### Architecture

LatentMoE wraps the routed expert path with shared linear projections:

1. **Down-projection** (W_down): `d=4096 → ℓ=1024` — applied locally on each GPU BEFORE dispatch
2. **Routing + dispatch all-to-all**: operates on latent tensors `[S, ℓ=1024]` (4x smaller)
3. **Expert computation**: experts are `[m × ℓ]` and `[ℓ × m]` (4x smaller weights)
4. **Combine all-to-all**: returns latent tensors `[S, ℓ=1024]` (4x smaller)
5. **Up-projection** (W_up): `ℓ=1024 → d=4096` — applied locally on each GPU AFTER combine

Router still operates on full hidden dim `d=4096`. Shared experts also operate in full `d`.

### EP Backend Compatibility

**All standard EP backends handle LatentMoE transparently** — the compression is orthogonal:

| Backend | Status | Notes |
|---------|--------|-------|
| NCCL all-to-all | Works | Payloads are `[S, ℓ]` instead of `[S, d]` |
| NCCL EP LL mode | Works | Per-token send/recv 4x smaller — major EFA advantage |
| DeepEP (LL/HT) | Works | Same dispatch protocol, just smaller tensors |
| Megatron EP | Works | Native `--moe-latent-size` parameter |

Per-token dispatch payload: 22 × 1024 × 2B = **45 KB** (vs 180 KB for standard MoE with d=4096).

### Framework Implementations

- **vLLM**: `NemotronHMoE` with `fc1_latent_proj` as `routed_input_transform`. PR #32790 adds shared/routed CUDA stream overlap. `--enable-expert-parallel` confirmed working (TP=4).
- **SGLang**: `--tp 4 --ep 4` confirmed in NVIDIA deployment guide.
- **TRT-LLM**: `--tp_size 2 --ep_size 2` (2-GPU) or `--tp_size 8 --ep_size 8` (8-GPU).
- **Megatron-LM/NeMo**: `--moe-latent-size` + `expert_model_parallel_size`.

### 2x p5en EFA Implications

- NCCL EP LL on EFA should work with ~4x less per-token communication vs Qwen3-235B
- 512 experts with EP=4 → ~128 experts per GPU (within EP group)
- W_down/W_up gradient all-reduce is additional training cost (small: 4096×1024 matrices)
- **Risk: LOW** — no special EP backend adaptation needed

### References

- [LatentMoE paper (Elango et al., Jan 2026)](https://arxiv.org/abs/2601.18089)
- [Multi-Head LatentMoE (Cui et al., Feb 2026)](https://arxiv.org/abs/2602.04870) — research extension, NOT used by Super
- [NVIDIA Advanced Deployment Guide](https://docs.nvidia.com/nemotron/nightly/usage-cookbook/Nemotron-3-Super/AdvancedDeploymentGuide/README.html)
- [vLLM PR #32790 — Shared/Routed Overlap for Latent MoE](https://github.com/vllm-project/vllm/pull/32790)

---

## Key References

- [Nemotron 3 Super HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8)
- [Nemotron 3 Nano HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [NVIDIA Nemotron GitHub (cookbooks)](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3-Super)
- [NeMo Megatron Bridge - Nemotron 3 Nano](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html)
- [NeMo Megatron Bridge - Nemotron H](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotronh.html)
- [NeMo RL](https://github.com/NVIDIA-NeMo/RL)
- [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)
- [Existing NeMo test case (repo pattern)](3.test_cases/megatron/nemo/)
