# Nemotron 3 Super — Kubernetes Deployment

This guide covers deploying Nemotron 3 Super (120B/12B active) LoRA SFT and GRPO training on Kubernetes (EKS) with AWS EFA networking.

## Prerequisites

- Amazon EKS cluster with GPU nodes (P5/P5en/P6)
- [Kubeflow Training Operator](https://github.com/kubeflow/training-operator) installed
- EFA device plugin enabled
- PersistentVolumeClaim for FSx storage (600GB+ free)
- ECR repository for container images

## Setup

### 1. Build and push container

```bash
# LoRA/SFT container
bash build.sh
bash push.sh

# GRPO container
bash build.sh Dockerfile.grpo
# Update IMAGE_NAME in push.sh or tag manually
```

### 2. Container images

| Purpose | Dockerfile | Base Image |
|---------|-----------|-----------|
| LoRA SFT | `../Dockerfile` | `nvcr.io/nvidia/nemo:26.02.nemotron_3_super` |
| GRPO/DAPO | `../Dockerfile.grpo` | `nvcr.io/nvidia/nemo-rl:v0.5.0` |

> **Note**: The LoRA/SFT container uses the dedicated Nemotron 3 Super NeMo image
> which includes all Mamba-2, LatentMoE, and MTP dependencies pre-installed.

### 3. LoRA SFT on Kubernetes

Use the NeMo-Run SkyPilot executor to launch on K8s:

```bash
python lora_sft.py \
    --cloud kubernetes \
    --gpus H200 \
    --num_nodes 1 \
    --pvc_name fsx-pvc \
    --hf_model_id nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
```

With EFA for multi-node:

```bash
python lora_sft.py \
    --cloud kubernetes \
    --gpus H200 \
    --num_nodes 2 \
    --pvc_name fsx-pvc \
    --enable_efa --efa_devices 4
```

> **Note**: FP8 model variant is recommended. Super requires significantly more
> memory than Nano — use H200 or B200 GPUs for single-node LoRA.

### 4. GRPO on Kubernetes

GRPO requires 2+ nodes for Super:

```bash
python grpo_training.py \
    --cloud kubernetes \
    --gpus H200 \
    --num_nodes 2 \
    --pvc_name fsx-pvc \
    --hf_model_id nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
```

With custom reward environment:

```bash
python grpo_training.py \
    --reward_env custom \
    --reward_script /mnt/nemo/my_reward.py \
    --num_nodes 2 \
    --pvc_name fsx-pvc
```

## Default Parallelism

| Parameter | Default | Notes |
|-----------|---------|-------|
| TP | 2 | Tensor parallelism (12B active params needs TP>1) |
| EP | 4 | Expert parallelism (512+1 experts) |
| PP | 1 | Pipeline parallelism |
| CP | 1 | Context parallelism |

These defaults target p5en (8x H200) with FP8 precision. Adjust based on
your GPU type and number of nodes.

## References

- [Nemotron 3 Super FP8 on HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8)
- [Nemotron 3 Super BF16 on HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16)
- [NeMo Megatron Bridge — Nemotron H](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotronh.html)
- [NVIDIA Nemotron Developer Repository](https://github.com/NVIDIA-NeMo/Nemotron)
