# Nemotron 3 Nano — Kubernetes Deployment

This guide covers deploying Nemotron 3 Nano LoRA SFT and GRPO training on Kubernetes (EKS) with AWS EFA networking.

## Prerequisites

- Amazon EKS cluster with GPU nodes (P5/P5en/P6)
- [Kubeflow Training Operator](https://github.com/kubeflow/training-operator) installed
- EFA device plugin enabled
- PersistentVolumeClaim for FSx storage
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
| LoRA SFT | `../Dockerfile` | `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano` |
| GRPO/DAPO | `../Dockerfile.grpo` | `nvcr.io/nvidia/nemo-rl:v0.5.0` |

### 3. LoRA SFT on Kubernetes

Use the NeMo-Run SkyPilot executor to launch on K8s:

```bash
python lora_sft.py \
    --cloud kubernetes \
    --gpus H100 \
    --num_nodes 1 \
    --pvc_name fsx-pvc \
    --hf_model_id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

### 4. GRPO on Kubernetes

```bash
python grpo_training.py \
    --cloud kubernetes \
    --gpus H100 \
    --num_nodes 2 \
    --pvc_name fsx-pvc \
    --hf_model_id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

## References

- [Nemotron 3 Nano on HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [NeMo Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html)
