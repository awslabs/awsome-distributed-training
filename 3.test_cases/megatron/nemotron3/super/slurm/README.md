# Nemotron 3 Super — LoRA SFT on Slurm

This guide covers LoRA fine-tuning of [NVIDIA Nemotron 3 Super](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8) (120B total / 12B active parameters) on a Slurm cluster with AWS EFA networking.

## Model Overview

Nemotron 3 Super is a hybrid LatentMoE + Mamba-2 + Attention + MTP model with:
- **120B total parameters**, 12B active per token
- **512 + 1 shared experts**, 22 active per token (LatentMoE routing)
- **1M token** context window
- **Multi-Token Prediction (MTP)** for speculative decoding-aware training
- Optimized for reasoning, code generation, and agentic AI

## Prerequisites

- A Slurm cluster on AWS (HyperPod or ParallelCluster)
- Docker with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- FSx for Lustre mounted at `/fsx` on all nodes
- Minimum: 1 node with 8x H100/H200/B200 GPUs (FP8 recommended)
- **600GB+ disk** space for model checkpoints

### Instance Compatibility

| Instance | GPUs | VRAM/GPU | LoRA SFT | Recommended Parallelism |
|----------|------|----------|----------|------------------------|
| p6-B200 | 8x B200 | 180GB | 1 node | TP=2, EP=TBD, PP=1 |
| p5en.48xlarge | 8x H200 | 141GB | 1 node | TP=2, EP=TBD, PP=1 |
| p5.48xlarge | 8x H100 | 80GB | 1 node (FP8 required) | TP=2, EP=TBD, PP=1 |
| p4de.24xlarge | 8x A100 | 80GB | Marginal | Not recommended |

> **Note**: Expert parallelism (EP) values are TBD pending validation with the
> 512-expert LatentMoE architecture. FP8 precision is strongly recommended on Hopper GPUs.

## Setup

### 1. Clone this repository

```bash
cd /fsx
git clone https://github.com/awslabs/awsome-distributed-training.git
cd awsome-distributed-training/3.test_cases/megatron/nemotron3/super/slurm
```

### 2. Build the container

```bash
docker build --progress=plain -t aws-nemotron3-super:26.02 -f ../Dockerfile ..
enroot import -o ~/aws-nemotron3-super.sqsh dockerd://aws-nemotron3-super:26.02
```

### 3. Install host dependencies

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev

python3.10 -m venv nemotron-env
source nemotron-env/bin/activate
bash venv.sh
```

### 4. Download and convert model checkpoint

First, download the HuggingFace model:

```bash
# Set HuggingFace token if needed for gated models
export HF_TOKEN=your_token_here

# Download FP8 model (recommended, requires ~300GB disk)
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 --local-dir /fsx/models/nemotron3-super-hf

# Or download BF16 model (larger, requires ~600GB disk)
# huggingface-cli download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16 --local-dir /fsx/models/nemotron3-super-hf
```

Then convert to Megatron format inside the container:

```bash
srun --partition=dev --nodes=1 --ntasks-per-node=1 --gpus=1 --mem=0 \
    --container-image=$HOME/aws-nemotron3-super.sqsh \
    --container-mounts=/fsx:/fsx \
    bash -c "cd /opt/Megatron-Bridge && python examples/conversion/convert_checkpoints.py import \
        --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
        --megatron-path /fsx/models/nemotron3-super-megatron \
        --trust-remote-code"
```

> **Note**: Checkpoint conversion for Super uses the `nemotronh` architecture
> in Megatron Bridge, which supports the LatentMoE + Mamba-2 + Attention + MTP
> hybrid layers.

## Launch LoRA Fine-Tuning

### Default dataset (SQuAD) with FP8

```bash
python run_lora_sft.py \
    --container_image ~/aws-nemotron3-super.sqsh \
    --nodes 1 \
    --partition dev \
    --megatron_ckpt_path /fsx/models/nemotron3-super-megatron \
    --max_steps 100 \
    --global_batch_size 64 \
    --precision fp8
```

### Custom dataset

You can point to any HuggingFace dataset:

```bash
python run_lora_sft.py \
    --container_image ~/aws-nemotron3-super.sqsh \
    --nodes 1 \
    --partition dev \
    --dataset gretelai/synthetic_text_to_sql \
    --megatron_ckpt_path /fsx/models/nemotron3-super-megatron \
    --max_steps 200
```

### Full fine-tuning (no LoRA)

For full parameter fine-tuning, use `--peft none` and increase nodes:

```bash
python run_lora_sft.py \
    --container_image ~/aws-nemotron3-super.sqsh \
    --nodes 2 \
    --partition dev \
    --peft none \
    --megatron_ckpt_path /fsx/models/nemotron3-super-megatron
```

## Launch GRPO Reinforcement Learning

GRPO requires 2+ nodes for Super due to the larger model size.

### Build the GRPO container

```bash
docker build --progress=plain -t aws-nemotron3-super-grpo:26.02 -f ../Dockerfile.grpo ..
enroot import -o ~/aws-nemotron3-super-grpo.sqsh dockerd://aws-nemotron3-super-grpo:26.02
```

### Launch GRPO

```bash
python run_grpo.py \
    --container_image ~/aws-nemotron3-super-grpo.sqsh \
    --nodes 2 \
    --partition dev \
    --model_path /fsx/models/nemotron3-super-hf
```

## Post-Training: Merge LoRA and Export

After training, merge the LoRA adapters back to the base model:

```bash
srun --partition=dev --nodes=1 --ntasks-per-node=1 --gpus=1 --mem=0 \
    --container-image=$HOME/aws-nemotron3-super.sqsh \
    --container-mounts=/fsx:/fsx \
    bash -c "cd /opt/Megatron-Bridge && python examples/peft/merge_lora.py \
        --hf-model-path nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
        --lora-checkpoint /fsx/results/nemotron3-super-lora/checkpoints/iter_XXXXXXX \
        --output /fsx/results/nemotron3-super-lora/merged_hf_checkpoint"
```

## Performance Tuning

### Environment Variables

The `env_vars.json` file contains NCCL and EFA tuning parameters:

| Variable | Value | Purpose |
|----------|-------|---------|
| `TORCH_NCCL_AVOID_RECORD_STREAMS` | 1 | Reduce NCCL memory usage |
| `NVTE_DP_AMAX_REDUCE_INTERVAL` | 0 | TransformerEngine FP8 amax tuning |
| `NVTE_ASYNC_AMAX_REDUCTION` | 1 | Async amax reduction |
| `NVTE_FUSED_ATTN` | 0 | Disable fused attention (compatibility) |
| `FI_EFA_USE_HUGE_PAGE` | 0 | Disable huge pages for EFA |
| `FI_PROVIDER` | efa | Force EFA provider |
| `NCCL_DEBUG` | INFO | NCCL debug logging |

### Troubleshooting

**EFA not detected:**
```bash
fi_info -p efa  # Should show EFA provider
```

**OOM errors:**
- Ensure FP8 precision is enabled: `--precision fp8` (critical for Super on H100)
- Reduce `--global_batch_size` or `--micro_batch_size`
- Increase `--tp` (tensor parallelism) to shard the model across more GPUs
- Increase `--pp` (pipeline parallelism) to split model across more GPU groups
- Use the FP8 model variant instead of BF16

**Slow training:**
- Verify EFA with `NCCL_DEBUG=INFO` in logs
- Check NVLink: `nvidia-smi nvlink -s`
- Ensure GPU clocks are not throttled: `nvidia-smi -q -d CLOCK`

**Disk space:**
- Super requires ~600GB for checkpoints (FP8 variant is smaller at ~300GB)
- Ensure `/fsx` volume has sufficient free space before downloading

## References

- [Nemotron 3 Super FP8 on HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8)
- [Nemotron 3 Super BF16 on HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16)
- [NeMo Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotronh.html)
- [Megatron Bridge GitHub](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
