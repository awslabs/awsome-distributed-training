# Nemotron 3 Nano — LoRA SFT on Slurm

This guide covers LoRA fine-tuning of [NVIDIA Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) (30B total / 3.5B active parameters) on a Slurm cluster with AWS EFA networking.

## Model Overview

Nemotron 3 Nano is a hybrid Mamba-2 + MoE + Attention model with:
- **30B total parameters**, 3.5B active per token
- **128 + 1 shared experts**, 5 active per token
- **1M token** context window
- Optimized for agentic AI, reasoning, and code generation

## Prerequisites

- A Slurm cluster on AWS (HyperPod or ParallelCluster)
- Docker with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- FSx for Lustre mounted at `/fsx` on all nodes
- Minimum: 1 node with 8x H100/H200/A100 80GB GPUs

### Instance Compatibility

| Instance | GPUs | VRAM/GPU | LoRA SFT | Recommended Parallelism |
|----------|------|----------|----------|------------------------|
| p6-B200 | 8x B200 | 180GB | 1 node | TP=1, EP=8, PP=1 |
| p5en.48xlarge | 8x H200 | 141GB | 1 node | TP=1, EP=8, PP=1 |
| p5.48xlarge | 8x H100 | 80GB | 1 node | TP=1, EP=8, PP=1 |
| p4de.24xlarge | 8x A100 | 80GB | 1 node | TP=1, EP=8, PP=1 |

## Setup

### 1. Clone this repository

```bash
cd /fsx
git clone https://github.com/awslabs/awsome-distributed-training.git
cd awsome-distributed-training/3.test_cases/megatron/nemotron3/nano/slurm
```

### 2. Build the container

```bash
docker build --progress=plain -t aws-nemotron3-nano:25.11 -f ../Dockerfile ..
enroot import -o ~/aws-nemotron3-nano.sqsh dockerd://aws-nemotron3-nano:25.11
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

# Download model (requires ~60GB disk)
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --local-dir /fsx/models/nemotron3-nano-hf
```

Then convert to Megatron format inside the container:

```bash
srun --partition=dev --nodes=1 --ntasks-per-node=1 --gpus=1 --mem=0 \
    --container-image=$HOME/aws-nemotron3-nano.sqsh \
    --container-mounts=/fsx:/fsx \
    bash -c "cd /opt/Megatron-Bridge && python examples/conversion/convert_checkpoints.py import \
        --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --megatron-path /fsx/models/nemotron3-nano-megatron \
        --trust-remote-code"
```

## Launch LoRA Fine-Tuning

### Default dataset (SQuAD)

```bash
python run_lora_sft.py \
    --container_image ~/aws-nemotron3-nano.sqsh \
    --nodes 1 \
    --partition dev \
    --megatron_ckpt_path /fsx/models/nemotron3-nano-megatron \
    --max_steps 100 \
    --global_batch_size 128
```

### Custom dataset

You can point to any HuggingFace dataset:

```bash
python run_lora_sft.py \
    --container_image ~/aws-nemotron3-nano.sqsh \
    --nodes 1 \
    --partition dev \
    --dataset gretelai/synthetic_text_to_sql \
    --megatron_ckpt_path /fsx/models/nemotron3-nano-megatron \
    --max_steps 200
```

### Full fine-tuning (no LoRA)

For full parameter fine-tuning, use `--peft none` and increase nodes:

```bash
python run_lora_sft.py \
    --container_image ~/aws-nemotron3-nano.sqsh \
    --nodes 2 \
    --partition dev \
    --peft none \
    --megatron_ckpt_path /fsx/models/nemotron3-nano-megatron
```

## Post-Training: Merge LoRA and Export

After training, merge the LoRA adapters back to the base model:

```bash
srun --partition=dev --nodes=1 --ntasks-per-node=1 --gpus=1 --mem=0 \
    --container-image=$HOME/aws-nemotron3-nano.sqsh \
    --container-mounts=/fsx:/fsx \
    bash -c "cd /opt/Megatron-Bridge && python examples/peft/merge_lora.py \
        --hf-model-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --lora-checkpoint /fsx/results/nemotron3-nano-lora/checkpoints/iter_XXXXXXX \
        --output /fsx/results/nemotron3-nano-lora/merged_hf_checkpoint"
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
- Reduce `--global_batch_size` or `--micro_batch_size`
- Enable FP8 precision: `--precision fp8`
- Increase `--pp` (pipeline parallelism) to split model across more GPU groups

**Slow training:**
- Verify EFA with `NCCL_DEBUG=INFO` in logs
- Check NVLink: `nvidia-smi nvlink -s`
- Ensure GPU clocks are not throttled: `nvidia-smi -q -d CLOCK`

## References

- [Nemotron 3 Nano on HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [NeMo Megatron Bridge — Nemotron 3 Nano](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html)
- [Megatron Bridge GitHub](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
