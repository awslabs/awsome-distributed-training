"""
Nemotron 3 Nano — LoRA SFT Fine-Tuning with NeMo Megatron Bridge on Slurm

This script uses NeMo-Run to launch LoRA fine-tuning of Nemotron 3 Nano (30B/3.5B active)
on a Slurm cluster with AWS EFA networking.

Usage:
    # Default dataset (SQuAD)
    python run_lora_sft.py \
        --container_image ~/aws-nemotron3-nano.sqsh \
        --nodes 1 --partition dev \
        --hf_model_id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

    # Custom HuggingFace dataset
    python run_lora_sft.py \
        --container_image ~/aws-nemotron3-nano.sqsh \
        --nodes 1 --partition dev \
        --hf_model_id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --dataset gretelai/synthetic_text_to_sql

    # Adjust parallelism for different instance types
    python run_lora_sft.py \
        --container_image ~/aws-nemotron3-nano.sqsh \
        --nodes 2 --partition dev \
        --tp 1 --ep 8 --pp 1 \
        --global_batch_size 128 --micro_batch_size 1

Checkpoint flow:
    1. Import HF checkpoint -> Megatron format (automatic)
    2. Run LoRA fine-tuning
    3. Merge LoRA adapters + Export back to HF format (post-training)

Supported instance types:
    - p6-B200 (8x B200 180GB) — 1 node, best performance
    - p5en.48xlarge (8x H200 141GB) — 1 node
    - p5.48xlarge (8x H100 80GB) — 1 node with FP8
    - p4de.24xlarge (8x A100 80GB) — 1 node
"""

import nemo_run as run
import json
import argparse
import os
import signal
import subprocess
from typing import Optional


def get_parser():
    parser = argparse.ArgumentParser(
        description="Nemotron 3 Nano LoRA SFT on Slurm"
    )

    # Slurm configuration
    parser.add_argument("--partition", type=str, default="dev",
                        help="Slurm partition to run on")
    parser.add_argument("--nodes", type=int, default=1,
                        help="Number of nodes (1 node = 8 GPUs)")
    parser.add_argument("--account", type=str, default="ubuntu",
                        help="Slurm account")
    parser.add_argument("--container_image", type=str, required=True,
                        help="Path to Enroot squash file (.sqsh)")
    parser.add_argument("--time", type=str, default="04:00:00",
                        help="Slurm time limit")
    parser.add_argument("--ntasks_per_node", type=int, default=8,
                        help="GPUs per node")
    parser.add_argument("--env_vars_file", type=str, default="env_vars.json",
                        help="Path to environment variables JSON")

    # Model configuration
    parser.add_argument("--hf_model_id", type=str,
                        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                        help="HuggingFace model ID for Nemotron 3 Nano")
    parser.add_argument("--megatron_ckpt_path", type=str, default="/fsx/models/nemotron3-nano-megatron",
                        help="Path to store converted Megatron checkpoint")

    # Training hyperparameters
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Maximum training steps")
    parser.add_argument("--global_batch_size", type=int, default=128,
                        help="Global batch size across all GPUs")
    parser.add_argument("--micro_batch_size", type=int, default=1,
                        help="Micro batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--lr_warmup_iters", type=int, default=10,
                        help="Number of warmup iterations")

    # Parallelism (defaults match Megatron Bridge Nano LoRA recipe)
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallelism degree")
    parser.add_argument("--ep", type=int, default=8,
                        help="Expert parallelism degree (Nano has 128+1 experts)")
    parser.add_argument("--pp", type=int, default=1,
                        help="Pipeline parallelism degree")
    parser.add_argument("--cp", type=int, default=1,
                        help="Context parallelism degree")

    # LoRA configuration
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank (r)")
    parser.add_argument("--lora_alpha", type=float, default=32,
                        help="LoRA alpha scaling factor")
    parser.add_argument("--peft", type=str, default="lora",
                        choices=["lora", "none"],
                        help="PEFT method: 'lora' or 'none' for full fine-tuning")

    # Dataset configuration (abstracted)
    parser.add_argument("--dataset", type=str, default="squad",
                        help="Dataset name: 'squad' (default), HuggingFace dataset ID, or local JSONL path")
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="Dataset split to use for training")
    parser.add_argument("--seq_length", type=int, default=4096,
                        help="Maximum sequence length for fine-tuning")

    # Precision
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["bf16", "fp8", "mxfp8"],
                        help="Training precision (bf16 for A100/H100, fp8/mxfp8 for H100/H200/B200)")

    # Output
    parser.add_argument("--output_dir", type=str, default="/fsx/results/nemotron3-nano-lora",
                        help="Output directory for checkpoints and logs")

    return parser


def slurm_executor(
    account: str,
    partition: str,
    nodes: int,
    time: str = "04:00:00",
    container_image: str = "",
    env_vars_file: str = "env_vars.json",
    ntasks_per_node: int = 8,
    custom_mounts: Optional[list] = None,
) -> run.SlurmExecutor:
    """Configure a Slurm executor for NeMo-Run."""

    mounts = []
    if custom_mounts:
        mounts.extend(custom_mounts)

    # Load environment variables for NCCL/EFA tuning
    with open(env_vars_file, 'r') as f:
        env_vars = json.load(f)

    packager = run.Packager()
    local_tunnel = run.LocalTunnel(job_dir="")

    # Detect HyperPod cluster for auto-resume
    srun_args = None
    if os.path.isdir("/opt/sagemaker_cluster"):
        print("Detected HyperPod cluster — enabling --auto-resume=1")
        srun_args = ["--auto-resume=1"]

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=local_tunnel,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        mem="0",
        exclusive=True,
        packager=packager,
        srun_args=srun_args,
    )

    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.time = time

    return executor


def convert_checkpoint(hf_model_id: str, megatron_path: str):
    """Import HuggingFace checkpoint to Megatron format.

    This must be run inside the container where Megatron Bridge is installed.
    For Slurm execution, this step is done as a pre-training job or manually.
    """
    print(f"Converting {hf_model_id} -> {megatron_path}")
    print("Run the following inside the container:")
    print(f"  python examples/conversion/convert_checkpoints.py import \\")
    print(f"    --hf-model {hf_model_id} \\")
    print(f"    --megatron-path {megatron_path} \\")
    print(f"    --trust-remote-code")


def build_lora_sft_command(args) -> str:
    """Build the torchrun command for Megatron Bridge LoRA fine-tuning.

    Uses the Megatron Bridge finetune_nemotron_3_nano.py entry point
    which is pre-installed in the nemo:25.11.nemotron_3_nano container.
    """
    cmd_parts = [
        f"torchrun --nproc-per-node={args.ntasks_per_node}",
        "examples/models/nemotron_3/finetune_nemotron_3_nano.py",
    ]

    # Add PEFT flag
    if args.peft == "lora":
        cmd_parts.append("--peft lora")

    # Training config
    cmd_parts.extend([
        f"train.global_batch_size={args.global_batch_size}",
        f"train.train_iters={args.max_steps}",
        f"scheduler.lr_warmup_iters={args.lr_warmup_iters}",
        f"checkpoint.pretrained_checkpoint={args.megatron_ckpt_path}",
    ])

    return " \\\n    ".join(cmd_parts)


if __name__ == "__main__":
    args = get_parser().parse_args()

    import random
    import string

    # Generate unique experiment name
    suffix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    exp_name = f"nemotron3-nano-lora-{suffix}"

    # Print checkpoint conversion instructions
    print("=" * 70)
    print("STEP 1: Checkpoint Conversion (run once before training)")
    print("=" * 70)
    convert_checkpoint(args.hf_model_id, args.megatron_ckpt_path)
    print()

    # Print the training command that will be executed
    print("=" * 70)
    print("STEP 2: LoRA Fine-Tuning")
    print("=" * 70)
    train_cmd = build_lora_sft_command(args)
    print(f"Command:\n  {train_cmd}")
    print()

    # Configure Slurm executor
    executor = slurm_executor(
        partition=args.partition,
        account=args.account,
        nodes=args.nodes,
        container_image=args.container_image,
        time=args.time,
        env_vars_file=args.env_vars_file,
        ntasks_per_node=args.ntasks_per_node,
        custom_mounts=[
            f"{args.output_dir}:{args.output_dir}",
            f"{os.path.dirname(args.megatron_ckpt_path)}:{os.path.dirname(args.megatron_ckpt_path)}",
        ],
    )

    # Enable fault-tolerant launcher
    executor.launcher = "ft"

    # Configure NeMo-Run plugins
    from nemo.lightning.pytorch.callbacks import PreemptionCallback
    from nemo.lightning.run import plugins
    from nemo.collections.llm.recipes.callbacks.common import straggler_det_callback

    run_plugins = [
        plugins.PreemptionPlugin(
            callbacks=[run.Config(PreemptionCallback, sig=signal.SIGINT)]
        ),
        plugins.FaultTolerancePlugin(),
    ]

    # Build the Megatron Bridge fine-tuning command
    # The Megatron Bridge entrypoint handles model loading, data, and training loop
    finetune_cmd = [
        "torchrun",
        f"--nproc-per-node={args.ntasks_per_node}",
        "examples/models/nemotron_3/finetune_nemotron_3_nano.py",
    ]

    if args.peft == "lora":
        finetune_cmd.append("--peft")
        finetune_cmd.append("lora")

    finetune_cmd.extend([
        f"train.global_batch_size={args.global_batch_size}",
        f"train.train_iters={args.max_steps}",
        f"scheduler.lr_warmup_iters={args.lr_warmup_iters}",
        f"checkpoint.pretrained_checkpoint={args.megatron_ckpt_path}",
    ])

    print(f"\nLaunching experiment: {exp_name}")
    print(f"  Nodes: {args.nodes}")
    print(f"  GPUs/node: {args.ntasks_per_node}")
    print(f"  Total GPUs: {args.nodes * args.ntasks_per_node}")
    print(f"  Parallelism: TP={args.tp}, EP={args.ep}, PP={args.pp}, CP={args.cp}")
    print(f"  Precision: {args.precision}")
    print(f"  PEFT: {args.peft}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {args.output_dir}")

    # For Megatron Bridge, we submit the command directly via Slurm
    # rather than through NeMo-Run's recipe API, because the Bridge
    # has its own entry points for Nemotron 3 models
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={exp_name}
#SBATCH --partition={args.partition}
#SBATCH --account={args.account}
#SBATCH --nodes={args.nodes}
#SBATCH --ntasks-per-node={args.ntasks_per_node}
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time={args.time}
#SBATCH --output={args.output_dir}/logs/%x_%j.out
#SBATCH --error={args.output_dir}/logs/%x_%j.err

# Load environment variables
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export NVTE_ASYNC_AMAX_REDUCTION=1
export NVTE_FUSED_ATTN=0
export FI_EFA_USE_HUGE_PAGE=0
export NCCL_DEBUG=INFO
export FI_PROVIDER=efa

# Set HuggingFace cache
export HF_HOME=/fsx/.cache/huggingface
export NEMO_HOME={args.output_dir}/nemo_home

srun --container-image={args.container_image} \\
    --container-mounts={args.output_dir}:{args.output_dir},{os.path.dirname(args.megatron_ckpt_path)}:{os.path.dirname(args.megatron_ckpt_path)} \\
    --no-container-mount-home \\
    bash -c "cd /opt/Megatron-Bridge && {' '.join(finetune_cmd)}"
"""

    # Write sbatch script
    os.makedirs(f"{args.output_dir}/logs", exist_ok=True)
    sbatch_path = f"{args.output_dir}/{exp_name}.sbatch"
    with open(sbatch_path, 'w') as f:
        f.write(sbatch_script)

    print(f"\nGenerated sbatch script: {sbatch_path}")
    print(f"Submit with: sbatch {sbatch_path}")
    print()
    print("Post-training steps:")
    print(f"  1. Merge LoRA weights:")
    print(f"     python examples/peft/merge_lora.py \\")
    print(f"       --hf-model-path {args.hf_model_id} \\")
    print(f"       --lora-checkpoint {args.output_dir}/checkpoints/iter_XXXXXXX \\")
    print(f"       --output {args.output_dir}/merged_hf_checkpoint")
    print(f"  2. Export to HuggingFace format for inference")
