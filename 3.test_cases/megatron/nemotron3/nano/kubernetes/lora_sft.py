"""
Nemotron 3 Nano — LoRA SFT Fine-Tuning on Kubernetes with NeMo-Run

This script uses NeMo-Run with a SkyPilot executor to launch LoRA (or full)
fine-tuning of Nemotron 3 Nano on a Kubernetes cluster (e.g. Amazon EKS with
GPU nodes and optional EFA networking).

The training flow is a two-step sequential experiment:
    1. Import the HuggingFace checkpoint to Megatron format (llm.import_ckpt)
    2. Run LoRA fine-tuning via the Megatron Bridge entry point

Usage:
    # Default: LoRA SFT on a single H100 node
    python lora_sft.py \\
        --pvc_name fsx-pvc \\
        --pvc_mount /mnt/nemo \\
        --hf_model_id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

    # Multi-node with EFA
    python lora_sft.py \\
        --num_nodes 2 \\
        --gpus H200 \\
        --pvc_name fsx-pvc \\
        --pvc_mount /mnt/nemo \\
        --enable_efa --efa_devices 4

    # Full fine-tuning (no LoRA)
    python lora_sft.py \\
        --pvc_name fsx-pvc \\
        --pvc_mount /mnt/nemo \\
        --peft none

    # Custom HuggingFace dataset
    python lora_sft.py \\
        --pvc_name fsx-pvc \\
        --pvc_mount /mnt/nemo \\
        --dataset gretelai/synthetic_text_to_sql

Supported GPU types:
    - H100 (p5.48xlarge)   — 1+ nodes
    - H200 (p5en.48xlarge) — 1+ nodes
    - B200 (p6-B200)       — 1+ nodes
    - A100 (p4de.24xlarge) — 1+ nodes

Container image:
    Uses the AWS-optimized Nemotron 3 Nano image built from ../Dockerfile,
    based on nvcr.io/nvidia/nemo:25.11.nemotron_3_nano with EFA support.
"""

import nemo_run as run
import json
import argparse
import os
import signal
from datetime import datetime
from typing import Optional

from nemo.collections import llm
from nemo.lightning.run import plugins
from nemo.lightning.pytorch.callbacks import PreemptionCallback
from nemo.utils import logging


# ---------------------------------------------------------------------------
# Default container image — matches kubernetes/build.sh output
# ---------------------------------------------------------------------------
DEFAULT_CONTAINER_IMAGE = "aws-nemotron3-nano:25.11"

# Default HuggingFace model identifier
DEFAULT_HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser for Nemotron 3 Nano LoRA SFT on Kubernetes."""

    parser = argparse.ArgumentParser(
        description="Nemotron 3 Nano LoRA SFT on Kubernetes (NeMo-Run + SkyPilot)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Kubernetes / SkyPilot configuration ---------------------------------
    k8s = parser.add_argument_group("Kubernetes / SkyPilot")
    k8s.add_argument(
        "--cloud", type=str, default="kubernetes",
        help="SkyPilot cloud backend (use 'kubernetes' for EKS)",
    )
    k8s.add_argument(
        "--gpus", type=str, default="H100",
        help="GPU type available on the cluster (e.g. H100, H200, A100, B200)",
    )
    k8s.add_argument(
        "--num_nodes", type=int, default=1,
        help="Number of nodes to schedule",
    )
    k8s.add_argument(
        "--gpu_devices", type=int, default=8,
        help="Number of GPUs per node",
    )
    k8s.add_argument(
        "--container_image", type=str, default=DEFAULT_CONTAINER_IMAGE,
        help="Container image URI (local tag or ECR URI)",
    )
    k8s.add_argument(
        "--env_vars_file", type=str, default="env_vars.json",
        help="Path to JSON file with environment variables for NCCL/EFA tuning",
    )

    # -- Persistent storage ---------------------------------------------------
    pvc = parser.add_argument_group("Persistent Volume")
    pvc.add_argument(
        "--pvc_name", type=str, default="fsx-pvc",
        help="Name of the Kubernetes PersistentVolumeClaim",
    )
    pvc.add_argument(
        "--pvc_mount", type=str, default="/mnt/nemo",
        help="Mount path for the PVC inside the container",
    )

    # -- EFA networking -------------------------------------------------------
    efa = parser.add_argument_group("EFA Networking")
    efa.add_argument(
        "--enable_efa", action="store_true", default=False,
        help="Enable AWS EFA by requesting vpc.amazonaws.com/efa resources",
    )
    efa.add_argument(
        "--efa_devices", type=int, default=4,
        help="Number of EFA devices per node (only used when --enable_efa is set)",
    )

    # -- Model ----------------------------------------------------------------
    model = parser.add_argument_group("Model")
    model.add_argument(
        "--hf_model_id", type=str, default=DEFAULT_HF_MODEL_ID,
        help="HuggingFace model ID for Nemotron 3 Nano",
    )
    model.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace token for gated model access",
    )

    # -- Training hyper-parameters -------------------------------------------
    train = parser.add_argument_group("Training")
    train.add_argument(
        "--max_steps", type=int, default=100,
        help="Maximum number of training steps",
    )
    train.add_argument(
        "--global_batch_size", type=int, default=128,
        help="Global batch size across all GPUs",
    )
    train.add_argument(
        "--micro_batch_size", type=int, default=1,
        help="Micro batch size per GPU",
    )
    train.add_argument(
        "--seq_length", type=int, default=4096,
        help="Maximum sequence length for fine-tuning",
    )
    train.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate",
    )
    train.add_argument(
        "--lr_warmup_iters", type=int, default=10,
        help="Number of learning-rate warmup iterations",
    )

    # -- PEFT / LoRA ----------------------------------------------------------
    peft_group = parser.add_argument_group("PEFT")
    peft_group.add_argument(
        "--peft", type=str, default="lora", choices=["lora", "none"],
        help="PEFT method: 'lora' for LoRA fine-tuning, 'none' for full fine-tuning",
    )
    peft_group.add_argument(
        "--lora_rank", type=int, default=16,
        help="LoRA rank (r)",
    )
    peft_group.add_argument(
        "--lora_alpha", type=float, default=32,
        help="LoRA alpha scaling factor",
    )

    # -- Parallelism ----------------------------------------------------------
    par = parser.add_argument_group("Parallelism")
    par.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
    par.add_argument("--ep", type=int, default=8, help="Expert parallelism degree")
    par.add_argument("--pp", type=int, default=1, help="Pipeline parallelism degree")
    par.add_argument("--cp", type=int, default=1, help="Context parallelism degree")

    # -- Dataset --------------------------------------------------------------
    data = parser.add_argument_group("Dataset")
    data.add_argument(
        "--dataset", type=str, default="squad",
        help="Dataset name: 'squad' (default), HuggingFace dataset ID, or local JSONL path",
    )
    data.add_argument(
        "--dataset_split", type=str, default="train",
        help="Dataset split to use for training",
    )

    # -- Precision ------------------------------------------------------------
    parser.add_argument(
        "--precision", type=str, default="bf16",
        choices=["bf16", "fp8", "mxfp8"],
        help="Training precision (bf16 for A100/H100, fp8/mxfp8 for H100/H200/B200)",
    )

    return parser


# ---------------------------------------------------------------------------
# Checkpoint import configuration
# ---------------------------------------------------------------------------

def configure_checkpoint_import(hf_model_id: str, pvc_mount: str) -> run.Partial:
    """Configure the HuggingFace -> Megatron checkpoint import step.

    This uses llm.import_ckpt to convert the HuggingFace model into
    the Megatron checkpoint format expected by the Megatron Bridge.

    Args:
        hf_model_id: HuggingFace model identifier (e.g. nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16).
        pvc_mount: PVC mount path where the converted checkpoint will be stored.

    Returns:
        A run.Partial wrapping llm.import_ckpt for deferred execution.
    """
    return run.Partial(
        llm.import_ckpt,
        model=llm.nemotron3_nano.model(),
        source=f"hf://{hf_model_id}",
        overwrite=False,
    )


# ---------------------------------------------------------------------------
# Fine-tuning recipe configuration
# ---------------------------------------------------------------------------

def configure_finetune_recipe(
    exp_name: str,
    work_dir: str,
    peft_scheme: Optional[str],
    lora_enabled: bool,
    max_steps: int,
    num_nodes: int,
    gpu_devices: int,
    global_batch_size: int,
    micro_batch_size: int,
    seq_length: int,
) -> llm.FineTuneRecipe:
    """Build the NeMo fine-tuning recipe for Nemotron 3 Nano.

    Configures the trainer, data, and parallelism settings for the
    Megatron Bridge finetune_nemotron_3_nano entry point.

    Args:
        exp_name: Experiment name for logging and checkpointing.
        work_dir: Base directory for experiment outputs.
        peft_scheme: PEFT method ('lora' or None for full fine-tuning).
        lora_enabled: Whether LoRA adapters are active.
        max_steps: Maximum number of training steps.
        num_nodes: Number of compute nodes.
        gpu_devices: GPUs per node.
        global_batch_size: Global batch size.
        micro_batch_size: Per-GPU micro batch size.
        seq_length: Maximum sequence length.

    Returns:
        Configured NeMo fine-tuning recipe.
    """
    finetune_recipe = llm.nemotron3_nano.finetune_recipe(
        num_nodes=num_nodes,
        name=exp_name,
        dir=work_dir,
        peft_scheme=peft_scheme,
    )

    # Trainer settings
    finetune_recipe.trainer.devices = gpu_devices
    finetune_recipe.trainer.num_sanity_val_steps = 0
    finetune_recipe.trainer.max_steps = max_steps
    finetune_recipe.trainer.val_check_interval = 10
    finetune_recipe.trainer.log_every_n_steps = 1
    finetune_recipe.trainer.strategy.context_parallel_size = 1

    # Batch sizes
    finetune_recipe.data.global_batch_size = global_batch_size
    finetune_recipe.data.micro_batch_size = micro_batch_size
    finetune_recipe.data.seq_length = seq_length

    # LoRA-specific: use Megatron DDP for parameter-efficient training
    if lora_enabled:
        finetune_recipe.trainer.strategy.ddp = "megatron"

    return finetune_recipe


# ---------------------------------------------------------------------------
# SkyPilot executor for Kubernetes
# ---------------------------------------------------------------------------

def skypilot_executor(
    cloud: str,
    num_nodes: int,
    pvc_mount: str,
    gpu_devices: int,
    gpus: str = "H100",
    container_image: str = DEFAULT_CONTAINER_IMAGE,
    env_vars_file: str = "env_vars.json",
    pvc_name: str = "fsx-pvc",
    enable_efa: bool = False,
    efa_devices: int = 4,
    hf_token: Optional[str] = None,
    custom_mounts: Optional[dict[str, str]] = None,
) -> run.SkypilotExecutor:
    """Configure a SkyPilot executor targeting a Kubernetes cluster.

    Sets up PVC mounts for persistent storage, environment variables for
    NCCL/EFA tuning, and optional EFA resource requests for multi-node
    training on AWS.

    Args:
        cloud: SkyPilot cloud backend ('kubernetes').
        num_nodes: Number of nodes to schedule.
        pvc_mount: Container-side mount path for the PVC.
        gpu_devices: Number of GPUs per node.
        gpus: GPU type string (e.g. 'H100', 'H200', 'A100').
        container_image: Container image URI.
        env_vars_file: Path to JSON file with environment variables.
        pvc_name: Kubernetes PVC name.
        enable_efa: Whether to request EFA devices.
        efa_devices: Number of EFA devices per node.
        hf_token: Optional HuggingFace API token.
        custom_mounts: Additional file mounts {container_path: local_path}.

    Returns:
        Configured run.SkypilotExecutor.
    """
    # File mounts
    mounts = {}
    if custom_mounts:
        mounts.update(custom_mounts)

    # Environment variables from JSON config
    with open(env_vars_file, "r") as f:
        env_vars = json.load(f)

    packager = run.GitArchivePackager()

    # Kubernetes pod configuration: PVC volume mount
    shared_pod_config = {
        "kubernetes": {
            "pod_config": {
                "spec": {
                    "containers": [{
                        "volumeMounts": [
                            {"name": "nemo-runs", "mountPath": pvc_mount},
                        ],
                    }],
                    "volumes": [{
                        "name": "nemo-runs",
                        "persistentVolumeClaim": {"claimName": pvc_name},
                    }],
                },
            },
        },
    }

    # Optional EFA resource requests for high-bandwidth inter-node communication
    if enable_efa:
        shared_pod_config["kubernetes"]["pod_config"]["spec"]["containers"][0]["resources"] = {
            "requests": {"vpc.amazonaws.com/efa": efa_devices},
            "limits": {"vpc.amazonaws.com/efa": efa_devices},
        }

    # Build executor
    executor = run.SkypilotExecutor(
        cloud=cloud,
        gpus=gpus,
        gpus_per_node=gpu_devices,
        num_nodes=num_nodes,
        packager=packager,
        cluster_config_overrides=shared_pod_config,
    )

    executor.container_image = container_image
    executor.file_mounts = mounts

    # Environment variables
    executor.env_vars = env_vars
    executor.env_vars["NEMORUN_HOME"] = pvc_mount
    executor.env_vars["NEMO_HOME"] = f"{pvc_mount}/nemo"
    executor.env_vars["NEMO_MODELS_CACHE"] = f"{pvc_mount}/nemo/cache"
    executor.env_vars["HF_HOME"] = f"{pvc_mount}/huggingface"
    executor.env_vars["HF_HUB_CACHE"] = f"{pvc_mount}/huggingface/hub"

    if hf_token:
        executor.env_vars["HF_TOKEN"] = hf_token
    else:
        logging.info("No HuggingFace token provided; gated repositories may be inaccessible.")

    return executor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = get_parser().parse_args()

    pvc_mount = args.pvc_mount
    work_dir = os.path.join(pvc_mount, "experiments")

    # Determine PEFT scheme
    lora_enabled = args.peft == "lora"
    peft_scheme = "lora" if lora_enabled else None
    peft_tag = "lora-sft" if lora_enabled else "full-sft"
    exp_name = f"nemotron3-nano-{peft_tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # -----------------------------------------------------------------------
    # Step 1: Checkpoint import (HuggingFace -> Megatron)
    # -----------------------------------------------------------------------
    import_ckpt = configure_checkpoint_import(
        hf_model_id=args.hf_model_id,
        pvc_mount=pvc_mount,
    )

    # -----------------------------------------------------------------------
    # Step 2: Fine-tuning recipe
    # -----------------------------------------------------------------------
    finetune_recipe = configure_finetune_recipe(
        exp_name=exp_name,
        work_dir=work_dir,
        peft_scheme=peft_scheme,
        lora_enabled=lora_enabled,
        max_steps=args.max_steps,
        num_nodes=args.num_nodes,
        gpu_devices=args.gpu_devices,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        seq_length=args.seq_length,
    )

    # -----------------------------------------------------------------------
    # Executor
    # -----------------------------------------------------------------------
    executor = skypilot_executor(
        cloud=args.cloud,
        num_nodes=args.num_nodes,
        gpus=args.gpus,
        gpu_devices=args.gpu_devices,
        container_image=args.container_image,
        env_vars_file=args.env_vars_file,
        pvc_name=args.pvc_name,
        pvc_mount=pvc_mount,
        enable_efa=args.enable_efa,
        efa_devices=args.efa_devices,
        hf_token=args.hf_token,
        custom_mounts={
            "/root/nemo": ".",
        },
    )

    # Multi-node: use torchrun launcher for distributed training
    if args.num_nodes > 1:
        executor.launcher = "torchrun"

    # Clone executor for the checkpoint import step (runs single-node)
    import_executor = executor.clone()

    # -----------------------------------------------------------------------
    # Launch experiment
    # -----------------------------------------------------------------------
    print(f"Experiment:  {exp_name}")
    print(f"Model:       {args.hf_model_id}")
    print(f"PEFT:        {args.peft}")
    print(f"Nodes:       {args.num_nodes} x {args.gpu_devices} GPUs ({args.gpus})")
    print(f"Batch size:  {args.global_batch_size} global / {args.micro_batch_size} micro")
    print(f"Max steps:   {args.max_steps}")
    print(f"Precision:   {args.precision}")
    print(f"EFA:         {'enabled' if args.enable_efa else 'disabled'}")
    print(f"PVC:         {args.pvc_name} -> {pvc_mount}")
    print(f"Work dir:    {work_dir}")

    with run.Experiment(exp_name, log_level="INFO") as exp:
        # Step 1: Import checkpoint
        exp.add(
            import_ckpt,
            executor=import_executor,
            name="checkpoint_import",
        )
        # Step 2: Fine-tuning
        exp.add(
            finetune_recipe,
            executor=executor,
            tail_logs=True,
            name="lora_finetuning" if lora_enabled else "full_finetuning",
        )
        # Run sequentially: import must complete before training starts
        exp.run(sequential=True, detach=True)
