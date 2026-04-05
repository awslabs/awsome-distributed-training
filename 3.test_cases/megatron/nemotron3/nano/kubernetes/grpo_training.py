"""
Nemotron 3 Nano — GRPO/DAPO Reinforcement Learning on Kubernetes with NeMo-Run

This script uses NeMo-Run with a SkyPilot executor to launch GRPO (Group
Relative Policy Optimization) or DAPO (Direct Advantage Policy Optimization)
training for Nemotron 3 Nano on a Kubernetes cluster (e.g. Amazon EKS).

The GRPO container (built from ../Dockerfile.grpo) is based on
nvcr.io/nvidia/nemo-rl:v0.5.0 and includes NeMo Gym for reward environments.

Training pipeline:
    1. Load base model (HuggingFace format)
    2. Generate rollouts using vLLM inference engine
    3. Compute rewards via NeMo Gym environment (math/code/custom)
    4. Update policy with GRPO/DAPO
    5. Repeat until convergence

Usage:
    # GRPO with math reasoning environment
    python grpo_training.py \\
        --gpus H100 \\
        --num_nodes 2 \\
        --pvc_name fsx-pvc \\
        --pvc_mount /mnt/nemo \\
        --hf_model_id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

    # DAPO algorithm
    python grpo_training.py \\
        --algorithm dapo \\
        --num_nodes 2 \\
        --pvc_name fsx-pvc \\
        --pvc_mount /mnt/nemo

    # Code reward environment with custom rollout settings
    python grpo_training.py \\
        --reward_env code \\
        --rollout_batch_size 256 \\
        --num_rollouts_per_prompt 16 \\
        --pvc_name fsx-pvc \\
        --pvc_mount /mnt/nemo

    # Custom reward function
    python grpo_training.py \\
        --reward_env custom \\
        --reward_script /mnt/nemo/my_reward.py \\
        --pvc_name fsx-pvc \\
        --pvc_mount /mnt/nemo

    # Multi-node with EFA
    python grpo_training.py \\
        --num_nodes 4 \\
        --enable_efa --efa_devices 4 \\
        --pvc_name fsx-pvc \\
        --pvc_mount /mnt/nemo

Supported GPU types:
    - H100 (p5.48xlarge)   — 1+ nodes
    - H200 (p5en.48xlarge) — 1+ nodes
    - B200 (p6-B200)       — 1+ nodes
    - A100 (p4de.24xlarge) — 2+ nodes (recommend multi-node for RL)

Container image:
    Uses the AWS-optimized GRPO image built from ../Dockerfile.grpo,
    based on nvcr.io/nvidia/nemo-rl:v0.5.0 with EFA support and NeMo Gym.
"""

import nemo_run as run
import json
import argparse
import os
from datetime import datetime
from typing import Optional

from nemo.utils import logging


# ---------------------------------------------------------------------------
# Default container image — matches kubernetes/build.sh Dockerfile.grpo
# ---------------------------------------------------------------------------
DEFAULT_GRPO_CONTAINER_IMAGE = "aws-nemotron3-nano-grpo:v0.5.0"

# Default HuggingFace model identifier
DEFAULT_HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser for Nemotron 3 Nano GRPO on Kubernetes."""

    parser = argparse.ArgumentParser(
        description="Nemotron 3 Nano GRPO/DAPO on Kubernetes (NeMo-Run + SkyPilot)",
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
        "--num_nodes", type=int, default=2,
        help="Number of nodes (recommend 2+ for RL workloads)",
    )
    k8s.add_argument(
        "--gpu_devices", type=int, default=8,
        help="Number of GPUs per node",
    )
    k8s.add_argument(
        "--container_image", type=str, default=DEFAULT_GRPO_CONTAINER_IMAGE,
        help="GRPO container image URI (local tag or ECR URI)",
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
        "--model_path", type=str, default=None,
        help="Local model path on PVC (if already downloaded); overrides --hf_model_id",
    )
    model.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace token for gated model access",
    )

    # -- Algorithm ------------------------------------------------------------
    algo = parser.add_argument_group("Algorithm")
    algo.add_argument(
        "--algorithm", type=str, default="grpo",
        choices=["grpo", "dapo"],
        help="RL algorithm: 'grpo' (Group Relative Policy Optimization) or 'dapo' (Direct Advantage Policy Optimization)",
    )

    # -- Training hyper-parameters -------------------------------------------
    train = parser.add_argument_group("Training")
    train.add_argument(
        "--max_steps", type=int, default=200,
        help="Maximum RL training steps",
    )
    train.add_argument(
        "--lr", type=float, default=1e-6,
        help="Policy learning rate",
    )
    train.add_argument(
        "--kl_coeff", type=float, default=0.01,
        help="KL divergence penalty coefficient",
    )

    # -- Rollout parameters ---------------------------------------------------
    rollout = parser.add_argument_group("Rollout")
    rollout.add_argument(
        "--rollout_batch_size", type=int, default=512,
        help="Number of prompts per rollout batch",
    )
    rollout.add_argument(
        "--num_rollouts_per_prompt", type=int, default=8,
        help="Number of rollout completions generated per prompt",
    )
    rollout.add_argument(
        "--max_prompt_length", type=int, default=1024,
        help="Maximum prompt length in tokens",
    )
    rollout.add_argument(
        "--max_response_length", type=int, default=2048,
        help="Maximum response length in tokens",
    )

    # -- Parallelism ----------------------------------------------------------
    par = parser.add_argument_group("Parallelism")
    par.add_argument("--tp", type=int, default=1, help="Tensor parallelism for actor")
    par.add_argument("--ep", type=int, default=8, help="Expert parallelism")
    par.add_argument("--pp", type=int, default=1, help="Pipeline parallelism")

    # -- Reward environment ---------------------------------------------------
    reward = parser.add_argument_group("Reward Environment")
    reward.add_argument(
        "--reward_env", type=str, default="math",
        choices=["math", "code", "custom"],
        help="Reward environment: 'math' (NeMo Gym), 'code', or 'custom'",
    )
    reward.add_argument(
        "--reward_script", type=str, default=None,
        help="Path to custom reward script (required when --reward_env=custom)",
    )

    # -- Dataset --------------------------------------------------------------
    data = parser.add_argument_group("Dataset")
    data.add_argument(
        "--train_data", type=str, default="nvidia/nemotron-post-training-v3",
        help="Training prompt dataset (HuggingFace ID or local path)",
    )

    return parser


# ---------------------------------------------------------------------------
# GRPO configuration builder
# ---------------------------------------------------------------------------

def build_grpo_config(args) -> dict:
    """Build the NeMo RL GRPO/DAPO configuration dictionary.

    This configuration is written to JSON and passed to the NeMo RL
    training entrypoint inside the container.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Configuration dictionary for NeMo RL training.
    """
    # Determine model path: use explicit path or derive from PVC mount
    model_path = args.model_path or f"{args.pvc_mount}/models/nemotron3-nano-hf"

    config = {
        "algorithm": args.algorithm.upper(),
        "model": {
            "name_or_path": model_path,
            "trust_remote_code": True,
        },
        "actor": {
            "learning_rate": args.lr,
            "tensor_parallel_size": args.tp,
            "expert_parallel_size": args.ep,
            "pipeline_parallel_size": args.pp,
        },
        "rollout": {
            "batch_size": args.rollout_batch_size,
            "num_rollouts_per_prompt": args.num_rollouts_per_prompt,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "engine": "vllm",
            "vllm_config": {
                "tensor_parallel_size": args.tp,
                "trust_remote_code": True,
                "gpu_memory_utilization": 0.85,
            },
        },
        "reward": {
            "environment": args.reward_env,
        },
        "training": {
            "max_steps": args.max_steps,
            "kl_coeff": args.kl_coeff,
            "save_interval": 50,
            "log_interval": 1,
            "output_dir": f"{args.pvc_mount}/results/nemotron3-nano-{args.algorithm}",
        },
    }

    if args.reward_env == "custom" and args.reward_script:
        config["reward"]["custom_script"] = args.reward_script

    return config


# ---------------------------------------------------------------------------
# NeMo RL training task (wrapped for NeMo-Run execution)
# ---------------------------------------------------------------------------

def build_nemorl_command(args, config_path: str) -> list[str]:
    """Build the NeMo RL training command for execution inside the container.

    Args:
        args: Parsed command-line arguments.
        config_path: Container-side path to the GRPO config JSON file.

    Returns:
        Command list for NeMo RL training.
    """
    model_path = args.model_path or f"{args.pvc_mount}/models/nemotron3-nano-hf"

    cmd = [
        "python", "-m", "nemo_rl.train",
        "--algorithm", args.algorithm,
        "--model-path", model_path,
        "--trust-remote-code",
        "--reward-env", args.reward_env,
        "--rollout-batch-size", str(args.rollout_batch_size),
        "--num-rollouts-per-prompt", str(args.num_rollouts_per_prompt),
        "--max-prompt-length", str(args.max_prompt_length),
        "--max-response-length", str(args.max_response_length),
        "--actor-lr", str(args.lr),
        "--kl-coeff", str(args.kl_coeff),
        "--max-steps", str(args.max_steps),
        "--output-dir", f"{args.pvc_mount}/results/nemotron3-nano-{args.algorithm}",
        "--tensor-parallel-size", str(args.tp),
        "--expert-parallel-size", str(args.ep),
        "--pipeline-parallel-size", str(args.pp),
    ]

    if args.reward_env == "custom" and args.reward_script:
        cmd.extend(["--reward-script", args.reward_script])

    return cmd


# ---------------------------------------------------------------------------
# SkyPilot executor for Kubernetes
# ---------------------------------------------------------------------------

def skypilot_executor(
    cloud: str,
    num_nodes: int,
    pvc_mount: str,
    gpu_devices: int,
    gpus: str = "H100",
    container_image: str = DEFAULT_GRPO_CONTAINER_IMAGE,
    env_vars_file: str = "env_vars.json",
    pvc_name: str = "fsx-pvc",
    enable_efa: bool = False,
    efa_devices: int = 4,
    hf_token: Optional[str] = None,
    custom_mounts: Optional[dict[str, str]] = None,
) -> run.SkypilotExecutor:
    """Configure a SkyPilot executor targeting a Kubernetes cluster for GRPO.

    Sets up PVC mounts for persistent storage, environment variables for
    NCCL/EFA tuning, and optional EFA resource requests.

    Args:
        cloud: SkyPilot cloud backend ('kubernetes').
        num_nodes: Number of nodes to schedule.
        pvc_mount: Container-side mount path for the PVC.
        gpu_devices: Number of GPUs per node.
        gpus: GPU type string (e.g. 'H100', 'H200', 'A100').
        container_image: GRPO container image URI.
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

    # Validate custom reward configuration
    if args.reward_env == "custom" and not args.reward_script:
        raise ValueError("--reward_script is required when --reward_env=custom")

    pvc_mount = args.pvc_mount
    output_dir = f"{pvc_mount}/results/nemotron3-nano-{args.algorithm}"
    config_dir = f"{pvc_mount}/configs"

    exp_name = f"nemotron3-nano-{args.algorithm}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # -----------------------------------------------------------------------
    # Build GRPO/DAPO configuration
    # -----------------------------------------------------------------------
    grpo_config = build_grpo_config(args)

    # Write config to a local file that will be packaged via GitArchivePackager
    config_filename = f"{exp_name}-config.json"
    os.makedirs("configs", exist_ok=True)
    local_config_path = os.path.join("configs", config_filename)
    with open(local_config_path, "w") as f:
        json.dump(grpo_config, f, indent=2)

    # -----------------------------------------------------------------------
    # Build training command
    # -----------------------------------------------------------------------
    container_config_path = f"{config_dir}/{config_filename}"
    train_cmd = build_nemorl_command(args, container_config_path)

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

    # -----------------------------------------------------------------------
    # Launch experiment
    # -----------------------------------------------------------------------
    print(f"Experiment:     {exp_name}")
    print(f"Algorithm:      {args.algorithm.upper()}")
    print(f"Model:          {args.hf_model_id}")
    print(f"Nodes:          {args.num_nodes} x {args.gpu_devices} GPUs ({args.gpus})")
    print(f"Parallelism:    TP={args.tp}, EP={args.ep}, PP={args.pp}")
    print(f"Reward env:     {args.reward_env}")
    print(f"Rollout:        {args.rollout_batch_size} prompts x {args.num_rollouts_per_prompt} completions")
    print(f"Max steps:      {args.max_steps}")
    print(f"KL coeff:       {args.kl_coeff}")
    print(f"EFA:            {'enabled' if args.enable_efa else 'disabled'}")
    print(f"PVC:            {args.pvc_name} -> {pvc_mount}")
    print(f"Config:         {local_config_path}")

    with run.Experiment(exp_name, log_level="INFO") as exp:
        # NeMo RL GRPO/DAPO training
        # The run.Script wrapper allows running arbitrary commands via NeMo-Run
        exp.add(
            run.Script(
                inline=" ".join(train_cmd),
            ),
            executor=executor,
            tail_logs=True,
            name=f"{args.algorithm}_training",
        )
        exp.run(detach=True)
