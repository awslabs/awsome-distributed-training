"""
Nemotron 3 Super — GRPO/DAPO Reinforcement Learning with NeMo RL on Slurm

This script launches GRPO (Group Relative Policy Optimization) or DAPO
(Direct Advantage Policy Optimization) training for Nemotron 3 Super
(120B total / 12B active) using NeMo RL and NeMo Gym on a Slurm cluster.

Architecture: LatentMoE + Mamba-2 + Attention + MTP hybrid.

Usage:
    # GRPO with math reasoning environment (FP8 model, 2 nodes minimum)
    python run_grpo.py \\
        --container_image ~/aws-nemotron3-super-grpo.sqsh \\
        --nodes 2 --partition dev \\
        --hf_model_id nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8

    # DAPO algorithm
    python run_grpo.py \\
        --container_image ~/aws-nemotron3-super-grpo.sqsh \\
        --nodes 2 --partition dev \\
        --algorithm dapo

    # Custom reward environment
    python run_grpo.py \\
        --container_image ~/aws-nemotron3-super-grpo.sqsh \\
        --nodes 4 --partition dev \\
        --reward_env custom \\
        --reward_script /fsx/my_reward.py

Training pipeline:
    1. Load base model (HuggingFace format)
    2. Generate rollouts using vLLM inference engine
    3. Compute rewards via NeMo Gym environment
    4. Update policy with GRPO/DAPO
    5. Repeat until convergence

Supported instance types (GRPO requires more resources than SFT):
    - p6-B200 (8x B200 180GB) — 1-2 nodes
    - p5en.48xlarge (8x H200 141GB) — 2+ nodes
    - p5.48xlarge (8x H100 80GB) — 2+ nodes (FP8 required)
    - p4de.24xlarge (8x A100 80GB) — Not recommended for Super

Note: Super (120B) requires significantly more memory than Nano (30B).
      2+ nodes is the minimum recommendation for GRPO with Super.
"""

import json
import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser(
        description="Nemotron 3 Super GRPO/DAPO on Slurm"
    )

    # Slurm configuration
    parser.add_argument("--partition", type=str, default="dev",
                        help="Slurm partition")
    parser.add_argument("--nodes", type=int, default=2,
                        help="Number of nodes (2+ recommended for Super GRPO)")
    parser.add_argument("--account", type=str, default="ubuntu",
                        help="Slurm account")
    parser.add_argument("--container_image", type=str, required=True,
                        help="Path to GRPO Enroot squash file (.sqsh)")
    parser.add_argument("--time", type=str, default="12:00:00",
                        help="Slurm time limit (longer default for Super)")
    parser.add_argument("--ntasks_per_node", type=int, default=8,
                        help="GPUs per node")
    parser.add_argument("--env_vars_file", type=str, default="env_vars.json",
                        help="Environment variables JSON")

    # Model — default to FP8 variant
    parser.add_argument("--hf_model_id", type=str,
                        default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
                        help="HuggingFace model ID "
                             "(also supports nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16)")
    parser.add_argument("--model_path", type=str, default="/fsx/models/nemotron3-super-hf",
                        help="Local path to downloaded HF model (~600GB)")

    # Algorithm
    parser.add_argument("--algorithm", type=str, default="grpo",
                        choices=["grpo", "dapo"],
                        help="RL algorithm: grpo or dapo")

    # Training hyperparameters
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Maximum RL training steps")
    parser.add_argument("--rollout_batch_size", type=int, default=256,
                        help="Number of prompts per rollout batch (reduced vs Nano)")
    parser.add_argument("--num_rollouts_per_prompt", type=int, default=8,
                        help="Number of rollout completions per prompt")
    parser.add_argument("--max_prompt_length", type=int, default=1024,
                        help="Maximum prompt length in tokens")
    parser.add_argument("--max_response_length", type=int, default=2048,
                        help="Maximum response length in tokens")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Policy learning rate")
    parser.add_argument("--kl_coeff", type=float, default=0.01,
                        help="KL divergence penalty coefficient")

    # Parallelism — Super defaults
    parser.add_argument("--tp", type=int, default=2,
                        help="Tensor parallelism for actor (default 2 for Super)")
    parser.add_argument("--ep", type=int, default=4,
                        help="Expert parallelism (TBD pending validation with 512 experts)")
    parser.add_argument("--pp", type=int, default=1,
                        help="Pipeline parallelism")

    # Reward environment (abstracted)
    parser.add_argument("--reward_env", type=str, default="math",
                        choices=["math", "code", "custom"],
                        help="Reward environment: math (NeMo Gym), code, or custom")
    parser.add_argument("--reward_script", type=str, default=None,
                        help="Path to custom reward script (when --reward_env=custom)")

    # Dataset
    parser.add_argument("--train_data", type=str,
                        default="nvidia/nemotron-post-training-v3",
                        help="Training prompt dataset (HF ID or local path)")

    # Output
    parser.add_argument("--output_dir", type=str,
                        default="/fsx/results/nemotron3-super-grpo",
                        help="Output directory")

    return parser


def build_grpo_config(args) -> dict:
    """Build NeMo RL GRPO configuration dictionary."""
    config = {
        "algorithm": args.algorithm.upper(),
        "model": {
            "name_or_path": args.model_path,
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
            "output_dir": args.output_dir,
        },
    }

    if args.reward_env == "custom" and args.reward_script:
        config["reward"]["custom_script"] = args.reward_script

    return config


if __name__ == "__main__":
    args = get_parser().parse_args()

    import random
    import string

    suffix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    exp_name = f"nemotron3-super-{args.algorithm}-{suffix}"

    # Load env vars
    with open(args.env_vars_file, 'r') as f:
        env_vars = json.load(f)

    # Build GRPO config
    grpo_config = build_grpo_config(args)

    # Write config to file
    os.makedirs(f"{args.output_dir}/configs", exist_ok=True)
    config_path = f"{args.output_dir}/configs/{exp_name}.json"
    with open(config_path, 'w') as f:
        json.dump(grpo_config, f, indent=2)

    # Build environment variable exports
    env_exports = "\n".join([f"export {k}={v}" for k, v in env_vars.items()])

    # NeMo RL uses an asynchronous architecture:
    # - Actor nodes: run policy training (Megatron-based)
    # - Rollout nodes: run vLLM inference for generating completions
    # - Reward: computed via NeMo Gym environments
    #
    # The NeMo RL launcher handles GPU allocation across these roles.
    # Super requires more GPU memory per role due to its 120B parameter count.
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

# Environment variables for NCCL/EFA
{env_exports}

# HuggingFace and NeMo paths
export HF_HOME=/fsx/.cache/huggingface
export NEMO_HOME={args.output_dir}/nemo_home

# NeMo RL GRPO training
# The nemo-rl container includes the training entrypoint
srun --container-image={args.container_image} \\
    --container-mounts={args.output_dir}:{args.output_dir},{os.path.dirname(args.model_path)}:{os.path.dirname(args.model_path)},/fsx/.cache:/fsx/.cache \\
    --no-container-mount-home \\
    python -m nemo_rl.train \\
        --algorithm {args.algorithm} \\
        --model-path {args.model_path} \\
        --trust-remote-code \\
        --reward-env {args.reward_env} \\
        --rollout-batch-size {args.rollout_batch_size} \\
        --num-rollouts-per-prompt {args.num_rollouts_per_prompt} \\
        --max-prompt-length {args.max_prompt_length} \\
        --max-response-length {args.max_response_length} \\
        --actor-lr {args.lr} \\
        --kl-coeff {args.kl_coeff} \\
        --max-steps {args.max_steps} \\
        --output-dir {args.output_dir} \\
        --tensor-parallel-size {args.tp} \\
        --expert-parallel-size {args.ep} \\
        --pipeline-parallel-size {args.pp}
"""

    # Write sbatch script
    os.makedirs(f"{args.output_dir}/logs", exist_ok=True)
    sbatch_path = f"{args.output_dir}/{exp_name}.sbatch"
    with open(sbatch_path, 'w') as f:
        f.write(sbatch_script)

    print(f"Experiment: {exp_name}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Model: {args.hf_model_id}")
    print(f"Nodes: {args.nodes} ({args.nodes * args.ntasks_per_node} GPUs)")
    print(f"Parallelism: TP={args.tp}, EP={args.ep}, PP={args.pp}")
    print(f"Reward env: {args.reward_env}")
    print(f"Config: {config_path}")
    print(f"Sbatch: {sbatch_path}")
    print(f"\nSubmit with: sbatch {sbatch_path}")
