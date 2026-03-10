#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -xeuo pipefail

# ---------------------------------------------------------------------------
# veRL GRPO training — instance-aware configurable recipe
#
# This script auto-detects the instance type and loads the appropriate
# hardware profile before submitting a GRPO training job to Ray.
#
# Profile loading order:
#   1. env_vars (cluster config, model paths, tokens)
#   2. Instance profile (FSDP strategy, offloading, TP, NCCL settings)
#   3. Explicit env var overrides (anything set after sourcing profile wins)
#
# See recipe/profiles/README.md for available profiles and how to create new ones.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Load instance profile --------------------------------------------------
PROFILE_ENV=$("${SCRIPT_DIR}/profiles/_detect.sh" "${SCRIPT_DIR}/profiles")
echo "Loading instance profile: ${PROFILE_ENV}"
source "$PROFILE_ENV"

# --- Project configuration --------------------------------------------------
project_name='GRPO'
exp_name="GRPO-${MODEL_NAME}"

# --- GRPO Algorithm parameters (task-specific, not instance-dependent) ------
adv_estimator=grpo
use_kl_in_reward=False
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl
entropy_coeff=0

# --- Token length configuration ---------------------------------------------
max_prompt_length=512
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
filter_overlong_prompts=True
truncation='error'

# --- Training configuration -------------------------------------------------
train_prompt_bsz=${TRAIN_BATCH_SIZE:-32}
gen_prompt_bsz=${GEN_BATCH_SIZE:-$train_prompt_bsz}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-2}
train_prompt_mini_bsz=16  # Must be <= train_prompt_bsz
train_prompt_micro_bsz_per_gpu=1

# --- Ray configuration from env_vars ----------------------------------------
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}

# --- Cluster configuration (from profile, overridable by env_vars) ----------
NNODES=${NUM_NODES:-4}
GPUS_PER_NODE=${NUM_GPU_PER_NODE:-8}

# --- Model and data paths from env_vars -------------------------------------
MODEL_NAME=${MODEL_NAME:-"Qwen3-8B"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
RAY_DATA_HOME=${RAY_DATA_HOME:-"/fsx/verl"}
CKPTS_DIR="${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"

# --- Data files --------------------------------------------------------------
TRAIN_FILE="${RAY_DATA_HOME}/data/gsm8k/train.parquet"
TEST_FILE="${RAY_DATA_HOME}/data/gsm8k/test.parquet"

# --- Performance parameters (from profile, overridable) ---------------------
gen_tp=${TENSOR_PARALLEL_SIZE:-2}
log_prob_micro_bsz_per_gpu=${LOG_PROB_MICRO_BSZ_PER_GPU:-32}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.6}
enforce_eager=${ENFORCE_EAGER:-False}

# --- FSDP / memory optimization (from profile) ------------------------------
actor_strategy=${ACTOR_STRATEGY:-fsdp}
model_dtype=${MODEL_DTYPE:-}
param_offload=${PARAM_OFFLOAD:-False}
optimizer_offload=${OPTIMIZER_OFFLOAD:-False}
offload_policy=${OFFLOAD_POLICY:-}
reshard_after_forward=${RESHARD_AFTER_FORWARD:-}
ref_param_offload=${REF_PARAM_OFFLOAD:-True}
rollout_dtype=${ROLLOUT_DTYPE:-}

# --- Checkpoint management (from profile) ------------------------------------
save_freq=${SAVE_FREQ:-1}
test_freq=${TEST_FREQ:-2}
max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-}
total_epochs=${TOTAL_EPOCHS:-2}
resume_mode=${RESUME_MODE:-}

# --- Print configuration for verification -----------------------------------
echo "=== GRPO Training Configuration ==="
echo "Project       : ${project_name}"
echo "Experiment    : ${exp_name}"
echo "Model         : ${MODEL_NAME} (${MODEL_PATH})"
echo "Profile       : ${PROFILE_ENV}"
echo "Nodes         : ${NNODES}"
echo "GPUs/node     : ${GPUS_PER_NODE}"
echo "Total GPUs    : $((NNODES * GPUS_PER_NODE))"
echo "Strategy      : ${actor_strategy}"
echo "Model dtype   : ${model_dtype:-default}"
echo "TP            : ${gen_tp}"
echo "gpu_mem_util  : ${gpu_memory_utilization}"
echo "enforce_eager : ${enforce_eager}"
echo "param_offload : ${param_offload}"
echo "optim_offload : ${optimizer_offload}"
echo "offload_policy: ${offload_policy:-not set}"
echo "ref_offload   : ${ref_param_offload}"
echo "NCCL_PROTO    : ${NCCL_PROTO:-default}"
echo "EFA RDMA      : ${FI_EFA_USE_DEVICE_RDMA:-not set}"
echo "save_freq     : ${save_freq}"
echo "Data home     : ${RAY_DATA_HOME}"
echo "Checkpoints   : ${CKPTS_DIR}"
echo "Ray address   : ${RAY_ADDRESS}"
echo "=================================="

# --- Build ray job submit command dynamically --------------------------------
# Start with required arguments
RAY_CMD=(
    ray job submit --no-wait
    --working-dir "${WORKING_DIR}"
    -- python3 -m verl.trainer.main_ppo
    algorithm.adv_estimator=${adv_estimator}
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=question
    data.train_batch_size=${train_prompt_bsz}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=${filter_overlong_prompts}
    data.truncation=${truncation}
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=${actor_strategy}
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_prompt_micro_bsz_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type}
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_param_offload}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    trainer.critic_warmup=0
    "trainer.logger=[\"console\"]"
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node=${GPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

# --- Conditionally add profile-specific overrides ----------------------------
# Only pass these if the profile set them (avoids sending empty/default values
# that might conflict with veRL's own defaults)

if [[ -n "${offload_policy}" ]]; then
    RAY_CMD+=(actor_rollout_ref.actor.fsdp_config.offload_policy=${offload_policy})
fi

if [[ -n "${model_dtype}" ]]; then
    RAY_CMD+=(actor_rollout_ref.actor.fsdp_config.model_dtype=${model_dtype})
    RAY_CMD+=(actor_rollout_ref.ref.fsdp_config.model_dtype=${model_dtype})
fi

if [[ -n "${reshard_after_forward}" ]]; then
    RAY_CMD+=(actor_rollout_ref.actor.fsdp_config.reshard_after_forward=${reshard_after_forward})
fi

if [[ "${enforce_eager}" == "True" ]]; then
    RAY_CMD+=(actor_rollout_ref.rollout.enforce_eager=True)
fi

if [[ -n "${rollout_dtype}" ]]; then
    RAY_CMD+=(actor_rollout_ref.rollout.dtype=${rollout_dtype})
fi

if [[ -n "${max_actor_ckpt_to_keep}" ]]; then
    RAY_CMD+=(trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep})
fi

if [[ -n "${resume_mode}" ]]; then
    RAY_CMD+=(trainer.resume_mode=${resume_mode})
fi

# --- Submit ------------------------------------------------------------------
"${RAY_CMD[@]}"
