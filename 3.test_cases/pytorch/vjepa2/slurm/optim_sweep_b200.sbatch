#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Optimization sweep for V-JEPA 2 on SageMaker HyperPod with B200 GPUs.
#
# Runs cumulative optimization phases with nsys profiling on rank 0.
# Each phase builds on the previous one to isolate the impact of each change.
#
# Environment variables:
#   PHASE - Which optimization phase to run (A, B, C, D). Required.
#
# Phase definitions (cumulative):
#   A = torch.compile (fix) + no activation checkpointing
#   B = A + fused AdamW + TF32
#   C = B + NCCL_NVLS_ENABLE=1 + NCCL_GRAPH_REGISTER=1 + CUDA_DEVICE_MAX_CONNECTIONS=1
#   D = C + gradient_as_bucket_view + prefetch_factor=4
#
# Usage:
#   PHASE=A sbatch slurm/optim_sweep_b200.sbatch
#   PHASE=B sbatch slurm/optim_sweep_b200.sbatch
#   PHASE=C sbatch slurm/optim_sweep_b200.sbatch
#   PHASE=D sbatch slurm/optim_sweep_b200.sbatch

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --job-name=vjepa2-optim
#SBATCH --output=logs/vjepa2_optim/%A.out
#SBATCH --error=logs/vjepa2_optim/%A.err
#SBATCH --exclusive
#SBATCH --partition=b200
#SBATCH --account=root
#SBATCH --time=01:00:00

set -euo pipefail

PHASE="${PHASE:?ERROR: PHASE env var must be set (A, B, C, or D)}"

###########################
###### User Variables #####
###########################

VJEPA2_DIR=/fsx/${USER}/vjepa2
VJEPA2_CODE=/fsx/${USER}/vjepa2_code
VJEPA_DEPS=/fsx/${USER}/vjepa_deps

# All phases use the optimized config (compile_model + no AC)
CONFIG=${VJEPA2_DIR}/configs/benchmark-vitg-8nodes-optimized.yaml

# nsys output named by phase
export NSYS_OUTPUT=${VJEPA2_DIR}/nsys/optim_phase_${PHASE}/rank0

###########################
## Container Variables ####
###########################

export CONTAINER_IMAGE=/fsx/dgxc-benchmark/images/nvidia+nemo+25.11.01-efa-nccl29.sqsh
export FSX_MOUNT=/fsx:/fsx

###########################
## Environment Variables ##
###########################

# --- Base env (same for all phases) ---
export NCCL_DEBUG=INFO
export FI_PROVIDER=efa
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export NCCL_SOCKET_IFNAME=^docker,lo,veth

# NCCL performance tuning for EFA on HyperPod B200
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_NET_CHUNKSIZE=2097152

# Blackwell-specific
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Limit torch.compile threads per process to prevent OOM when 8 processes
# compile simultaneously on the same node (each spawns many threads by default)
export TORCH_COMPILE_THREADS=1

# PyTorch memory allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Add V-JEPA 2 deps and source code to Python path
export PYTHONPATH=${VJEPA_DEPS}:${VJEPA2_CODE}:${PYTHONPATH}

# --- Phase A: torch.compile (fix) + no AC ---
# (compile_model is correctly under model: in the config now)
# NCCL conservative settings (same as baseline)
export NCCL_NVLS_ENABLE=0
export NCCL_GRAPH_REGISTER=0

# --- Phase B: A + fused optimizer + TF32 ---
# Note: max-autotune causes CUDA allocator assertion failures on B200 with
# no-AC (75+ GB GPU memory + autotune temp buffers exceeds limits).
# Using default compile mode (reduce-overhead) instead.
if [[ "$PHASE" =~ ^[BCD]$ ]]; then
    export VJEPA_FUSED_OPTIMIZER=1
    export VJEPA_TF32=1
fi

# --- Phase C: B + NCCL optimizations ---
if [[ "$PHASE" =~ ^[CD]$ ]]; then
    export NCCL_NVLS_ENABLE=1
    export NCCL_GRAPH_REGISTER=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
fi

# --- Phase D: C + DDP + data loader tuning ---
if [[ "$PHASE" == "D" ]]; then
    export VJEPA_GRAD_BUCKET_VIEW=1
    export VJEPA_PREFETCH_FACTOR=4
fi

###########################
## Container Args #########
###########################

declare -a ARGS=(
    --container-image $CONTAINER_IMAGE
    --container-mounts $FSX_MOUNT
    --container-workdir ${VJEPA2_CODE}
    --no-container-mount-home
    --mpi=pmix
)

###########################
## Training Args ##########
###########################

declare -a TRAINING_ARGS=(
    --fname ${CONFIG}
)

###########################
## HyperPod Auto-Resume ###
###########################

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi

echo "==============================="
echo "  V-JEPA 2 Optimization Sweep"
echo "  Phase: ${PHASE}"
echo "  Config: ${CONFIG}"
echo "  nsys output: ${NSYS_OUTPUT}.nsys-rep"
echo ""
echo "  Phase A: compile_model (fixed) + no AC"
[[ "$PHASE" =~ ^[BCD]$ ]] && echo "  Phase B: + fused AdamW + TF32"
[[ "$PHASE" =~ ^[CD]$ ]]  && echo "  Phase C: + NCCL_NVLS=1, GRAPH_REG=1, MAX_CONN=1"
[[ "$PHASE" == "D" ]]     && echo "  Phase D: + grad_bucket_view + prefetch=4"
echo "==============================="

mkdir -p logs/vjepa2_optim
mkdir -p "$(dirname "${NSYS_OUTPUT}")"

# Create the training output folder so CSVLogger on non-rank-0 doesn't fail
mkdir -p ${VJEPA2_DIR}/benchmark/optim_sweep

# Delete stale checkpoint to prevent loading previous state
rm -f ${VJEPA2_DIR}/benchmark/optim_sweep/latest.pt

srun ${AUTO_RESUME} -l "${ARGS[@]}" \
    bash ${VJEPA2_DIR}/scripts/nsys_wrapper.sh \
    python ${VJEPA2_DIR}/scripts/run_train.py "${TRAINING_ARGS[@]}"
