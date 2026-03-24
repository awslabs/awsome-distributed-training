#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Optimization sweep for V-JEPA 2.1 on SageMaker HyperPod with B200 GPUs.
#
# Runs cumulative optimization phases with nsys profiling on rank 0.
# Each phase builds on the previous one to isolate the impact of each change.
#
# Environment variables:
#   PHASE - Which optimization phase to run (1, 2, 3, 4, 5). Required.
#
# Phase definitions (cumulative):
#   1 = AC off + num_workers 16 + prefetch_factor 4 + SDPA logging
#   2 = Phase 1 + torch.compile (compile_model in config)
#   3 = Phase 2 + fused AdamW + TF32
#   4 = Phase 3 + batch size increase (video 24->36, image 72->108)
#   5 = Phase 4 + NCCL tuning (NVLS=1, GRAPH_REGISTER=1) + DDP tuning
#
# Usage:
#   PHASE=1 sbatch slurm/optim_sweep_b200.sbatch
#   PHASE=2 sbatch slurm/optim_sweep_b200.sbatch
#   PHASE=3 sbatch slurm/optim_sweep_b200.sbatch

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --job-name=vjepa21-optim
#SBATCH --output=logs/vjepa21_optim/%A.out
#SBATCH --error=logs/vjepa21_optim/%A.err
#SBATCH --exclusive
#SBATCH --partition=b200
#SBATCH --account=root
#SBATCH --time=01:00:00

set -ex

PHASE="${PHASE:?ERROR: PHASE env var must be set (1, 2, 3, 4, or 5)}"

###########################
###### User Variables #####
###########################

VJEPA21_DIR=/fsx/${USER}/vjepa2.1
VJEPA2_CODE=/fsx/${USER}/vjepa2_code
VJEPA_DEPS=/fsx/${USER}/vjepa_deps

# All phases use the optimized config (AC off + compile + workers 16)
CONFIG=${VJEPA21_DIR}/configs/benchmark-50k-optimized.yaml

# nsys output named by phase
export NSYS_OUTPUT=${VJEPA21_DIR}/nsys/optim_phase_${PHASE}/rank0

###########################
## Container Variables ####
###########################

export CONTAINER_IMAGE=/fsx/dgxc-benchmark/images/nvidia+nemo+25.11.01-efa-nccl29.sqsh
export FSX_MOUNT=/fsx:/fsx
export CODE_MOUNT=${VJEPA2_CODE}:/workspace

###########################
## Environment Variables ##
###########################

# --- Required NCCL and EFA flags (always set) ---
export NCCL_DEBUG=INFO
export FI_PROVIDER=efa
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export NCCL_SOCKET_IFNAME=^docker,lo,veth
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so

# --- NVTE cuDNN normalization kernels ---
export NVTE_NORM_BWD_USE_CUDNN=1
export NVTE_NORM_FWD_USE_CUDNN=1

# --- Blackwell-specific ---
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# --- PyTorch memory allocator ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Limit torch.compile threads per process to prevent OOM ---
# 8 processes per node compiling simultaneously exhausts host CPU memory
export TORCH_COMPILE_THREADS=1

# --- V-JEPA 2.1 PYTHONPATH ---
export PYTHONPATH=/workspace:${VJEPA_DEPS}:${VJEPA2_CODE}:${PYTHONPATH}

# --- Always log SDPA backend status ---
export VJEPA_LOG_SDPA=1

# --- Always set prefetch_factor=4 (all phases) ---
export VJEPA_PREFETCH_FACTOR=4

# --- Phase 1: AC off + num_workers 16 + prefetch 4 + SDPA check ---
# (AC off and num_workers 16 are set in the config file)
# NCCL conservative settings (same as baseline)
export NCCL_NVLS_ENABLE=0
export NCCL_GRAPH_REGISTER=0

# --- Phase 2: Phase 1 + torch.compile ---
# (compile_model: true is set in the config file; Phase 1 runs without it)
# For Phase 1, we override compile_model via a separate config or env.
# Actually, we use TWO configs: baseline-like (no compile) for Phase 1,
# and the optimized config (with compile) for Phase 2+.
if [[ "$PHASE" == "1" ]]; then
    # Phase 1: use a config without compile_model
    CONFIG=${VJEPA21_DIR}/configs/benchmark-50k-phase1.yaml
fi

# --- Phase 3: Phase 2 + fused optimizer + TF32 ---
if [[ "$PHASE" =~ ^[345]$ ]]; then
    export VJEPA_FUSED_OPTIMIZER=1
    export VJEPA_TF32=1
fi

# --- Phase 4: Phase 3 + batch size increase ---
if [[ "$PHASE" =~ ^[45]$ ]]; then
    CONFIG=${VJEPA21_DIR}/configs/benchmark-50k-phase4.yaml
fi

# --- Phase 5: Phase 4 + NCCL tuning + DDP tuning ---
if [[ "$PHASE" == "5" ]]; then
    export NCCL_NVLS_ENABLE=1
    export NCCL_GRAPH_REGISTER=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export VJEPA_GRAD_BUCKET_VIEW=1
fi

###########################
## Container Args #########
###########################

declare -a ARGS=(
    --container-image $CONTAINER_IMAGE
    --container-mounts "${FSX_MOUNT},${CODE_MOUNT}"
    --container-workdir /workspace
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

echo "======================================="
echo "  V-JEPA 2.1 Optimization Sweep"
echo "  Phase: ${PHASE}"
echo "  Config: ${CONFIG}"
echo "  nsys output: ${NSYS_OUTPUT}.nsys-rep"
echo ""
echo "  Phase 1: AC off + workers 16 + prefetch 4 + SDPA check"
[[ "$PHASE" =~ ^[2345]$ ]] && echo "  Phase 2: + torch.compile"
[[ "$PHASE" =~ ^[345]$ ]]  && echo "  Phase 3: + fused AdamW + TF32"
[[ "$PHASE" =~ ^[45]$ ]]   && echo "  Phase 4: + batch size increase (video 36, image 108)"
[[ "$PHASE" == "5" ]]      && echo "  Phase 5: + NCCL_NVLS=1, GRAPH_REG=1, grad_bucket_view"
echo "======================================="

mkdir -p logs/vjepa21_optim
mkdir -p "$(dirname "${NSYS_OUTPUT}")"

# Delete stale checkpoint to prevent loading previous state
rm -f ${VJEPA21_DIR}/benchmark/optim_sweep/latest.pt
rm -f ${VJEPA21_DIR}/benchmark/optim_sweep/latest.pth.tar

###########################
## Storage Monitoring #####
###########################

# Background storage monitoring (lightweight, writes to log file, killed on exit)
STORAGE_LOG=${VJEPA21_DIR}/nsys/optim_phase_${PHASE}/storage_monitor.log
mkdir -p "$(dirname "${STORAGE_LOG}")"

# Only run monitoring on the first node (SLURM_NODEID=0 equivalent: head node runs sbatch)
(
    echo "=== Storage monitoring started at $(date) ===" > "${STORAGE_LOG}"
    while true; do
        echo "--- $(date) ---" >> "${STORAGE_LOG}"
        # Lustre stats (if available)
        if [ -d /proc/fs/lustre ]; then
            cat /proc/fs/lustre/llite/*/stats 2>/dev/null >> "${STORAGE_LOG}" || true
        fi
        # IO stats
        iostat -x 1 1 >> "${STORAGE_LOG}" 2>/dev/null || true
        sleep 30
    done
) &
MONITOR_PID=$!

# Cleanup monitor on exit
cleanup() {
    kill $MONITOR_PID 2>/dev/null || true
    wait $MONITOR_PID 2>/dev/null || true
    echo "Storage monitor stopped" >> "${STORAGE_LOG}"
}
trap cleanup EXIT

###########################
## Launch Training ########
###########################

srun ${AUTO_RESUME} -l "${ARGS[@]}" \
    bash ${VJEPA21_DIR}/scripts/nsys_wrapper.sh \
    python ${VJEPA21_DIR}/scripts/run_train.py "${TRAINING_ARGS[@]}"
