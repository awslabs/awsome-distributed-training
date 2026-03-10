#!/bin/bash
#SBATCH --nodes=2                    # number of nodes
#SBATCH --ntasks-per-node=8          # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=8
#SBATCH --exclusive                   # exclusive node access
#SBATCH --output slurm-esm2-train-%j.out

###########################
###### Instance Profile ###
###########################
# Auto-detect instance type and source the matching profile.
# Profiles set: GPUS_PER_NODE, EFA/NCCL vars.
# Override with: export INSTANCE_PROFILE=g5-12xlarge (before sbatch)
# See ../profiles/README.md for details.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILES_DIR="${SCRIPT_DIR}/../profiles"
PROFILE_LOADED=0

if [[ -d "$PROFILES_DIR" ]]; then
    if PROFILE_ENV=$("${PROFILES_DIR}/_detect.sh" "${PROFILES_DIR}"); then
        echo "Sourcing instance profile: $PROFILE_ENV"
        source "$PROFILE_ENV"
        PROFILE_LOADED=1
    else
        echo "WARNING: Profile detection failed. Using defaults (8 GPU, EFA enabled)."
    fi
else
    echo "WARNING: No profiles/ directory found. Using defaults (8 GPU, EFA enabled)."
fi

# Fallback defaults when no profile is loaded
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# EFA — configured by profile or legacy defaults
if [[ "$PROFILE_LOADED" != "1" ]]; then
    #export FI_EFA_USE_HUGE_PAGE=0 #Uncomment if you get os.fork() memory error
    export FI_PROVIDER=efa
    export NCCL_DEBUG=INFO
fi

#Path to store data and checkpoints
export DATA_HOME_DIR=/fsxl/awsankur/bionemo

###########################
###### User Variables #####
###########################

# default variables for Enroot
: "${IMAGE:=${DATA_HOME_DIR}/bionemo.sqsh}"
: "${DATA_PATH:=/fsxl}"
: "${FSX_MOUNT:=$DATA_PATH:$DATA_PATH}"

declare -a ARGS=(
    --container-image $IMAGE
    --container-mount-home
    --container-mounts ${DATA_HOME_DIR}:${DATA_HOME_DIR}
)

# Enable fused attention in transformer engine for speed-up
DATA_DIR=$(find $DATA_HOME_DIR -type d -name "*untar*" -print -quit)

srun -l "${ARGS[@]}"  python3 /workspace/bionemo2/sub-packages/bionemo-esm2/src/bionemo/esm2/scripts/train_esm2.py \
    --train-cluster-path ${DATA_DIR}/2024_03_sanity/train_clusters_sanity.parquet \
    --train-database-path ${DATA_DIR}/2024_03_sanity/train_sanity.db \
    --valid-cluster-path ${DATA_DIR}/2024_03_sanity/valid_clusters.parquet \
    --valid-database-path ${DATA_DIR}/2024_03_sanity/validation.db \
    --precision="bf16-mixed" \
    --num-gpus ${GPUS_PER_NODE} \
    --num-nodes ${SLURM_JOB_NUM_NODES} \
    --num-steps 100 \
    --val-check-interval 25 \
    --max-seq-length 1024 \
    --limit-val-batches 2 \
    --micro-batch-size 2 \
    --num-layers 33 \
    --hidden-size 1280 \
    --num-attention-head 20 \
    --ffn-hidden-size 5120 \
    --tensor-model-parallel-size 1 \
    --create-tensorboard-logger \
    --result-dir ${DATA_HOME_DIR}
