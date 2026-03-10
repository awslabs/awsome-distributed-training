#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=2                   # number of nodes to use, 2 p4d(e) = 16 A100 GPUs
#SBATCH --job-name=megatron_llama2  # name of your job
#SBATCH --exclusive                 # job has exclusive use of the resource, no sharing
#SBATCH --wait-all-nodes=1

set -exuo pipefail

###########################
###### Instance Profile ###
###########################
# Auto-detect instance type and source the matching profile.
# Profiles set: GPUS_PER_NODE, EFA vars, NCCL settings, and default TP/PP.
# Note: The model architecture presets below override the profile's TP/PP.
# Override with: export INSTANCE_PROFILE=g5-12xlarge (before sbatch)
# See profiles/README.md for details.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILES_DIR="${SCRIPT_DIR}/../../profiles"
PROFILE_LOADED=0

if [[ -d "$PROFILES_DIR" ]]; then
    if PROFILE_ENV=$("${PROFILES_DIR}/_detect.sh" "${PROFILES_DIR}"); then
        echo "Sourcing instance profile: $PROFILE_ENV"
        source "$PROFILE_ENV"
        PROFILE_LOADED=1
    else
        echo "WARNING: Profile detection failed. Using defaults (8 GPU, p4de-style)."
    fi
else
    echo "WARNING: No profiles/ directory found. Using defaults (8 GPU, p4de-style)."
fi

# Fallback defaults when no profile is loaded (assumes P4de-class instance)
GPUS_PER_NODE=${GPUS_PER_NODE:-8}


##################################################
###### Model architectures (example presets) #####
##################################################
# Feel free to choose one of the sample presents, or completely define your own
# custom model size.

## llama2-7b-hf
#declare -a MEGATRON_ARGS=(
#    --num-layers 32
#    --hidden-size 4096
#    --num-attention-heads 32
#
#    --tensor-model-parallel-size 1
#    --pipeline-model-parallel-size 1
#)

## llama2-13b-hf
#declare -a MEGATRON_ARGS=(
#    --num-layers 40
#    --hidden-size 5120
#    --num-attention-heads 40
#
#    --tensor-model-parallel-size 2
#    --pipeline-model-parallel-size 1
#    --sequence-parallel
#
#    --use-distributed-optimizer
#    --overlap-grad-reduce
#    --overlap-param-gather
#)

# llama2-70b-hf.
declare -a MEGATRON_ARGS=(
    --num-layers 80
    --hidden-size 8192
    --num-attention-heads 64
    --group-query-attention
    --num-query-groups 8

    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 4
    --sequence-parallel

    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

# Required for Llama2-style architecture. Do not comment or remove.
MEGATRON_ARGS+=(
   --untie-embeddings-and-output-weights
   --position-embedding-type rope
   --no-position-embedding
   --normalization RMSNorm
   --swiglu
   --no-masked-softmax-fusion
)

# Additional flags to make it possible to test with as few nodes as possible
MEGATRON_ARGS+=(
    --no-rope-fusion
    --use-flash-attn
    --transformer-impl transformer_engine
)


###########################
###### User Variables #####
###########################

: "${SEQ_LENGTH:=4096}"
: "${MAX_POSITION_EMBEDDINGS:=4096}"
: "${MICRO_BATCH_SIZE:=1}"
: "${GLOBAL_BATCH_SIZE:=2048}"

# default variables for Enroot
: "${IMAGE:=$(pwd)/megatron-training.sqsh}"
: "${DATA_PATH:=/fsx}"
: "${FSX_MOUNT:=$(pwd):$DATA_PATH}"


###########################
## Environment Variables ##
###########################

# EFA networking — configured by profile or defaults.
if [[ "$PROFILE_LOADED" == "1" ]]; then
    # Profile was sourced — trust its EFA/NCCL settings.
    true
else
    # No profile — use legacy defaults (P4de)
    export FI_PROVIDER=efa
    export FI_EFA_USE_HUGE_PAGE=0
fi

# https://discuss.pytorch.org/t/nccl-network-is-unreachable-connection-refused-when-initializing-ddp/137352
# https://github.com/pytorch/pytorch/issues/68893
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_AVOID_RECORD_STREAMS=1          # torch<2.2
export TORCH_NCCL_AVOID_RECORD_STREAMS=1    # torch>=2.2
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"^docker,lo,veth"}

# async runtime error ...
export CUDA_DEVICE_MAX_CONNECTIONS=1


#########################
## Command and Options ##
#########################

declare -a ARGS=(
    --container-image $IMAGE
    --container-mounts $FSX_MOUNT
)

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=$GPUS_PER_NODE
    --nnodes=$SLURM_JOB_NUM_NODES
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$(hostname)
)

MEGATRON_ARGS+=(
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE

    # Example how to control training duration using steps rather than number of samples.
    --train-iters 5

    # Example how to disable all validations, hence only training steps performed.
    --split 100,0,0
)
[[ -f ${IMAGE} ]] || { echo "Could not find enroot image: $IMAGE" ; exit -1 ; }
srun -l "${ARGS[@]}" python -m torch.distributed.run "${TORCHRUN_ARGS[@]}" /workspace/Megatron-LM/pretrain_gpt.py \
        "${MEGATRON_ARGS[@]}" \
        --use-mcore-models \
        --log-throughput \
        --lr 6.0e-5 \
        --min-lr 6.0e-6 \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters 0 \
        --data-path ${DATA_PATH}/llama2/my-llama2_text_document \
        --tokenizer-type Llama2Tokenizer \
        --tokenizer-model ${DATA_PATH}/llama2/tokenizer.model \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --wandb-exp-name llama2-megatron-lm \
        --wandb-project "megatron-lm" \
        --init-method-std 0.006 \
        --fp16
