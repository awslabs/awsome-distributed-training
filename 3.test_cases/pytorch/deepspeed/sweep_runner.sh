#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# sweep_runner.sh - Automated parameter sweep for DeepSpeed 103B pretraining
# Runs all parallelism and environment flag configurations, collects results.
#
# Usage: bash sweep_runner.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/pretrain_gpt_103b.sbatch"
RESULTS_DIR="${SCRIPT_DIR}/sweep_results"
NODES=8
PARTITION="${PARTITION:-dev}"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    echo "[DRY RUN] Will print commands without submitting"
fi

mkdir -p "${RESULTS_DIR}" logs

# ============================================================
# Helper: submit a sweep configuration
# ============================================================
submit_config() {
    local config_name="$1"
    local tp="$2"
    local pp="$3"
    local zero="$4"
    local mbs="$5"
    local gbs="$6"
    local act_ckpt="${7:-0}"
    local seq_par="${8:-0}"
    local overlap="${9:-0}"
    shift 9 || true
    local extra_env="${*:-}"

    echo "============================================"
    echo "Submitting: ${config_name}"
    echo "  TP=${tp} PP=${pp} ZeRO=${zero} MBS=${mbs} GBS=${gbs}"
    echo "  ActCkpt=${act_ckpt} SeqPar=${seq_par} Overlap=${overlap}"
    [ -n "${extra_env}" ] && echo "  Extra env: ${extra_env}"
    echo "============================================"

    local env_exports=""
    env_exports+="TP=${tp},"
    env_exports+="PP=${pp},"
    env_exports+="ZERO_STAGE=${zero},"
    env_exports+="MICRO_BATCH_SIZE=${mbs},"
    env_exports+="GLOBAL_BATCH_SIZE=${gbs},"
    env_exports+="USE_ACTIVATION_CHECKPOINTING=${act_ckpt},"
    env_exports+="USE_SEQUENCE_PARALLEL=${seq_par},"
    env_exports+="USE_OVERLAP_COMM=${overlap},"
    env_exports+="CONFIG_NAME=${config_name}"

    local sbatch_cmd="sbatch"
    sbatch_cmd+=" --partition=${PARTITION}"
    sbatch_cmd+=" --nodes=${NODES}"
    sbatch_cmd+=" --export=ALL,${env_exports}"
    sbatch_cmd+=" --job-name=sweep_${config_name}"

    # Add extra env vars for NCCL tuning
    if [ -n "${extra_env}" ]; then
        sbatch_cmd+=" --export=ALL,${env_exports},${extra_env}"
    fi

    sbatch_cmd+=" ${SBATCH_SCRIPT}"

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "[DRY RUN] ${sbatch_cmd}"
        echo ""
        return
    fi

    local job_output
    job_output=$(eval "${sbatch_cmd}")
    local job_id
    job_id=$(echo "${job_output}" | awk '{print $NF}')
    echo "Submitted job ${job_id} for config ${config_name}"
    echo "${job_id},${config_name},${tp},${pp},${zero},${mbs},${gbs},${act_ckpt},${seq_par},${overlap}" >> "${RESULTS_DIR}/sweep_jobs.csv"
}

# ============================================================
# Initialize tracking file
# ============================================================
echo "job_id,config_name,tp,pp,zero,mbs,gbs,act_ckpt,seq_par,overlap" > "${RESULTS_DIR}/sweep_jobs.csv"

# ============================================================
# PARALLELISM SWEEP (Configs 1-11)
# ============================================================
echo ""
echo "========== PARALLELISM SWEEP =========="
echo ""

#            config_name         TP PP ZeRO MBS GBS ACT SEQ OVR
submit_config "01_baseline"       8  2  0    1   64  0   0   0
submit_config "02_more_pp"        8  4  0    1   64  0   0   0
submit_config "03_zero1"          8  2  1    1   64  0   0   0
submit_config "04_larger_mbs"     8  2  1    2  128  0   0   0
submit_config "05_pp4_zero1"      8  4  1    1  128  0   0   0
submit_config "06_zero2"          8  2  2    1   64  0   0   0
submit_config "07_full_pp"        8  8  0    1   64  0   0   0
submit_config "08_tp4_pp4"        4  4  1    1   64  0   0   0
submit_config "09_act_ckpt"       8  2  1    1   64  1   0   0
submit_config "10_seq_parallel"   8  2  1    1   64  0   1   0
submit_config "11_overlap_comm"   8  2  1    1   64  0   0   1

# ============================================================
# Wait for parallelism sweep to determine best config
# If not waiting, env sweep uses config 03 (TP8/PP2/ZeRO1) as default
# ============================================================
echo ""
echo "========== ENVIRONMENT FLAGS SWEEP =========="
echo "(Using TP=8 PP=2 ZeRO=1 as base for env flag sweep)"
echo ""

# Base parallelism for env sweep
BASE_TP=8
BASE_PP=2
BASE_ZERO=1
BASE_MBS=1
BASE_GBS=64

#            config_name              TP       PP       ZeRO     MBS      GBS      ACT SEQ OVR extra_env
submit_config "12_nccl_ring"          ${BASE_TP} ${BASE_PP} ${BASE_ZERO} ${BASE_MBS} ${BASE_GBS} 0 0 0 "NCCL_ALGO=Ring"
submit_config "13_nccl_tree"          ${BASE_TP} ${BASE_PP} ${BASE_ZERO} ${BASE_MBS} ${BASE_GBS} 0 0 0 "NCCL_ALGO=Tree"
submit_config "14_nccl_no_tuner"      ${BASE_TP} ${BASE_PP} ${BASE_ZERO} ${BASE_MBS} ${BASE_GBS} 0 0 0 "NCCL_TUNER_PLUGIN="
submit_config "15_nccl_chunk_4mb"     ${BASE_TP} ${BASE_PP} ${BASE_ZERO} ${BASE_MBS} ${BASE_GBS} 0 0 0 "NCCL_P2P_NET_CHUNKSIZE=4194304"
submit_config "16_cuda_max_conn_1"    ${BASE_TP} ${BASE_PP} ${BASE_ZERO} ${BASE_MBS} ${BASE_GBS} 0 0 0 "CUDA_DEVICE_MAX_CONNECTIONS=1"
submit_config "17_nccl_buf_16mb"      ${BASE_TP} ${BASE_PP} ${BASE_ZERO} ${BASE_MBS} ${BASE_GBS} 0 0 0 "NCCL_BUFFERSIZE=16777216"
submit_config "18_nccl_buf_32mb"      ${BASE_TP} ${BASE_PP} ${BASE_ZERO} ${BASE_MBS} ${BASE_GBS} 0 0 0 "NCCL_BUFFERSIZE=33554432"
submit_config "19_nccl_min_ch_16"     ${BASE_TP} ${BASE_PP} ${BASE_ZERO} ${BASE_MBS} ${BASE_GBS} 0 0 0 "NCCL_MIN_NCHANNELS=16"
submit_config "20_nccl_min_ch_32"     ${BASE_TP} ${BASE_PP} ${BASE_ZERO} ${BASE_MBS} ${BASE_GBS} 0 0 0 "NCCL_MIN_NCHANNELS=32"

echo ""
echo "========== SWEEP SUBMITTED =========="
echo "Job tracking file: ${RESULTS_DIR}/sweep_jobs.csv"
echo ""
echo "To monitor: watch 'squeue -u \$USER'"
echo "When all jobs finish, run: python parse_results.py"
