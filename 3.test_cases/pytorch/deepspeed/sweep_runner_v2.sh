#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# sweep_runner_v2.sh - Sweep v2: ZeRO-2 (no PP), ZeRO-3, memory push, fusion ops
#
# All configs use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:true (set in sbatch).
# Optimal NCCL flags are the defaults already in the sbatch script.
#
# Usage: bash sweep_runner_v2.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/pretrain_gpt_103b.sbatch"
RESULTS_DIR="${SCRIPT_DIR}/sweep_results"
NODES=8
PARTITION="b200"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    echo "[DRY RUN] Will print commands without submitting"
fi

mkdir -p "${RESULTS_DIR}" logs

# ============================================================
# Helper: submit a sweep configuration
# Extends v1 helper with seq_length and enable_fusions params.
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
    local seq_length="${10:-2048}"
    local enable_fusions="${11:-0}"
    shift 11 || true
    local extra_env="${*:-}"

    echo "============================================"
    echo "Submitting: ${config_name}"
    echo "  TP=${tp} PP=${pp} ZeRO=${zero} MBS=${mbs} GBS=${gbs}"
    echo "  SeqLen=${seq_length} Fusions=${enable_fusions}"
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
    env_exports+="SEQ_LENGTH=${seq_length},"
    env_exports+="ENABLE_FUSIONS=${enable_fusions},"
    env_exports+="CONFIG_NAME=${config_name}"

    local sbatch_cmd="sbatch"
    sbatch_cmd+=" --partition=${PARTITION}"
    sbatch_cmd+=" --nodes=${NODES}"

    # Add extra env vars (NCCL overrides etc.)
    if [ -n "${extra_env}" ]; then
        sbatch_cmd+=" --export=ALL,${env_exports},${extra_env}"
    else
        sbatch_cmd+=" --export=ALL,${env_exports}"
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
    echo "${job_id},${config_name},${tp},${pp},${zero},${mbs},${gbs},${act_ckpt},${seq_par},${overlap},${seq_length},${enable_fusions}" >> "${RESULTS_DIR}/sweep_jobs_v2.csv"
}

# ============================================================
# Initialize tracking file
# ============================================================
echo "job_id,config_name,tp,pp,zero,mbs,gbs,act_ckpt,seq_par,overlap,seq_length,enable_fusions" > "${RESULTS_DIR}/sweep_jobs_v2.csv"

# ============================================================
# ZeRO-2 WITHOUT PIPELINE PARALLELISM (PP=1)
# ============================================================
echo ""
echo "========== ZeRO-2 SWEEP (PP=1) =========="
echo ""

#            config_name              TP PP ZeRO MBS GBS ACT SEQ OVR SEQ_LEN FUSE
submit_config "21_zero2_tp8_pp1"      8  1  2    1   64  0   0   0   2048    0
submit_config "22_zero2_tp8_pp1_mbs2" 8  1  2    2   64  0   0   0   2048    0
submit_config "23_zero2_tp4_pp1"      4  1  2    1   64  0   0   0   2048    0

# ============================================================
# ZeRO-3 (PP=1)
# ============================================================
echo ""
echo "========== ZeRO-3 SWEEP (PP=1) =========="
echo ""

#            config_name                  TP PP ZeRO MBS GBS ACT SEQ OVR SEQ_LEN FUSE
submit_config "24_zero3_tp8_pp1"          8  1  3    1   64  0   0   0   2048    0
submit_config "25_zero3_tp8_pp1_mbs2"     8  1  3    2   64  0   0   0   2048    0
submit_config "26_zero3_tp4_pp1"          4  1  3    1   64  0   0   0   2048    0
submit_config "27_zero3_tp8_pp1_overlap"  8  1  3    1   64  0   0   1   2048    0

# ============================================================
# MEMORY PUSH / SEQ LENGTH / FUSIONS
# ============================================================
echo ""
echo "========== MEMORY PUSH SWEEP =========="
echo ""

#            config_name              TP PP ZeRO MBS GBS ACT SEQ OVR SEQ_LEN FUSE
submit_config "28_mem_seq4k_tp8_pp2"  8  2  0    1   64  0   0   0   4096    0
submit_config "29_mem_fused_tp8_pp8"  8  8  0    1   64  0   0   0   2048    1

# ============================================================
# EXPANDABLE SEGMENTS IMPACT ON BEST CONFIG
# Re-test best config (TP8/PP8/ZeRO0) — now with expandable_segments
# enabled automatically via the updated sbatch.
# ============================================================
echo ""
echo "========== EXPANDABLE SEGMENTS IMPACT =========="
echo ""

#            config_name              TP PP ZeRO MBS GBS ACT SEQ OVR SEQ_LEN FUSE
submit_config "30_best_expand_seg"    8  8  0    1   64  0   0   0   2048    0

echo ""
echo "========== SWEEP V2 SUBMITTED =========="
echo "Job tracking file: ${RESULTS_DIR}/sweep_jobs_v2.csv"
echo ""
echo "Total configs: 10"
echo "To monitor: watch 'squeue -u \$USER'"
echo "When all jobs finish, run: python parse_results.py --jobs-csv sweep_results/sweep_jobs_v2.csv"
