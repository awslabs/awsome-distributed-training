#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# sync_profiles.sh — Copy the canonical instance_detect.sh to all test cases.
#
# Run this after editing 3.test_cases/shared/instance_detect.sh to propagate
# changes to all test case profile directories.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANONICAL="${SCRIPT_DIR}/instance_detect.sh"

if [[ ! -f "$CANONICAL" ]]; then
    echo "ERROR: Canonical source not found: $CANONICAL" >&2
    exit 1
fi

# All test case profile directories that contain _detect.sh
TARGETS=(
    "${SCRIPT_DIR}/../pytorch/verl/hyperpod-eks/rlvr/recipe/profiles/_detect.sh"
    "${SCRIPT_DIR}/../pytorch/verl/kubernetes/rlvr/recipe/profiles/_detect.sh"
    "${SCRIPT_DIR}/../pytorch/FSDP/profiles/_detect.sh"
    "${SCRIPT_DIR}/../pytorch/torchtitan/profiles/_detect.sh"
    "${SCRIPT_DIR}/../pytorch/nanoVLM/profiles/_detect.sh"
    "${SCRIPT_DIR}/../pytorch/trl/profiles/_detect.sh"
    "${SCRIPT_DIR}/../megatron/megatron-lm/profiles/_detect.sh"
    "${SCRIPT_DIR}/../megatron/bionemo/profiles/_detect.sh"
    "${SCRIPT_DIR}/../pytorch/deepspeed/examples_megatron_deepspeed/profiles/_detect.sh"
)

SYNCED=0
SKIPPED=0

for target in "${TARGETS[@]}"; do
    target_resolved="$(cd "$(dirname "$target")" 2>/dev/null && pwd)/$(basename "$target")" 2>/dev/null || true
    if [[ -d "$(dirname "$target")" ]]; then
        cp "$CANONICAL" "$target"
        chmod +x "$target"
        echo "  Synced: $target"
        ((SYNCED++))
    else
        echo "  Skipped (dir not found): $target"
        ((SKIPPED++))
    fi
done

echo ""
echo "Done: $SYNCED synced, $SKIPPED skipped."
