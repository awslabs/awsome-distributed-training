#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# _detect.sh — Auto-detect EC2 instance type and resolve the matching profile.
#
# Usage (from a recipe script):
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   PROFILE_ENV=$("${SCRIPT_DIR}/profiles/_detect.sh" "${SCRIPT_DIR}/profiles")
#   source "$PROFILE_ENV"
#
# Detection order:
#   1. INSTANCE_PROFILE env var (explicit override, e.g. "g5-12xlarge")
#   2. INSTANCE_TYPE env var (from env_vars, e.g. "g5.12xlarge")
#   3. EC2 instance metadata API (works on bare metal and K8s with host networking)
#   4. GPU name from nvidia-smi (fallback when metadata is unavailable)
#
# Outputs the path to the profile .env file on stdout.
# Exits non-zero if no profile can be resolved.
# ---------------------------------------------------------------------------
set -euo pipefail

PROFILES_DIR="${1:-.}"

# --- Step 1: Check for explicit INSTANCE_PROFILE override -------------------
if [[ -n "${INSTANCE_PROFILE:-}" ]]; then
    PROFILE_NAME="$INSTANCE_PROFILE"
    echo "Instance profile override: ${PROFILE_NAME}" >&2
else
    # --- Step 2: Try INSTANCE_TYPE from env_vars ----------------------------
    INSTANCE_TYPE="${INSTANCE_TYPE:-}"

    # --- Step 3: Try EC2 instance metadata API ------------------------------
    if [[ -z "$INSTANCE_TYPE" ]]; then
        # IMDSv2: get a token first, then query
        TOKEN=$(curl -s --connect-timeout 2 -X PUT \
            "http://169.254.169.254/latest/api/token" \
            -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null) || true
        if [[ -n "$TOKEN" ]]; then
            INSTANCE_TYPE=$(curl -s --connect-timeout 2 \
                -H "X-aws-ec2-metadata-token: $TOKEN" \
                "http://169.254.169.254/latest/meta-data/instance-type" 2>/dev/null) || true
        fi
    fi

    # --- Step 4: Fallback — detect from GPU name ----------------------------
    if [[ -z "$INSTANCE_TYPE" ]]; then
        if command -v nvidia-smi &>/dev/null; then
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) || true
            case "${GPU_NAME:-}" in
                *A10G*)     INSTANCE_TYPE="g5.12xlarge" ;;
                *A100*80*)  INSTANCE_TYPE="p4de.24xlarge" ;;
                *A100*40*)  INSTANCE_TYPE="p4d.24xlarge" ;;
                *H100*)     INSTANCE_TYPE="p5.48xlarge" ;;
                *H200*)     INSTANCE_TYPE="p5en.48xlarge" ;;
                *L40S*)     INSTANCE_TYPE="g6e.12xlarge" ;;
                *L4*)       INSTANCE_TYPE="g6.12xlarge" ;;
                *)
                    echo "ERROR: Could not determine instance type from GPU: '${GPU_NAME:-unknown}'" >&2
                    echo "Set INSTANCE_TYPE or INSTANCE_PROFILE in env_vars." >&2
                    exit 1
                    ;;
            esac
            echo "Detected GPU '${GPU_NAME}' -> assuming ${INSTANCE_TYPE}" >&2
        else
            echo "ERROR: Cannot detect instance type (no metadata, no nvidia-smi)." >&2
            echo "Set INSTANCE_TYPE or INSTANCE_PROFILE in env_vars." >&2
            exit 1
        fi
    fi

    # Convert instance type to profile name: "g5.12xlarge" -> "g5-12xlarge"
    PROFILE_NAME="${INSTANCE_TYPE//./-}"
    echo "Detected instance type: ${INSTANCE_TYPE} -> profile: ${PROFILE_NAME}" >&2
fi

# --- Resolve profile file path ----------------------------------------------
PROFILE_PATH="${PROFILES_DIR}/${PROFILE_NAME}.env"

if [[ ! -f "$PROFILE_PATH" ]]; then
    echo "ERROR: No profile found at ${PROFILE_PATH}" >&2
    echo "" >&2
    echo "Available profiles:" >&2
    ls -1 "${PROFILES_DIR}"/*.env 2>/dev/null | sed 's/.*\//  /' >&2 || echo "  (none)" >&2
    echo "" >&2
    echo "To create a new profile, copy an existing one and adjust the values:" >&2
    echo "  cp ${PROFILES_DIR}/p5en-48xlarge.env ${PROFILE_PATH}" >&2
    exit 1
fi

# Output the resolved path (recipe script will source it)
echo "$PROFILE_PATH"
