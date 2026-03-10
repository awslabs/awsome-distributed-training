#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Extracted helper functions for deploy.sh
# This file is sourced by deploy.sh and can be sourced independently
# in tests for unit testing.

###########################
## Node Type Resolution ###
###########################

# resolve_node_profile <node_type>
# Sets ACCEL_INSTANCE_TYPE and ACCEL_INSTANCE_COUNT based on node type.
# Returns 0 on success, 1 on invalid node type.
resolve_node_profile() {
    local node_type="$1"

    case "${node_type}" in
        g5)
            ACCEL_INSTANCE_TYPE="ml.g5.8xlarge"
            ACCEL_INSTANCE_COUNT=4
            ;;
        p5)
            ACCEL_INSTANCE_TYPE="ml.p5.48xlarge"
            ACCEL_INSTANCE_COUNT=2
            ;;
        *)
            echo "Error: --node-type must be 'g5' or 'p5' (got: ${node_type})"
            return 1
            ;;
    esac
    return 0
}

###########################
## Helm Profile Resolution
###########################

# resolve_helm_profile <node_type>
# Sets Helm template variables for the slurm-values.yaml.template.
# Returns 0 on success, 1 on invalid node type.
resolve_helm_profile() {
    local node_type="$1"

    MGMT_INSTANCE_TYPE="ml.m5.4xlarge"
    PVC_NAME="fsx-claim"

    case "${node_type}" in
        g5)
            HELM_ACCEL_INSTANCE_TYPE="ml.g5.8xlarge"
            GPU_COUNT=1
            EFA_COUNT=1
            GPU_GRES="gpu:a10g:1"
            REPLICAS=4
            ;;
        p5)
            HELM_ACCEL_INSTANCE_TYPE="ml.p5.48xlarge"
            GPU_COUNT=8
            EFA_COUNT=32
            GPU_GRES="gpu:h100:8"
            REPLICAS=2
            ;;
        *)
            echo "Error: --node-type must be 'g5' or 'p5' (got: ${node_type})"
            return 1
            ;;
    esac
    return 0
}

###########################
## Prerequisite Checks ####
###########################

# check_command <command_name>
# Verifies that a command exists in PATH.
# Returns 0 if found, 1 if not found.
check_command() {
    if ! command -v "$1" &>/dev/null; then
        echo "Error: '$1' is required but not installed."
        return 1
    fi
    return 0
}

###########################
## AZ Validation ##########
###########################

# validate_az_id <az_id> <comma_separated_az_ids>
# Checks if the specified AZ ID exists in the comma-separated list.
# Returns 0 if found, 1 if not found.
validate_az_id() {
    local az_id="$1"
    local az_ids="$2"

    if echo "${az_ids}" | tr ',' '\n' | grep -q "^${az_id}$"; then
        return 0
    else
        return 1
    fi
}

###########################
## CFN Param Resolution ###
###########################

# resolve_cfn_params <params_file> <az_ids> <az_id> <accel_type> <accel_count>
# Runs the jq filter to substitute AZ IDs and instance overrides.
# Outputs resolved JSON to stdout.
# Returns 0 on success, 1 on failure.
resolve_cfn_params() {
    local params_file="$1"
    local az_ids="$2"
    local az_id="$3"
    local accel_type="$4"
    local accel_count="$5"

    if [[ ! -f "${params_file}" ]]; then
        echo "Error: Parameters file not found: ${params_file}" >&2
        return 1
    fi

    jq \
        --arg az_ids "${az_ids}" \
        --arg az_id "${az_id}" \
        --arg accel_type "${accel_type}" \
        --argjson accel_count "${accel_count}" \
        '
        map(
            if .ParameterKey == "AvailabilityZoneIds" then
                .ParameterValue = $az_ids
            elif .ParameterKey == "FsxAvailabilityZoneId" then
                .ParameterValue = $az_id
            elif .ParameterKey == "InstanceGroupSettings1" then
                .ParameterValue = (
                    .ParameterValue | fromjson |
                    map(
                        .TargetAvailabilityZoneId = $az_id |
                        if .InstanceGroupName == "accelerated-instance-group-1" then
                            .InstanceType = $accel_type |
                            .InstanceCount = $accel_count
                        else .
                        end
                    ) |
                    tojson
                )
            else .
            end
        )
        ' "${params_file}"
}

###########################
## TF Var Resolution ######
###########################

# resolve_tf_vars <target_file> <region> <az_id> <accel_type> <accel_count> <node_type>
# Patches a tfvars file in-place with region, AZ, and instance overrides.
# The target file must already exist (copied from source before calling).
# Returns 0 on success, 1 on failure.
resolve_tf_vars() {
    local target_file="$1"
    local region="$2"
    local az_id="$3"
    local accel_type="$4"
    local accel_count="$5"
    local node_type="$6"

    if [[ ! -f "${target_file}" ]]; then
        echo "Error: tfvars file not found: ${target_file}" >&2
        return 1
    fi

    # Override the aws_region
    sed -i.bak \
        "s|aws_region.*=.*|aws_region            = \"${region}\"|" \
        "${target_file}"

    # Override availability_zone_id in all instance groups
    sed -i.bak \
        "s|availability_zone_id.*=.*|availability_zone_id      = \"${az_id}\",|" \
        "${target_file}"

    # Apply p5 overrides: patch the first instance group's type and count.
    # NOTE: GNU sed supports "0,/pattern/" for first-occurrence-only, but
    # macOS (BSD) sed does not. Use awk for portability.
    if [[ "${node_type}" == "p5" ]]; then
        # Replace the first occurrence of the g5 instance type
        awk -v new_type="${accel_type}" '
            /instance_type.*=.*"ml\.g5\.8xlarge"/ && !type_done {
                sub(/"ml\.g5\.8xlarge"/, "\"" new_type "\"")
                type_done = 1
            }
            { print }
        ' "${target_file}" > "${target_file}.tmp" && mv "${target_file}.tmp" "${target_file}"

        # Replace the first occurrence of instance_count = 4
        awk -v new_count="${accel_count}" '
            /instance_count.*=.*4/ && !count_done {
                sub(/=.*/, "= " new_count ",")
                count_done = 1
            }
            { print }
        ' "${target_file}" > "${target_file}.tmp" && mv "${target_file}.tmp" "${target_file}"
    fi

    # Clean up sed backup files
    rm -f "${target_file}.bak"

    return 0
}
