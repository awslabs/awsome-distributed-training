#!/usr/bin/env bats
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Unit tests for deploy.sh and lib/deploy_helpers.sh
# Run: bats tests/test_deploy.bats

load 'helpers/setup'

###########################
## resolve_node_profile ###
###########################

@test "resolve_node_profile: g5 sets correct instance type and count" {
    resolve_node_profile "g5"
    assert_equal "${ACCEL_INSTANCE_TYPE}" "ml.g5.8xlarge"
    assert_equal "${ACCEL_INSTANCE_COUNT}" "4"
}

@test "resolve_node_profile: p5 sets correct instance type and count" {
    resolve_node_profile "p5"
    assert_equal "${ACCEL_INSTANCE_TYPE}" "ml.p5.48xlarge"
    assert_equal "${ACCEL_INSTANCE_COUNT}" "2"
}

@test "resolve_node_profile: invalid node type returns 1" {
    run resolve_node_profile "p4"
    assert_failure
    assert_output --partial "Error: --node-type must be 'g5' or 'p5'"
}

@test "resolve_node_profile: empty node type returns 1" {
    run resolve_node_profile ""
    assert_failure
}

###########################
## check_command ##########
###########################

@test "check_command: finds existing command (bash)" {
    run check_command "bash"
    assert_success
}

@test "check_command: fails for nonexistent command" {
    run check_command "definitely_not_a_real_command_xyz"
    assert_failure
    assert_output --partial "Error: 'definitely_not_a_real_command_xyz' is required but not installed."
}

@test "check_command: finds jq" {
    run check_command "jq"
    assert_success
}

###########################
## validate_az_id #########
###########################

@test "validate_az_id: returns 0 when AZ exists in list" {
    run validate_az_id "usw2-az2" "usw2-az1,usw2-az2,usw2-az3,usw2-az4"
    assert_success
}

@test "validate_az_id: returns 1 when AZ not in list" {
    run validate_az_id "usw2-az9" "usw2-az1,usw2-az2,usw2-az3,usw2-az4"
    assert_failure
}

@test "validate_az_id: handles single-element list" {
    run validate_az_id "usw2-az1" "usw2-az1"
    assert_success
}

@test "validate_az_id: no partial matches" {
    run validate_az_id "usw2-az1" "usw2-az10,usw2-az11"
    assert_failure
}

@test "validate_az_id: handles different regions" {
    run validate_az_id "use1-az2" "use1-az1,use1-az2,use1-az3"
    assert_success
}

###########################
## resolve_cfn_params #####
###########################

@test "resolve_cfn_params: substitutes AvailabilityZoneIds" {
    local result
    result=$(resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "use1-az1,use1-az2,use1-az3" \
        "use1-az2" \
        "ml.g5.8xlarge" \
        4)

    local az_ids
    az_ids=$(echo "${result}" | jq -r \
        '.[] | select(.ParameterKey == "AvailabilityZoneIds") | .ParameterValue')

    assert_equal "${az_ids}" "use1-az1,use1-az2,use1-az3"
}

@test "resolve_cfn_params: substitutes FsxAvailabilityZoneId" {
    local result
    result=$(resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "use1-az1,use1-az2,use1-az3" \
        "use1-az2" \
        "ml.g5.8xlarge" \
        4)

    local fsx_az
    fsx_az=$(echo "${result}" | jq -r \
        '.[] | select(.ParameterKey == "FsxAvailabilityZoneId") | .ParameterValue')

    assert_equal "${fsx_az}" "use1-az2"
}

@test "resolve_cfn_params: sets TargetAvailabilityZoneId in all instance groups" {
    local result
    result=$(resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "use1-az1,use1-az2,use1-az3" \
        "use1-az2" \
        "ml.g5.8xlarge" \
        4)

    local ig_json
    ig_json=$(echo "${result}" | jq -r \
        '.[] | select(.ParameterKey == "InstanceGroupSettings1") | .ParameterValue')

    # Both instance groups should have use1-az2
    local az_values
    az_values=$(echo "${ig_json}" | jq -r '.[].TargetAvailabilityZoneId')

    # All values should be use1-az2
    while IFS= read -r line; do
        assert_equal "${line}" "use1-az2"
    done <<< "${az_values}"
}

@test "resolve_cfn_params: g5 keeps default instance type in accelerated group" {
    local result
    result=$(resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" \
        "usw2-az2" \
        "ml.g5.8xlarge" \
        4)

    local ig_json
    ig_json=$(echo "${result}" | jq -r \
        '.[] | select(.ParameterKey == "InstanceGroupSettings1") | .ParameterValue')

    local accel_type
    accel_type=$(echo "${ig_json}" | jq -r \
        '.[] | select(.InstanceGroupName == "accelerated-instance-group-1") | .InstanceType')

    assert_equal "${accel_type}" "ml.g5.8xlarge"
}

@test "resolve_cfn_params: p5 overrides accelerated group instance type" {
    local result
    result=$(resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" \
        "usw2-az2" \
        "ml.p5.48xlarge" \
        2)

    local ig_json
    ig_json=$(echo "${result}" | jq -r \
        '.[] | select(.ParameterKey == "InstanceGroupSettings1") | .ParameterValue')

    local accel_type
    accel_type=$(echo "${ig_json}" | jq -r \
        '.[] | select(.InstanceGroupName == "accelerated-instance-group-1") | .InstanceType')

    assert_equal "${accel_type}" "ml.p5.48xlarge"
}

@test "resolve_cfn_params: p5 overrides accelerated group instance count" {
    local result
    result=$(resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" \
        "usw2-az2" \
        "ml.p5.48xlarge" \
        2)

    local ig_json
    ig_json=$(echo "${result}" | jq -r \
        '.[] | select(.ParameterKey == "InstanceGroupSettings1") | .ParameterValue')

    local accel_count
    accel_count=$(echo "${ig_json}" | jq -r \
        '.[] | select(.InstanceGroupName == "accelerated-instance-group-1") | .InstanceCount')

    assert_equal "${accel_count}" "2"
}

@test "resolve_cfn_params: general group unchanged after p5 override" {
    local result
    result=$(resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" \
        "usw2-az2" \
        "ml.p5.48xlarge" \
        2)

    local ig_json
    ig_json=$(echo "${result}" | jq -r \
        '.[] | select(.ParameterKey == "InstanceGroupSettings1") | .ParameterValue')

    local general_type
    general_type=$(echo "${ig_json}" | jq -r \
        '.[] | select(.InstanceGroupName == "general-instance-group-2") | .InstanceType')

    local general_count
    general_count=$(echo "${ig_json}" | jq -r \
        '.[] | select(.InstanceGroupName == "general-instance-group-2") | .InstanceCount')

    assert_equal "${general_type}" "ml.m5.2xlarge"
    assert_equal "${general_count}" "2"
}

@test "resolve_cfn_params: fails when params file not found" {
    run resolve_cfn_params \
        "/nonexistent/params.json" \
        "usw2-az1,usw2-az2" \
        "usw2-az2" \
        "ml.g5.8xlarge" \
        4

    assert_failure
    assert_output --partial "Error: Parameters file not found"
}

@test "resolve_cfn_params: preserves all 40 parameters" {
    local result
    result=$(resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" \
        "usw2-az2" \
        "ml.g5.8xlarge" \
        4)

    local count
    count=$(echo "${result}" | jq 'length')

    assert_equal "${count}" "40"
}

###########################
## resolve_tf_vars ########
###########################

@test "resolve_tf_vars: overrides aws_region" {
    local target="${TEST_TEMP_DIR}/custom.tfvars"
    cp "${FIXTURE_DIR}/custom.tfvars" "${target}"

    resolve_tf_vars "${target}" "us-east-1" "use1-az2" "ml.g5.8xlarge" 4 "g5"

    run grep 'aws_region' "${target}"
    assert_output --partial 'us-east-1'
}

@test "resolve_tf_vars: overrides availability_zone_id" {
    local target="${TEST_TEMP_DIR}/custom.tfvars"
    cp "${FIXTURE_DIR}/custom.tfvars" "${target}"

    resolve_tf_vars "${target}" "us-east-1" "use1-az2" "ml.g5.8xlarge" 4 "g5"

    run grep 'availability_zone_id' "${target}"
    # Both instance groups should have use1-az2
    assert_output --partial 'use1-az2'
    # Verify no leftover usw2-az2
    run grep 'usw2-az2' "${target}"
    assert_failure
}

@test "resolve_tf_vars: g5 preserves default instance type" {
    local target="${TEST_TEMP_DIR}/custom.tfvars"
    cp "${FIXTURE_DIR}/custom.tfvars" "${target}"

    resolve_tf_vars "${target}" "us-west-2" "usw2-az2" "ml.g5.8xlarge" 4 "g5"

    run grep 'ml.g5.8xlarge' "${target}"
    assert_success
}

@test "resolve_tf_vars: p5 overrides accelerated instance type" {
    local target="${TEST_TEMP_DIR}/custom.tfvars"
    cp "${FIXTURE_DIR}/custom.tfvars" "${target}"

    resolve_tf_vars "${target}" "us-west-2" "usw2-az2" "ml.p5.48xlarge" 2 "p5"

    run grep 'ml.p5.48xlarge' "${target}"
    assert_success
}

@test "resolve_tf_vars: p5 overrides accelerated instance count" {
    local target="${TEST_TEMP_DIR}/custom.tfvars"
    cp "${FIXTURE_DIR}/custom.tfvars" "${target}"

    resolve_tf_vars "${target}" "us-west-2" "usw2-az2" "ml.p5.48xlarge" 2 "p5"

    # The first instance_count should be 2 (accelerated group).
    # Extract just the number from the first instance_count line.
    local first_count
    first_count=$(awk '/instance_count/ { match($0, /[0-9]+/); print substr($0, RSTART, RLENGTH); exit }' "${target}")

    assert_equal "${first_count}" "2"
}

@test "resolve_tf_vars: p5 does not change general group instance type" {
    local target="${TEST_TEMP_DIR}/custom.tfvars"
    cp "${FIXTURE_DIR}/custom.tfvars" "${target}"

    resolve_tf_vars "${target}" "us-west-2" "usw2-az2" "ml.p5.48xlarge" 2 "p5"

    run grep 'ml.m5.2xlarge' "${target}"
    assert_success
}

@test "resolve_tf_vars: fails when file not found" {
    run resolve_tf_vars "/nonexistent/custom.tfvars" "us-west-2" "usw2-az2" "ml.g5.8xlarge" 4 "g5"
    assert_failure
    assert_output --partial "Error: tfvars file not found"
}

@test "resolve_tf_vars: cleans up sed .bak files" {
    local target="${TEST_TEMP_DIR}/custom.tfvars"
    cp "${FIXTURE_DIR}/custom.tfvars" "${target}"

    resolve_tf_vars "${target}" "us-east-1" "use1-az2" "ml.g5.8xlarge" 4 "g5"

    # No .bak files should remain
    run ls "${TEST_TEMP_DIR}"/*.bak 2>/dev/null
    assert_failure
}

###########################
## deploy.sh arg parsing ##
###########################

@test "deploy.sh: --help exits 0 and prints usage" {
    run bash "${PROJECT_DIR}/deploy.sh" --help
    assert_success
    assert_output --partial "Usage:"
}

@test "deploy.sh: fails when --node-type is missing" {
    run bash "${PROJECT_DIR}/deploy.sh" --infra cfn
    assert_failure
    assert_output --partial "Error: --node-type is required"
}

@test "deploy.sh: fails when --infra is missing" {
    run bash "${PROJECT_DIR}/deploy.sh" --node-type g5
    assert_failure
    assert_output --partial "Error: --infra is required"
}

@test "deploy.sh: fails with invalid --infra value" {
    run bash "${PROJECT_DIR}/deploy.sh" --node-type g5 --infra docker
    assert_failure
    assert_output --partial "Error: --infra must be 'cfn' or 'tf'"
}

@test "deploy.sh: fails with unknown option" {
    run bash "${PROJECT_DIR}/deploy.sh" --foobar
    assert_failure
    assert_output --partial "Error: Unknown option"
}

###########################
## resolve_helm_profile ###
###########################

@test "resolve_helm_profile: g5 sets correct GPU/EFA/GRES/replicas" {
    resolve_helm_profile "g5"
    assert_equal "${HELM_ACCEL_INSTANCE_TYPE}" "ml.g5.8xlarge"
    assert_equal "${GPU_COUNT}" "1"
    assert_equal "${EFA_COUNT}" "1"
    assert_equal "${GPU_GRES}" "gpu:a10g:1"
    assert_equal "${REPLICAS}" "4"
    assert_equal "${MGMT_INSTANCE_TYPE}" "ml.m5.2xlarge"
    assert_equal "${PVC_NAME}" "fsx-claim"
}

@test "resolve_helm_profile: p5 sets correct GPU/EFA/GRES/replicas" {
    resolve_helm_profile "p5"
    assert_equal "${HELM_ACCEL_INSTANCE_TYPE}" "ml.p5.48xlarge"
    assert_equal "${GPU_COUNT}" "8"
    assert_equal "${EFA_COUNT}" "32"
    assert_equal "${GPU_GRES}" "gpu:h100:8"
    assert_equal "${REPLICAS}" "2"
}

@test "resolve_helm_profile: invalid node type returns 1" {
    run resolve_helm_profile "p4"
    assert_failure
    assert_output --partial "Error: --node-type must be 'g5' or 'p5'"
}

###########################
## values template sed ####
###########################

@test "values template: g5 substitution produces valid YAML" {
    resolve_helm_profile "g5"

    sed -e "s|\${image_repository}|123456789012.dkr.ecr.us-west-2.amazonaws.com/dlc-slurmd|g" \
        -e "s|\${image_tag}|25.11.1-ubuntu24.04|g" \
        -e "s|\${ssh_key}|ssh-ed25519 AAAAC3test test@example.com|g" \
        -e "s|\${mgmt_instance_type}|${MGMT_INSTANCE_TYPE}|g" \
        -e "s|\${accel_instance_type}|${HELM_ACCEL_INSTANCE_TYPE}|g" \
        -e "s|\${gpu_count}|${GPU_COUNT}|g" \
        -e "s|\${efa_count}|${EFA_COUNT}|g" \
        -e "s|\${gpu_gres}|${GPU_GRES}|g" \
        -e "s|\${replicas}|${REPLICAS}|g" \
        -e "s|\${pvc_name}|${PVC_NAME}|g" \
        "${FIXTURE_DIR}/slurm-values.yaml.template" > "${TEST_TEMP_DIR}/slurm-values.yaml"

    # Verify key values were substituted
    run grep 'ml.g5.8xlarge' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success

    run grep 'ml.m5.2xlarge' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success

    run grep 'gpu:a10g:1' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success

    run grep 'replicas: 4' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success

    # No unsubstituted template variables remain
    run grep -c '${' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_failure
}

@test "values template: p5 substitution has correct instance type and GPU count" {
    resolve_helm_profile "p5"

    sed -e "s|\${image_repository}|123456789012.dkr.ecr.us-west-2.amazonaws.com/dlc-slurmd|g" \
        -e "s|\${image_tag}|25.11.1-ubuntu24.04|g" \
        -e "s|\${ssh_key}|ssh-ed25519 AAAAC3test test@example.com|g" \
        -e "s|\${mgmt_instance_type}|${MGMT_INSTANCE_TYPE}|g" \
        -e "s|\${accel_instance_type}|${HELM_ACCEL_INSTANCE_TYPE}|g" \
        -e "s|\${gpu_count}|${GPU_COUNT}|g" \
        -e "s|\${efa_count}|${EFA_COUNT}|g" \
        -e "s|\${gpu_gres}|${GPU_GRES}|g" \
        -e "s|\${replicas}|${REPLICAS}|g" \
        -e "s|\${pvc_name}|${PVC_NAME}|g" \
        "${FIXTURE_DIR}/slurm-values.yaml.template" > "${TEST_TEMP_DIR}/slurm-values.yaml"

    run grep 'ml.p5.48xlarge' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success

    run grep 'gpu:h100:8' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success

    run grep 'replicas: 2' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success

    run grep 'vpc.amazonaws.com/efa: 32' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success
}

###########################
## setup.sh arg parsing ###
###########################

@test "setup.sh: --help exits 0 and prints usage" {
    run bash "${PROJECT_DIR}/setup.sh" --help
    assert_success
    assert_output --partial "Usage:"
}

@test "setup.sh: fails when --node-type is missing" {
    run bash "${PROJECT_DIR}/setup.sh" --infra cfn
    assert_failure
    assert_output --partial "Error: --node-type is required"
}

@test "setup.sh: fails when --infra is missing" {
    run bash "${PROJECT_DIR}/setup.sh" --node-type g5
    assert_failure
    assert_output --partial "Error: --infra is required"
}

###########################
## install.sh arg parsing #
###########################

@test "install.sh: --help exits 0 and prints usage" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "Usage:"
}

###########################
## destroy.sh arg parsing #
###########################

@test "destroy.sh: --help exits 0 and prints usage" {
    run bash "${PROJECT_DIR}/destroy.sh" --help
    assert_success
    assert_output --partial "Usage:"
}

@test "destroy.sh: fails when --infra is missing" {
    run bash "${PROJECT_DIR}/destroy.sh"
    assert_failure
    assert_output --partial "Error: --infra is required"
}
