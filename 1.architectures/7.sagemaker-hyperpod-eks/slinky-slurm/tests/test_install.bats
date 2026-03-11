#!/usr/bin/env bats
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Unit tests for install.sh
# Run: bats tests/test_install.bats

load 'helpers/setup'

###########################
## install.sh arg parsing #
###########################

@test "install.sh: --help exits 0 and prints usage" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "Usage:"
}

@test "install.sh: --help mentions --skip-setup" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "--skip-setup"
}

@test "install.sh: --help mentions pass-through options" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "--node-type"
    assert_output --partial "--infra"
    assert_output --partial "--local-build"
    assert_output --partial "--skip-build"
}

@test "install.sh: fails with unknown option" {
    run bash "${PROJECT_DIR}/install.sh" --foobar
    assert_failure
    assert_output --partial "Error: Unknown option"
}

@test "install.sh: --skip-setup fails when slurm-values.yaml is missing" {
    # Ensure slurm-values.yaml does not exist in the project directory.
    # The mock aws and mock kubectl won't be present, but the script should
    # fail before reaching any kubectl/helm calls because the values file
    # doesn't exist.
    local temp_project="${TEST_TEMP_DIR}/project"
    mkdir -p "${temp_project}"
    cp "${PROJECT_DIR}/install.sh" "${temp_project}/"

    run bash "${temp_project}/install.sh" --skip-setup
    assert_failure
    assert_output --partial "slurm-values.yaml not found"
}

###########################
## pass-through flags #####
###########################

@test "install.sh: accepts --repo-name as pass-through flag" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "--repo-name"
}

@test "install.sh: accepts --tag as pass-through flag" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "--tag"
}

@test "install.sh: accepts --region flag" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "--region"
}

###########################
## version constants ######
###########################

@test "install.sh: defines CERT_MANAGER_VERSION" {
    run grep 'CERT_MANAGER_VERSION=' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial '1.19.2'
}

@test "install.sh: defines LB_CONTROLLER_CHART_VERSION" {
    run grep 'LB_CONTROLLER_CHART_VERSION=' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial '1.11.0'
}

@test "install.sh: defines LB_CONTROLLER_IAM_ROLE_NAME" {
    run grep 'LB_CONTROLLER_IAM_ROLE_NAME=' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial 'AmazonEKS_LB_Controller_Role_slinky'
}

@test "install.sh: defines LB_CONTROLLER_IAM_POLICY_NAME" {
    run grep 'LB_CONTROLLER_IAM_POLICY_NAME=' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial 'AWSLoadBalancerControllerIAMPolicy_slinky'
}

@test "install.sh: defines EBS_CSI_IAM_ROLE_NAME" {
    run grep 'EBS_CSI_IAM_ROLE_NAME=' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial 'AmazonEKS_EBS_CSI_DriverRole_slinky'
}

@test "install.sh: defines EBS_CSI_INLINE_POLICY_NAME" {
    run grep 'EBS_CSI_INLINE_POLICY_NAME=' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial 'SageMakerHyperPodVolumeAccess'
}

###########################
## install order ##########
###########################

@test "install.sh: cert-manager is installed before LB Controller" {
    # Verify the cert-manager section appears before the LB Controller section
    local cert_line lb_line
    cert_line=$(grep -n 'Install cert-manager' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    lb_line=$(grep -n 'Install LB Controller' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    [[ "${cert_line}" -lt "${lb_line}" ]]
}

@test "install.sh: LB Controller is installed before EBS CSI Driver" {
    local lb_line ebs_line
    lb_line=$(grep -n 'Install LB Controller' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    ebs_line=$(grep -n 'Install EBS CSI Driver' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    [[ "${lb_line}" -lt "${ebs_line}" ]]
}

@test "install.sh: EBS CSI Driver is installed before FSx PVC" {
    local ebs_line fsx_line
    ebs_line=$(grep -n 'Install EBS CSI Driver' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    fsx_line=$(grep -n 'Apply FSx PVC' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    [[ "${ebs_line}" -lt "${fsx_line}" ]]
}

@test "install.sh: LB Controller is installed before MariaDB" {
    local lb_line mariadb_line
    lb_line=$(grep -n 'Install LB Controller' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    mariadb_line=$(grep -n 'Install MariaDB' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    [[ "${lb_line}" -lt "${mariadb_line}" ]]
}

@test "install.sh: FSx PVC is applied before MariaDB" {
    local fsx_line mariadb_line
    fsx_line=$(grep -n 'Apply FSx PVC' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    mariadb_line=$(grep -n 'Install MariaDB' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    [[ "${fsx_line}" -lt "${mariadb_line}" ]]
}

@test "install.sh: requires EKS_CLUSTER_NAME from env_vars.sh" {
    run grep 'EKS_CLUSTER_NAME' "${PROJECT_DIR}/install.sh"
    assert_success
}

@test "install.sh: requires VPC_ID from env_vars.sh" {
    run grep 'VPC_ID' "${PROJECT_DIR}/install.sh"
    assert_success
}
