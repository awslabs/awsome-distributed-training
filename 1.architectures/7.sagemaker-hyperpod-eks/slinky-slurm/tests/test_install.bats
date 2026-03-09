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
