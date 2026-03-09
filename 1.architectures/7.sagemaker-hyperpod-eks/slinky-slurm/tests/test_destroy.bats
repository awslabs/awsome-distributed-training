#!/usr/bin/env bats
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Unit tests for destroy.sh
# Run: bats tests/test_destroy.bats

load 'helpers/setup'

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

@test "destroy.sh: fails with invalid --infra value" {
    run bash "${PROJECT_DIR}/destroy.sh" --infra docker
    assert_failure
    assert_output --partial "Error: --infra must be 'cfn' or 'tf'"
}

@test "destroy.sh: fails with unknown option" {
    run bash "${PROJECT_DIR}/destroy.sh" --foobar
    assert_failure
    assert_output --partial "Error: Unknown option"
}

@test "destroy.sh: --help mentions --region flag" {
    run bash "${PROJECT_DIR}/destroy.sh" --help
    assert_success
    assert_output --partial "--region"
}

@test "destroy.sh: --help mentions --stack-name flag" {
    run bash "${PROJECT_DIR}/destroy.sh" --help
    assert_success
    assert_output --partial "--stack-name"
}

@test "destroy.sh: aborts when confirmation prompt is non-interactive" {
    # In a non-interactive shell, read from /dev/null returns empty,
    # which doesn't match [Yy], so the script should abort.
    run bash -c "echo 'n' | bash '${PROJECT_DIR}/destroy.sh' --infra cfn"
    assert_success
    assert_output --partial "Aborted"
}
