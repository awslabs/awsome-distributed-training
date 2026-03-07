#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# AWS CLI mock for bats-core unit tests.
# Source this file and call mock_aws() to override the `aws` command
# with canned responses. Override per-test as needed.

mock_aws() {
    aws() {
        case "$*" in
            *"sts get-caller-identity"*"--query Account"*)
                echo "123456789012"
                ;;
            *"sts get-caller-identity"*)
                echo '{"UserId":"AIDEXAMPLE","Account":"123456789012","Arn":"arn:aws:iam::123456789012:user/test"}'
                ;;
            *"ec2 describe-availability-zones"*)
                printf "usw2-az1\tusw2-az2\tusw2-az3\tusw2-az4"
                ;;
            *"cloudformation create-stack"*)
                echo '{"StackId": "arn:aws:cloudformation:us-west-2:123456789012:stack/test-stack/guid-1234"}'
                ;;
            *"cloudformation wait"*)
                return 0
                ;;
            *"cloudformation describe-stacks"*"OutputEKSClusterName"*)
                echo "mock-eks-cluster"
                ;;
            *"cloudformation describe-stacks"*"OutputVpcId"*)
                echo "vpc-mock12345"
                ;;
            *"cloudformation describe-stacks"*"OutputPrivateSubnetIds"*)
                echo "subnet-mock1,subnet-mock2"
                ;;
            *"cloudformation describe-stacks"*"OutputSecurityGroupId"*)
                echo "sg-mock12345"
                ;;
            *"cloudformation describe-stacks"*)
                echo "mock-value"
                ;;
            *)
                echo "UNMOCKED AWS CALL: aws $*" >&2
                return 1
                ;;
        esac
    }
    export -f aws
}
