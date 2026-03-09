#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -euo pipefail

###########################
###### Default Values #####
###########################

AWS_REGION=$(aws configure get region 2>/dev/null || echo "us-west-2")
INFRA=""
STACK_NAME="hp-eks-slinky-stack"
CODEBUILD_STACK_NAME="slurmd-codebuild-stack"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

###########################
###### Usage Function #####
###########################

usage() {
    cat <<EOF
Usage: $0 --infra <cfn|tf> [OPTIONS]

Tear down the Slurm cluster and HyperPod EKS infrastructure in reverse
order: Slurm cluster -> Slurm operator -> MariaDB -> CodeBuild stack ->
HyperPod infrastructure stack.

Required:
  --infra <cfn|tf>          Infrastructure method used for deployment

Optional:
  --region <region>         AWS region (default: AWS CLI configured or us-west-2)
  --stack-name <name>       HyperPod CFN stack name (default: hp-eks-slinky-stack)
  --help                    Show this help message

Examples:
  # Destroy CloudFormation-based deployment
  $0 --infra cfn

  # Destroy Terraform-based deployment with custom stack name
  $0 --infra tf --stack-name my-slinky-stack
EOF
    exit 0
}

###########################
###### Parse Arguments ####
###########################

while [[ $# -gt 0 ]]; do
    case $1 in
        --infra)
            INFRA="$2"
            shift 2
            ;;
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --stack-name)
            STACK_NAME="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Run '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

###########################
### Validate Arguments ####
###########################

if [[ -z "${INFRA}" ]]; then
    echo "Error: --infra is required (cfn or tf)"
    exit 1
fi

if [[ "${INFRA}" != "cfn" && "${INFRA}" != "tf" ]]; then
    echo "Error: --infra must be 'cfn' or 'tf' (got: ${INFRA})"
    exit 1
fi

echo ""
echo "=========================================="
echo "  Slurm Cluster Teardown"
echo "=========================================="
echo ""
echo "  Infrastructure: ${INFRA}"
echo "  Region: ${AWS_REGION}"
echo "  Stack name: ${STACK_NAME}"
echo ""

read -r -p "This will destroy all Slurm and infrastructure resources. Continue? [y/N] " response
if [[ ! "${response}" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

###########################
## Uninstall Slurm ########
###########################

echo ""
echo "Uninstalling Slurm cluster..."
helm uninstall slurm -n slurm 2>/dev/null || echo "  Slurm cluster not found (already removed)."

echo "Uninstalling Slurm operator..."
helm uninstall slurm-operator -n slinky 2>/dev/null || echo "  Slurm operator not found (already removed)."

###########################
## Uninstall MariaDB ######
###########################

echo ""
echo "Deleting MariaDB instance..."
kubectl delete -f "${SCRIPT_DIR}/mariadb.yaml" 2>/dev/null || echo "  MariaDB instance not found."

echo "Uninstalling MariaDB operator..."
helm uninstall mariadb-operator -n mariadb 2>/dev/null || echo "  MariaDB operator not found (already removed)."

###########################
## Delete Namespaces ######
###########################

echo ""
echo "Deleting namespaces..."
kubectl delete namespace slurm 2>/dev/null || echo "  Namespace slurm not found."
kubectl delete namespace slinky 2>/dev/null || echo "  Namespace slinky not found."
kubectl delete namespace mariadb 2>/dev/null || echo "  Namespace mariadb not found."

###########################
## Delete CodeBuild Stack #
###########################

echo ""
echo "Deleting CodeBuild infrastructure..."

if [[ "${INFRA}" == "cfn" ]]; then
    if aws cloudformation describe-stacks \
        --stack-name "${CODEBUILD_STACK_NAME}" \
        --region "${AWS_REGION}" &>/dev/null; then
        echo "  Deleting CodeBuild CloudFormation stack..."
        aws cloudformation delete-stack \
            --stack-name "${CODEBUILD_STACK_NAME}" \
            --region "${AWS_REGION}"
        aws cloudformation wait stack-delete-complete \
            --stack-name "${CODEBUILD_STACK_NAME}" \
            --region "${AWS_REGION}"
        echo "  CodeBuild stack deleted."
    else
        echo "  CodeBuild stack not found (already removed)."
    fi
else
    local cb_dir="${SCRIPT_DIR}/codebuild-tf"
    if [[ -d "${cb_dir}" ]] && [[ -f "${cb_dir}/terraform.tfstate" ]]; then
        echo "  Destroying CodeBuild Terraform resources..."
        terraform -chdir="${cb_dir}" destroy -auto-approve
        echo "  CodeBuild resources destroyed."
    else
        echo "  CodeBuild Terraform state not found (already removed)."
    fi
fi

###########################
## Delete Infrastructure ##
###########################

echo ""
echo "Deleting HyperPod EKS infrastructure..."

if [[ "${INFRA}" == "cfn" ]]; then
    if aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --region "${AWS_REGION}" &>/dev/null; then
        echo "  Deleting CloudFormation stack '${STACK_NAME}'..."
        aws cloudformation delete-stack \
            --stack-name "${STACK_NAME}" \
            --region "${AWS_REGION}"
        echo "  Stack deletion initiated."
        echo "  (This typically takes 15-20 minutes)"
        echo ""
        echo "  To monitor progress:"
        echo "    aws cloudformation describe-stacks --stack-name ${STACK_NAME} --region ${AWS_REGION}"
    else
        echo "  Stack '${STACK_NAME}' not found (already removed)."
    fi
else
    local tf_dir="${SCRIPT_DIR}/../terraform-modules/hyperpod-eks-tf"
    if [[ -d "${tf_dir}" ]]; then
        echo "  Destroying Terraform infrastructure..."
        terraform -chdir="${tf_dir}" plan -destroy -var-file="custom.tfvars"

        read -r -p "  Apply Terraform destroy? [y/N] " tf_response
        if [[ "${tf_response}" =~ ^[Yy]$ ]]; then
            terraform -chdir="${tf_dir}" destroy -var-file="custom.tfvars" -auto-approve
            echo "  Terraform resources destroyed."
        else
            echo "  Terraform destroy aborted."
        fi
    else
        echo "  Terraform directory not found: ${tf_dir}"
    fi
fi

###########################
## Clean Up Local Files ###
###########################

echo ""
echo "Cleaning up local generated files..."
rm -f "${SCRIPT_DIR}/slurm-values.yaml"
rm -f "${SCRIPT_DIR}/slurm-login-service-patch.yaml"
rm -f "${SCRIPT_DIR}/env_vars.sh"
echo "  Cleaned up."

echo ""
echo "=========================================="
echo "  Teardown Complete"
echo "=========================================="
echo ""
