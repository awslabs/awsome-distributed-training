#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -euo pipefail

###########################
###### Default Values #####
###########################

AWS_REGION="us-west-2"
AZ_ID="usw2-az2"
NODE_TYPE=""
INFRA=""
STACK_NAME="hp-eks-slinky-stack"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default profile: g5
ACCEL_INSTANCE_TYPE="ml.g5.8xlarge"
ACCEL_INSTANCE_COUNT=4

###########################
## Source Helper Library ###
###########################

source "${SCRIPT_DIR}/lib/deploy_helpers.sh"

###########################
###### Usage Function #####
###########################

usage() {
    cat <<EOF
Usage: $0 --node-type <g5|p5> --infra <cfn|tf> [OPTIONS]

Deploy HyperPod EKS infrastructure using CloudFormation or Terraform.

Required:
  --node-type <g5|p5>       Instance profile to deploy
  --infra <cfn|tf>          Infrastructure deployment method

Optional:
  --region <region>         AWS region (default: us-west-2)
  --az-id <az-id>           Availability zone ID for instance groups and FSx
                            (default: usw2-az2)
  --stack-name <name>       CloudFormation stack name (default: hp-eks-slinky-stack)
  --help                    Show this help message

Examples:
  # Deploy g5 instances in us-west-2 using CloudFormation
  $0 --node-type g5 --infra cfn

  # Deploy p5 instances in us-east-1 using Terraform
  $0 --node-type p5 --infra tf --region us-east-1 --az-id use1-az2

  # Deploy with custom stack name
  $0 --node-type g5 --infra cfn --stack-name my-slinky-stack
EOF
    exit 0
}

###########################
###### Parse Arguments ####
###########################

while [[ $# -gt 0 ]]; do
    case $1 in
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --az-id)
            AZ_ID="$2"
            shift 2
            ;;
        --node-type)
            NODE_TYPE="$2"
            shift 2
            ;;
        --infra)
            INFRA="$2"
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

if [[ -z "${NODE_TYPE}" ]]; then
    echo "Error: --node-type is required (g5 or p5)"
    exit 1
fi

if [[ -z "${INFRA}" ]]; then
    echo "Error: --infra is required (cfn or tf)"
    exit 1
fi

if [[ "${INFRA}" != "cfn" && "${INFRA}" != "tf" ]]; then
    echo "Error: --infra must be 'cfn' or 'tf' (got: ${INFRA})"
    exit 1
fi

# Resolve node profile (sets ACCEL_INSTANCE_TYPE and ACCEL_INSTANCE_COUNT)
resolve_node_profile "${NODE_TYPE}"

###########################
## Check Prerequisites ####
###########################

echo "Checking prerequisites..."
check_command "aws" || exit 1

if [[ "${INFRA}" == "cfn" ]]; then
    check_command "jq" || exit 1
elif [[ "${INFRA}" == "tf" ]]; then
    check_command "terraform" || exit 1
fi

# Validate AWS credentials
if ! aws sts get-caller-identity --region "${AWS_REGION}" &>/dev/null; then
    echo "Error: Invalid AWS credentials or unable to reach AWS."
    exit 1
fi

echo "  AWS CLI: OK"
echo "  AWS credentials: OK"
echo "  Region: ${AWS_REGION}"
echo "  Node type: ${NODE_TYPE}"
echo "  Accelerated instance: ${ACCEL_INSTANCE_TYPE} x ${ACCEL_INSTANCE_COUNT}"
echo "  Infrastructure: ${INFRA}"
echo "  AZ ID: ${AZ_ID}"

###########################
## Resolve AZ IDs #########
###########################

echo ""
echo "Resolving availability zones for region ${AWS_REGION}..."

AZ_IDS=$(aws ec2 describe-availability-zones \
    --region "${AWS_REGION}" \
    --filters "Name=opt-in-status,Values=opt-in-not-required" \
    --query "AvailabilityZones[?ZoneType=='availability-zone'].ZoneId | sort(@) | [:5]" \
    --output text | tr '\t' ',')

if [[ -z "${AZ_IDS}" ]]; then
    echo "Error: No availability zones found in region ${AWS_REGION}."
    exit 1
fi

echo "  Resolved AZ IDs: ${AZ_IDS}"
echo "  Instance/FSx AZ ID: ${AZ_ID}"

# Verify the specified AZ_ID exists in the resolved list
if ! validate_az_id "${AZ_ID}" "${AZ_IDS}"; then
    echo ""
    echo "WARNING: AZ ID '${AZ_ID}' was not found in the resolved AZ list for ${AWS_REGION}."
    echo "  Available AZs: ${AZ_IDS}"
    echo "  Specify a valid AZ ID with --az-id or press Ctrl+C to abort."
    read -r -p "  Continue anyway? [y/N] " response
    if [[ ! "${response}" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

###########################
## Deploy via CFN #########
###########################

deploy_cfn() {
    local params_file="${SCRIPT_DIR}/params.json"

    if [[ ! -f "${params_file}" ]]; then
        echo "Error: Parameters file not found: ${params_file}"
        exit 1
    fi

    echo ""
    echo "Preparing CloudFormation parameters..."
    echo "  Source: ${params_file}"
    echo "  Stack name: ${STACK_NAME}"

    # Resolve AZ values and node-type overrides in the params file
    local resolved_params
    resolved_params=$(resolve_cfn_params \
        "${params_file}" \
        "${AZ_IDS}" \
        "${AZ_ID}" \
        "${ACCEL_INSTANCE_TYPE}" \
        "${ACCEL_INSTANCE_COUNT}")

    # Write resolved params to a temp file
    local resolved_file
    resolved_file=$(mktemp /tmp/resolved-params-XXXXXX.json)
    echo "${resolved_params}" > "${resolved_file}"

    echo "  Resolved params written to: ${resolved_file}"
    echo ""

    # Construct the S3 template URL
    local template_url="https://aws-sagemaker-hyperpod-cluster-setup-${AWS_REGION}-prod.s3.${AWS_REGION}.amazonaws.com/templates/main-stack-eks-based-template.yaml"

    echo "Deploying CloudFormation stack '${STACK_NAME}'..."
    echo "  Template: ${template_url}"
    echo ""

    aws cloudformation create-stack \
        --region "${AWS_REGION}" \
        --stack-name "${STACK_NAME}" \
        --template-url "${template_url}" \
        --parameters "file://${resolved_file}" \
        --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND

    echo ""
    echo "Stack creation initiated. Waiting for completion..."
    echo "  (This typically takes 20-30 minutes)"
    echo ""

    aws cloudformation wait stack-create-complete \
        --region "${AWS_REGION}" \
        --stack-name "${STACK_NAME}"

    echo "Stack '${STACK_NAME}' created successfully."
    echo ""

    # Extract and export stack outputs
    extract_cfn_outputs

    # Clean up temp file
    rm -f "${resolved_file}"
}

###########################
## Extract CFN Outputs ####
###########################

extract_cfn_outputs() {
    echo "Extracting stack outputs..."
    echo ""

    local account_id
    account_id=$(aws sts get-caller-identity --query Account --output text)

    local eks_cluster_name
    eks_cluster_name=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" --region "${AWS_REGION}" \
        --query "Stacks[0].Outputs[?OutputKey=='OutputEKSClusterName'].OutputValue" \
        --output text)

    local vpc_id
    vpc_id=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" --region "${AWS_REGION}" \
        --query "Stacks[0].Outputs[?OutputKey=='OutputVpcId'].OutputValue" \
        --output text)

    local private_subnet_id
    private_subnet_id=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" --region "${AWS_REGION}" \
        --query "Stacks[0].Outputs[?OutputKey=='OutputPrivateSubnetIds'].OutputValue" \
        --output text | cut -d',' -f1)

    local security_group_id
    security_group_id=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" --region "${AWS_REGION}" \
        --query "Stacks[0].Outputs[?OutputKey=='OutputSecurityGroupId'].OutputValue" \
        --output text)

    # Write env_vars.sh for sourcing
    local env_file="${SCRIPT_DIR}/env_vars.sh"
    cat > "${env_file}" <<EOF
export AWS_ACCOUNT_ID="${account_id}"
export AWS_REGION="${AWS_REGION}"
export EKS_CLUSTER_NAME="${eks_cluster_name}"
export VPC_ID="${vpc_id}"
export PRIVATE_SUBNET_ID="${private_subnet_id}"
export SECURITY_GROUP_ID="${security_group_id}"
export STACK_ID="${STACK_NAME}"
EOF

    echo "Environment variables written to: ${env_file}"
    echo ""
    echo "  AWS_ACCOUNT_ID=${account_id}"
    echo "  AWS_REGION=${AWS_REGION}"
    echo "  EKS_CLUSTER_NAME=${eks_cluster_name}"
    echo "  VPC_ID=${vpc_id}"
    echo "  PRIVATE_SUBNET_ID=${private_subnet_id}"
    echo "  SECURITY_GROUP_ID=${security_group_id}"
    echo ""
    echo "To load these variables into your shell, run:"
    echo "  source ${env_file}"
}

###########################
## Deploy via Terraform ###
###########################

deploy_tf() {
    local tfvars_file="${SCRIPT_DIR}/custom.tfvars"
    local tf_dir="${SCRIPT_DIR}/../terraform-modules/hyperpod-eks-tf"

    if [[ ! -f "${tfvars_file}" ]]; then
        echo "Error: Terraform variables file not found: ${tfvars_file}"
        exit 1
    fi

    if [[ ! -d "${tf_dir}" ]]; then
        echo "Error: Terraform modules directory not found: ${tf_dir}"
        echo "  Expected at: ${tf_dir}"
        echo "  Make sure you have cloned the awsome-distributed-training repo."
        exit 1
    fi

    echo ""
    echo "Preparing Terraform deployment..."
    echo "  Source tfvars: ${tfvars_file}"
    echo "  Terraform dir: ${tf_dir}"

    # Copy the tfvars file into the TF module directory and patch it
    local target_tfvars="${tf_dir}/custom.tfvars"
    cp "${tfvars_file}" "${target_tfvars}"

    resolve_tf_vars \
        "${target_tfvars}" \
        "${AWS_REGION}" \
        "${AZ_ID}" \
        "${ACCEL_INSTANCE_TYPE}" \
        "${ACCEL_INSTANCE_COUNT}" \
        "${NODE_TYPE}"

    echo "  Resolved tfvars written to: ${target_tfvars}"
    echo ""

    echo "Initializing Terraform..."
    terraform -chdir="${tf_dir}" init

    echo ""
    echo "Generating Terraform plan..."
    terraform -chdir="${tf_dir}" plan -var-file="custom.tfvars"

    echo ""
    read -r -p "Apply this Terraform plan? [y/N] " response
    if [[ ! "${response}" =~ ^[Yy]$ ]]; then
        echo "Aborted. You can apply manually with:"
        echo "  cd ${tf_dir}"
        echo "  terraform apply -var-file=custom.tfvars"
        exit 0
    fi

    echo ""
    echo "Applying Terraform..."
    terraform -chdir="${tf_dir}" apply -var-file="custom.tfvars" -auto-approve

    echo ""
    echo "Terraform apply completed successfully."
    echo ""

    # Run terraform_outputs.sh if it exists
    local outputs_script="${SCRIPT_DIR}/../terraform-modules/terraform_outputs.sh"
    if [[ -f "${outputs_script}" ]]; then
        echo "Extracting Terraform outputs..."
        chmod +x "${outputs_script}"
        (cd "${SCRIPT_DIR}/../terraform-modules" && ./terraform_outputs.sh)

        local env_file="${SCRIPT_DIR}/../terraform-modules/env_vars.sh"
        if [[ -f "${env_file}" ]]; then
            # Copy env_vars.sh to slinky-slurm directory for convenience
            cp "${env_file}" "${SCRIPT_DIR}/env_vars.sh"
            echo "Environment variables written to: ${SCRIPT_DIR}/env_vars.sh"
            echo ""
            echo "To load these variables into your shell, run:"
            echo "  source ${SCRIPT_DIR}/env_vars.sh"
        fi
    else
        echo "Note: terraform_outputs.sh not found at ${outputs_script}"
        echo "  You may need to manually extract outputs."
    fi
}

###########################
###### Main Execution #####
###########################

echo ""
echo "=========================================="
echo "  HyperPod EKS Infrastructure Deployment"
echo "=========================================="
echo ""

if [[ "${INFRA}" == "cfn" ]]; then
    deploy_cfn
elif [[ "${INFRA}" == "tf" ]]; then
    deploy_tf
fi

echo ""
echo "=========================================="
echo "  Deployment Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Source the environment variables:"
echo "       source ${SCRIPT_DIR}/env_vars.sh"
echo "  2. Update your kubectl context:"
echo "       aws eks update-kubeconfig --name \$EKS_CLUSTER_NAME --region ${AWS_REGION}"
echo "  3. Verify node connectivity:"
echo "       kubectl get nodes"
echo ""
