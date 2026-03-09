#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -euo pipefail

###########################
###### Default Values #####
###########################

RUN_SETUP=true
SETUP_ARGS=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_REGION=$(aws configure get region 2>/dev/null || echo "us-west-2")
SLURM_OPERATOR_VERSION="1.0.1"
SLURM_CHART_VERSION="1.0.1"
MARIADB_OPERATOR_VERSION="25.10.4"

###########################
###### Usage Function #####
###########################

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Install the Slurm cluster on HyperPod EKS. Runs setup.sh first (unless
--skip-setup is specified), then deploys MariaDB, the Slurm operator, and
the Slurm cluster via Helm.

Optional:
  --skip-setup              Use previously generated slurm-values.yaml
  --region <region>         AWS region (default: AWS CLI configured or us-west-2)
  --help                    Show this help message

Options passed through to setup.sh:
  --node-type <g5|p5>       Instance profile
  --infra <cfn|tf>          Infrastructure method for CodeBuild stack
  --repo-name <name>        ECR repository name
  --tag <tag>               Image tag
  --local-build             Build image locally instead of CodeBuild
  --skip-build              Skip image build (use existing image in ECR)

Examples:
  # Full install: build image + deploy Slurm (g5 via CloudFormation)
  $0 --node-type g5 --infra cfn

  # Skip setup (slurm-values.yaml already generated)
  $0 --skip-setup

  # Build locally, then install
  $0 --node-type p5 --infra tf --local-build
EOF
    exit 0
}

###########################
###### Parse Arguments ####
###########################

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-setup)
            RUN_SETUP=false
            shift
            ;;
        --region)
            AWS_REGION="$2"
            SETUP_ARGS="${SETUP_ARGS} --region $2"
            shift 2
            ;;
        --help)
            usage
            ;;
        --node-type|--infra|--repo-name|--tag)
            SETUP_ARGS="${SETUP_ARGS} $1 $2"
            shift 2
            ;;
        --local-build|--skip-build)
            SETUP_ARGS="${SETUP_ARGS} $1"
            shift
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Run '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

echo ""
echo "=========================================="
echo "  Slurm Cluster Installation"
echo "=========================================="
echo ""

###########################
## Run Setup ##############
###########################

if [[ "${RUN_SETUP}" == "true" ]]; then
    echo "Running setup.sh..."
    echo ""
    # shellcheck disable=SC2086
    bash "${SCRIPT_DIR}/setup.sh" ${SETUP_ARGS}
else
    echo "Skipping setup (--skip-setup)..."
    if [[ ! -f "${SCRIPT_DIR}/slurm-values.yaml" ]]; then
        echo "Error: slurm-values.yaml not found."
        echo "  Run without --skip-setup, or generate it manually."
        exit 1
    fi
    echo "  Using existing slurm-values.yaml"
fi

echo ""

###########################
## Install MariaDB ########
###########################

echo "Installing MariaDB operator (v${MARIADB_OPERATOR_VERSION})..."

helm repo add mariadb-operator \
    https://mariadb-operator.github.io/mariadb-operator 2>/dev/null || true
helm repo update

if ! helm status mariadb-operator -n mariadb &>/dev/null; then
    helm install mariadb-operator mariadb-operator/mariadb-operator \
        --version="${MARIADB_OPERATOR_VERSION}" \
        --namespace=mariadb --create-namespace \
        --set crds.enabled=true \
        --wait
    echo "  MariaDB operator installed."
else
    echo "  MariaDB operator already installed."
fi

echo ""
echo "Creating Slurm namespace and MariaDB instance..."

kubectl create namespace slurm 2>/dev/null || true
kubectl apply -f "${SCRIPT_DIR}/mariadb.yaml"
echo "  MariaDB instance applied."

# Wait for MariaDB to be ready
echo "  Waiting for MariaDB to be ready..."
kubectl wait --for=condition=Ready mariadb/mariadb \
    -n slurm --timeout=300s 2>/dev/null || \
    echo "  WARNING: MariaDB readiness check timed out. Proceeding anyway."

###########################
## Install Slurm Operator #
###########################

echo ""
echo "Installing Slurm Operator (v${SLURM_OPERATOR_VERSION})..."

# Delete stale CRDs if upgrading from an older version
kubectl delete crd clusters.slinky.slurm.net 2>/dev/null || true
kubectl delete crd nodesets.slinky.slurm.net 2>/dev/null || true

if ! helm status slurm-operator -n slinky &>/dev/null; then
    helm install slurm-operator \
        oci://ghcr.io/slinkyproject/charts/slurm-operator \
        --version="${SLURM_OPERATOR_VERSION}" \
        --namespace=slinky --create-namespace \
        --wait
    echo "  Slurm operator installed."
else
    echo "  Slurm operator already installed."
fi

###########################
## Install Slurm Cluster ##
###########################

echo ""
echo "Installing Slurm Cluster (v${SLURM_CHART_VERSION})..."

if ! helm status slurm -n slurm &>/dev/null; then
    helm install slurm \
        oci://ghcr.io/slinkyproject/charts/slurm \
        --values="${SCRIPT_DIR}/slurm-values.yaml" \
        --version="${SLURM_CHART_VERSION}" \
        --namespace=slurm
    echo "  Slurm cluster installed."
else
    echo "  Slurm cluster already installed."
fi

###########################
## Configure NLB ##########
###########################

echo ""
echo "Configuring login service NLB..."

# Wait for the login service to exist
echo "  Waiting for slurm-login-slinky service..."
until kubectl get service slurm-login-slinky -n slurm &>/dev/null; do
    sleep 5
done
echo "  Service found."

# Get public IP for NLB source range restriction
IP_ADDRESS="$(curl -s https://checkip.amazonaws.com)"
echo "  Source IP: ${IP_ADDRESS}"

# Generate and apply service patch
sed "s|\${ip_address}|${IP_ADDRESS}|g" \
    "${SCRIPT_DIR}/slurm-login-service-patch.yaml.template" \
    > "${SCRIPT_DIR}/slurm-login-service-patch.yaml"

kubectl patch service slurm-login-slinky -n slurm \
    --patch-file "${SCRIPT_DIR}/slurm-login-service-patch.yaml"
echo "  Login service patched with NLB annotations."

###########################
## Wait for NLB ###########
###########################

echo ""
echo "Waiting for NLB endpoint..."

SLURM_LOGIN_HOSTNAME=""
for i in $(seq 1 60); do
    SLURM_LOGIN_HOSTNAME=$(kubectl get services -n slurm \
        -l app.kubernetes.io/instance=slurm,app.kubernetes.io/name=login \
        -o jsonpath="{.items[0].status.loadBalancer.ingress[0].hostname}" 2>/dev/null || true)

    if [[ -n "${SLURM_LOGIN_HOSTNAME}" ]]; then
        break
    fi
    sleep 5
done

echo ""
echo "=========================================="
echo "  Installation Complete"
echo "=========================================="
echo ""

if [[ -n "${SLURM_LOGIN_HOSTNAME}" ]]; then
    echo "  Login endpoint: ${SLURM_LOGIN_HOSTNAME}"
    echo ""
    echo "  SSH into the Slurm login node:"
    echo "    ssh -i ~/.ssh/id_ed25519_slurm root@${SLURM_LOGIN_HOSTNAME}"
else
    echo "  WARNING: NLB hostname not yet available."
    echo "  Check with: kubectl get svc slurm-login-slinky -n slurm"
fi

echo ""
echo "  Verify cluster status:"
echo "    kubectl -n slurm get pods -l app.kubernetes.io/instance=slurm"
echo ""
