#!/bin/bash

# Build the AWS-optimized Nemotron 3 Nano container for Kubernetes
# Includes EFA support optimizations for P4/P5/P6 instances

set -e

IMAGE_NAME="aws-nemotron3-nano"
TAG="25.11"
DOCKERFILE="${1:-Dockerfile}"

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
echo "Dockerfile: ${DOCKERFILE}"
echo "This may take several minutes..."

CMD="docker build --progress=plain -t ${IMAGE_NAME}:${TAG} -f ${DOCKERFILE} ."
if [ ! "$verbose" == "false" ]; then echo -e "\n${CMD}\n"; fi
eval "${CMD}"

echo "Build completed successfully!"
echo "Image: ${IMAGE_NAME}:${TAG}"
echo ""
echo "To build the GRPO container instead:"
echo "  bash build.sh Dockerfile.grpo"
