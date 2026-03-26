#!/usr/bin/env bash
set -ex

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

PYTHON_VERSION=$(python3 --version | awk '{print $2}' | awk -F'.' '{print $1"."$2}')

sudo apt install -y python${PYTHON_VERSION}-venv

# Create and activate Python virtual environment
python3 -m venv pt
source ./pt/bin/activate

# Install required packages
pip install torch==2.10.0+cpu torchvision==0.25.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install numpy mlflow==2.13.2 sagemaker-mlflow==0.1.0
