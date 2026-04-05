#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Setup script for Nemotron 3 Nano Kubernetes launchers.
# Installs NeMo-Run, SkyPilot (with Kubernetes backend), and dependencies
# needed to submit training jobs from a local machine to an EKS cluster.
#
# Usage:
#   python -m venv .venv && source .venv/bin/activate
#   bash venv.sh

set -e

echo "Installing Nemotron 3 Nano Kubernetes dependencies..."

# Install SkyPilot with Kubernetes backend
pip install 'skypilot[kubernetes]'

# Install NeMo-Run for experiment orchestration
pip install git+https://github.com/NVIDIA/NeMo-Run.git

# Install PyTorch (must match container CUDA version)
pip install torch==2.6.0

# Install Megatron-LM core
pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git

# Install NeMo Toolkit
pip install nemo_toolkit['all']

# Install Megatron Bridge (for Nemotron 3 Nano support)
pip install git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git

# Install NVIDIA Resiliency Extension for fault tolerance
pip install nvidia-resiliency-ext

# Install OpenCC (tokenizer dependency)
pip install opencc==1.1.6

echo "Nemotron 3 Nano Kubernetes environment setup complete."
