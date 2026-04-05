#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -e

# Install NeMo-Run for experiment orchestration
pip install git+https://github.com/NVIDIA/NeMo-Run.git

# Install PyTorch (must match container CUDA version)
pip install torch==2.6.0

# Install Megatron-LM core
pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git

# Install NeMo Toolkit
pip install nemo_toolkit['all']

# Install Megatron Bridge (for Nemotron 3 Super / nemotronh support)
pip install git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git

# Install NVIDIA Resiliency Extension for fault tolerance
pip install nvidia-resiliency-ext

# Install OpenCC (tokenizer dependency)
pip install opencc==1.1.6

# Mamba-2 / SSM dependencies for the hybrid architecture
pip install mamba-ssm causal-conv1d || echo "mamba-ssm packages already present or build skipped"

echo "Nemotron 3 Super environment setup complete."
