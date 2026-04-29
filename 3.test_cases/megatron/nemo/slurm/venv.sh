#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Host-side venv for NeMo-Run. The actual training runs inside the
# `nvcr.io/nvidia/nemo:26.02` container (built by ../Dockerfile); this venv
# only needs to satisfy NeMo-Run's import-time deps on the head node.

set -e

# Pin to NeMo-Run v0.9.0 (Apr 2026 tag) instead of an arbitrary commit, so
# `bash venv.sh` produces the same environment across runs.
pip install "nemo-run==0.9.0"

# Torch is a NeMo-Run import-time dep on the host; CUDA flavor doesn't matter
# here because all GPU work happens inside the container.
pip install "torch==2.10.0"

# Megatron-LM pinned to the same release tag the container ships with.
pip install --no-deps "git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.17.0"

# NeMo Toolkit. PERFORMANCE.md (in this directory's parent) lists 2.5+ as
# recommended on the NeMo 26.02 container. 2.7.3 is the latest patch in 2.x.
pip install "nemo_toolkit[all]==2.7.3"

pip install "opencc==1.1.6"

# NVIDIA Resiliency Extension for fault-tolerance plugins used in run.py.
pip install "nvidia-resiliency-ext==0.4.1"

echo "Environment setup complete."
