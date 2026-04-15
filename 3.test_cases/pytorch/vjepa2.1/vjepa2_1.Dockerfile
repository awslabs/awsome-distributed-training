# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# V-JEPA 2.1 reuses the same container as V-JEPA 2, since both training
# apps live in the same facebookresearch/vjepa2 repository.
#
# V-JEPA 2.1 requires Python >= 3.11
# Using NVIDIA PyTorch container as base for CUDA 13 + NCCL + EFA compatibility
FROM nvcr.io/nvidia/pytorch:25.03-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install EFA
ARG EFA_INSTALLER_VERSION=1.47.0
RUN cd /tmp && \
    curl -sL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz | tar xz && \
    cd aws-efa-installer && \
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify && \
    cd /tmp && rm -rf aws-efa-installer

# Install NCCL OFI plugin for EFA
RUN apt-get update && apt-get install -y libnccl-dev && rm -rf /var/lib/apt/lists/* || true

# Install V-JEPA 2 / 2.1 dependencies
RUN pip install --no-cache-dir \
    tensorboard wandb iopath pyyaml \
    opencv-python submitit braceexpand webdataset timm transformers \
    peft decord pandas einops beartype psutil h5py fire python-box \
    scikit-image ftfy eva-decord Pillow

# Clone V-JEPA 2 (includes V-JEPA 2.1 code under app/vjepa_2_1/)
RUN git clone https://github.com/facebookresearch/vjepa2.git /vjepa2
WORKDIR /vjepa2
RUN pip install -e .

# Copy launcher scripts into the container
COPY scripts/run_train.py /vjepa2/scripts/run_train.py

ENV PYTHONPATH="/vjepa2:${PYTHONPATH}"
