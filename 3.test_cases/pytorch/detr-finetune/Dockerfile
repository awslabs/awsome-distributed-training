# DETR-ResNet50 Object Detection Training Container
# Base image: HPC-optimized with pre-configured EFA and NCCL for distributed training
FROM public.ecr.aws/hpc-cloud/nccl-tests:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

WORKDIR /workspace

# Install system dependencies for OpenCV/Pillow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies with pinned versions
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchmetrics==1.6.1 \
    qai-hub-models[detr-resnet50]==0.19.1 \
    Pillow==11.1.0 \
    numpy==1.26.4 \
    timm==1.0.12

# Copy training script
COPY detr_main.py /workspace/detr_main.py

# Create working directories
RUN mkdir -p /workspace/data \
    /workspace/outputs \
    /workspace/checkpoints \
    /workspace/logs

# Verify installations
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python3 -c "import qai_hub_models; print('QAI Hub models installed successfully')" && \
    python3 -c "import torchmetrics; print(f'torchmetrics version: {torchmetrics.__version__}')"

CMD ["python3", "detr_main.py", "--help"]
