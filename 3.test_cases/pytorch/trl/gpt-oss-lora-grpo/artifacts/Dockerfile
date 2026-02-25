# 1. Use AWS Image (EFA/NCCL/PyTorch pre-installed)

FROM public.ecr.aws/deep-learning-containers/base:12.9.1-gpu-py312-cu129-ubuntu22.04-ec2-v1.31

## Install NCCL-tests
ARG NCCL_TESTS_VERSION=v2.16.9
RUN git clone -b ${NCCL_TESTS_VERSION} https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && cd /opt/nccl-tests \
    && make -j $(nproc) \
    MPI=1 \
    MPI_HOME=/opt/amazon/openmpi/ \
    CUDA_HOME=/usr/local/cuda \
    NCCL_HOME=/opt/nccl/build \
    NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100"
RUN rm -rf /var/lib/apt/lists/*

# 2. System tools

RUN apt-get update && apt-get install -y \
    git \
    wget \
    python3-pip \
    python3-dev \  
    && rm -rf /var/lib/apt/lists/*

# 3. Setup Workspace
WORKDIR /app
COPY requirements.txt /app/

# 4. Install Dependencies
# CRITICAL FIX: We remove flash-attn from the list first, then install it manually with --no-build-isolation
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install packaging ninja && \
    sed -i '/flash-attn/d' requirements.txt && \
    pip install -r requirements.txt

# 5. Install Flash Attention (The FIX)
# This flag tells pip: "Don't create a new empty environment; use the PyTorch that is already here."
RUN pip install flash-attn --no-build-isolation

# 6. Install HyperPod Agent & DeepSpeed (if not in requirements)
RUN pip install hyperpod-elastic-agent trackio langdetect

# 7. Copy Code & Set Entrypoint
COPY src/ /app/
ENV PYTHONPATH="${PYTHONPATH}:/app"
#CMD ["python", "sft.py", "--config", "configs/sft_lora.yaml"]
