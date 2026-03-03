FROM public.ecr.aws/hpc-cloud/nccl-tests:latest AS base

RUN apt update && apt install -y nvtop

RUN mkdir -p /fsdp /checkpoints
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY src/ /fsdp/
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /fsdp/requirements.txt

WORKDIR /fsdp

# HyperPod-specific image with elastic agent
FROM base AS hpto

# Install HPEA from test.pypi without deps first (test.pypi has a broken
# 'FASTAPI' package that shadows the real fastapi), then resolve deps from PyPI.
RUN pip install "fastapi>=0.113.0" "uvicorn[standard]>=0.30.6" "pip<25.3" \
    && pip install --no-deps --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ hyperpod-elastic-agent==1.1.1
