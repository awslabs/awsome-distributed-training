FROM public.ecr.aws/hpc-cloud/nccl-tests:latest

RUN apt update && apt install -y nvtop

RUN pip install torch numpy torchvision pillow datasets huggingface-hub transformers wandb einops accelerate loguru lmms_eval
RUN pip install sagemaker-mlflow
RUN mkdir -p /nanoVLM 
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY nanoVLM/ /nanoVLM/

WORKDIR /nanoVLM
