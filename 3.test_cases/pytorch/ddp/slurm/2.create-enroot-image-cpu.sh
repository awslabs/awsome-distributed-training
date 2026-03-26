#!/bin/bash
set -ex

# 1. Pre-download MNIST to a local folder named 'data'
# This ensures we have the files ready to COPY into the Docker image
python3 -c "from torchvision import datasets; datasets.MNIST(root='./data', train=True, download=True)"

# 2. Build the Docker image
docker build -t pytorch-ddp -f ../Dockerfile.cpu ..

# 3. Convert to Enroot squashfs file
if [ -f pytorch.sqsh ] ; then rm pytorch.sqsh; fi
enroot import -o pytorch.sqsh dockerd://pytorch-ddp:latest