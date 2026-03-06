#!/bin/bash
#SBATCH --job-name=nccl-sendrecv-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=nccl_test_%j.out
#SBATCH --error=nccl_test_%j.err
#SBATCH --wait-all-nodes=1

###############################################################################
# NCCL Send/Recv Ring Test — SageMaker HyperPod (Slurm + EFA)
#
# Current config: p4d.24xlarge (8 A100 GPUs/node, 2 nodes)
###############################################################################

# ---------- EFA / Network settings ----------
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1

# ---------- NCCL tuning for AWS EFA ----------
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^lo,docker0

# Let NCCL auto-select the best algorithm and protocol
# (previously forced Ring/Simple which limited performance)

# Increase cross-node channels to better utilise 4x EFA NICs
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=16

# Larger buffers for 64MB messages
export NCCL_BUFFSIZE=8388608

# Use all 4 EFA interfaces on p4d.24xlarge
export NCCL_NET_GDR_LEVEL=SYS

# ---------- CUDA visibility ----------
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ---------- Build ----------
WORKDIR="$(pwd)"
echo "=== Building on $(hostname) ==="
make -C "${WORKDIR}" clean
make -C "${WORKDIR}"

# ---------- Run ----------
echo "=== Running NCCL Send/Recv test ==="
echo "Nodes: ${SLURM_JOB_NUM_NODES}, Tasks: ${SLURM_NTASKS}, GPUs/node: ${SLURM_GPUS_PER_NODE}"

srun \
    --mpi=pmix \
    --gpus-per-node=${SLURM_GPUS_PER_NODE} \
    --ntasks-per-node=${SLURM_NTASKS_PER_NODE} \
    "${WORKDIR}/nccl_sendrecv_test"
