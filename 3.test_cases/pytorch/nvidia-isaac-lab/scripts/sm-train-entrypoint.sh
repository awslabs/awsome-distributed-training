#!/bin/bash
set -e

echo "=== SageMaker Training Job ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
nvidia-smi -L

# === NCCL / Networking Configuration ===
# g6 instances don't have EFA, so use TCP sockets for inter-node NCCL
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_PROTO=Simple

# Isaac Sim base image sets NVIDIA_VISIBLE_DEVICES=void — override
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=all

# NVIDIA EULA
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

# Enable full error tracebacks
export HYDRA_FULL_ERROR=1
export TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error.json

# Serialize Isaac Sim pip env initialization to avoid race condition
# (FileExistsError on /isaac-sim/kit/data/Kit/Isaac-Sim/4.5/pip3-envs/default)
export ISAACLAB_INIT_LOCK=/tmp/isaaclab_init.lock

echo "=== Network Interfaces ==="
ip addr show eth0 | head -5
echo "=== /dev/shm ==="
df -h /dev/shm
echo ""

# SageMaker BYOC doesn't inject SM_* env vars automatically.
# For multi-node, we need to read from /opt/ml/input/config/resourceconfig.json
CONFIG_FILE="/opt/ml/input/config/resourceconfig.json"
if [ -f "$CONFIG_FILE" ]; then
  echo "=== Resource Config ==="
  cat "$CONFIG_FILE"
  echo ""

  # Use Isaac Sim's bundled python
  PYTHON="/isaac-sim/python.sh"

  CURRENT_HOST=$($PYTHON -c "import json; cfg=json.load(open('$CONFIG_FILE')); print(cfg['current_host'])")
  ALL_HOSTS=$($PYTHON -c "import json; cfg=json.load(open('$CONFIG_FILE')); print(','.join(cfg['hosts']))")
  NNODES=$($PYTHON -c "import json; cfg=json.load(open('$CONFIG_FILE')); print(len(cfg['hosts']))")
  NODE_RANK=$($PYTHON -c "import json; cfg=json.load(open('$CONFIG_FILE')); print(cfg['hosts'].index(cfg['current_host']))")
  MASTER_HOST=$($PYTHON -c "import json; cfg=json.load(open('$CONFIG_FILE')); print(cfg['hosts'][0])")
  NPROC=$($PYTHON -c "
import subprocess
result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
print(len([l for l in result.stdout.strip().split(chr(10)) if l.startswith('GPU')]))
")
else
  echo "No resource config found, assuming single node"
  CURRENT_HOST=$(hostname)
  MASTER_HOST=$(hostname)
  NNODES=1
  NODE_RANK=0
  NPROC=4
fi

MASTER_PORT=29500

echo "=== Training Configuration ==="
echo "CURRENT_HOST=$CURRENT_HOST"
echo "MASTER_HOST=$MASTER_HOST"
echo "ALL_HOSTS=$ALL_HOSTS"
echo "NNODES=$NNODES"
echo "NODE_RANK=$NODE_RANK"
echo "NPROC=$NPROC"
echo "MASTER_PORT=$MASTER_PORT"
echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "NCCL_DEBUG=$NCCL_DEBUG"
echo "MAX_ITERATIONS=${MAX_ITERATIONS:-1000}"

# Test cross-node connectivity before starting training
if [ "$NNODES" -gt 1 ]; then
  echo "=== Testing Cross-Node Connectivity ==="
  MASTER_IP=$(getent hosts $MASTER_HOST | awk '{print $1}')
  echo "Master IP: $MASTER_IP"
  # Test if we can reach the master port (non-blocking, just informational)
  timeout 5 bash -c "echo > /dev/tcp/$MASTER_HOST/$MASTER_PORT" 2>/dev/null && echo "Master port reachable" || echo "Master port not yet open (expected if we start before master)"
  # Also test basic ICMP
  ping -c 1 -W 2 $MASTER_HOST 2>/dev/null && echo "Ping to master OK" || echo "Ping to master failed (ICMP may be blocked, not fatal)"
fi

echo "=== Starting Isaac Lab H1 Training ==="
cd /workspace/IsaacLab

/isaac-sim/python.sh -m torch.distributed.run \
  --nproc_per_node=$NPROC \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --rdzv_id=sm-isaaclab \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_HOST:$MASTER_PORT \
  scripts/reinforcement_learning/skrl/train.py \
  --distributed \
  --task=Isaac-Velocity-Rough-H1-v0 \
  --max_iterations=${MAX_ITERATIONS:-1000} \
  --headless

TRAIN_EXIT=$?
echo "=== Training Exit Code: $TRAIN_EXIT ==="

# Check for error files
if [ -f /tmp/torch_elastic_error.json ]; then
  echo "=== Torch Elastic Error File ==="
  cat /tmp/torch_elastic_error.json
fi

echo "=== Copying Checkpoints ==="
mkdir -p /opt/ml/model
cp -rv /workspace/IsaacLab/logs/* /opt/ml/model/ 2>/dev/null || true
echo "=== Checkpoints ==="
find /opt/ml/model -name "*.pt" -ls 2>/dev/null
echo "Done!"
