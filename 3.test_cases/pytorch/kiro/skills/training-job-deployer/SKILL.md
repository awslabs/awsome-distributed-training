---
name: training-job-deployer
description: Deploy distributed training jobs on EKS with support for PyTorchJob (torchrun) and Ray (KubeRay). Orchestrates cluster validation, framework setup, storage configuration, job deployment, and monitoring.
license: MIT
compatibility: kiro
metadata:
  category: deployment
  author: opencode
  orchestrator: true
  dependencies:
    - k8s-cluster-manager
    - ray-cluster-manager
    - pytorchjob-manager
    - checkpoint-manager
    - training-monitor
    - hyperpod-manager
---

## What I do

I am an **orchestrator skill** that coordinates multiple specialized skills to deploy distributed training jobs on EKS. I don't do the work myself - I delegate to focused sub-skills:

1. **k8s-cluster-manager** - Validates cluster health and resources
2. **ray-cluster-manager** - Sets up Ray/KubeRay infrastructure
3. **pytorchjob-manager** - Manages Kubeflow PyTorchJobs
4. **checkpoint-manager** - Configures persistent storage
5. **training-monitor** - Monitors and auto-restarts failed jobs
6. **hyperpod-manager** - Leverages HyperPod-specific features (including instance group scaling)

I also handle **post-training cleanup**: deleting RayClusters/PyTorchJobs and optionally scaling down HyperPod instance groups to save costs.

## When to use me

Use me when you want to:
- **Deploy training with one command** - I handle all the complexity
- **Get started quickly** - Don't worry about which sub-skills to call
- **Ensure proper setup** - I validate and configure everything in the right order
- **Clean up after training** - Delete workloads and scale down instances

**Use individual sub-skills directly when:**
- You need fine-grained control
- You're debugging specific issues
- You want to customize the deployment flow

## Quick Start

### Deploy with Ray (Recommended for VERL)
```bash
python3 ~/.config/opencode/skills/training-job-deployer/src/deploy.py \
  --cluster_name my-cluster \
  --image_uri 123456789.dkr.ecr.us-west-2.amazonaws.com/verl:latest \
  --num_nodes 4 \
  --use_ray \
  --auto_monitor
```

### Deploy with PyTorchJob (Native Kubeflow)
```bash
python3 ~/.config/opencode/skills/training-job-deployer/src/deploy.py \
  --cluster_name my-cluster \
  --image_uri 123456789.dkr.ecr.us-west-2.amazonaws.com/fsdp:latest \
  --num_nodes 4 \
  --use_pytorchjob \
  --auto_monitor
```

### Cleanup: Delete Workloads Only
```bash
python3 ~/.config/opencode/skills/training-job-deployer/src/deploy.py \
  --cluster_name my-cluster \
  --image_uri dummy \
  --job_name verl-grpo-training \
  --cleanup
```

### Cleanup: Delete Workloads AND Scale Down Instances
```bash
python3 ~/.config/opencode/skills/training-job-deployer/src/deploy.py \
  --cluster_name my-cluster \
  --image_uri dummy \
  --job_name verl-grpo-training \
  --cleanup \
  --scale_down \
  --hyperpod_cluster test-cluster-eks-try2 \
  --hyperpod_group test-cluster-eks-try2-ig-8x
```

## How It Works

When you run me in **deploy mode**, I execute this workflow:

```
Step 1: k8s-cluster-manager
        ↓ Check cluster health, GPU availability, EFA status
        
Step 2: checkpoint-manager  
        ↓ Setup persistent storage (PVC) for checkpoints
        
Step 3: ray-cluster-manager OR pytorchjob-manager
        ↓ Install framework (Ray/PyTorchJob) if needed
        ↓ Create cluster/job with proper configuration
        
Step 4: Deploy training job
        ↓ Submit job with resume configuration
        
Step 5: training-monitor (if --auto_monitor)
        ↓ Start monitoring and auto-restart loop
```

When you run me in **cleanup mode** (`--cleanup`), I execute:

```
Step 1: Delete RayCluster (if exists)
        ↓ ray-cluster-manager.delete_raycluster()
        
Step 2: Delete PyTorchJob (if exists)
        ↓ kubectl delete pytorchjob
        
Step 3: Scale down instances (if --scale_down)
        ↓ hyperpod-manager.scale_instance_group() → 0
```

## Parameters

### Required
- `--cluster_name`: EKS cluster name
- `--image_uri`: Docker image URI for training

### Framework Selection
- `--use_ray`: Use Ray/KubeRay (default if neither specified)
- `--use_pytorchjob`: Use Kubeflow PyTorchJob

### Configuration
- `--num_nodes`: Number of nodes (default: 4)
- `--gpu_per_node`: GPUs per node (default: 1)
- `--job_name`: Job name (default: training-job)
- `--checkpoint_dir`: Checkpoint directory (default: /checkpoints/GRPO/<job_name>)
- `--storage_class`: Storage class for PVC (default: fsx-sc)
- `--storage_size`: Storage size (default: 100Gi)

### Monitoring
- `--auto_monitor`: Start auto-restart monitor after deployment
- `--max_retries`: Max restart attempts (default: 10)
- `--retry_delay`: Seconds between retries (default: 60)

### EFA (High-Performance Networking)
- `--use_efia`: Enable EFA (default: True for g5/p4d/p5)
- `--efa_device`: EFA devices per node (default: 1)

### Training Configuration
- `--model_path`: Model path (default: Qwen/Qwen2.5-0.5B)
- `--batch_size`: Training batch size (default: 8)
- `--save_freq`: Checkpoint save frequency (default: 10)

### Cleanup / Teardown
- `--cleanup`: Run cleanup mode instead of deployment
- `--scale_down`: Scale down HyperPod instance group during cleanup
- `--scale_down_target`: Target instance count (default: 0)
- `--hyperpod_cluster`: HyperPod cluster name (auto-detected if only one exists)
- `--hyperpod_group`: HyperPod instance group name (auto-detected from node labels)

## Examples

### Basic Ray Deployment
```bash
python3 ~/.config/opencode/skills/training-job-deployer/src/deploy.py \
  --cluster_name sagemaker-test-cluster-eks-try2-3a6aa148-eks \
  --image_uri 975049888767.dkr.ecr.us-west-2.amazonaws.com/verl-rlvr:latest \
  --num_nodes 4 \
  --use_ray \
  --auto_monitor
```

### VERL Training with Auto-Restart
```bash
python3 ~/.config/opencode/skills/training-job-deployer/src/deploy.py \
  --cluster_name my-cluster \
  --image_uri my-verl-image:latest \
  --job_name verl-grpo-training \
  --num_nodes 4 \
  --use_ray \
  --auto_monitor \
  --max_retries 10 \
  --checkpoint_dir /checkpoints/GRPO/verl-grpo-training
```

### PyTorchJob FSDP Training
```bash
python3 ~/.config/opencode/skills/training-job-deployer/src/deploy.py \
  --cluster_name my-cluster \
  --image_uri my-fsdp-image:latest \
  --job_name llama-fsdp \
  --num_nodes 8 \
  --use_pytorchjob \
  --model_path meta-llama/Llama-2-7b-hf \
  --auto_monitor
```

### Resume from Checkpoint
```bash
python3 ~/.config/opencode/skills/training-job-deployer/src/deploy.py \
  --cluster_name my-cluster \
  --image_uri my-image:latest \
  --use_ray \
  --auto_monitor \
  --resume_from_checkpoint /checkpoints/GRPO/my-job/global_step_220
```

## Architecture

### Sub-Skills I Coordinate

| Skill | Responsibility | When Called |
|-------|---------------|-------------|
| k8s-cluster-manager | Cluster validation | Step 1 - Always (deploy) |
| checkpoint-manager | Storage setup | Step 2 - Always (deploy) |
| ray-cluster-manager | Ray setup & teardown | Step 3 (deploy) / Step 1 (cleanup) |
| pytorchjob-manager | PyTorchJob setup | Step 3 - If --use_pytorchjob |
| training-monitor | Job monitoring | Step 5 - If --auto_monitor |
| hyperpod-manager | HyperPod features & scaling | Auto-detected (deploy) / Step 3 (cleanup --scale_down) |

### Why This Architecture?

**Benefits of modular design:**
- ✅ Each sub-skill is focused and testable (~150 lines each)
- ✅ Can use sub-skills independently for debugging
- ✅ Easy to add new frameworks (e.g., DeepSpeed, Megatron)
- ✅ Clear separation of concerns
- ✅ Parallel development possible

**Trade-offs:**
- Slightly more complex than monolithic skill
- Need to understand sub-skill dependencies
- More files to maintain

## Troubleshooting

### "Which sub-skill failed?"
I print clear step indicators:
```
[Step 1/5] Validating cluster with k8s-cluster-manager...
[Step 2/5] Setting up storage with checkpoint-manager...
[Step 3/5] Creating Ray cluster with ray-cluster-manager...
...
```

### "I want to debug a specific step"
Run the sub-skill directly:
```bash
# Just check cluster
python3 ~/.config/opencode/skills/k8s-cluster-manager/src/check_cluster.py my-cluster

# Just setup Ray
python3 ~/.config/opencode/skills/ray-cluster-manager/src/ray_manager.py create

# Just monitor
python3 ~/.config/opencode/skills/training-monitor/src/monitor.py --job_id xxx
```

### "Deployment failed, how do I recover?"
Check which step failed, then:
```bash
# If cluster check failed
python3 ~/.config/opencode/skills/k8s-cluster-manager/src/check_cluster.py my-cluster --full

# If Ray setup failed
python3 ~/.config/opencode/skills/ray-cluster-manager/src/ray_manager.py status my-job

# If monitoring needed
python3 ~/.config/opencode/skills/training-monitor/src/monitor.py --head_pod xxx
```

## Advanced Usage

### Python API
```python
from training_job_deployer.src.deploy import deploy_training_job

result = deploy_training_job({
    'cluster_name': 'my-cluster',
    'image_uri': 'my-image:latest',
    'use_ray': True,
    'num_nodes': 4,
    'auto_monitor': True,
    'max_retries': 10
})
```

### Skip Validation (Fast Deploy)
```bash
python3 ~/.config/opencode/skills/training-job-deployer/src/deploy.py \
  --cluster_name my-cluster \
  --image_uri my-image:latest \
  --skip_validation \
  --use_ray
```

### Custom Configuration
```bash
python3 ~/.config/opencode/skills/training-job-deployer/src/deploy.py \
  --cluster_name my-cluster \
  --image_uri my-image:latest \
  --use_ray \
  --num_nodes 8 \
  --gpu_per_node 2 \
  --batch_size 16 \
  --save_freq 5 \
  --storage_size 500Gi \
  --auto_monitor \
  --max_retries 20
```

## Dependencies

I require these skills to be installed:
```bash
# All sub-skills should be in:
~/.config/opencode/skills/k8s-cluster-manager/
~/.config/opencode/skills/ray-cluster-manager/
~/.config/opencode/skills/pytorchjob-manager/
~/.config/opencode/skills/checkpoint-manager/
~/.config/opencode/skills/training-monitor/
~/.config/opencode/skills/hyperpod-manager/
```

Each sub-skill is standalone and can be used independently.

## Known Issues

### ⚠️ Ray Job Submit vs kubectl exec

**Issue**: When deploying Ray training jobs, using `ray job submit` isolates the job from the Ray cluster's GPU resources. This causes training to fail with "0 GPUs available" even though the cluster has GPUs.

**Root Cause**: Ray jobs run in isolated processes that don't inherit the cluster's resource pool. This is a known limitation of `ray job submit`.

**Solution**: This skill now uses `kubectl exec` to run training directly in the head pod, which provides full access to cluster resources including GPUs.

**Verification**: After deployment, the skill automatically verifies GPU utilization:
- Checks `ray status` to confirm GPUs are allocated
- Warns if GPUs show 0% utilization after startup
- Displays Ray resource allocation for troubleshooting

**Manual Verification**:
```bash
# Check Ray resources
kubectl exec <head-pod> -- ray status

# Check GPU utilization
kubectl exec <head-pod> -- nvidia-smi

# Expected: GPUs should show >0% utilization within 2 minutes
```

### EFA Silent Fallback to TCP Sockets

**Issue**: EFA device can be present and ACTIVE on all nodes, but NCCL silently falls back to TCP sockets. This causes:
- Slower inter-node communication
- NCCL timeout errors under heavy load (ALLGATHER timeouts after 600s)
- Training failures at scale

**Root Cause**: Without `NCCL_NET=ofi` and correct `LD_LIBRARY_PATH`, NCCL doesn't load the aws-ofi-nccl plugin.

**Required Environment Variables**:
```bash
NCCL_NET=ofi
LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
FI_PROVIDER=efa
FI_EFA_USE_DEVICE_RDMA=0   # g5 instances (no GPUDirect RDMA)
FI_EFA_USE_DEVICE_RDMA=1   # p4d/p5 instances (full GPUDirect RDMA)
```

**Verification**: Check NCCL logs for:
```bash
kubectl exec <head-pod> -- grep 'NET/OFI' /tmp/ray/session_latest/logs/worker*.out
# Should see: "NET/OFI Selected provider is efa"
```

## Best Practices

1. **Always use --auto_monitor for long training** - Handles EFA failures and restarts automatically
2. **Set appropriate --max_retries** - Longer training = more retries needed
3. **Verify GPU utilization** - Check that GPUs are being used after deployment
4. **Use persistent storage** - Checkpoints survive pod restarts
5. **Check logs** - Each sub-skill has detailed logging
6. **Start with validation** - Let me check everything before deploying

## See Also

- **k8s-cluster-manager** - For cluster validation and health checks
- **ray-cluster-manager** - For Ray/KubeRay specific operations
- **pytorchjob-manager** - For PyTorchJob specific operations
- **checkpoint-manager** - For storage and checkpoint management
- **training-monitor** - For job monitoring and auto-restart
- **hyperpod-manager** - For HyperPod-specific features
