# AWS Parallel Computing Service Distributed Training Reference Architecture

This repository provides reference architectures and deployment templates for setting up distributed training clusters using [AWS Parallel Computing Service (PCS)](https://aws.amazon.com/pcs/). These architectures are optimized for machine learning workloads and include configurations for high-performance computing instances (P and Trn EC2 families) with shared filesystems (FSx for Lustre and OpenZFS).

## Key Features

- **Pre-configured for ML workloads**: Optimized for distributed training with Slurm scheduler
- **High-performance storage**: FSx for Lustre (shared scratch) and OpenZFS (home directories)
- **Flexible compute options**: Support for On-Demand, On-Demand Capacity Reservations (ODCR), and Capacity Blocks for ML
- **Advanced networking**: Elastic Fabric Adapter (EFA) support for multi-node training
- **Custom AMI building**: Automated DLAMI creation with PCS agent, Slurm, Enroot, and Pyxis
- **Modular deployment**: Deploy complete clusters or individual components via nested CloudFormation stacks

## Architecture

![AWS PCS diagram](./images/ml-pcs-architecture.png)

The architecture includes:
- VPC with public/private subnets
- FSx for Lustre for high-performance shared storage
- FSx for OpenZFS for home directories
- PCS cluster with Slurm scheduler (24.05, 24.11, or 25.05)
- Login node group (public subnet)
- Compute node groups (private subnet)
- Optional custom DLAMI with ML frameworks and container runtime

## Deployment Options

### Option 1: Complete Cluster (Recommended)

Deploy the complete PCS ML cluster with a single nested CloudFormation stack:

[<kbd> <br> 1-Click Deploy Complete Cluster 🚀 <br> </kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://ws-assets-prod-iad-r-iad-ed304a55c2ca1aee.s3.us-east-1.amazonaws.com/2457970d-002f-4794-9e70-3610f2df74ac/pcs-ml-cluster-deploy-all.yaml&stackName=pcs-ml-cluster&param_S3BucketName=ws-assets-prod-iad-r-iad-ed304a55c2ca1aee&param_S3KeyPrefix=2457970d-002f-4794-9e70-3610f2df74ac/)

**What gets deployed:**
- ✅ VPC with public/private subnets, NAT gateway, S3 endpoint
- ✅ FSx for Lustre (high-performance shared storage)
- ✅ FSx for OpenZFS (home directories)
- ✅ Custom DLAMI with PCS agent and Slurm (optional)
- ✅ AWS PCS cluster with Slurm scheduler
- ✅ Login node group (m6i.4xlarge)
- ✅ CPU compute node group (c6i.4xlarge)
- ⚙️ Additional on-demand GPU compute node group (optional, e.g., g5.12xlarge)
- ⚙️ Additional capacity block P5 compute node group (optional, for P5.48xlarge)

**Key Parameters:**
- `PrimarySubnetAZ`: Availability Zone for deployment (required)
- `BuildAMI`: Build custom DLAMI (`true`/`false`, default: `true`)
- `DeployOnDemandCNG`: Deploy additional on-demand GPU queue (`true`/`false`, default: `false`)
- `DeployCapacityBlockCNG`: Deploy capacity block P5 queue (`true`/`false`, default: `false`)
- `CapacityReservationId`: Capacity Reservation ID (required if deploying capacity block)

**Example deployment with GPU queue:**
```bash
aws cloudformation create-stack \
  --stack-name my-pcs-cluster \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=us-east-1a \
    ParameterKey=DeployOnDemandCNG,ParameterValue=true \
    ParameterKey=OnDemandInstanceType,ParameterValue=g5.12xlarge \
    ParameterKey=OnDemandMaxCount,ParameterValue=8 \
  --capabilities CAPABILITY_IAM
```

### Option 2: Individual Components

Deploy components separately for more control:

| Component | Description | Deploy | When to Use |
|-----------|-------------|--------|-------------|
| **Prerequisites** | VPC, subnets, security groups, FSx filesystems | [<kbd>Deploy</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/ml-cluster-prerequisites.yaml&stackName=pcs-prerequisites) | Use existing VPC or customize networking |
| **DLAMI for PCS** | Custom AMI with PCS agent, Slurm 24.11/25.05, Enroot, Pyxis | [<kbd>Deploy</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/dlami-for-pcs.yaml&stackName=pcs-dlami) | Build custom AMI with specific configurations |
| **PCS Cluster** | Main cluster with login and compute nodes | [<kbd>Deploy</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/cluster.yaml&stackName=pcs-cluster) | Deploy cluster to existing VPC/FSx |
| **Add On-Demand CNG** | Additional on-demand compute node group | [<kbd>Deploy</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/add-cng.yaml&stackName=pcs-add-cng) | Add GPU/CPU queues to existing cluster |
| **Add Capacity Block CNG** | P5 compute nodes with capacity blocks | [<kbd>Deploy</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/add-cng-cbml-p5.yaml&stackName=pcs-add-cng-cb) | Add P5 instances with capacity reservation |

### Option 3: Manual Step-by-Step

For detailed step-by-step deployment instructions, see the [AI/ML for AWS Parallel Computing Service Workshop](https://catalog.workshops.aws/ml-on-pcs/).

---

## CloudFormation Templates

### Main Templates

| Template | Purpose | Nested Stacks |
|----------|---------|---------------|
| [`pcs-ml-cluster-deploy-all.yaml`](./assets/pcs-ml-cluster-deploy-all.yaml) | All-in-one nested stack deployment | Prerequisites + DLAMI + Cluster + Optional CNGs |
| [`ml-cluster-prerequisites.yaml`](./assets/ml-cluster-prerequisites.yaml) | VPC, subnets, FSx for Lustre/OpenZFS | Standalone |
| [`dlami-for-pcs.yaml`](./assets/dlami-for-pcs.yaml) | EC2 Image Builder for custom AMI | Standalone |
| [`cluster.yaml`](./assets/cluster.yaml) | PCS cluster with login and compute nodes | Standalone |

### Add-on Templates

| Template | Purpose | Prerequisites |
|----------|---------|---------------|
| [`add-cng.yaml`](./assets/add-cng.yaml) | Add on-demand compute node group (any instance type) | Existing PCS cluster |
| [`add-cng-cbml-p5.yaml`](./assets/add-cng-cbml-p5.yaml) | Add P5 compute nodes with Capacity Blocks for ML | Existing PCS cluster + Capacity Reservation |

---

## Supported Compute Options

### 1. On-Demand Instances
Standard on-demand pricing with auto-scaling support. Suitable for:
- Development and testing
- Workloads with unpredictable demand
- Short-duration training jobs

**Recommended instance types:**
- **CPU**: `c6i.32xlarge`, `c7i.48xlarge`
- **GPU**: `g5.12xlarge`, `g5.48xlarge`, `p4d.24xlarge`
- **Trainium**: `trn1.32xlarge`, `trn1n.32xlarge`

### 2. On-Demand Capacity Reservations (ODCR)
Reserved capacity with on-demand flexibility:
- Guaranteed capacity in specific AZ
- No long-term commitment
- Pay on-demand rates when using reserved capacity

### 3. Capacity Blocks for ML
Time-bound GPU capacity reservations:
- Reserved for 1-14 days
- Up to 64 P5.48xlarge instances
- Ideal for scheduled large-scale training
- Requires advance purchase

**Supported P5 instances:**
- `p5.48xlarge`: 8x NVIDIA H100 GPUs (32 EFA interfaces)
- `p5e.48xlarge`: 8x NVIDIA H200 GPUs (32 EFA interfaces)  
- `p5en.48xlarge`: 8x NVIDIA H200 GPUs with NVSwitch (16 EFA interfaces)

---

## Custom DLAMI Components

The custom DLAMI built by `dlami-for-pcs.yaml` includes:

| Component | Version | Purpose |
|-----------|---------|---------|
| **Base Image** | DLAMI Base GPU (Ubuntu 24.04 / AL2023) | Pre-installed NVIDIA drivers and CUDA |
| **AWS PCS Agent** | Latest | Node lifecycle management |
| **Slurm** | 24.11 + 25.05 | Workload scheduler (both versions installed) |
| **Enroot** | 3.5.0 | Unprivileged container runtime |
| **Pyxis** | 0.20.0 | Slurm plugin for container jobs |
| **EFS Utils** | Latest | Mount EFS filesystems |
| **CloudWatch Agent** | Latest | Metrics and log collection |
| **SSM Agent** | Latest | Remote management |

**Slurm PATH configuration:**
- Slurm 24.11 is first in PATH for maximum compatibility
- Both versions available: `/opt/aws/pcs/scheduler/slurm-24.11` and `/opt/aws/pcs/scheduler/slurm-25.05`

---

## Usage Examples

### Example 1: Basic CPU Cluster

```bash
aws cloudformation create-stack \
  --stack-name cpu-training-cluster \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=us-east-1a \
    ParameterKey=ComputeNodeInstanceType,ParameterValue=c7i.48xlarge \
  --capabilities CAPABILITY_IAM
```

### Example 2: GPU Cluster with G5 Instances

```bash
aws cloudformation create-stack \
  --stack-name gpu-training-cluster \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=us-east-1a \
    ParameterKey=DeployOnDemandCNG,ParameterValue=true \
    ParameterKey=OnDemandCngName,ParameterValue=gpu-g5 \
    ParameterKey=OnDemandQueueName,ParameterValue=gpu \
    ParameterKey=OnDemandInstanceType,ParameterValue=g5.48xlarge \
    ParameterKey=OnDemandMaxCount,ParameterValue=16 \
  --capabilities CAPABILITY_IAM
```

### Example 3: P5 Cluster with Capacity Block

```bash
# First, purchase a capacity block and get the reservation ID
CAPACITY_RESERVATION_ID="cr-0a1b2c3d4e5f6g7h8"

aws cloudformation create-stack \
  --stack-name p5-training-cluster \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=us-east-1a \
    ParameterKey=DeployCapacityBlockCNG,ParameterValue=true \
    ParameterKey=CapacityReservationId,ParameterValue=${CAPACITY_RESERVATION_ID} \
    ParameterKey=CapacityBlockInstanceType,ParameterValue=p5.48xlarge \
    ParameterKey=NetworkInterfaceCount,ParameterValue=32 \
    ParameterKey=CapacityBlockMaxCount,ParameterValue=32 \
  --capabilities CAPABILITY_IAM
```

### Example 4: Multi-Queue Cluster (CPU + GPU + P5)

```bash
aws cloudformation create-stack \
  --stack-name multi-queue-cluster \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=us-east-1a \
    ParameterKey=ComputeNodeInstanceType,ParameterValue=c7i.48xlarge \
    ParameterKey=DeployOnDemandCNG,ParameterValue=true \
    ParameterKey=OnDemandCngName,ParameterValue=gpu-g5 \
    ParameterKey=OnDemandInstanceType,ParameterValue=g5.48xlarge \
    ParameterKey=DeployCapacityBlockCNG,ParameterValue=true \
    ParameterKey=CapacityReservationId,ParameterValue=cr-xxxxx \
    ParameterKey=CapacityBlockCngName,ParameterValue=gpu-p5 \
    ParameterKey=CapacityBlockInstanceType,ParameterValue=p5.48xlarge \
  --capabilities CAPABILITY_IAM
```

---

## Accessing the Cluster

After deployment completes, access your cluster via:

### 1. AWS Systems Manager Session Manager (Recommended)

Click the `Ec2ConsoleUrl` output link to access login nodes directly in the browser.

### 2. SSH (if configured)

```bash
# Get the login node public IP from EC2 console
ssh -i your-key.pem ubuntu@<login-node-public-ip>
```

### 3. PCS Console

Click the `PcsConsoleUrl` output link to view cluster status and metrics.

---

## Monitoring and Operations

### View Cluster Status

```bash
# Connect to login node, then:
sinfo                    # View partition and node status
squeue                   # View job queue
sacct                    # View job accounting (if enabled)
```

### Submit Jobs

```bash
# CPU job
sbatch --partition=cpu1 --nodes=4 --ntasks-per-node=96 my_job.sh

# GPU job (G5)
sbatch --partition=gpu --nodes=4 --gres=gpu:4 my_gpu_job.sh

# P5 job with capacity block
sbatch --partition=p5 --nodes=8 --gres=gpu:8 my_p5_job.sh
```

### Container Jobs (Enroot + Pyxis)

```bash
# Run PyTorch container
sbatch --partition=gpu \
  --nodes=4 \
  --container-image=nvcr.io/nvidia/pytorch:24.01-py3 \
  --container-mounts=/shared:/shared \
  my_container_job.sh
```

---

## Cost Optimization

1. **Use dynamic scaling**: Set `MinCount=0` for compute node groups
2. **Right-size instances**: Choose appropriate instance types for your workload
3. **FSx for Lustre**: Use compression (`LZ4`) to reduce storage costs
4. **AMI building**: Set `BuildSchedule=Manual` to avoid unnecessary builds
5. **Capacity Blocks**: Purchase in advance for predictable large-scale training

---

## Cleanup

To delete the entire cluster:

```bash
aws cloudformation delete-stack --stack-name pcs-ml-cluster
```

**Note**: Nested stacks will be deleted automatically. Manual backups of data in FSx filesystems are recommended before deletion.

---

## Additional Resources

- [AWS Parallel Computing Service Documentation](https://docs.aws.amazon.com/pcs/)
- [AI/ML for AWS PCS Workshop](https://catalog.workshops.aws/ml-on-pcs/)
- [Slurm Documentation](https://slurm.schedmd.com/documentation.html)
- [Enroot Documentation](https://github.com/NVIDIA/enroot)
- [Pyxis Documentation](https://github.com/NVIDIA/pyxis)
- [Capacity Blocks for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html)
