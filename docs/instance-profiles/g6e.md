# g6e Instance Family — L40S GPUs

> Covers: g6e.xlarge, g6e.2xlarge, g6e.4xlarge, g6e.8xlarge, g6e.12xlarge,
> g6e.16xlarge, g6e.24xlarge, g6e.48xlarge

## Hardware Summary

| Dimension | g6e.12xlarge | g6e.48xlarge |
|-----------|--------------|--------------|
| GPU | NVIDIA L40S | NVIDIA L40S |
| GPU VRAM | 48 GB GDDR6 | 48 GB GDDR6 |
| GPU Count | 4 | 8 |
| GPUDirect RDMA | **No** | **No** |
| EFA Adapters | 1 | 1 |
| NVLink | **None** (PCIe only) | **None** (PCIe only) |
| Node CPU Memory | ~168 Gi allocatable | ~768 Gi allocatable |
| Local Storage | 1 x 7.6 TB NVMe | 2 x 7.6 TB NVMe |

## Key Characteristics

- **Middle ground** between g5 (24 GB) and p4de (80 GB)
- **48 GB VRAM**: Many models that OOM on g5 (24 GB) will fit on g6e without
  full offloading — but still need offloading for >40B models
- **No NVLink**: Same as g5; GPU-to-GPU over PCIe only
- **No GPUDirect RDMA**: Same NCCL settings as g5
- **Single EFA adapter**: Same inter-node bandwidth limitations as g5
- **Good for development and testing** workloads that don't require
  multi-node high-bandwidth communication

## Required NCCL / EFA Settings

```bash
# Same as g5 — no GPUDirect RDMA
export FI_EFA_USE_DEVICE_RDMA=0
export NCCL_PROTO=simple

export FI_PROVIDER=efa
export NCCL_DEBUG=WARN
```

## Differences from g5

| Aspect | g6e.12xlarge | g5.12xlarge |
|--------|--------------|-------------|
| GPU | L40S 48 GB | A10G 24 GB |
| VRAM | 48 GB | 24 GB |
| Offloading needed | Models >30B | Models >10B |
| FSDP strategy | FSDP1 may work | Must use FSDP2 |
| enforce_eager | May not be needed | Required |
| NCCL settings | Identical | Identical |
| EFA | Identical | Identical |

## Memory Optimization Strategies

With 48 GB per GPU, optimization requirements are less aggressive than g5:

1. **FSDP1 may suffice** for models up to ~30B parameters
2. **Offloading**: May only need `ref_param_offload=True` instead of full
   offloading
3. **bf16**: Still recommended to enable explicitly if the framework defaults
   to fp32
4. **CUDA graphs**: May work for smaller models; test on a case-by-case basis
5. **TP=4**: Same as g5 (4 GPUs per node on g6e.12xlarge)

## Resource Requests (Kubernetes)

```yaml
# g6e.12xlarge — 4 GPUs, ~168 Gi allocatable
resources:
  limits:
    nvidia.com/gpu: 4
    vpc.amazonaws.com/efa: 1
  requests:
    memory: "150Gi"
    cpu: "40"
```

## Tested Workloads

| Test Case | Model | Nodes | Status | Notes |
|-----------|-------|-------|--------|-------|
| — | — | — | Untested | No test cases validated on g6e yet |

> If you validate a workload on g6e, please update this table and the
> [compatibility matrix](../instance-compatibility.md).
