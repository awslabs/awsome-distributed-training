# p4de Instance Family — A100 GPUs

> Covers: p4d.24xlarge, p4de.24xlarge

## Hardware Summary

| Dimension | p4de.24xlarge | p4d.24xlarge |
|-----------|---------------|--------------|
| GPU | NVIDIA A100 | NVIDIA A100 |
| GPU VRAM | 80 GB HBM2e | 40 GB HBM2e |
| GPU Count | 8 | 8 |
| GPUDirect RDMA | **Yes** | **Yes** |
| EFA Adapters | 4 | 4 |
| NVLink | NVSwitch (600 GB/s bisection) | NVSwitch (600 GB/s bisection) |
| Node CPU Memory | ~1100 Gi allocatable | ~1100 Gi allocatable |
| Local Storage | 8 x 1 TB NVMe | 8 x 1 TB NVMe |

## Key Characteristics

- **GPUDirect RDMA supported**: Same NCCL settings as p5/p5en
- **Fewer EFA adapters** (4 vs 32): Inter-node bandwidth is lower than p5;
  may need to reduce batch sizes for large multi-node runs
- **NVSwitch**: Full-mesh GPU-to-GPU connectivity (slightly lower bandwidth
  than p5's NVSwitch generation)
- **80 GB VRAM (p4de)**: Same fitting characteristics as p5 for most models
- **40 GB VRAM (p4d)**: May require offloading for models >30B; intermediate
  between g5 (24 GB) and p4de (80 GB)
- **Higher CPU memory** (~1100 Gi): More headroom for CPU offloading than p5

## Required NCCL / EFA Settings

```bash
# p4de supports GPUDirect RDMA
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export NCCL_DEBUG=WARN
```

## Differences from p5en

| Aspect | p4de.24xlarge | p5en.48xlarge |
|--------|---------------|---------------|
| GPU | A100 80 GB | H200 80 GB |
| EFA adapters | 4 | 32 |
| NVSwitch bandwidth | 600 GB/s | 900 GB/s |
| Node CPU memory | ~1100 Gi | ~700 Gi |
| Training parameters | Usually identical | Baseline |

Most p5en profiles work on p4de without modification. For large multi-node
runs, the lower EFA count (4 vs 32) may become a bottleneck — consider:
- Reducing gradient accumulation batch size
- Using gradient compression
- Overlapping communication with computation

## Resource Requests (Kubernetes)

```yaml
# p4de.24xlarge — 8 GPUs, ~1100 Gi allocatable
resources:
  limits:
    nvidia.com/gpu: 8
    vpc.amazonaws.com/efa: 4
  requests:
    memory: "900Gi"
    cpu: "90"
```

## Tested Workloads

| Test Case | Model | Nodes | Status | Notes |
|-----------|-------|-------|--------|-------|
| FSDP | Llama 2/3 | Various | Tested | See FSDP README |
| NeMo 1.0 | Various | Various | Tested | Primary target for NeMo 1.0 |
| BioNeMo | Various | Various | Tested | See BioNeMo README |
| distillation | Various | Various | Tested | Explicitly listed |
| Stable Diffusion | SD models | Various | Tested | Performance comparison available |
