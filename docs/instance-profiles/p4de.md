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
- **Fewer EFA adapters** (4 vs 32 on p5): Lower inter-node bandwidth
- **NVSwitch**: Full-mesh GPU-to-GPU connectivity (slightly lower bandwidth
  than p5's NVSwitch generation)
- **80 GB VRAM (p4de)** / **40 GB VRAM (p4d)**
- **Higher CPU memory** (~1100 Gi): More headroom for CPU offloading than p5

## Required NCCL / EFA Settings

```bash
# p4de supports GPUDirect RDMA
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export NCCL_DEBUG=WARN
```

## Comparison with p5en

| Aspect | p4de.24xlarge | p5en.48xlarge |
|--------|---------------|---------------|
| GPU | A100 80 GB | H200 141 GB |
| EFA adapters | 4 | 32 |
| NVSwitch bandwidth | 600 GB/s | 900 GB/s |
| Node CPU memory | ~1100 Gi | ~700 Gi |
