# p5en Instance Family — H200 GPUs

> Covers: p5en.48xlarge

## Hardware Summary

| Dimension | p5en.48xlarge |
|-----------|---------------|
| GPU | NVIDIA H200 |
| GPU VRAM | 141 GB HBM3e |
| GPU Count | 8 |
| GPUDirect RDMA | **Yes** |
| EFA Adapters | 32 |
| NVLink | NVSwitch (full-mesh, 900 GB/s bisection) |
| Node CPU Memory | ~700 Gi allocatable |
| Local Storage | 8 x 3.84 TB NVMe |

## Key Characteristics

- **141 GB HBM3e**: Highest VRAM capacity among current instances;
  4.8 TB/s memory bandwidth
- **NVSwitch full-mesh**: All 8 GPUs have equal bandwidth to each other;
  tensor parallelism is highly efficient
- **32 EFA adapters**: Maximum inter-node bandwidth; ideal for large-scale
  multi-node training
- **GPUDirect RDMA**: GPU memory can be read/written directly by the NIC,
  bypassing the CPU for collective operations

## Required NCCL / EFA Settings

```bash
# p5en supports GPUDirect RDMA — use defaults
export FI_EFA_USE_DEVICE_RDMA=1
# NCCL_PROTO can be left at default (LL/LL128)

# Standard EFA settings
export FI_PROVIDER=efa
export NCCL_DEBUG=WARN
```
