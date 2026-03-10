# p5en Instance Family — H200 GPUs

> Covers: p5en.48xlarge

## Hardware Summary

| Dimension | p5en.48xlarge |
|-----------|---------------|
| GPU | NVIDIA H200 |
| GPU VRAM | 80 GB HBM3e |
| GPU Count | 8 |
| GPUDirect RDMA | **Yes** |
| EFA Adapters | 32 |
| NVLink | NVSwitch (full-mesh, 900 GB/s bisection) |
| Node CPU Memory | ~700 Gi allocatable |
| Local Storage | 8 x 3.84 TB NVMe |

## Key Characteristics

- **Current primary target** for most test cases in this repository
- **NVSwitch full-mesh**: All 8 GPUs have equal bandwidth to each other;
  tensor parallelism is highly efficient
- **32 EFA adapters**: Maximum inter-node bandwidth; ideal for large-scale
  multi-node training
- **GPUDirect RDMA**: GPU memory can be read/written directly by the NIC,
  bypassing the CPU for collective operations
- **80 GB HBM3e**: Most models up to 70B fit without offloading when using
  appropriate parallelism

## Required NCCL / EFA Settings

```bash
# p5en supports GPUDirect RDMA — use defaults
export FI_EFA_USE_DEVICE_RDMA=1
# NCCL_PROTO can be left at default (LL/LL128)

# Standard EFA settings
export FI_PROVIDER=efa
export NCCL_DEBUG=WARN
```

## Memory Optimization Strategies

With 80 GB per GPU, offloading is typically not needed:

1. **FSDP1 is sufficient** for most workloads
2. **Offloading optional**: Only needed for very large models (>100B) or
   when running with small TP degree
3. **`ref_param_offload=True`**: Common optimization — offload reference
   model to CPU since it's only used for KL divergence computation
4. **TP=2**: Typical for 8B-70B models; NVSwitch makes TP very efficient
5. **Standard save_freq**: Disk space is ample with 8 x 3.84 TB NVMe

## Resource Requests (Kubernetes)

```yaml
# p5en.48xlarge — 8 GPUs, ~700 Gi allocatable
resources:
  limits:
    nvidia.com/gpu: 8
    vpc.amazonaws.com/efa: 32
  requests:
    memory: "600Gi"
    cpu: "180"
```

## Tested Workloads

| Test Case | Model | Nodes | Status | Notes |
|-----------|-------|-------|--------|-------|
| veRL GRPO | Qwen3-8B | 4 | Tested | FSDP1, TP=2, ref_offload only |
| FSDP | Llama 2/3, Mixtral | Various | Tested | Primary CI target |
| NeMo 2.0 | Various | Various | Tested | See NeMo test case |
| distillation | Various | Various | Tested | P5en supported |
