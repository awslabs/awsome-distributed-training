# Instance Compatibility Guide

This guide helps you choose the right EC2 instance type for each test case
in this repository and understand the parameter changes required when moving
between instance families.

Most test cases are developed and tested on **p5en.48xlarge** (8 x H200 141GB).
Running them on smaller or differently-configured instances (g5, p4de, g6e)
requires adjustments to FSDP strategy, offloading, tensor parallelism, NCCL
flags, and checkpoint frequency. This document captures those differences
systematically.

## Quick Reference: Instance Hardware Profiles

| Instance | GPU | VRAM | GPUs | GPUDirect RDMA | EFA | NVLink | Node RAM | Notes |
|----------|-----|------|------|----------------|-----|--------|----------|-------|
| [p5en.48xlarge](instance-profiles/p5en.md) | H200 | 141 GB | 8 | Yes | 32 | NVSwitch | ~700 Gi | Current primary target |
| [p5.48xlarge](instance-profiles/p5.md) | H100 | 80 GB | 8 | Yes | 32 | NVSwitch | ~700 Gi | Same profile as p5en for most workloads |
| [p4de.24xlarge](instance-profiles/p4de.md) | A100 | 80 GB | 8 | Yes | 4 | NVSwitch | ~1100 Gi | Fewer EFA adapters than p5 |
| [g5.12xlarge](instance-profiles/g5.md) | A10G | 24 GB | 4 | No | 1 | None | ~168 Gi | Requires aggressive offloading for >10B models |
| [g6e.12xlarge](instance-profiles/g6e.md) | L40S | 48 GB | 4 | No | 1 | None | ~168 Gi | Middle ground between g5 and p4de |
| [trn1.32xlarge](instance-profiles/trn1.md) | Trainium v1 | 32 GB HBM | 16 | Yes | 8 | NeuronLink | ~480 Gi | Neuron SDK only |

> See [docs/instance-profiles/](instance-profiles/) for detailed per-instance
> specifications, NCCL settings, and tuning recommendations.

## The 6 Dimensions That Differ Across Instances

When porting a workload between instance types, these are the six hardware
dimensions that drive parameter changes:

| # | Dimension | Why It Matters | Example Impact |
|---|-----------|---------------|----------------|
| 1 | **GPU VRAM** | Determines FSDP strategy, offloading needs, TP degree, batch sizes | A10G 24 GB needs FSDP2 + full offload; H200 141 GB does not |
| 2 | **GPUDirect RDMA** | Controls NCCL transport protocol and EFA device flags | g5: `RDMA=0, PROTO=simple`; p5: `RDMA=1, PROTO=default` |
| 3 | **EFA Count** | Inter-node bandwidth; affects multi-node scaling efficiency | g5: 1 EFA adapter; p5en: 32 EFA adapters |
| 4 | **NVLink Topology** | Intra-node GPU-to-GPU bandwidth; affects TP efficiency | g5: no NVLink (PCIe only); p5: NVSwitch full-mesh |
| 5 | **Node CPU Memory** | Feasibility of CPU offloading for optimizer states and params | g5: ~168 Gi allocatable; p5en: ~700 Gi |
| 6 | **Storage Size** | Checkpoint frequency limits; large models produce huge checkpoints | 117 GB/checkpoint for 20B model; save_freq=1 fills 1.2 TB in 9 steps |

## Test Case Compatibility Matrix

The table below shows which instance types have been tested with each test case.
Status key: **Tested** = validated end-to-end, **—** = not yet validated,
**N/A** = not applicable (e.g., Neuron test cases on NVIDIA instances).

### PyTorch Test Cases

| Test Case | p5en.48xl (H200) | p5.48xl (H100) | p4de.24xl (A100) | g5.12xl (A10G) | g6e.12xl (L40S) | trn1/trn2 |
|-----------|:-:|:-:|:-:|:-:|:-:|:-:|
| [FSDP](../3.test_cases/pytorch/FSDP/) | Tested | Tested | Tested | Tested | — | N/A |
| [veRL GRPO](../3.test_cases/pytorch/verl/) | Tested | — | — | Tested | — | N/A |
| [DDP](../3.test_cases/pytorch/ddp/) | — | — | — | — | — | N/A |
| [DeepSpeed](../3.test_cases/pytorch/deepspeed/) | — | — | — | — | — | N/A |
| [torchtitan](../3.test_cases/pytorch/torchtitan/) | — | — | — | — | — | N/A |
| [TRL](../3.test_cases/pytorch/trl/) | — | — | — | — | — | N/A |
| [distillation](../3.test_cases/pytorch/distillation/) | Tested | — | Tested | — | — | N/A |
| [nanoVLM](../3.test_cases/pytorch/nanoVLM/) | — | — | — | Tested | — | N/A |
| [MosaicML Composer](../3.test_cases/pytorch/mosaicml-composer/) | — | — | — | — | — | N/A |
| [Picotron](../3.test_cases/pytorch/picotron/) | — | — | — | — | — | N/A |

### Megatron Test Cases

| Test Case | p5en.48xl (H200) | p5.48xl (H100) | p4de.24xl (A100) | g5.12xl (A10G) | g6e.12xl (L40S) | trn1/trn2 |
|-----------|:-:|:-:|:-:|:-:|:-:|:-:|
| [NeMo 2.0](../3.test_cases/megatron/nemo/) | Tested | Tested | — | — | — | N/A |
| [NeMo 1.0](../3.test_cases/megatron/nemo1.0/) | — | — | Tested | — | — | N/A |
| [Megatron-LM](../3.test_cases/megatron/megatron-lm/) | — | — | — | — | — | N/A |
| [BioNeMo](../3.test_cases/megatron/bionemo/) | — | — | Tested | — | — | N/A |

### Neuron Test Cases

| Test Case | trn1.32xl | trn1n.32xl | trn2.48xl | trn2.3xl | NVIDIA |
|-----------|:-:|:-:|:-:|:-:|:-:|
| [optimum-neuron](../3.test_cases/pytorch/optimum-neuron/) | Tested | Tested | Tested | Tested | N/A |
| [neuronx-distributed](../3.test_cases/pytorch/neuronx-distributed/) | — | — | — | — | N/A |

### Other Test Cases

| Test Case | p5en.48xl (H200) | p5.48xl (H100) | p4de.24xl (A100) | g5.12xl (A10G) |
|-----------|:-:|:-:|:-:|:-:|
| [JAX/Paxml](../3.test_cases/jax/) | — | — | — | — |
| [ESM2](../3.test_cases/23.SMHP-esm2/) | — | — | — | Tested |

## Common Parameter Adjustments by Instance Type

This section summarizes the most frequently needed changes when moving from
the default p5en configuration to other instance types.

### Moving to g5.12xlarge (A10G 24 GB)

The most impactful change. Requires aggressive memory optimization:

| Parameter | p5en Value | g5 Value | Rationale |
|-----------|-----------|----------|-----------|
| FSDP strategy | `fsdp` (FSDP1) | `fsdp2` | FSDP1 explicitly disables CPUOffload for actor role |
| `offload_policy` | not set | `True` | FSDP2-specific flag; enables proper CPU offloading |
| `model_dtype` | default (fp32) | `bf16` | veRL defaults to fp32; 24 GB requires explicit bf16 |
| `gpu_memory_utilization` | 0.6 | 0.6 | Fraction of TOTAL GPU; 0.3 x 23 GB = 6.9 GB < model shard |
| `enforce_eager` | `False` | `True` | CUDA graphs need extra workspace; OOM on 24 GB |
| `tensor_parallel_size` | 2 | 4 | Shard model across all 4 GPUs per node |
| `param_offload` | `False` | `True` | Offload params to CPU to fit in 24 GB |
| `optimizer_offload` | `False` | `True` | Offload optimizer states to CPU |
| `NCCL_PROTO` | default | `simple` | Required when `FI_EFA_USE_DEVICE_RDMA=0` |
| `FI_EFA_USE_DEVICE_RDMA` | `1` | `0` | g5 does not support GPUDirect RDMA |
| `save_freq` | 1 | 20+ | 117 GB/ckpt for 20B model; save_freq=1 fills disk fast |
| `WORKER_MEMORY` | 200 Gi+ | 150 Gi | g5.12xl allocatable is ~168 Gi, not 200 Gi |
| `nnodes` | node count | worker count only | Head pod without GPUs causes NCCL hang |

### Moving to p4de.24xlarge (A100 80 GB)

Moderate changes. Same VRAM as p5 but fewer EFA adapters:

| Parameter | p5en Value | p4de Value | Rationale |
|-----------|-----------|-----------|-----------|
| EFA adapter count | 32 | 4 | Lower inter-node bandwidth |
| `FI_EFA_USE_DEVICE_RDMA` | `1` | `1` | p4de supports GPUDirect RDMA |
| `NCCL_PROTO` | default | default | RDMA available |
| Batch sizes | as-is | may need reduction | Less inter-node bandwidth for gradient sync |

### Moving to g6e.12xlarge (L40S 48 GB)

Moderate changes. More VRAM than g5 but still no RDMA:

| Parameter | p5en Value | g6e Value | Rationale |
|-----------|-----------|----------|-----------|
| FSDP strategy | `fsdp` | `fsdp` or `fsdp2` | 48 GB may fit without offloading for models <30B |
| `NCCL_PROTO` | default | `simple` | No GPUDirect RDMA on g6e |
| `FI_EFA_USE_DEVICE_RDMA` | `1` | `0` | g6e does not support GPUDirect RDMA |
| `tensor_parallel_size` | 2 | 4 | 4 GPUs per node |
| `gpu_memory_utilization` | 0.6 | 0.6-0.7 | More headroom than g5 |

## Lessons Learned: veRL GRPO on g5 (11 OOM iterations)

The Instance Compatibility Framework was motivated by the experience of
porting veRL GRPO training from p5en.48xlarge to g5.12xlarge. It required
11 iterations to resolve cascading OOM failures. Each failure mapped to a
parameter that differs by instance type.

Key findings:

1. **FSDP2 vs FSDP1**: FSDP1 explicitly disables `CPUOffload` for the actor
   role in veRL. On 24 GB GPUs, this is fatal for models >10B params.

2. **offload_policy=True**: FSDP2-specific flag. Without it, the actor model
   stays on GPU even when FSDP2 is selected.

3. **model_dtype=bf16**: veRL defaults the actor to fp32. On 80 GB GPUs this
   wastes space; on 24 GB GPUs it causes instant OOM.

4. **gpu_memory_utilization**: Fraction of TOTAL GPU memory, not just KV
   cache. `0.3 x 23 GB = 6.9 GB` which is less than a 10 GB model shard.

5. **enforce_eager=True**: CUDA graphs require extra workspace memory that
   pushes 24 GB GPUs into OOM.

6. **NCCL_PROTO=simple**: Required when `FI_EFA_USE_DEVICE_RDMA=0` (g5, g6e).
   Without it, NCCL hangs on collective operations.

7. **Checkpoint size**: Full FSDP state for a 20B model across 12 GPUs
   produces ~117 GB per checkpoint. `save_freq=1` on 1.2 TB FSx fills the
   disk in 9 training steps.

8. **nnodes must exclude the non-GPU head pod**: In Ray on K8s, the head pod
   often has no GPUs. Including it in the `nnodes` count causes NCCL hangs.

9. **WORKER_MEMORY**: g5.12xlarge allocatable memory is ~168 Gi, not 200 Gi.
   Requesting 200 Gi causes pod scheduling failures.

10. **expandable_segments incompatible with vLLM**: Setting
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` causes
    `CuMemAllocator` assertion failures in vLLM.

## Contributing

When you validate a test case on a new instance type:

1. Add a row to the test case's "Tested Configurations" table in its README
2. Update the [compatibility matrix](#test-case-compatibility-matrix) above
3. If you needed new parameters, document them in the per-test-case README
4. Consider adding an [instance profile](instance-profiles/) if the instance
   family is not yet documented
