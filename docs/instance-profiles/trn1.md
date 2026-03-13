# trn1 / trn1n Instance Family — AWS Trainium v1

> Covers: trn1.2xlarge, trn1.32xlarge, trn1n.32xlarge

## Hardware at a Glance

| Spec | trn1.2xlarge | trn1.32xlarge | trn1n.32xlarge |
|------|--------------|---------------|----------------|
| Accelerator | AWS Trainium v1 | AWS Trainium v1 | AWS Trainium v1 |
| Chips | 1 | 16 | 16 |
| NeuronCores (v2) | 2 | 32 | 32 |
| HBM per chip | 32 GB | 32 GB | 32 GB |
| Total accelerator memory | 32 GB | 512 GB | 512 GB |
| Memory bandwidth per chip | 820 GB/s | 820 GB/s | 820 GB/s |
| BF16 TFLOPS per chip | ~190 | ~190 | ~190 |
| FP32 TFLOPS per chip | ~47.5 | ~47.5 | ~47.5 |
| Chip interconnect | N/A (single chip) | NeuronLink v2, 2D torus (768 GB/s) | NeuronLink v2, 2D torus (768 GB/s) |
| EFA | None | 8 adapters (EFAv2) | 16 adapters (EFAv2) |
| Network bandwidth | Up to 12.5 Gbps | 800 Gbps | 1,600 Gbps |
| vCPUs | 8 (Intel Xeon 3rd Gen) | 128 (Intel Xeon 3rd Gen) | 128 (Intel Xeon 3rd Gen) |
| System memory | 32 GiB | 512 GiB | 512 GiB |
| Local NVMe | 1 x 474 GB | 4 x 1.9 TB | 4 x 1.9 TB |

## Key Characteristics

- **Neuron SDK only** — these instances use the AWS Neuron compiler and runtime,
  not CUDA. NVIDIA-targeted test cases are not applicable
- **NeuronLink v2** (trn1.32xl / trn1n.32xl) — 2D torus topology connecting
  all 16 chips at 768 GB/s aggregate; analogous to NVSwitch for NVIDIA
- **trn1n vs trn1**: trn1n doubles EFA bandwidth (1,600 vs 800 Gbps) and
  adapter count (16 vs 8); ~20% faster multi-node training. All other specs
  are identical
- **Compiler-driven optimization** — memory management and parallelism are
  handled primarily by the Neuron compiler rather than manual tuning

## Distributed Training Considerations

- **Software stack**: Uses `torch-neuronx` (XLA-based), `neuronx-distributed`
  (NxD Training), `optimum-neuron`, or `neuronx-nemo-megatron`
- **Max model size**: 100B+ with NxD Training parallelism strategies (TP + PP +
  DP) across multiple trn1n.32xl nodes
- **Choose trn1n over trn1** for multi-node training — the 2x network bandwidth
  significantly reduces gradient synchronization time
- **Compilation overhead**: First-run compilation can take significant time;
  compiled graphs are cached for subsequent runs

## Configuration

```bash
# Neuron runtime configuration
export NEURON_RT_NUM_CORES=32  # Adjust per instance type
export NEURON_CC_FLAGS="--model-type=transformer"

# EFA for multi-node (trn1.32xl / trn1n.32xl only)
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
```

## Differences from NVIDIA Instances

| NVIDIA Concept | Trainium Equivalent |
|----------------|---------------------|
| CUDA | Neuron SDK |
| NCCL | XLA collective communication |
| CUDA graphs | Neuron compiler graph extraction |
| `nvidia-smi` | `neuron-top`, `neuron-monitor` |
| NVLink / NVSwitch | NeuronLink |
| FSDP / DeepSpeed | NxD Training (tensor/pipeline/data parallelism) |
| Transformer Engine (FP8) | Neuron compiler auto-casting |
| PyTorch (native) | `torch-neuronx` (XLA) or native PyTorch (private preview) |
