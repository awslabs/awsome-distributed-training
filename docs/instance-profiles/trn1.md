# trn1 / trn2 Instance Family — AWS Trainium

> Covers: trn1.2xlarge, trn1.32xlarge, trn1n.32xlarge, trn2.3xlarge,
> trn2.48xlarge

## Hardware Summary

| Dimension | trn1.32xlarge | trn1n.32xlarge | trn2.48xlarge | trn2.3xlarge |
|-----------|---------------|----------------|---------------|--------------|
| Accelerator | Trainium v1 | Trainium v1 | Trainium v2 | Trainium v2 |
| Accelerator Memory | 32 GB HBM per core | 32 GB HBM per core | 32 GB HBM per core | 32 GB HBM per core |
| NeuronCores | 32 (16 devices x 2) | 32 (16 devices x 2) | 64 (32 devices x 2) | 4 (2 devices x 2) |
| EFA Adapters | 8 | 16 | 16 | 1 |
| NeuronLink | Yes (intra-node) | Yes (intra-node) | Yes (intra-node) | No |
| Node CPU Memory | ~480 Gi | ~480 Gi | ~960 Gi | ~30 Gi |

## Key Characteristics

- **Neuron SDK only**: These instances use the AWS Neuron compiler and
  runtime, not CUDA. NVIDIA-targeted test cases are **not applicable**
- **Different software stack**: Uses `neuronx-distributed`, `optimum-neuron`,
  or `neuronx-nemo-megatron` instead of PyTorch FSDP / Megatron-LM
- **NeuronLink**: Intra-node device-to-device communication (analogous to
  NVLink for NVIDIA)
- **EFA**: Inter-node communication; trn1n and trn2 have more adapters
- **Compiler-driven optimization**: Memory optimization is handled primarily
  by the Neuron compiler rather than manual FSDP/offloading configuration

## Required Settings

```bash
# Neuron-specific environment variables
export NEURON_RT_NUM_CORES=32  # Adjust per instance type
export NEURON_CC_FLAGS="--model-type=transformer"

# EFA for multi-node
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1  # Trainium supports EFA direct
```

## Differences from NVIDIA Instances

Trainium instances use a fundamentally different software stack. The
parameter adjustment patterns described in the
[main compatibility guide](../instance-compatibility.md) (FSDP strategy,
NCCL settings, etc.) do not apply. Instead:

| NVIDIA Concept | Trainium Equivalent |
|----------------|-------------------|
| FSDP / DeepSpeed | `neuronx-distributed` tensor/pipeline parallelism |
| NCCL | `xla` collective communication |
| CUDA graphs | Neuron compiler graph extraction |
| `nvidia-smi` | `neuron-top`, `neuron-monitor` |
| TP / PP configuration | Set via Neuron distributed config |

## Resource Requests (Kubernetes)

```yaml
# trn1.32xlarge — 16 Neuron devices
resources:
  limits:
    aws.amazon.com/neuron: 16
    vpc.amazonaws.com/efa: 8
  requests:
    memory: "400Gi"
    cpu: "120"
```

## Tested Workloads

| Test Case | Model | Instance | Status | Notes |
|-----------|-------|----------|--------|-------|
| optimum-neuron | Various | trn1.32xl, trn1n, trn2.48xl, trn2.3xl | Tested | Multiple instance types validated |
| neuronx-distributed | Various | Various | Untested | Expected to work |
