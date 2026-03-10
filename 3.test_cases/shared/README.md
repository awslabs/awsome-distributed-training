# Shared Utilities

Shared scripts and utilities used across multiple test cases.

## Instance Detection (`instance_detect.sh`)

Auto-detects the EC2 instance type and resolves a matching instance profile
(`.env` file) for the current test case. Each test case has its own
`profiles/` directory with instance-specific `.env` files, and a copy of the
detection script at `profiles/_detect.sh`.

### Canonical Source

**`instance_detect.sh`** in this directory is the single source of truth.
Copies exist in each test case's `profiles/_detect.sh`. To update all copies
after editing the canonical source:

```bash
bash 3.test_cases/shared/sync_profiles.sh
```

### Detection Order

1. `INSTANCE_PROFILE` env var — explicit override (e.g., `g5-12xlarge`)
2. `INSTANCE_TYPE` env var — from env_vars (e.g., `g5.12xlarge`)
3. EC2 Instance Metadata API (IMDSv2) — works on bare metal and K8s
4. GPU name from `nvidia-smi` — fallback mapping (A10G -> g5, H100 -> p5, etc.)

### Test Cases Using Profiles

| Test Case | Profiles Directory |
|-----------|-------------------|
| [FSDP](../pytorch/FSDP/) | `pytorch/FSDP/profiles/` |
| [veRL (HyperPod EKS)](../pytorch/verl/hyperpod-eks/rlvr/) | `pytorch/verl/hyperpod-eks/rlvr/recipe/profiles/` |
| [veRL (Kubernetes)](../pytorch/verl/kubernetes/rlvr/) | `pytorch/verl/kubernetes/rlvr/recipe/profiles/` |
| [torchtitan](../pytorch/torchtitan/) | `pytorch/torchtitan/profiles/` |
| [nanoVLM](../pytorch/nanoVLM/) | `pytorch/nanoVLM/profiles/` |
| [TRL](../pytorch/trl/) | `pytorch/trl/profiles/` |
| [Megatron-LM](../megatron/megatron-lm/) | `megatron/megatron-lm/profiles/` |
| [BioNeMo](../megatron/bionemo/) | `megatron/bionemo/profiles/` |
| [Megatron-DeepSpeed](../pytorch/deepspeed/examples_megatron_deepspeed/) | `pytorch/deepspeed/examples_megatron_deepspeed/profiles/` |
