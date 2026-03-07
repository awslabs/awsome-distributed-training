---
id: deployment-automation
status: in_progress
started: 2026-03-06
---

# Deployment Automation Scripts — Plan

## Steps

- [x] 1. Create `deploy.sh` (Phase 0: Infrastructure deployment)
- [x] 1a. Extract testable functions into `lib/deploy_helpers.sh`
- [x] 1b. Unit test `deploy.sh` with bats-core (34 tests passing)
- [ ] 2. Create `setup.sh` (Phase 1: Docker build + SSH keys + values template)
- [ ] 3. Create `install.sh` (Phase 2: Helm installs + k8s Day-2 config)
- [ ] 4. Create `destroy.sh` (Phase 3: Reverse teardown)
- [ ] 5. Update README.md to reference all scripts
- [ ] 6. Mark complete

## Completed: deploy.sh

`deploy.sh` handles infrastructure deployment via CloudFormation or Terraform
with automatic AZ resolution.

### Interface

```bash
./deploy.sh --node-type <g5|p5> --infra <cfn|tf> [OPTIONS]

Options:
  --region <region>       AWS region (default: us-west-2)
  --az-id <az-id>         AZ for instance groups + FSx (default: usw2-az2)
  --stack-name <name>     CFN stack name (default: hp-eks-slinky-stack)
  --help                  Show usage
```

### Features

- Resolves up to 5 non-opt-in AZs via `aws ec2 describe-availability-zones`
- Validates the specified `--az-id` exists in the resolved AZ list
- CFN path: `jq` substitution of `AvailabilityZoneIds`,
  `FsxAvailabilityZoneId`, `TargetAvailabilityZoneId`, and instance
  type/count in `InstanceGroupSettings1`
- TF path: copies tfvars, overrides `aws_region`, `availability_zone_id`,
  and instance type/count via `sed`, runs `terraform init/plan/apply`
- Extracts stack outputs to `env_vars.sh`

### Also completed in this phase

- Consolidated `g5/g5-params.json` and `p5/p5-params.json` into a single
  `params.json` at the project root (defaults to g5 settings; `deploy.sh`
  overrides instance type/count for p5 at deploy time)
- Consolidated `g5/g5-custom.tfvars` and `p5/p5-custom.tfvars` into a
  single `custom.tfvars` at the project root (same override pattern)
- Updated `params.json` to include all 40 parameters (23 previously missing)
  matching the reference configuration
- Default `AvailabilityZoneIds` set to `usw2-az1,usw2-az2,usw2-az3,usw2-az4`
- Bumped `KubernetesVersion` to `1.34` in both params.json and custom.tfvars
- Enabled all features: observability, logging, training operator, inference
  addon, task governance, GPU operator
- Updated README.md to reference `deploy.sh` as primary deployment method
  with manual commands preserved as collapsible fallback

## Remaining: setup.sh, install.sh, destroy.sh

These scripts will be created in subsequent iterations following the design
in `idea.md`.

## Completed: Unit Testing (bats-core)

34 unit tests covering all 5 extracted helper functions and `deploy.sh`
argument parsing. Test infrastructure includes:

- `tests/test_deploy.bats` — Test file (34 tests across 6 categories)
- `tests/helpers/setup.bash` — Common setup/teardown with library guard
- `tests/helpers/mock_aws.bash` — AWS CLI mock with canned responses
- `tests/fixtures/params.json` — Independent fixture copy
- `tests/fixtures/custom.tfvars` — Independent fixture copy
- `tests/install_bats_libs.sh` — Installs bats-assert + bats-support
- `.gitignore` updated to exclude `tests/bats/`

Bug fixed: `resolve_tf_vars` used GNU sed `0,/pattern/` syntax (first
occurrence only) which silently fails on macOS (BSD sed). Replaced with
portable awk approach.
