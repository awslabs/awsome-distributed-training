#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Thin launcher for V-JEPA 2 training via srun.

This script loads a YAML config and calls app.vjepa.train.main() directly,
which reads SLURM_LOCALID, SLURM_NTASKS, and SLURM_PROCID from the
environment to configure CUDA device selection and torch.distributed.

Why not use `python -m app.main --devices cuda:0`?
    app/main.py spawns a subprocess that passes rank_and_world_size=(0, 1) to
    init_distributed(), bypassing SLURM env vars. This causes each process to
    see world_size=1 instead of the actual SLURM world size. Calling
    app.vjepa.train.main() directly avoids this issue.

Usage with srun:
    srun --ntasks-per-node=8 ... python scripts/run_train.py \
        --fname /path/to/config.yaml
"""

import argparse
import pprint

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, required=True, help="Path to YAML config file")

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.fname, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(params)

    # Import train module - this triggers CUDA_VISIBLE_DEVICES setup from SLURM_LOCALID
    from app.vjepa.train import main as train_main

    train_main(args=params, resume_preempt=False)
