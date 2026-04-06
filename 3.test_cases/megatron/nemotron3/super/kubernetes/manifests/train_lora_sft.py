"""
Nemotron 3 Super LoRA SFT — Direct training script for PyTorchJob execution.

This script runs inside the NeMo 26.02 nemotron_3_super container and performs:
  1. Download/cache the HuggingFace model (if not already on FSx)
  2. Import HF checkpoint to Megatron format (if not already converted)
  3. Run LoRA fine-tuning using the nemotronh recipe

Usage (inside container):
  torchrun --nproc-per-node=8 --nnodes=1 --node-rank=0 \
    --master-addr=$MASTER_ADDR --master-port=29500 \
    /scripts/train_lora_sft.py \
    --hf-model-id nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
    --work-dir /fsx/nchkumar/nemotron3-super \
    --max-steps 20 \
    --peft lora

Environment: nvcr.io/nvidia/nemo:26.02.nemotron_3_super (with EFA)
"""
import argparse
import os
import sys
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Nemotron 3 Super LoRA SFT")
    parser.add_argument("--hf-model-id", type=str,
                        default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8")
    parser.add_argument("--work-dir", type=str, default="/fsx/nchkumar/nemotron3-super",
                        help="Base directory on FSx for checkpoints, logs, cache")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Training steps (keep low for validation)")
    parser.add_argument("--global-batch-size", type=int, default=8,
                        help="Global batch size (reduced for validation)")
    parser.add_argument("--micro-batch-size", type=int, default=1,
                        help="Per-GPU micro batch size")
    parser.add_argument("--seq-length", type=int, default=2048,
                        help="Sequence length (reduced for validation)")
    parser.add_argument("--peft", type=str, default="lora",
                        choices=["lora", "none"],
                        help="PEFT method")
    parser.add_argument("--tp", type=int, default=2,
                        help="Tensor parallelism")
    parser.add_argument("--ep", type=int, default=4,
                        help="Expert parallelism")
    parser.add_argument("--pp", type=int, default=1,
                        help="Pipeline parallelism")
    parser.add_argument("--skip-import", action="store_true",
                        help="Skip checkpoint import (if already done)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine rank info
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Setup directories
    work_dir = args.work_dir
    cache_dir = os.path.join(work_dir, "cache")
    results_dir = os.path.join(work_dir, "results")
    nemo_home = os.path.join(work_dir, "nemo_home")

    if global_rank == 0:
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(nemo_home, exist_ok=True)
        print(f"[rank {global_rank}] Nemotron 3 Super LoRA SFT Validation")
        print(f"  Model:     {args.hf_model_id}")
        print(f"  Work dir:  {work_dir}")
        print(f"  World:     {world_size} GPUs")
        print(f"  TP={args.tp}, EP={args.ep}, PP={args.pp}")
        print(f"  Batch:     {args.global_batch_size} global, {args.micro_batch_size} micro")
        print(f"  Steps:     {args.max_steps}")
        print(f"  SeqLen:    {args.seq_length}")
        print(f"  PEFT:      {args.peft}")
        print(f"  CUDA:      {torch.cuda.device_count()} devices")
        sys.stdout.flush()

    # Set NeMo/HF cache paths
    os.environ["NEMORUN_HOME"] = nemo_home
    os.environ["NEMO_HOME"] = nemo_home
    os.environ["HF_HOME"] = os.path.join(cache_dir, "huggingface")
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "huggingface", "hub")

    # Import NeMo after setting env vars
    from nemo.collections import llm
    from nemo.lightning.pytorch.callbacks import PreemptionCallback

    # --- Step 1: Checkpoint Import (rank 0 only, others wait) ---
    peft_scheme = "lora" if args.peft == "lora" else None

    if not args.skip_import:
        if global_rank == 0:
            print(f"[rank 0] Starting HF -> Megatron checkpoint import...")
            print(f"  Source: hf://{args.hf_model_id}")
            sys.stdout.flush()

        # The import_ckpt call handles distributed coordination internally
        # It will download on rank 0 and broadcast/shard as needed
        try:
            llm.import_ckpt(
                model=llm.nemotronh.model(),
                source=f"hf://{args.hf_model_id}",
                overwrite=False,
            )
            if global_rank == 0:
                print("[rank 0] Checkpoint import completed.")
                sys.stdout.flush()
        except Exception as e:
            if global_rank == 0:
                print(f"[rank 0] Checkpoint import error: {e}")
                print("[rank 0] Proceeding to finetune (may use cached checkpoint)...")
                sys.stdout.flush()

    # --- Step 2: Configure Fine-tuning Recipe ---
    if global_rank == 0:
        print(f"[rank 0] Configuring finetune recipe...")
        sys.stdout.flush()

    num_nodes = world_size // torch.cuda.device_count()
    gpu_devices = torch.cuda.device_count()

    finetune_recipe = llm.nemotronh.finetune_recipe(
        num_nodes=num_nodes,
        name="nemotron3-super-lora-validation",
        dir=results_dir,
        peft_scheme=peft_scheme,
    )

    # Override trainer settings for validation run
    finetune_recipe.trainer.devices = gpu_devices
    finetune_recipe.trainer.num_sanity_val_steps = 0
    finetune_recipe.trainer.max_steps = args.max_steps
    finetune_recipe.trainer.val_check_interval = args.max_steps  # validate at end only
    finetune_recipe.trainer.log_every_n_steps = 1
    finetune_recipe.trainer.strategy.context_parallel_size = 1

    # Batch sizes
    finetune_recipe.data.global_batch_size = args.global_batch_size
    finetune_recipe.data.micro_batch_size = args.micro_batch_size
    finetune_recipe.data.seq_length = args.seq_length

    # LoRA-specific settings
    if args.peft == "lora":
        finetune_recipe.trainer.strategy.ddp = "megatron"

    if global_rank == 0:
        print(f"[rank 0] Recipe configured. Starting training...")
        sys.stdout.flush()

    # --- Step 3: Run training ---
    # NeMo-Run handles the training loop internally
    # For direct execution, we use the recipe's run method
    import nemo_run as run
    from nemo.collections.llm.api import finetune

    finetune(
        model=finetune_recipe.model,
        data=finetune_recipe.data,
        trainer=finetune_recipe.trainer,
        optim=finetune_recipe.optim,
        peft=finetune_recipe.peft,
        log=finetune_recipe.log,
    )

    if global_rank == 0:
        print("[rank 0] Training completed successfully!")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
