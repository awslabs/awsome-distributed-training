#!/usr/bin/env python3
"""
Training Job Deployer - Orchestrator Skill
Coordinates sub-skills to deploy distributed training jobs on EKS.
"""

import argparse
import sys
import os
import json
import subprocess
import time
from typing import Dict, Optional, Tuple
from datetime import datetime

# Add sub-skills to path
SKILLS_BASE = os.path.expanduser('~/.config/opencode/skills')
sys.path.insert(0, os.path.join(SKILLS_BASE, 'k8s_cluster_manager', 'src'))
sys.path.insert(0, os.path.join(SKILLS_BASE, 'ray-cluster-manager', 'src'))
sys.path.insert(0, os.path.join(SKILLS_BASE, 'pytorchjob-manager', 'src'))
sys.path.insert(0, os.path.join(SKILLS_BASE, 'checkpoint-manager', 'src'))
sys.path.insert(0, os.path.join(SKILLS_BASE, 'training-monitor', 'src'))
sys.path.insert(0, os.path.join(SKILLS_BASE, 'hyperpod-manager', 'src'))

# Import sub-skills
try:
    from check_cluster import check_cluster_health, check_gpu_availability, check_efa_availability
    from ray_manager import check_kuberay_installed, install_kuberay, create_raycluster, generate_raycluster_yaml, delete_raycluster
    from pytorchjob_manager import check_pytorchjob_crd, generate_pytorchjob_yaml, deploy_pytorchjob
    from checkpoint_manager import create_pvc, find_latest_checkpoint, find_latest_checkpoint_on_pod
    from monitor import auto_restart, get_training_health, print_training_health
    from hyperpod_manager import is_hyperpod_cluster, get_hyperpod_nodes, describe_cluster as describe_hyperpod_cluster, scale_instance_group
except ImportError as e:
    print(f"Error: Missing sub-skill. {e}")
    print("Please ensure all sub-skills are installed:")
    print("  - k8s-cluster-manager")
    print("  - ray-cluster-manager")
    print("  - pytorchjob-manager")
    print("  - checkpoint-manager")
    print("  - training-monitor")
    print("  - hyperpod-manager")
    sys.exit(1)


class OrchestratorLogger:
    """Simple logger for orchestrator."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
    def log(self, message: str, level: str = "INFO"):
        """Log with timestamp."""
        if not self.verbose and level == "DEBUG":
            return
            
        colors = {
            "INFO": "\033[92m",    # Green
            "WARN": "\033[93m",    # Yellow
            "ERROR": "\033[91m",   # Red
            "STEP": "\033[94m",    # Blue
            "SUCCESS": "\033[92m", # Green
            "RESET": "\033[0m"
        }
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = colors.get(level, colors["INFO"])
        reset = colors["RESET"]
        
        print(f"{color}[{level}]{reset} {timestamp} - {message}")
        
    def step(self, step_num: int, total: int, message: str):
        """Log a workflow step."""
        self.log(f"[{step_num}/{total}] {message}", "STEP")


class TrainingJobDeployer:
    """Orchestrates deployment of training jobs using sub-skills."""
    
    def __init__(self, args):
        self.args = args
        self.logger = OrchestratorLogger(verbose=args.verbose)
        self.head_pod = None
        self.job_id = None
        
    def run(self) -> int:
        """Execute the deployment workflow."""
        try:
            # If cleanup mode, run cleanup only
            if getattr(self.args, 'cleanup', False):
                return self._run_cleanup()
            
            self.logger.log("=" * 70)
            self.logger.log("Training Job Deployer - Starting Deployment")
            self.logger.log("=" * 70)
            
            # Step 1: Validate cluster
            if not self.args.skip_validation:
                if not self._step1_validate_cluster():
                    return 1
            else:
                self.logger.log("Skipping cluster validation (--skip_validation)", "WARN")
            
            # Step 2: Setup storage
            if not self._step2_setup_storage():
                return 1
            
            # Step 3: Setup framework and deploy
            if self.args.use_ray:
                if not self._step3_deploy_ray():
                    return 1
            elif self.args.use_pytorchjob:
                if not self._step3_deploy_pytorchjob():
                    return 1
            else:
                # Default to Ray
                self.logger.log("No framework specified, defaulting to Ray")
                if not self._step3_deploy_ray():
                    return 1
            
            # Step 4: Start monitoring (if requested)
            if self.args.auto_monitor:
                if not self._step4_start_monitoring():
                    return 1
            else:
                self.logger.log("Auto-monitoring disabled. Job deployed but not monitored.")
                self.logger.log(f"To monitor manually: kubectl exec {self.head_pod} -- ray job status {self.job_id}")
            
            self.logger.log("=" * 70)
            self.logger.log("Deployment completed successfully!")
            self.logger.log("=" * 70)
            return 0
            
        except KeyboardInterrupt:
            self.logger.log("Deployment interrupted by user", "WARN")
            return 130
        except Exception as e:
            self.logger.log(f"Deployment failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return 1
    
    def _step1_validate_cluster(self) -> bool:
        """Step 1: Validate cluster health and resources."""
        self.logger.step(1, 4, "Validating cluster with k8s-cluster-manager...")
        
        # Check cluster health
        self.logger.log("Checking cluster health...")
        status = check_cluster_health(self.args.cluster_name, self.args.region)
        
        if not status.get('healthy', False):
            self.logger.log(f"Cluster health check failed: {status.get('error', 'Unknown error')}", "ERROR")
            return False
        
        self.logger.log(f"✓ Cluster is healthy ({status.get('node_count', 0)} nodes)")
        
        # Check GPU availability
        self.logger.log("Checking GPU availability...")
        gpu_info = check_gpu_availability()
        
        if not gpu_info.get('available', False):
            self.logger.log("No GPUs available on cluster!", "ERROR")
            return False
        
        total_gpus = sum(node.get('gpu_capacity', 0) for node in gpu_info.get('nodes', []))
        self.logger.log(f"✓ Found {total_gpus} GPUs across {len(gpu_info.get('nodes', []))} nodes")
        
        # Check EFA if requested
        if self.args.use_efia:
            self.logger.log("Checking EFA availability...")
            efa_info = check_efa_availability()
            
            if not efa_info.get('available', False):
                self.logger.log("EFA not available, continuing without EFA...", "WARN")
            else:
                self.logger.log(f"✓ EFA is available on {len(efa_info.get('nodes', []))} nodes")
        
        # Check if HyperPod
        try:
            if is_hyperpod_cluster():
                self.logger.log("✓ Running on HyperPod cluster")
        except:
            pass
        
        return True
    
    def _step2_setup_storage(self) -> bool:
        """Step 2: Setup persistent storage for checkpoints."""
        self.logger.step(2, 4, "Setting up storage with checkpoint-manager...")
        
        # Create PVC if it doesn't exist
        pvc_name = f"{self.args.job_name}-checkpoints"
        
        self.logger.log(f"Creating PVC '{pvc_name}' ({self.args.storage_size})...")
        
        try:
            success = create_pvc(
                name=pvc_name,
                storage_size=self.args.storage_size,
                storage_class=self.args.storage_class,
                namespace=self.args.namespace
            )
            
            if success:
                self.logger.log(f"✓ PVC '{pvc_name}' is ready")
            else:
                self.logger.log(f"PVC creation may have failed, continuing anyway...", "WARN")
                
        except Exception as e:
            self.logger.log(f"PVC creation error: {e}", "WARN")
            self.logger.log("Continuing without PVC (checkpoints may not persist)...", "WARN")
        
        return True
    
    def _step3_deploy_ray(self) -> bool:
        """Step 3: Deploy using Ray/KubeRay."""
        self.logger.step(3, 4, "Deploying with Ray (ray-cluster-manager)...")
        
        # Check if KubeRay is installed
        self.logger.log("Checking KubeRay installation...")
        if not check_kuberay_installed():
            self.logger.log("KubeRay not found, installing...")
            if not install_kuberay(self.args.cluster_name, self.args.region):
                self.logger.log("Failed to install KubeRay", "ERROR")
                return False
            self.logger.log("✓ KubeRay installed")
        else:
            self.logger.log("✓ KubeRay is already installed")
        
        # Check for existing checkpoint
        checkpoint_path = None
        if not self.args.skip_resume:
            try:
                checkpoint_path, step = find_latest_checkpoint(self.args.checkpoint_dir)
                if checkpoint_path:
                    self.logger.log(f"Found checkpoint at step {step}: {checkpoint_path}")
            except:
                pass
        
        # Create RayCluster
        self.logger.log(f"Creating RayCluster '{self.args.job_name}'...")
        
        config = {
            'job_name': self.args.job_name,
            'image_uri': self.args.image_uri,
            'num_nodes': self.args.num_nodes,
            'use_efia': self.args.use_efia,
            'checkpoint_pvc': f"{self.args.job_name}-checkpoints"
        }
        
        try:
            cluster_created = create_raycluster(config)
            if not cluster_created:
                self.logger.log("Failed to create RayCluster", "ERROR")
                return False
        except Exception as e:
            self.logger.log(f"Error creating RayCluster: {e}", "ERROR")
            return False
        
        self.logger.log("✓ RayCluster created")
        
        # Get head pod name
        self.head_pod = f"{self.args.job_name}-head-0"
        
        # Wait for cluster to be ready
        self.logger.log("Waiting for Ray cluster to be ready...")
        time.sleep(30)  # Give it time to start
        
        # Submit job
        self.logger.log("Submitting training job...")
        
        resume_args = ""
        if checkpoint_path and not self.args.skip_resume:
            resume_args = f'trainer.resume_mode=auto trainer.resume_from_path="{checkpoint_path}"'
            self.logger.log(f"Will resume from: {checkpoint_path}")
        
        # Build training command - run directly in pod (NOT as Ray job)
        # IMPORTANT: Using kubectl exec instead of ray job submit to ensure GPU access
        self.logger.log("Starting training directly in head pod...")
        
        cmd = f"""kubectl exec {self.head_pod} -- bash -c '
export RAY_DATA_HOME="/checkpoints"
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
cd /workspace/verl
python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo data.train_files="/checkpoints/data/gsm8k/train.parquet" data.val_files="/checkpoints/data/gsm8k/test.parquet" data.prompt_key=question data.train_batch_size={self.args.batch_size} data.max_prompt_length=512 data.max_response_length=1024 data.filter_overlong_prompts=True data.truncation=error actor_rollout_ref.model.path="{self.args.model_path}" actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size={self.args.batch_size} actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 actor_rollout_ref.rollout.tensor_model_parallel_size=2 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.n=2 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 actor_rollout_ref.ref.fsdp_config.param_offload=True algorithm.use_kl_in_reward=False trainer.critic_warmup=0 trainer.logger="[console]" trainer.project_name="GRPO" trainer.experiment_name="{self.args.job_name}" trainer.n_gpus_per_node=1 trainer.nnodes={self.args.num_nodes} trainer.default_local_dir="{self.args.checkpoint_dir}" trainer.save_freq={self.args.save_freq} trainer.test_freq=2 trainer.total_epochs=2 {resume_args}
' &"""
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            output = result.stdout + result.stderr
            
            # Extract job ID
            import re
            match = re.search(r"raysubmit_[a-zA-Z0-9]+", output)
            if match:
                self.job_id = match.group(0)
                self.logger.log(f"✓ Job submitted: {self.job_id}")
            else:
                self.logger.log("Job submitted but couldn't extract job ID", "WARN")
                self.logger.log(f"Output: {output[:200]}")
                
        except Exception as e:
            self.logger.log(f"Error starting training: {e}", "ERROR")
            return False
        
        # Verify GPUs are being used
        self.logger.log("Verifying GPU utilization...")
        time.sleep(10)  # Give training time to start
        
        try:
            # Check Ray resources
            import subprocess
            result = subprocess.run(
                f'kubectl exec {self.head_pod} -- ray status',
                shell=True, capture_output=True, text=True, timeout=30
            )
            
            if 'GPU' in result.stdout:
                # Parse GPU usage
                for line in result.stdout.split('\n'):
                    if 'GPU' in line and '/' in line:
                        self.logger.log(f"Ray resources: {line.strip()}")
                        if '0.0' in line.split('/')[0]:
                            self.logger.log("WARNING: GPUs available but 0% utilized!", "WARN")
                            self.logger.log("Training may not be using GPUs correctly.", "WARN")
                        else:
                            self.logger.log("✓ GPUs are being utilized")
                        break
            else:
                self.logger.log("No GPU info in Ray status", "WARN")
                
        except Exception as e:
            self.logger.log(f"Could not verify GPU utilization: {e}", "WARN")
        
        return True
    
    def _step3_deploy_pytorchjob(self) -> bool:
        """Step 3: Deploy using PyTorchJob."""
        self.logger.step(3, 4, "Deploying with PyTorchJob (pytorchjob-manager)...")
        
        # Check if PyTorchJob CRD exists
        self.logger.log("Checking PyTorchJob CRD...")
        if not check_pytorchjob_crd():
            self.logger.log("PyTorchJob CRD not found. Is Kubeflow installed?", "ERROR")
            return False
        
        self.logger.log("✓ PyTorchJob CRD is available")
        
        # Generate PyTorchJob YAML
        self.logger.log(f"Generating PyTorchJob YAML...")
        
        config = {
            'job_name': self.args.job_name,
            'image_uri': self.args.image_uri,
            'num_nodes': self.args.num_nodes,
            'command': self._generate_torchrun_command(),
            'checkpoint_pvc': f"{self.args.job_name}-checkpoints"
        }
        
        yaml_content = generate_pytorchjob_yaml(config)
        
        # Deploy
        self.logger.log("Deploying PyTorchJob...")
        if not deploy_pytorchjob(yaml_content):
            self.logger.log("Failed to deploy PyTorchJob", "ERROR")
            return False
        
        self.logger.log("✓ PyTorchJob deployed")
        self.logger.log(f"Monitor with: kubectl get pytorchjob {self.args.job_name}")
        
        return True
    
    def _step4_start_monitoring(self) -> bool:
        """Step 4: Start auto-restart monitoring."""
        self.logger.step(4, 4, "Starting auto-restart monitor (training-monitor)...")
        
        if not self.head_pod:
            self.logger.log("Cannot start monitoring - head pod not identified", "ERROR")
            return False
        
        self.logger.log(f"Starting monitoring for job: {self.args.job_name}")
        self.logger.log(f"Head pod: {self.head_pod}")
        self.logger.log(f"Max retries: {self.args.max_retries}")
        
        # Start monitoring in background
        try:
            auto_restart(
                head_pod=self.head_pod,
                job_name=self.args.job_name,
                checkpoint_dir=self.args.checkpoint_dir,
                max_retries=self.args.max_retries,
                retry_delay=self.args.retry_delay
            )
        except KeyboardInterrupt:
            self.logger.log("Monitoring interrupted", "WARN")
        
        return True
    
    def _generate_torchrun_command(self) -> str:
        """Generate torchrun command for PyTorchJob."""
        return (
            f"torchrun "
            f"--nnodes={self.args.num_nodes} "
            f"--nproc_per_node={self.args.gpu_per_node} "
            f"--master_addr=$MASTER_ADDR "
            f"--master_port=$MASTER_PORT "
            f"train.py "
            f"--model_path={self.args.model_path} "
            f"--batch_size={self.args.batch_size}"
        )

    # ------------------------------------------------------------------
    # Cleanup / Teardown
    # ------------------------------------------------------------------

    def _run_cleanup(self) -> int:
        """Execute cleanup workflow: delete workloads, optionally scale down instances."""
        self.logger.log("=" * 70)
        self.logger.log("Training Job Deployer - Cleanup Mode")
        self.logger.log("=" * 70)

        total_steps = 2 + (1 if self.args.scale_down else 0)
        step = 0

        # Step 1: Delete RayCluster (if it exists)
        step += 1
        self.logger.step(step, total_steps, "Checking for RayCluster to delete...")
        ray_name = self.args.job_name
        try:
            result = subprocess.run(
                ["kubectl", "get", "raycluster", ray_name, "-o", "name"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                self.logger.log(f"Deleting RayCluster '{ray_name}'...")
                delete_raycluster(ray_name, namespace=self.args.namespace)
                self.logger.log(f"RayCluster '{ray_name}' deleted", "SUCCESS")
            else:
                self.logger.log(f"No RayCluster '{ray_name}' found, skipping")
        except Exception as e:
            self.logger.log(f"Error deleting RayCluster: {e}", "WARN")

        # Step 2: Delete PyTorchJob (if it exists)
        step += 1
        self.logger.step(step, total_steps, "Checking for PyTorchJob to delete...")
        try:
            result = subprocess.run(
                ["kubectl", "get", "pytorchjob", ray_name, "-o", "name"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                self.logger.log(f"Deleting PyTorchJob '{ray_name}'...")
                subprocess.run(
                    ["kubectl", "delete", "pytorchjob", ray_name,
                     "-n", self.args.namespace, "--timeout=60s"],
                    capture_output=True, text=True, timeout=90,
                )
                self.logger.log(f"PyTorchJob '{ray_name}' deleted", "SUCCESS")
            else:
                self.logger.log(f"No PyTorchJob '{ray_name}' found, skipping")
        except Exception as e:
            self.logger.log(f"Error deleting PyTorchJob: {e}", "WARN")

        # Step 3 (optional): Scale down HyperPod instance group
        if self.args.scale_down:
            step += 1
            self.logger.step(step, total_steps, "Scaling down HyperPod instance group...")

            hp_cluster = self.args.hyperpod_cluster
            hp_group = self.args.hyperpod_group

            if not hp_cluster or not hp_group:
                # Try to auto-detect from current nodes
                self.logger.log("Auto-detecting HyperPod cluster and instance group...")
                try:
                    nodes = get_hyperpod_nodes()
                    if nodes:
                        hp_group = hp_group or nodes[0].get("node_group")
                    if not hp_cluster:
                        # Try to find the cluster from the EKS cluster name
                        clusters = []
                        try:
                            from hyperpod_manager import list_clusters
                            clusters = list_clusters(region=self.args.region)
                        except Exception:
                            pass
                        if len(clusters) == 1:
                            hp_cluster = clusters[0]["name"]
                        elif len(clusters) > 1:
                            self.logger.log(
                                f"Multiple HyperPod clusters found: {[c['name'] for c in clusters]}. "
                                f"Please specify --hyperpod_cluster.", "ERROR"
                            )
                            return 1
                except Exception as e:
                    self.logger.log(f"Auto-detection failed: {e}", "ERROR")

            if not hp_cluster or not hp_group:
                self.logger.log(
                    "Cannot scale down: --hyperpod_cluster and --hyperpod_group are required "
                    "(or auto-detection must succeed).", "ERROR"
                )
                return 1

            target = self.args.scale_down_target
            try:
                result = scale_instance_group(
                    cluster_name=hp_cluster,
                    group_name=hp_group,
                    target_count=target,
                    region=self.args.region,
                )
                self.logger.log(
                    f"Instance group '{hp_group}' scaling: "
                    f"{result['old_count']} -> {target}", "SUCCESS"
                )
            except Exception as e:
                self.logger.log(f"Failed to scale instance group: {e}", "ERROR")
                return 1

        self.logger.log("=" * 70)
        self.logger.log("Cleanup completed successfully!", "SUCCESS")
        self.logger.log("=" * 70)
        return 0


def main():
    parser = argparse.ArgumentParser(
        description='Deploy distributed training jobs on EKS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy with Ray
  python deploy.py --cluster_name my-cluster --image_uri my-image:latest --use_ray
  
  # Deploy with auto-monitoring
  python deploy.py --cluster_name my-cluster --image_uri my-image:latest --use_ray --auto_monitor
  
  # Deploy with PyTorchJob
  python deploy.py --cluster_name my-cluster --image_uri my-image:latest --use_pytorchjob

  # Cleanup: delete workloads and scale down instances
  python deploy.py --cluster_name my-cluster --image_uri dummy --job_name my-job --cleanup --scale_down \\
    --hyperpod_cluster my-hp-cluster --hyperpod_group my-ig-8x
        """
    )
    
    # Required
    parser.add_argument('--cluster_name', required=True, help='EKS cluster name')
    parser.add_argument('--image_uri', required=True, help='Docker image URI')
    
    # Framework selection
    parser.add_argument('--use_ray', action='store_true', help='Use Ray/KubeRay')
    parser.add_argument('--use_pytorchjob', action='store_true', help='Use PyTorchJob')
    
    # Configuration
    parser.add_argument('--num_nodes', type=int, default=4, help='Number of nodes')
    parser.add_argument('--gpu_per_node', type=int, default=1, help='GPUs per node')
    parser.add_argument('--job_name', default='training-job', help='Job name')
    parser.add_argument('--checkpoint_dir', help='Checkpoint directory (auto-generated if not specified)')
    parser.add_argument('--storage_class', default='fsx-sc', help='Storage class for PVC')
    parser.add_argument('--storage_size', default='100Gi', help='Storage size')
    parser.add_argument('--namespace', default='default', help='Kubernetes namespace')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    
    # Monitoring
    parser.add_argument('--auto_monitor', action='store_true', help='Start auto-restart monitor')
    parser.add_argument('--max_retries', type=int, default=10, help='Max restart attempts')
    parser.add_argument('--retry_delay', type=int, default=60, help='Seconds between retries')
    
    # EFA
    parser.add_argument('--use_efia', action='store_true', default=True, help='Enable EFA')
    parser.add_argument('--efa_device', type=int, default=1, help='EFA devices per node')
    
    # Training config
    parser.add_argument('--model_path', default='Qwen/Qwen2.5-0.5B', help='Model path')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--save_freq', type=int, default=10, help='Checkpoint save frequency')
    
    # Control
    parser.add_argument('--skip_validation', action='store_true', help='Skip cluster validation')
    parser.add_argument('--skip_resume', action='store_true', help='Skip checkpoint resume')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    
    # Cleanup / Teardown
    parser.add_argument('--cleanup', action='store_true', help='Run cleanup mode: delete workloads and optionally scale down instances')
    parser.add_argument('--scale_down', action='store_true', help='Scale down HyperPod instance group during cleanup')
    parser.add_argument('--scale_down_target', type=int, default=0, help='Target instance count when scaling down (default: 0)')
    parser.add_argument('--hyperpod_cluster', default=None, help='HyperPod cluster name (for scale down). Auto-detected if only one cluster exists.')
    parser.add_argument('--hyperpod_group', default=None, help='HyperPod instance group name (for scale down). Auto-detected from node labels if possible.')
    
    args = parser.parse_args()
    
    # Auto-generate checkpoint dir if not specified
    if not args.checkpoint_dir:
        args.checkpoint_dir = f"/checkpoints/GRPO/{args.job_name}"
    
    # Deploy
    deployer = TrainingJobDeployer(args)
    sys.exit(deployer.run())


if __name__ == '__main__':
    main()
