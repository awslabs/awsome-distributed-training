#!/usr/bin/env python3
"""
Claude Code Command: Deploy Training Job
Deploy distributed training jobs to EKS with support for PyTorchJob and Ray (KubeRay).
Supports cleanup mode for deleting workloads and scaling down HyperPod instance groups.

This command uses the training-job-deployer skill which orchestrates multiple sub-skills:
- k8s-cluster-manager: Validates cluster health and resources
- ray-cluster-manager: Sets up Ray/KubeRay infrastructure  
- pytorchjob-manager: Manages Kubeflow PyTorchJobs
- checkpoint-manager: Configures persistent storage
- training-monitor: Monitors and auto-restarts failed jobs
- hyperpod-manager: Leverages HyperPod features (including instance group scaling)

Examples:
    # Deploy with PyTorchJob (default)
    deploy_training_job(cluster_name="my-cluster", image_uri="my-image:latest")
    
    # Deploy with Ray/KubeRay
    deploy_training_job(cluster_name="my-cluster", image_uri="my-image:latest", use_ray=True)
    
    # Deploy with auto-monitoring and checkpoint resume
    deploy_training_job(
        cluster_name="my-cluster",
        image_uri="my-image:latest",
        use_ray=True,
        auto_monitor=True,
        max_retries=10
    )
    
    # Cleanup: delete workloads only
    cleanup_training_job(cluster_name="my-cluster", job_name="verl-grpo-training")
    
    # Cleanup: delete workloads AND scale down HyperPod instances
    cleanup_training_job(
        cluster_name="my-cluster",
        job_name="verl-grpo-training",
        scale_down=True,
        hyperpod_cluster="my-hp-cluster",
        hyperpod_group="my-hp-group"
    )
"""

from typing import Optional
import sys
import os
import subprocess


def deploy_training_job(
    cluster_name: str,
    image_uri: str,
    job_name: str = "training-job",
    num_nodes: int = 4,
    gpu_per_node: int = 1,
    use_ray: bool = False,
    use_pytorchjob: bool = False,
    auto_monitor: bool = False,
    max_retries: int = 10,
    checkpoint_dir: Optional[str] = None,
    storage_size: str = "100Gi",
    storage_class: str = "fsx-sc",
    model_path: str = "Qwen/Qwen2.5-0.5B",
    batch_size: int = 8,
    save_freq: int = 10,
    use_efia: bool = True,
    skip_validation: bool = False,
    namespace: str = "default",
    region: str = "us-west-2",
    verbose: bool = True
) -> str:
    """
    Deploy distributed training job to EKS using PyTorchJob or Ray (KubeRay).
    
    This function orchestrates the deployment through the training-job-deployer skill,
    which coordinates multiple sub-skills for cluster validation, storage setup,
    framework deployment, and monitoring.
    
    Args:
        cluster_name: EKS cluster name (required)
        image_uri: Docker image URI for training (required)
        job_name: Job name for identification (default: "training-job")
        num_nodes: Number of nodes for distributed training (default: 4)
        gpu_per_node: Number of GPUs per node (default: 1)
        use_ray: Use Ray/KubeRay for distributed training (default: False)
        use_pytorchjob: Use Kubeflow PyTorchJob (default: False, uses Ray if both False)
        auto_monitor: Start auto-restart monitor after deployment (default: False)
        max_retries: Maximum restart attempts for auto-monitor (default: 10)
        checkpoint_dir: Directory for checkpoints (auto-generated if not specified)
        storage_size: Storage size for PVC (default: "100Gi")
        storage_class: Storage class for PVC (default: "fsx-sc")
        model_path: Model path or HuggingFace model name (default: "Qwen/Qwen2.5-0.5B")
        batch_size: Training batch size (default: 8)
        save_freq: Checkpoint save frequency (default: 10)
        use_efia: Enable EFA for high-performance networking (default: True)
        skip_validation: Skip cluster validation for faster deployment (default: False)
        namespace: Kubernetes namespace (default: "default")
        region: AWS region (default: "us-west-2")
        verbose: Enable verbose output (default: True)
    
    Returns:
        str: Deployment status and monitoring information
    
    Examples:
        # Basic PyTorchJob deployment
        deploy_training_job(cluster_name="my-cluster", image_uri="my-image:latest")
        
        # Ray deployment with monitoring
        deploy_training_job(
            cluster_name="my-cluster",
            image_uri="my-image:latest",
            use_ray=True,
            auto_monitor=True
        )
        
        # VERL training with auto-restart
        deploy_training_job(
            cluster_name="my-cluster",
            image_uri="my-verl-image:latest",
            job_name="verl-grpo",
            num_nodes=4,
            use_ray=True,
            auto_monitor=True,
            max_retries=10
        )
        
        # FSDP training with PyTorchJob
        deploy_training_job(
            cluster_name="my-cluster",
            image_uri="my-fsdp-image:latest",
            job_name="llama-fsdp",
            num_nodes=8,
            use_pytorchjob=True,
            model_path="meta-llama/Llama-2-7b-hf"
        )
    """
    
    if not cluster_name:
        return "❌ Error: cluster_name is required"
    
    if not image_uri:
        return "❌ Error: image_uri is required"
    
    # Build command to call the skill
    skill_path = os.path.join(
        os.path.dirname(__file__), '..', 'opencode', 'skills',
        'training-job-deployer', 'src', 'deploy.py'
    )
    
    cmd = [
        'python3',
        skill_path,
        '--cluster_name', cluster_name,
        '--image_uri', image_uri,
        '--job_name', job_name,
        '--num_nodes', str(num_nodes),
        '--gpu_per_node', str(gpu_per_node),
        '--storage_size', storage_size,
        '--storage_class', storage_class,
        '--model_path', model_path,
        '--batch_size', str(batch_size),
        '--save_freq', str(save_freq),
        '--namespace', namespace,
        '--region', region,
        '--max_retries', str(max_retries)
    ]
    
    # Add framework selection
    if use_ray:
        cmd.append('--use_ray')
    elif use_pytorchjob:
        cmd.append('--use_pytorchjob')
    
    # Add optional flags
    if auto_monitor:
        cmd.append('--auto_monitor')
    
    if use_efia:
        cmd.append('--use_efia')
    
    if skip_validation:
        cmd.append('--skip_validation')
    
    if verbose:
        cmd.append('--verbose')
    
    # Add checkpoint directory if specified
    if checkpoint_dir:
        cmd.extend(['--checkpoint_dir', checkpoint_dir])
    
    try:
        # Run the skill
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for deployment
        )
        
        if result.returncode == 0:
            status = "✅ Deployment successful"
            if auto_monitor:
                status += " with auto-monitoring enabled"
            return f"{status}\n\n{result.stdout}"
        else:
            return f"❌ Deployment failed (exit code: {result.returncode})\n\n{result.stderr}\n\n{result.stdout}"
    
    except subprocess.TimeoutExpired:
        return "❌ Deployment timeout after 600 seconds"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def deploy_with_checkpoints(
    cluster_name: str,
    image_uri: str,
    job_name: str = "training-job",
    num_nodes: int = 4,
    storage_size: str = "100Gi",
    namespace: str = "default"
) -> str:
    """
    Deploy training job with persistent checkpoint storage.
    
    This is a convenience wrapper that sets up PVC for checkpoints before deployment.
    
    Args:
        cluster_name: EKS cluster name (required)
        image_uri: Docker image URI (required)
        job_name: Job name (default: "training-job")
        num_nodes: Number of nodes (default: 4)
        storage_size: Storage size for checkpoints (default: "100Gi")
        namespace: Kubernetes namespace (default: "default")
    
    Returns:
        str: Deployment status
    """
    
    skill_path = os.path.join(
        os.path.dirname(__file__), '..', 'opencode', 'skills',
        'training-job-deployer', 'src', 'deploy_with_checkpoints.py'
    )
    
    cmd = [
        'python3',
        skill_path,
        '--cluster_name', cluster_name,
        '--job_name', job_name,
        '--image_uri', image_uri,
        '--num_nodes', str(num_nodes),
        '--namespace', namespace,
        '--storage_size', storage_size
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            return f"✅ Checkpoint storage setup successful\n\n{result.stdout}"
        else:
            return f"❌ Setup failed\n\n{result.stderr}"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"


def monitor_training_job(
    head_pod: str,
    job_name: str = "training-job",
    checkpoint_dir: str = "/checkpoints",
    max_retries: int = 10,
    retry_delay: int = 60
) -> str:
    """
    Monitor training job and auto-restart on failure.
    
    Args:
        head_pod: Name of the Ray head pod (required)
        job_name: Job name for logging (default: "training-job")
        checkpoint_dir: Directory for checkpoints (default: "/checkpoints")
        max_retries: Maximum restart attempts (default: 10)
        retry_delay: Seconds between retries (default: 60)
    
    Returns:
        str: Monitoring status
    """
    
    monitor_skill_src = os.path.join(
        os.path.dirname(__file__), '..', 'opencode', 'skills',
        'training-monitor', 'src'
    )
    sys.path.insert(0, monitor_skill_src)
    
    try:
        from monitor import auto_restart
        
        result = auto_restart(
            head_pod=head_pod,
            job_name=job_name,
            checkpoint_dir=checkpoint_dir,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        if result.get('success'):
            return (
                f"Training completed successfully!\n"
                f"  Final step: {result.get('final_step')}\n"
                f"  Total retries: {result.get('total_retries')}\n"
                f"  Job ID: {result.get('job_id')}"
            )
        else:
            return (
                f"Training ended: {result.get('reason')}\n"
                f"  Final step: {result.get('final_step')}\n"
                f"  Total retries: {result.get('total_retries')}\n"
                f"  Job ID: {result.get('job_id')}"
            )
    except ImportError as e:
        return f"Error: Could not import training monitor: {e}"
    except KeyboardInterrupt:
        return "Monitoring stopped by user"
    except Exception as e:
        return f"Error: {str(e)}"


def cleanup_training_job(
    cluster_name: str,
    job_name: str = "training-job",
    scale_down: bool = False,
    scale_down_target: int = 0,
    hyperpod_cluster: Optional[str] = None,
    hyperpod_group: Optional[str] = None,
    namespace: str = "default",
    region: str = "us-west-2"
) -> str:
    """
    Clean up training workloads and optionally scale down HyperPod instance groups.
    
    Deletes RayClusters and PyTorchJobs by name, and optionally scales down
    HyperPod instance groups to save costs after training is complete.
    
    Args:
        cluster_name: EKS cluster name (required)
        job_name: Job name to delete (default: "training-job")
        scale_down: Also scale down HyperPod instance group (default: False)
        scale_down_target: Target instance count for scale down (default: 0)
        hyperpod_cluster: HyperPod cluster name (auto-detected if only one exists)
        hyperpod_group: HyperPod instance group name (auto-detected from node labels)
        namespace: Kubernetes namespace (default: "default")
        region: AWS region (default: "us-west-2")
    
    Returns:
        str: Cleanup status
    
    Examples:
        # Delete workloads only
        cleanup_training_job(cluster_name="my-cluster", job_name="verl-grpo")
        
        # Delete workloads and scale down instances
        cleanup_training_job(
            cluster_name="my-cluster",
            job_name="verl-grpo",
            scale_down=True
        )
        
        # Scale down specific HyperPod instance group
        cleanup_training_job(
            cluster_name="my-cluster",
            job_name="verl-grpo",
            scale_down=True,
            hyperpod_cluster="my-hp-cluster",
            hyperpod_group="my-hp-ig-8x"
        )
    """
    
    skill_path = os.path.join(
        os.path.dirname(__file__), '..', 'opencode', 'skills',
        'training-job-deployer', 'src', 'deploy.py'
    )
    
    cmd = [
        'python3',
        skill_path,
        '--cluster_name', cluster_name,
        '--image_uri', 'cleanup-placeholder',
        '--job_name', job_name,
        '--namespace', namespace,
        '--region', region,
        '--cleanup'
    ]
    
    if scale_down:
        cmd.append('--scale_down')
        cmd.extend(['--scale_down_target', str(scale_down_target)])
    
    if hyperpod_cluster:
        cmd.extend(['--hyperpod_cluster', hyperpod_cluster])
    
    if hyperpod_group:
        cmd.extend(['--hyperpod_group', hyperpod_group])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            msg = f"Cleanup successful for '{job_name}'"
            if scale_down:
                msg += f" (scaled down to {scale_down_target} instances)"
            return f"{msg}\n\n{result.stdout}"
        else:
            return f"Cleanup failed (exit code: {result.returncode})\n\n{result.stderr}\n\n{result.stdout}"
    
    except subprocess.TimeoutExpired:
        return "Cleanup timeout after 120 seconds"
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Deploy training job to EKS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy with PyTorchJob
  python deploy_training_job.py --cluster_name my-cluster --image_uri my-image:latest
  
  # Deploy with Ray and monitoring
  python deploy_training_job.py --cluster_name my-cluster --image_uri my-image:latest --use_ray --auto_monitor
  
  # Deploy with custom configuration
  python deploy_training_job.py --cluster_name my-cluster --image_uri my-image:latest --num_nodes 8 --job_name llama-training
        """
    )
    
    # Required arguments
    parser.add_argument('--cluster_name', required=True, help='EKS cluster name')
    parser.add_argument('--image_uri', required=True, help='Docker image URI')
    
    # Optional configuration
    parser.add_argument('--job_name', default='training-job', help='Job name')
    parser.add_argument('--num_nodes', type=int, default=4, help='Number of nodes')
    parser.add_argument('--gpu_per_node', type=int, default=1, help='GPUs per node')
    
    # Framework selection
    parser.add_argument('--use_ray', action='store_true', help='Use Ray/KubeRay')
    parser.add_argument('--use_pytorchjob', action='store_true', help='Use PyTorchJob')
    
    # Monitoring
    parser.add_argument('--auto_monitor', action='store_true', help='Enable auto-monitoring')
    parser.add_argument('--max_retries', type=int, default=10, help='Max restart attempts')
    
    # Storage
    parser.add_argument('--checkpoint_dir', default=None, help='Checkpoint directory')
    parser.add_argument('--storage_size', default='100Gi', help='Storage size')
    parser.add_argument('--storage_class', default='fsx-sc', help='Storage class')
    
    # Training config
    parser.add_argument('--model_path', default='Qwen/Qwen2.5-0.5B', help='Model path')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--save_freq', type=int, default=10, help='Save frequency')
    
    # Other options
    parser.add_argument('--use_efia', action='store_true', default=True, help='Enable EFA')
    parser.add_argument('--skip_validation', action='store_true', help='Skip validation')
    parser.add_argument('--namespace', default='default', help='Namespace')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    
    # Cleanup / teardown
    parser.add_argument('--cleanup', action='store_true', help='Run cleanup mode (delete workloads)')
    parser.add_argument('--scale_down', action='store_true', help='Scale down HyperPod instances during cleanup')
    parser.add_argument('--scale_down_target', type=int, default=0, help='Target instance count for scale down')
    parser.add_argument('--hyperpod_cluster', default=None, help='HyperPod cluster name')
    parser.add_argument('--hyperpod_group', default=None, help='HyperPod instance group name')
    
    args = parser.parse_args()
    
    if args.cleanup:
        print(cleanup_training_job(
            cluster_name=args.cluster_name,
            job_name=args.job_name,
            scale_down=args.scale_down,
            scale_down_target=args.scale_down_target,
            hyperpod_cluster=args.hyperpod_cluster,
            hyperpod_group=args.hyperpod_group,
            namespace=args.namespace,
            region=args.region
        ))
    else:
        # Remove cleanup-specific args before passing to deploy
        deploy_args = {k: v for k, v in vars(args).items() 
                       if k not in ('cleanup', 'scale_down', 'scale_down_target', 
                                    'hyperpod_cluster', 'hyperpod_group')}
        print(deploy_training_job(**deploy_args))
