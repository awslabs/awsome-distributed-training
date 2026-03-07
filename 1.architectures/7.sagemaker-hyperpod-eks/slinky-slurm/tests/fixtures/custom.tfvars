kubernetes_version    = "1.34"
eks_cluster_name      = "slinky-eks-cluster"
hyperpod_cluster_name = "slinky-hp-cluster"
resource_name_prefix  = "slinky-hp-eks"
aws_region            = "us-west-2"
instance_groups = [
  {
    name                      = "accelerated-instance-group-1"
    instance_type             = "ml.g5.8xlarge",
    instance_count            = 4,
    availability_zone_id      = "usw2-az2",
    ebs_volume_size_in_gb     = 500,
    threads_per_core          = 2,
    enable_stress_check       = true,
    enable_connectivity_check = true,
    lifecycle_script          = "on_create.sh"
  },
  {
    name                      = "general-instance-group-2"
    instance_type             = "ml.m5.2xlarge",
    instance_count            = 2,
    availability_zone_id      = "usw2-az2",
    ebs_volume_size_in_gb     = 500,
    threads_per_core          = 1,
    enable_stress_check       = false,
    enable_connectivity_check = false,
    lifecycle_script          = "on_create.sh"
  }
]
create_observability_module               = true
network_metric_level                      = "ADVANCED"
logging_enabled                           = true
create_task_governance_module             = true
create_hyperpod_training_operator_module  = true
create_hyperpod_inference_operator_module = true
enable_guardduty_cleanup                  = true
create_new_fsx_filesystem                 = true
