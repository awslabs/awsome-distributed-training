data "aws_partition" "current" {}
data "aws_region" "current" {}
data "aws_caller_identity" "current" {}

# Add delay to allow IAM role propagation
resource "time_sleep" "wait_for_iam_role" {
  create_duration = "30s"
}

locals {
  # Training plan target group (by name/key in var.instance_groups)
  training_group_name   = var.training_plan_instance_group_name
  training_group_config = try(var.instance_groups[local.training_group_name], null)

  training_group_instance_type  = try(local.training_group_config.instance_type, null)
  training_group_instance_count = try(local.training_group_config.instance_count, null)

  # Create configurations for each instance group
  # - instance group "name" comes from the map key of var.instance_groups
  # - training_plan_arn is attached ONLY when:
  #     use_training_plan = true
  #     training_plan_arn is provided (not null)
  #     group name matches var.training_plan_instance_group_name (default: "compute")
  instance_groups_list = [
    for name, config in var.instance_groups :
    merge(
      {
        instance_group_name = name
        instance_type       = config.instance_type
        instance_count      = config.instance_count
        threads_per_core    = config.threads_per_core
        execution_role      = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:role/${var.sagemaker_iam_role_name}"

        instance_storage_configs = [
          {
            ebs_volume_config = {
              volume_size_in_gb = config.ebs_volume_size
            }
          }
        ]

        life_cycle_config = {
          on_create     = config.lifecycle_script
          source_s3_uri = "s3://${var.s3_bucket_name}/LifecycleScripts/base-config/"
        }
      },

      # Attach Training Plan ONLY to the configured instance group name
      (var.use_training_plan &&
        var.training_plan_arn != null &&
        name == var.training_plan_instance_group_name) ? {
        training_plan_arn = var.training_plan_arn
      } : {}
    )
  ]
}

resource "awscc_sagemaker_cluster" "hyperpod_cluster" {
  depends_on = [time_sleep.wait_for_iam_role]

  cluster_name = var.hyperpod_cluster_name

  instance_groups = local.instance_groups_list

  node_recovery = var.node_recovery

  vpc_config = {
    security_group_ids = [var.security_group_id]
    subnets            = [var.private_subnet_id]
  }

  tags = [
    {
      key   = "Name"
      value = "${var.resource_name_prefix}-hyperpod-cluster"
    }
  ]

  lifecycle {
    precondition {
      condition = (
        !var.use_training_plan
        ||
        (
          var.training_plan_arn != null
          && local.training_group_config != null
          && (var.training_plan_expected_instance_type == null || var.training_plan_expected_instance_type == local.training_group_instance_type)
          && (var.training_plan_expected_instance_count == null || var.training_plan_expected_instance_count == local.training_group_instance_count)
        )
      )

      error_message = "Training Plan is enabled, but validation failed. Ensure: (1) training_plan_arn is set, (2) instance group '${var.training_plan_instance_group_name}' exists in var.instance_groups, and (3) its instance_type/instance_count match the Training Plan requirements (if expected values are provided)."
    }
  }
}
