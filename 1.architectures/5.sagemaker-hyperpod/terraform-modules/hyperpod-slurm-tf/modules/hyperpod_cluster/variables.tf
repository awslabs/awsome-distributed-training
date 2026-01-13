variable "resource_name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "hyperpod_cluster_name" {
  description = "Name of the HyperPod cluster"
  type        = string
}

variable "node_recovery" {
  description = "Node recovery mode"
  type        = string
}

variable "instance_groups" {
  description = "Map of instance group configurations"
  type = map(object({
    instance_type    = string
    instance_count   = number
    ebs_volume_size  = number
    threads_per_core = number
    lifecycle_script = string
  }))
}

variable "private_subnet_id" {
  description = "ID of the private subnet"
  type        = string
}

variable "security_group_id" {
  description = "ID of the security group"
  type        = string
}

variable "s3_bucket_name" {
  description = "Name of the S3 bucket for lifecycle scripts"
  type        = string
}

variable "sagemaker_iam_role_name" {
  description = "Name of the SageMaker IAM role"
  type        = string
}

variable "use_training_plan" {
  description = "Whether to attach a SageMaker Training Plan to cluster instance groups."
  type        = bool
  default     = false
}

variable "training_plan_arn" {
  description = "ARN of the SageMaker Training Plan to attach (when enabled)."
  type        = string
  default     = null
}

variable "training_plan_instance_group_name" {
  description = "Instance group name that should receive the SageMaker Training Plan."
  type        = string
  default     = "compute"
}

variable "training_plan_expected_instance_type" {
  description = "Expected instance type required by the Training Plan (optional validation)."
  type        = string
  default     = null
}

variable "training_plan_expected_instance_count" {
  description = "Expected instance count required by the Training Plan (optional validation)."
  type        = number
  default     = null
}

