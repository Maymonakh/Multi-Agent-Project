variable "name_prefix" {
  description = "Prefix for resource names."
  type        = string
}

variable "aws_region" {
  description = "AWS region (used by the start-up script)."
  type        = string
}

variable "ami_id" {
  description = "Amazon Linux 2023 (x86_64) AMI id."
  type        = string
}

variable "instance_type" {
  description = "Server size."
  type        = string
}

variable "key_pair_name" {
  description = "Existing EC2 key pair name for SSH, or null."
  type        = string
  default     = null
}

variable "root_volume_size_gb" {
  description = "Size of the system disk in GB."
  type        = number
}

variable "subnet_id" {
  description = "Subnet where the server runs."
  type        = string
}

variable "security_group_id" {
  description = "Security group of the server."
  type        = string
}

variable "data_volume_id" {
  description = "Id of the persistent data disk to attach."
  type        = string
}

variable "secret_arn" {
  description = "ARN of the secret the server may read."
  type        = string
}

variable "secret_name" {
  description = "Name of the secret the start-up script reads."
  type        = string
}

variable "container_image" {
  description = "Full image address with tag, for example 1234.dkr.ecr.us-east-1.amazonaws.com/app:latest."
  type        = string
}

variable "ecr_registry" {
  description = "Registry host used for docker login."
  type        = string
}

variable "app_environment" {
  description = "Non-secret environment variables of the application."
  type        = map(string)
  default     = {}
}

variable "docker_compose_version" {
  description = "Docker Compose plugin version installed on the server."
  type        = string
  default     = "2.29.7"
}
