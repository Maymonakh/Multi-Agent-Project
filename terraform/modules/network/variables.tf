variable "name_prefix" {
  description = "Prefix for resource names."
  type        = string
}

variable "vpc_cidr" {
  description = "IP range of the VPC."
  type        = string
}

variable "public_subnet_cidr" {
  description = "IP range of the public subnet."
  type        = string
}

variable "availability_zone" {
  description = "Availability zone of the subnet."
  type        = string
}

variable "allowed_app_cidrs" {
  description = "CIDR blocks allowed to reach the UI (8501) and API (8000)."
  type        = list(string)
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed to use SSH (22). Empty = closed."
  type        = list(string)
  default     = []
}
