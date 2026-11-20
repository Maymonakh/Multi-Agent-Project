variable "name_prefix" {
  description = "Prefix for resource names."
  type        = string
}

variable "availability_zone" {
  description = "Availability zone of the data disk (must be the same as the server)."
  type        = string
}

variable "data_volume_size_gb" {
  description = "Size of the data disk in GB."
  type        = number
}

variable "secret_recovery_days" {
  description = "Days Secrets Manager keeps a deleted secret (0 = delete immediately, otherwise 7 to 30)."
  type        = number
  default     = 7

  validation {
    condition     = var.secret_recovery_days == 0 || (var.secret_recovery_days >= 7 && var.secret_recovery_days <= 30)
    error_message = "secret_recovery_days must be 0 or between 7 and 30."
  }
}

variable "images_to_keep" {
  description = "How many image versions the registry keeps."
  type        = number
  default     = 10
}
