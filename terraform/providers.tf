terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Local state is used because nothing is deployed. For a real, shared deployment, store the
  # state remotely (see README, "Going to production"):
  # backend "s3" { bucket = "..." key = "multi-agent-rag/terraform.tfstate" region = "us-east-1" }
}

provider "aws" {
  region = var.aws_region

  # offline_mode = true lets `terraform plan` work WITHOUT an AWS account: Terraform does not
  # contact AWS to check credentials, and mock keys are used. Set it to false for a real deployment.
  access_key                  = var.offline_mode ? "mock-access-key" : null
  secret_key                  = var.offline_mode ? "mock-secret-key" : null
  skip_credentials_validation = var.offline_mode
  skip_requesting_account_id  = var.offline_mode
  skip_metadata_api_check     = var.offline_mode
  skip_region_validation      = var.offline_mode

  default_tags {
    tags = local.common_tags
  }
}
