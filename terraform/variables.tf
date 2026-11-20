# ---------------------------------------------------------------- general
variable "project_name" {
  description = "Short name of the project. Used as a prefix in every resource name."
  type        = string
  default     = "multi-agent-rag"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,30}$", var.project_name))
    error_message = "project_name must be 3-31 characters: lowercase letters, digits and hyphens, starting with a letter."
  }
}

variable "environment" {
  description = "Deployment environment."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be dev, staging or prod."
  }
}

variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "availability_zone" {
  description = "Availability zone for the subnet, the server and the data disk. null = the first zone (a) of aws_region."
  type        = string
  default     = null
}

variable "offline_mode" {
  description = "true = no AWS account needed (fake credentials, nothing is contacted). false = real deployment."
  type        = bool
  default     = true
}

variable "tags" {
  description = "Extra tags added to every resource."
  type        = map(string)
  default     = {}
}

# ---------------------------------------------------------------- network
variable "vpc_cidr" {
  description = "IP range of the VPC."
  type        = string
  default     = "10.0.0.0/16"

  validation {
    condition     = can(cidrnetmask(var.vpc_cidr))
    error_message = "vpc_cidr must be a valid IPv4 CIDR block, for example 10.0.0.0/16."
  }
}

variable "public_subnet_cidr" {
  description = "IP range of the public subnet (must be inside vpc_cidr)."
  type        = string
  default     = "10.0.1.0/24"

  validation {
    condition     = can(cidrnetmask(var.public_subnet_cidr))
    error_message = "public_subnet_cidr must be a valid IPv4 CIDR block, for example 10.0.1.0/24."
  }
}

variable "allowed_app_cidrs" {
  description = "Who can open the Streamlit UI (8501) and the FastAPI service (8000). Restrict this in real use."
  type        = list(string)
  default     = ["0.0.0.0/0"]

  validation {
    condition     = alltrue([for cidr in var.allowed_app_cidrs : can(cidrnetmask(cidr))])
    error_message = "Every entry of allowed_app_cidrs must be a valid IPv4 CIDR block."
  }
}

variable "allowed_ssh_cidrs" {
  description = "Who can use SSH (port 22). Empty list = SSH closed (use AWS Systems Manager Session Manager instead)."
  type        = list(string)
  default     = []

  validation {
    condition     = alltrue([for cidr in var.allowed_ssh_cidrs : can(cidrnetmask(cidr))])
    error_message = "Every entry of allowed_ssh_cidrs must be a valid IPv4 CIDR block."
  }
}

# ---------------------------------------------------------------- server
variable "ami_id" {
  description = "Amazon Linux 2023 (x86_64) AMI id of your region. The default is only a placeholder; see README for the command that finds the real one."
  type        = string
  default     = "ami-0123456789abcdef0"

  validation {
    condition     = can(regex("^ami-[0-9a-f]{8,17}$", var.ami_id))
    error_message = "ami_id must look like ami-0123456789abcdef0."
  }
}

variable "instance_type" {
  description = "Server size. The app runs embedding, reranker and speech models on CPU, so 16 GB of RAM (t3.xlarge) is recommended."
  type        = string
  default     = "t3.xlarge"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]*\\.[a-z0-9]+$", var.instance_type))
    error_message = "instance_type must look like t3.xlarge."
  }
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair for SSH. null = no key pair (use Session Manager)."
  type        = string
  default     = null
}

variable "root_volume_size_gb" {
  description = "Size of the system disk (operating system and Docker images)."
  type        = number
  default     = 40

  validation {
    condition     = var.root_volume_size_gb >= 30
    error_message = "root_volume_size_gb must be at least 30 GB (the Docker image is large)."
  }
}

variable "data_volume_size_gb" {
  description = "Size of the data disk (vector database, uploaded documents, reports, downloaded AI models, Redis)."
  type        = number
  default     = 60

  validation {
    condition     = var.data_volume_size_gb >= 20
    error_message = "data_volume_size_gb must be at least 20 GB."
  }
}

# ---------------------------------------------------------------- application
variable "container_image_tag" {
  description = "Tag of the application image in the container registry."
  type        = string
  default     = "latest"
}

variable "app_environment" {
  description = "Non-secret settings of the application (same names as in .env.example). Secrets are NOT set here, see README."
  type        = map(string)
  default = {
    OPENROUTER_MODEL  = "openrouter/free"
    CACHE_THRESHOLD   = "0.95"
    CACHE_TTL_SECONDS = "3600"
    EMBEDDING_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANKER_MODEL    = "BAAI/bge-reranker-v2-m3"
    WHISPER_MODEL     = "small"
    REPORTS_DIR       = "reports"
    MAX_UPLOAD_MB     = "10"
  }
}
