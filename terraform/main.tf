locals {
  name_prefix       = "${var.project_name}-${var.environment}"
  availability_zone = coalesce(var.availability_zone, "${var.aws_region}a")
  ecr_registry      = split("/", module.storage.ecr_repository_url)[0]

  common_tags = merge(
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    },
    var.tags
  )
}

# Network: VPC, public subnet, internet gateway, firewall (security group)
module "network" {
  source = "./modules/network"

  name_prefix        = local.name_prefix
  vpc_cidr           = var.vpc_cidr
  public_subnet_cidr = var.public_subnet_cidr
  availability_zone  = local.availability_zone
  allowed_app_cidrs  = var.allowed_app_cidrs
  allowed_ssh_cidrs  = var.allowed_ssh_cidrs
}

# Storage: image registry, secret container and the persistent data disk
module "storage" {
  source = "./modules/storage"

  name_prefix         = local.name_prefix
  availability_zone   = local.availability_zone
  data_volume_size_gb = var.data_volume_size_gb
}

# Compute: the server that runs Redis, the FastAPI service and the Streamlit UI with Docker Compose
module "compute" {
  source     = "./modules/compute"
  depends_on = [module.network]

  name_prefix         = local.name_prefix
  aws_region          = var.aws_region
  ami_id              = var.ami_id
  instance_type       = var.instance_type
  key_pair_name       = var.key_pair_name
  root_volume_size_gb = var.root_volume_size_gb
  subnet_id           = module.network.public_subnet_id
  security_group_id   = module.network.security_group_id
  data_volume_id      = module.storage.data_volume_id
  secret_arn          = module.storage.secret_arn
  secret_name         = module.storage.secret_name
  container_image     = "${module.storage.ecr_repository_url}:${var.container_image_tag}"
  ecr_registry        = local.ecr_registry
  app_environment     = var.app_environment
}
