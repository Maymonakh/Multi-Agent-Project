# Registry for the application image (built from the project's Dockerfile)
resource "aws_ecr_repository" "app" {
  name                 = var.name_prefix
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the last ${var.images_to_keep} images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = var.images_to_keep
      }
      action = {
        type = "expire"
      }
    }]
  })
}

# Only the secret CONTAINER is created here. The values (OPENROUTER_API_KEY, API_KEY) are added
# afterwards with the AWS CLI (see README), so they never appear in the Terraform state.
resource "aws_secretsmanager_secret" "app_env" {
  name                    = "${var.name_prefix}/app-env"
  description             = "Secret settings of the application: OPENROUTER_API_KEY and optional API_KEY as JSON"
  recovery_window_in_days = var.secret_recovery_days
}

# Persistent data disk: Chroma vector database, uploaded documents, reports, Redis data, downloaded AI models
resource "aws_ebs_volume" "data" {
  availability_zone = var.availability_zone
  size              = var.data_volume_size_gb
  type              = "gp3"
  encrypted         = true

  tags = {
    Name = "${var.name_prefix}-data"
  }
}
