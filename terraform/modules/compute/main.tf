# ---------------------------------------------------------------- permissions (IAM)
data "aws_iam_policy_document" "assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "read_secret" {
  statement {
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [var.secret_arn]
  }
}

resource "aws_iam_role" "app" {
  name               = "${var.name_prefix}-ec2-role"
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}

# Pull images from the registry
resource "aws_iam_role_policy_attachment" "ecr_read" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# Open a shell with Session Manager (no SSH needed)
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Read ONLY the application secret
resource "aws_iam_role_policy" "read_secret" {
  name   = "${var.name_prefix}-read-secret"
  role   = aws_iam_role.app.id
  policy = data.aws_iam_policy_document.read_secret.json
}

resource "aws_iam_instance_profile" "app" {
  name = "${var.name_prefix}-instance-profile"
  role = aws_iam_role.app.name
}

# ---------------------------------------------------------------- server
resource "aws_instance" "app" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [var.security_group_id]
  iam_instance_profile   = aws_iam_instance_profile.app.name
  key_name               = var.key_pair_name

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    aws_region      = var.aws_region
    ecr_registry    = var.ecr_registry
    image           = var.container_image
    secret_name     = var.secret_name
    compose_version = var.docker_compose_version
    env_lines       = join("\n", [for key in sort(keys(var.app_environment)) : "${key}=${var.app_environment[key]}"])
  })

  # IMDSv2 only (protects the instance credentials)
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_size           = var.root_volume_size_gb
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  tags = {
    Name = "${var.name_prefix}-app"
  }

  lifecycle {
    ignore_changes = [ami]
  }
}

resource "aws_volume_attachment" "data" {
  device_name = "/dev/sdf"
  volume_id   = var.data_volume_id
  instance_id = aws_instance.app.id
}

# Fixed public IP address (it does not change when the server restarts)
resource "aws_eip" "app" {
  domain   = "vpc"
  instance = aws_instance.app.id

  tags = {
    Name = "${var.name_prefix}-eip"
  }
}
