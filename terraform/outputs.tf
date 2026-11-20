output "ui_url" {
  description = "Streamlit web interface."
  value       = "http://${module.compute.public_ip}:8501"
}

output "api_url" {
  description = "FastAPI service (interactive documentation is at /docs)."
  value       = "http://${module.compute.public_ip}:8000"
}

output "server_public_ip" {
  description = "Fixed public IP address of the server (Elastic IP)."
  value       = module.compute.public_ip
}

output "server_instance_id" {
  description = "EC2 instance id. Use it to open a shell with Session Manager."
  value       = module.compute.instance_id
}

output "session_manager_command" {
  description = "Command that opens a shell on the server without SSH."
  value       = "aws ssm start-session --target ${module.compute.instance_id} --region ${var.aws_region}"
}

output "ecr_repository_url" {
  description = "Container registry address of the application image."
  value       = module.storage.ecr_repository_url
}

output "secret_name" {
  description = "Secrets Manager secret that must hold OPENROUTER_API_KEY (and optionally API_KEY)."
  value       = module.storage.secret_name
}

output "data_volume_id" {
  description = "Persistent data disk (vector database, documents, reports, models)."
  value       = module.storage.data_volume_id
}

output "vpc_id" {
  description = "VPC id."
  value       = module.network.vpc_id
}

output "image_push_commands" {
  description = "Commands that build and push the application image (run them from the project root)."
  value       = <<-EOT
    aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${local.ecr_registry}
    docker build -t ${module.storage.ecr_repository_url}:${var.container_image_tag} .
    docker push ${module.storage.ecr_repository_url}:${var.container_image_tag}
  EOT
}
