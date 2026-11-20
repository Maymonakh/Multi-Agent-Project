output "ecr_repository_url" {
  description = "Address of the image registry."
  value       = aws_ecr_repository.app.repository_url
}

output "secret_arn" {
  description = "ARN of the application secret."
  value       = aws_secretsmanager_secret.app_env.arn
}

output "secret_name" {
  description = "Name of the application secret."
  value       = aws_secretsmanager_secret.app_env.name
}

output "data_volume_id" {
  description = "Id of the persistent data disk."
  value       = aws_ebs_volume.data.id
}
