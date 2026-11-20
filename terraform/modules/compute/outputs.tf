output "instance_id" {
  description = "EC2 instance id."
  value       = aws_instance.app.id
}

output "public_ip" {
  description = "Elastic IP of the server."
  value       = aws_eip.app.public_ip
}

output "iam_role_arn" {
  description = "ARN of the server's IAM role."
  value       = aws_iam_role.app.arn
}
