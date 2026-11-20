output "vpc_id" {
  description = "VPC id."
  value       = aws_vpc.this.id
}

output "public_subnet_id" {
  description = "Public subnet id."
  value       = aws_subnet.public.id
}

output "security_group_id" {
  description = "Security group of the application server."
  value       = aws_security_group.app.id
}
