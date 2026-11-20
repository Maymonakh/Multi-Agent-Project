# Infrastructure as Code (Terraform) for the Multi-Agent Document Assistant

This folder prepares the cloud infrastructure that the application will need in the future.
**Nothing is deployed.** No cloud account is needed and `terraform apply` is never run.
Terraform is only used to `init`, `validate` and `plan` (a dry run that shows what *would* be created).

## 1. What the application needs

The project (see `docker-compose.yml` in the project root) runs three containers:

| Container | Job | Port |
|---|---|---|
| `api` | FastAPI service: agents, OCR, voice, reports | 8000 |
| `ui` | Streamlit web interface (talks to `api`) | 8501 |
| `redis` | Semantic cache | 6379 (internal only) |

It also needs: a place for the application image, a secret (`OPENROUTER_API_KEY`), a persistent disk
(Chroma vector database, uploaded documents, reports, downloaded AI models, Redis data) and enough RAM,
because the embedding, reranker and speech models run on the CPU.

## 2. Infrastructure design (AWS)

```
                    Internet
                       |
                 [ Elastic IP ]
                       |
   VPC 10.0.0.0/16     |
  +--------------------|-----------------------------------+
  |  Public subnet 10.0.1.0/24 (+ Internet Gateway)         |
  |  +----------------------------------------------------+ |
  |  |  EC2 server (Amazon Linux 2023, Docker Compose)    | |
  |  |   ui :8501 --> api :8000 --> redis (internal)      | |
  |  |   Security group: only 8501 and 8000 are open      | |
  |  |   IAM role: pull images, read ONE secret, SSM      | |
  |  +--------------------|-------------------------------+ |
  |                       | mounted at /opt/app/data        |
  |              [ EBS data disk, encrypted ]               |
  +---------------------------------------------------------+

  Outside the VPC:  ECR (image registry)   Secrets Manager (API keys)
```

**Why one server with Docker Compose?** It is the same setup the project already uses locally, so nothing
changes in the application. It is simple and cheap. If the project grows, the same modules can be replaced
by ECS or Kubernetes later.

## 3. Project structure

```
terraform/
├── providers.tf              Terraform and provider versions, offline mode
├── variables.tf              All inputs, with descriptions, defaults and validation
├── main.tf                   Connects the three modules
├── outputs.tf                URLs, ids and helper commands
├── terraform.tfvars.example  Example values (copy to terraform.tfvars)
└── modules/
    ├── network/   VPC, subnet, internet gateway, route table, security group
    ├── storage/   Image registry (ECR), secret container, persistent data disk
    └── compute/   IAM role, EC2 server, disk attachment, Elastic IP, start-up script
```

| Module | Resources |
|---|---|
| network | `aws_vpc`, `aws_internet_gateway`, `aws_subnet`, `aws_route_table`, `aws_route_table_association`, `aws_security_group` |
| storage | `aws_ecr_repository`, `aws_ecr_lifecycle_policy`, `aws_secretsmanager_secret`, `aws_ebs_volume` |
| compute | `aws_iam_role`, 2 x `aws_iam_role_policy_attachment`, `aws_iam_role_policy`, `aws_iam_instance_profile`, `aws_instance`, `aws_volume_attachment`, `aws_eip` |

Total: 18 resources.

## 4. Variables

All variables have defaults, so the plan works with no input at all.

| Variable | Default | Meaning |
|---|---|---|
| `project_name` | `multi-agent-rag` | Prefix of all resource names |
| `environment` | `dev` | `dev`, `staging` or `prod` |
| `aws_region` | `us-east-1` | Region |
| `availability_zone` | `null` (= region + `a`) | Zone of subnet, server and disk |
| `offline_mode` | `true` | `true` = no AWS account needed. `false` = real deployment |
| `vpc_cidr` / `public_subnet_cidr` | `10.0.0.0/16` / `10.0.1.0/24` | Network ranges |
| `allowed_app_cidrs` | `["0.0.0.0/0"]` | Who can open ports 8501 and 8000 (restrict it in real use) |
| `allowed_ssh_cidrs` | `[]` | Who can use SSH. Empty = closed, use Session Manager |
| `ami_id` | placeholder | Amazon Linux 2023 image of your region (see section 7) |
| `instance_type` | `t3.xlarge` | 4 CPU, 16 GB RAM (the AI models need memory) |
| `key_pair_name` | `null` | Optional SSH key pair |
| `root_volume_size_gb` / `data_volume_size_gb` | `40` / `60` | Disk sizes |
| `container_image_tag` | `latest` | Image tag to run |
| `app_environment` | see `variables.tf` | Non-secret app settings (same names as `.env.example`) |
| `tags` | `{}` | Extra tags |

Invalid values are rejected with a clear message (wrong CIDR, wrong AMI format, too small disks, ...).

## 5. Outputs

`ui_url`, `api_url`, `server_public_ip`, `server_instance_id`, `session_manager_command`,
`ecr_repository_url`, `secret_name`, `data_volume_id`, `vpc_id`, `image_push_commands`.

## 6. How to initialize, validate and plan (no cloud account)

1. Install Terraform 1.5 or newer (Windows: `winget install HashiCorp.Terraform`, then open a new terminal).
2. Run from this folder:

```
terraform init
terraform fmt -recursive
terraform validate
terraform plan
```

- `init` downloads the AWS provider from the internet (this does not need an AWS account).
- `fmt` fixes the formatting.
- `validate` must print `Success! The configuration is valid.`
- `plan` must end with `Plan: 18 to add, 0 to change, 0 to destroy.`

**Why does `plan` work without an AWS account?** In `providers.tf`, `offline_mode = true` (the default) gives the
provider fake credentials and turns off the checks that contact AWS. The configuration also avoids anything that
must read from AWS (no AMI lookups, no account lookups). Do NOT run `terraform apply`.

Optional: `copy terraform.tfvars.example terraform.tfvars` and change values.

## 7. Going to production later (not part of this task)

1. Set `offline_mode = false` and configure AWS credentials (`aws configure`).
2. Find the real AMI and set `ami_id`:
   `aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 --query Parameter.Value --output text`
3. Restrict `allowed_app_cidrs` to your own IP range.
4. Use a remote state backend (S3 + DynamoDB lock) and keep `.terraform.lock.hcl` in git.
5. `terraform apply`, then push the image with the `image_push_commands` output.
6. Store the secrets (they are never written in Terraform files or state):
   `aws secretsmanager put-secret-value --secret-id <secret_name> --secret-string '{"OPENROUTER_API_KEY":"...","API_KEY":"..."}'`
7. Start the app: open a shell with the `session_manager_command` output and run `sudo /opt/app/deploy.sh`.

## 8. Security and design notes

- Redis is not exposed: only the other containers on the server can reach it.
- Secrets live in Secrets Manager. The server's IAM role can read only that one secret.
- All disks are encrypted. The server accepts only IMDSv2. SSH is closed by default (Session Manager is used).
- The data disk is separate from the server, so the vector database, documents and downloaded models survive
  when the server is replaced.
- Resource names and tags come from one prefix, so several environments can live side by side.
- A single server is not highly available. It is a deliberate choice for cost and simplicity.
