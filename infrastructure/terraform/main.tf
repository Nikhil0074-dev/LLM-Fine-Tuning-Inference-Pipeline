# Reference infrastructure for the cloud path. NOT applied or validated by the author:
# run `terraform init && terraform validate && terraform plan` and read the plan before applying.
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws     = { source = "hashicorp/aws", version = "~> 5.0" }
    archive = { source = "hashicorp/archive", version = "~> 2.4" }
  }
}

provider "aws" { region = var.region }

variable "region" { default = "us-east-1" }
variable "name" { default = "llmp" }
variable "bucket_name" { type = string }
variable "training_image" { type = string }
variable "api_url" { type = string }
variable "internal_api_key" {
  type      = string
  sensitive = true
}
variable "s3_prefix" { default = "llm-pipeline" }

data "aws_caller_identity" "me" {}

# ---------- storage ----------
resource "aws_s3_bucket" "data" { bucket = var.bucket_name }

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

# ---------- SageMaker training role (only this bucket + ECR pull + logs) ----------
data "aws_iam_policy_document" "sm_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker" {
  name               = "${var.name}-sagemaker"
  assume_role_policy = data.aws_iam_policy_document.sm_assume.json
}

data "aws_iam_policy_document" "sm_access" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/${var.s3_prefix}/*"]
  }
  statement {
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    actions   = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
    resources = ["arn:aws:ecr:${var.region}:${data.aws_caller_identity.me.account_id}:repository/*"]
  }
  statement {
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${data.aws_caller_identity.me.account_id}:log-group:/aws/sagemaker/*"]
  }
}

resource "aws_iam_role_policy" "sagemaker" {
  role   = aws_iam_role.sagemaker.id
  policy = data.aws_iam_policy_document.sm_access.json
}

# ---------- callback Lambda ----------
data "archive_file" "lambda" {
  type        = "zip"
  source_dir  = "${path.module}/../../lambda_functions"
  output_path = "${path.module}/callback.zip"
}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${var.name}-callback"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_s3" {
  role = aws_iam_role.lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = ["s3:GetObject"], Resource = "${aws_s3_bucket.data.arn}/${var.s3_prefix}/*" }]
  })
}

resource "aws_lambda_function" "callback" {
  function_name    = "${var.name}-callback"
  role             = aws_iam_role.lambda.arn
  runtime          = "python3.12"
  handler          = "callback.handler"
  filename         = data.archive_file.lambda.output_path
  source_code_hash = data.archive_file.lambda.output_base64sha256
  timeout          = 30
  environment {
    variables = {
      API_URL          = var.api_url
      INTERNAL_API_KEY = var.internal_api_key # consider Secrets Manager for production
      S3_BUCKET        = aws_s3_bucket.data.bucket
      S3_PREFIX        = var.s3_prefix
    }
  }
}

# ---------- state machine ----------
data "aws_iam_policy_document" "sfn_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sfn" {
  name               = "${var.name}-sfn"
  assume_role_policy = data.aws_iam_policy_document.sfn_assume.json
}

resource "aws_iam_role_policy" "sfn" {
  role = aws_iam_role.sfn.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["lambda:InvokeFunction"], Resource = aws_lambda_function.callback.arn },
      { Effect = "Allow", Action = ["sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob", "sagemaker:StopTrainingJob", "sagemaker:AddTags"], Resource = "arn:aws:sagemaker:${var.region}:${data.aws_caller_identity.me.account_id}:training-job/*" },
      { Effect = "Allow", Action = ["iam:PassRole"], Resource = aws_iam_role.sagemaker.arn },
      { Effect = "Allow", Action = ["events:PutTargets", "events:PutRule", "events:DescribeRule"], Resource = "arn:aws:events:${var.region}:${data.aws_caller_identity.me.account_id}:rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule" }
    ]
  })
}

resource "aws_sfn_state_machine" "training" {
  name     = "${var.name}-training"
  role_arn = aws_iam_role.sfn.arn
  definition = templatefile("${path.module}/../../workflows/training_state_machine.asl.json", {
    CALLBACK_LAMBDA_ARN = aws_lambda_function.callback.arn
    SAGEMAKER_ROLE_ARN  = aws_iam_role.sagemaker.arn
    TRAINING_IMAGE      = var.training_image
  })
}

# ---------- permissions the API runtime (ECS task / App Runner) needs ----------
data "aws_iam_policy_document" "api_runtime" {
  statement {
    actions   = ["states:StartExecution", "states:StopExecution"]
    resources = [aws_sfn_state_machine.training.arn, "arn:aws:states:${var.region}:${data.aws_caller_identity.me.account_id}:execution:${aws_sfn_state_machine.training.name}:*"]
  }
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/${var.s3_prefix}/*"]
  }
}

resource "aws_iam_policy" "api_runtime" {
  name   = "${var.name}-api-runtime"
  policy = data.aws_iam_policy_document.api_runtime.json
}

output "state_machine_arn" { value = aws_sfn_state_machine.training.arn }
output "bucket" { value = aws_s3_bucket.data.bucket }
output "api_runtime_policy_arn" { value = aws_iam_policy.api_runtime.arn }
