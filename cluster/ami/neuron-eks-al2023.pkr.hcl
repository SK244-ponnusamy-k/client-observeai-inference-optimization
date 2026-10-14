# ==============================================================================
# cluster/ami/neuron-eks-al2023.pkr.hcl
#
# Builds a custom EKS AL2023 AMI with Neuron driver 2.31 for Trainium instances.
# Based on the existing EKS Neuron AMI (v20261003) with an updated Neuron driver.
#
# Why: The latest AWS-published Neuron AMI in ap-southeast-4 ships driver 2.30.2
# which is incompatible with vLLM-Neuron sdk2.31 container images required for
# gpt-oss-20b inference. This AMI upgrades the driver to 2.31.
#
# Prerequisites:
#   - Packer installed: https://developer.hashicorp.com/packer/downloads
#   - AWS credentials with EC2 permissions
#   - VPC/subnet with internet access for package downloads
#
# Usage:
#   bash cluster/ami/build-neuron-ami.sh
# ==============================================================================

packer {
  required_plugins {
    amazon = {
      version = ">= 1.3.0"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

variable "aws_region" {
  type    = string
  default = "ap-southeast-4"
}

variable "eks_version" {
  type    = string
  default = "1.36"
}

variable "base_ami_id" {
  type        = string
  description = "Base EKS Neuron AMI ID (ap-southeast-4 v20261003)"
  default     = "ami-012051d1c93d39733"
}

variable "subnet_id" {
  type        = string
  description = "Subnet ID for the build instance (needs internet access - use public subnet)"
  default     = "subnet-03882e829605afe28"  # ap-southeast-4c PUBLIC subnet
}

variable "instance_type" {
  type    = string
  default = "m5.xlarge"
}

variable "neuron_driver_version" {
  type    = string
  default = "2.19"  # DKMS package version for Neuron SDK 2.31
}

locals {
  timestamp = formatdate("YYYYMMDD", timestamp())
  ami_name  = "amazon-eks-node-al2023-x86_64-neuron-${var.eks_version}-neuron-sdk231-v2-${local.timestamp}"
}

source "amazon-ebs" "neuron_ami" {
  region        = var.aws_region
  source_ami    = var.base_ami_id
  instance_type = var.instance_type
  subnet_id     = var.subnet_id

  ami_name        = local.ami_name
  ami_description = "EKS AL2023 AMI with Neuron driver 2.31 for trn2 instances (EKS ${var.eks_version})"

  ssh_username = "ec2-user"
  associate_public_ip_address = true

  launch_block_device_mappings {
    device_name           = "/dev/xvda"
    volume_size           = 30
    volume_type           = "gp3"
    delete_on_termination = true
  }

  tags = {
    Name                  = local.ami_name
    NeuronSDK             = "2.31"
    EKSVersion            = var.eks_version
    BaseAMI               = var.base_ami_id
    Project               = "observeai-inference-optimization"
    ManagedBy             = "packer"
  }
}

build {
  sources = ["source.amazon-ebs.neuron_ami"]

  # Step 1: Configure Neuron yum repository
  provisioner "shell" {
    inline = [
      "echo 'Configuring Neuron yum repository...'",
      "sudo tee /etc/yum.repos.d/neuron.repo > /dev/null << 'EOF'",
      "[neuron]",
      "name=Neuron YUM Repository",
      "baseurl=https://yum.repos.neuron.amazonaws.com",
      "enabled=1",
      "metadata_expire=0",
      "gpgcheck=0",
      "repo_gpgcheck=0",
      "EOF",
      "sudo dnf clean metadata --disablerepo='*' --enablerepo=neuron 2>/dev/null || true"
    ]
  }

  # Step 2: Update Neuron driver to 2.31-compatible version
  provisioner "shell" {
    inline = [
      "echo 'Checking current Neuron driver...'",
      "modinfo neuron 2>/dev/null | grep -E 'version|filename' || echo 'Driver info not available'",
      "",
      "echo 'Available Neuron driver versions:'",
      "sudo dnf list available aws-neuronx-dkms --enablerepo=neuron 2>/dev/null | grep neuronx || true",
      "",
      "echo 'Installing latest Neuron driver from Neuron repo...'",
      "sudo dnf install -y --allowerasing --enablerepo=neuron aws-neuronx-dkms 2>&1 | tail -20",
      "",
      "echo 'Verifying installation...'",
      "rpm -qa | grep neuron",
      "",
      "echo 'Rebuilding DKMS modules...'",
      "sudo modprobe -r neuron 2>/dev/null || true",
      "sudo modprobe neuron 2>/dev/null && echo 'Neuron module loaded OK' || echo 'Module will load on reboot'"
    ]
  }

  # Step 3: Verify driver version
  provisioner "shell" {
    inline = [
      "echo 'Final verification...'",
      "cat /proc/driver/neuron/version 2>/dev/null || echo 'Driver version file not yet available (normal before reboot)'",
      "rpm -q aws-neuronx-dkms 2>/dev/null || true",
      "echo 'Build complete.'"
    ]
  }
}
