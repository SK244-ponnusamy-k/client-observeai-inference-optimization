# ==============================================================================
# cluster/ami/g7-eks-al2023.pkr.hcl
#
# Packer template to build an EKS-optimized AL2023 GPU AMI with NVIDIA Driver 595+
# required by EC2 G7 / G7e (NVIDIA Blackwell) instances.
#
# Usage:
#   packer build cluster/ami/g7-eks-al2023.pkr.hcl
# ==============================================================================

packer {
  required_plugins {
    amazon = {
      version = ">= 1.2.0"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

variable "aws_region" {
  type    = string
  default = "us-east-2"
}

variable "eks_version" {
  type    = string
  default = "1.36"
}

variable "nvidia_driver_version" {
  type    = string
  default = "595.45"
}

source "amazon-ebs" "al2023_g7_ami" {
  ami_name      = "eks-al2023-nvidia-595-${var.eks_version}-{{timestamp}}"
  instance_type = "c6i.2xlarge"
  region        = var.aws_region

  source_ami_filter {
    filters = {
      name                = "amazon-eks-node-al2023-x86_64-standard-${var.eks_version}-*"
      root-device-type    = "ebs"
      virtualization-type = "hvm"
    }
    most_recent = true
    owners      = ["602401143452"] # AWS EKS official AMI owner
  }

  ssh_username = "ec2-user"

  launch_block_device_mappings {
    device_name           = "/dev/xvda"
    volume_size           = 50
    volume_type           = "gp3"
    delete_on_termination = true
  }

  tags = {
    Name        = "eks-al2023-nvidia-595-${var.eks_version}"
    Project     = "observeai-inference-optimization"
    Engagement  = "shellkode-sow"
    Environment = "sandbox"
    Driver      = "NVIDIA-595"
  }
}

build {
  name    = "build-g7-ami"
  sources = ["source.amazon-ebs.al2023_g7_ami"]

  provisioner "shell" {
    inline = [
      # Step 1: Update the OS. This may upgrade the kernel package, but the
      # instance is still running on the pre-update kernel until a reboot.
      "sudo dnf update -y --allowerasing",

      # Step 2: Enable the AWS-maintained AL2023 NVIDIA repository.
      # 'nvidia-release' installs the repo config + GPG keys and is the
      # recommended approach per the AL2023 NVIDIA driver docs. It mirrors
      # the official NVIDIA CUDA repo but is qualified and maintained by AWS.
      "echo 'Enabling AWS AL2023 NVIDIA repository...'",
      "sudo dnf install -y nvidia-release",
      "sudo dnf clean all",

      # Step 3: Install build dependencies.
      # Do NOT use kernel-devel-$(uname -r) here — after dnf update the running
      # kernel version may no longer be available in the repos (superseded).
      # Install 'kernel-devel' without a version pin so dnf resolves the headers
      # that match the kernel it just installed (not the currently-running one).
      # DKMS will compile against the installed kernel headers on first boot.
      "sudo dnf install -y --allowerasing gcc make tar wget dkms kernel-devel",

      # Step 4: Install the NVIDIA driver via the AL2023 NVIDIA repo.
      # 'nvidia-driver-cuda' is the correct package name for AL2023 — the
      # 'nvidia-driver-latest-dkms' name from the raw CUDA developer repo
      # does not exist in this repository.
      "echo 'Installing NVIDIA Driver 595+ from AL2023 NVIDIA repository...'",
      "sudo dnf install -y --allowerasing nvidia-driver-cuda",
      "sudo systemctl enable nvidia-persistenced"
    ]
  }
}
