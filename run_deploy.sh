#!/bin/bash
set -e

echo "Starting deployment..."

# Source gcloud
source ~/Dev/nanochat-1/google-cloud-sdk/path.zsh.inc

# Set environment
export GCP_PROJECT_ID=lofty-outcome-190515
export NANOCHAT_BASE_DIR=/Users/richardhsu/Desktop/nanochat-artifacts

# Run Python script with unbuffered output
python3 -u deployment/deploy_to_cloud_run.py \
    --project-id $GCP_PROJECT_ID \
    --base-dir $NANOCHAT_BASE_DIR