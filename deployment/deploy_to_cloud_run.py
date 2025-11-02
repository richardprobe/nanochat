#!/usr/bin/env python3
"""
Deploy NanoChat web application to Google Cloud Run.

Usage:
    python deployment/deploy_to_cloud_run.py --project-id YOUR_PROJECT_ID
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")

    # Ensure gcloud is in PATH
    env = os.environ.copy()
    gcloud_path = os.path.expanduser("~/Dev/nanochat-1/google-cloud-sdk/bin")
    if gcloud_path not in env.get("PATH", ""):
        env["PATH"] = f"{gcloud_path}:{env.get('PATH', '')}"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result


def deploy_to_cloud_run(project_id, region="us-central1", service_name="nanochat-web", base_dir=None):
    """
    Build and deploy NanoChat web application to Cloud Run.

    Args:
        project_id: Google Cloud project ID
        region: GCP region for deployment
        service_name: Cloud Run service name
        base_dir: Base directory for nanochat cache
    """
    if base_dir is None:
        base_dir = os.path.expanduser("~/.cache/nanochat")

    # Check if model artifacts exist - try multiple possible locations
    possible_model_paths = [
        os.path.join(base_dir, "sft", "model.pth"),
        os.path.join(base_dir, "chatsft_checkpoints", "d20", "model_000650.pt"),
        os.path.join(base_dir, "chatsft_checkpoints", "d20", "model.pt"),
    ]

    sft_model = None
    for path in possible_model_paths:
        if os.path.exists(path):
            sft_model = path
            break

    if not sft_model:
        print(f"Error: Model not found in any expected location:")
        for path in possible_model_paths:
            print(f"  - {path}")
        print("Please run the training pipeline first (speedrun.sh)")
        return False

    print(f"Found model at: {sft_model}")

    # Set up paths
    project_root = Path(__file__).parent.parent
    dockerfile_path = project_root / "deployment" / "Dockerfile"

    # Configure Docker for Artifact Registry
    print("\n1. Configuring Docker for Google Artifact Registry...")
    run_command(f"gcloud auth configure-docker us-central1-docker.pkg.dev --quiet")

    # Create Artifact Registry repository if it doesn't exist
    print("\n   Ensuring Artifact Registry repository exists...")
    create_repo_cmd = f"""
    gcloud artifacts repositories create nanochat \
        --repository-format=docker \
        --location=us-central1 \
        --project={project_id} \
        --quiet 2>/dev/null || echo "Repository already exists"
    """
    run_command(create_repo_cmd, check=False)

    # Build Docker image with model artifacts
    image_tag = f"us-central1-docker.pkg.dev/{project_id}/nanochat/{service_name}:latest"

    print(f"\n2. Building Docker image: {image_tag}")
    print("   This will take several minutes...")

    # Create a temporary directory for model artifacts
    temp_model_dir = project_root / "temp_model_artifacts"
    temp_model_dir.mkdir(exist_ok=True)

    try:
        # Copy model artifacts to temp directory
        print("   Copying model artifacts...")

        # Handle different model locations
        if "chatsft_checkpoints" in sft_model:
            # Copy from chatsft_checkpoints structure
            (temp_model_dir / "sft").mkdir(exist_ok=True)
            shutil.copy(sft_model, temp_model_dir / "sft" / "model.pth")

            # Also copy meta file if exists
            meta_path = sft_model.replace("model_000650.pt", "meta_000650.json").replace("model.pt", "meta.json")
            if os.path.exists(meta_path):
                shutil.copy(meta_path, temp_model_dir / "sft" / "meta.json")
        else:
            # Copy from standard sft directory
            shutil.copytree(
                os.path.join(base_dir, "sft"),
                temp_model_dir / "sft",
                dirs_exist_ok=True
            )

        tokenizer_path = os.path.join(base_dir, "tokenizer", "tok65536.model")
        if os.path.exists(tokenizer_path):
            (temp_model_dir / "tokenizer").mkdir(exist_ok=True)
            shutil.copy(tokenizer_path, temp_model_dir / "tokenizer" / "tok65536.model")

        # Build Docker image for x86_64 platform (Cloud Run architecture)
        build_cmd = f"""
        cd {project_root} && \
        docker buildx build \
            --platform linux/amd64 \
            -f {dockerfile_path} \
            -t {image_tag} \
            --build-arg NANOCHAT_BASE_DIR=/app/model_cache \
            --load \
            .
        """
        run_command(build_cmd)

        # Copy model artifacts into the container
        print("\n3. Adding model artifacts to container...")
        container_name = "temp_nanochat_container"

        # Create a container from the image
        run_command(f"docker create --name {container_name} {image_tag}")

        # Copy model files into the container
        run_command(f"docker cp {temp_model_dir}/sft/. {container_name}:/app/model_cache/sft/")

        if (temp_model_dir / "tokenizer").exists():
            run_command(f"docker cp {temp_model_dir}/tokenizer/. {container_name}:/app/model_cache/tokenizer/")

        # Commit the container with model files to a new image
        run_command(f"docker commit {container_name} {image_tag}")

        # Clean up temporary container
        run_command(f"docker rm {container_name}", check=False)

    finally:
        # Clean up temporary directory
        if temp_model_dir.exists():
            shutil.rmtree(temp_model_dir)

    # Push image to Artifact Registry
    print(f"\n4. Pushing image to Google Artifact Registry...")
    run_command(f"docker push {image_tag}")

    # Deploy to Cloud Run
    print(f"\n5. Deploying to Cloud Run (region: {region})...")

    deploy_cmd = f"""
    gcloud run deploy {service_name} \
        --image {image_tag} \
        --platform managed \
        --region {region} \
        --allow-unauthenticated \
        --memory 4Gi \
        --cpu 2 \
        --timeout 3600 \
        --port 8080 \
        --project {project_id}
    """

    result = run_command(deploy_cmd)

    # Get the service URL
    print("\n6. Getting service URL...")
    url_cmd = f"gcloud run services describe {service_name} --region {region} --format 'value(status.url)' --project {project_id}"
    url_result = run_command(url_cmd)

    if url_result.stdout:
        print(f"\n✅ Deployment successful!")
        print(f"🌐 Your NanoChat web app is available at: {url_result.stdout.strip()}")
    else:
        print("\n✅ Deployment completed. Check Cloud Console for the service URL.")

    return True


def main():
    parser = argparse.ArgumentParser(description="Deploy NanoChat to Google Cloud Run")
    parser.add_argument(
        "--project-id",
        type=str,
        required=False,
        default=os.environ.get("GCP_PROJECT_ID"),
        help="Google Cloud project ID (or set GCP_PROJECT_ID environment variable)"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-central1",
        help="GCP region for deployment (default: us-central1)"
    )
    parser.add_argument(
        "--service-name",
        type=str,
        default="nanochat-web",
        help="Cloud Run service name (default: nanochat-web)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat")),
        help="Base directory for nanochat cache"
    )

    args = parser.parse_args()

    if not args.project_id:
        print("Error: Google Cloud project ID not provided.")
        print("Either pass --project-id or set GCP_PROJECT_ID environment variable")
        sys.exit(1)

    # Check if gcloud is installed and authenticated
    result = run_command("gcloud auth list --filter=status:ACTIVE --format='value(account)'", check=False)
    if not result.stdout:
        print("Error: Not authenticated with gcloud. Please run: gcloud auth login")
        sys.exit(1)

    print(f"Deploying as: {result.stdout.strip()}")

    # Check if Docker is installed
    result = run_command("docker --version", check=False)
    if result.returncode != 0:
        print("Error: Docker is not installed. Please install Docker Desktop.")
        sys.exit(1)

    success = deploy_to_cloud_run(
        args.project_id,
        args.region,
        args.service_name,
        args.base_dir
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()