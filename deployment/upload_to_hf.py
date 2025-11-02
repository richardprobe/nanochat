#!/usr/bin/env python3
"""
Upload trained NanoChat model to HuggingFace Hub.

Usage:
    python deployment/upload_to_hf.py --username YOUR_HF_USERNAME
"""

import os
import sys
import shutil
import argparse
from huggingface_hub import HfApi, create_repo


def upload_model_to_hf(username, repo_name="nanochat-speedrun", base_dir=None):
    """
    Upload the trained NanoChat model to HuggingFace Hub.

    Args:
        username: HuggingFace username
        repo_name: Repository name (default: nanochat-speedrun)
        base_dir: Base directory for nanochat cache (default: ~/.cache/nanochat)
    """
    if base_dir is None:
        base_dir = os.path.expanduser("~/.cache/nanochat")

    api = HfApi()
    repo_id = f"{username}/{repo_name}"

    print(f"Preparing to upload model to: https://huggingface.co/{repo_id}")

    # Create the repo if it doesn't exist
    try:
        create_repo(repo_id, exist_ok=True, repo_type="model")
        print(f"Repository {repo_id} ready")
    except Exception as e:
        print(f"Note: {e}")

    # Prepare model directory
    model_dir = os.path.join(base_dir, "export_hf")
    # Remove existing export directory to ensure clean copy
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    # Copy the entire chatsft_checkpoints directory
    chatsft_source = os.path.join(base_dir, "chatsft_checkpoints")
    if os.path.exists(chatsft_source):
        chatsft_dest = os.path.join(model_dir, "chatsft_checkpoints")
        print(f"Copying chatsft_checkpoints from {chatsft_source}")
        shutil.copytree(chatsft_source, chatsft_dest)
        print(f"✓ Copied chatsft_checkpoints directory")
    else:
        print(f"Warning: chatsft_checkpoints not found at {chatsft_source}")

    # Copy the entire tokenizer directory
    tokenizer_source = os.path.join(base_dir, "tokenizer")
    if os.path.exists(tokenizer_source):
        tokenizer_dest = os.path.join(model_dir, "tokenizer")
        print(f"Copying tokenizer from {tokenizer_source}")
        shutil.copytree(tokenizer_source, tokenizer_dest)
        print(f"✓ Copied tokenizer directory")
    else:
        print(f"Warning: tokenizer not found at {tokenizer_source}")

    # Check if at least one directory was copied
    if not os.path.exists(chatsft_source) and not os.path.exists(tokenizer_source):
        print(f"Error: Neither chatsft_checkpoints nor tokenizer found in {base_dir}")
        return False

    # Create a model card
    model_card_path = os.path.join(model_dir, "README.md")
    with open(model_card_path, "w") as f:
        f.write("""# NanoChat Speedrun Model

This model was trained using the NanoChat speedrun script.

## Model Details
- Architecture: d20 (561M parameters)
- Training Pipeline: Pretraining → Midtraining → SFT
- Tokenizer: Custom BPE with vocab size 65,536
- Training Time: ~4 hours on 8xH100 GPUs

## Usage

```python
# Load the model (example - adjust based on your setup)
import torch
model = torch.load('model.pth', map_location='cpu')
```

See [nanochat repository](https://github.com/karpathy/nanochat) for full usage instructions.

## Training Report
""")

        # Append the report if it exists
        report_path = os.path.join(base_dir, "report", "report.md")
        if os.path.exists(report_path):
            print(f"Appending training report from {report_path}")
            with open(report_path, "r") as report:
                f.write("\n")
                f.write(report.read())
        else:
            print("Training report not found, skipping...")

    print(f"Created model card at {model_card_path}")

    # Upload to HuggingFace
    print(f"Uploading to HuggingFace Hub...")
    try:
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"✓ Model successfully uploaded to: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"Error uploading to HuggingFace: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload NanoChat model to HuggingFace Hub")
    parser.add_argument(
        "--username",
        type=str,
        required=False,
        default=os.environ.get("HF_USERNAME"),
        help="HuggingFace username (or set HF_USERNAME environment variable)"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="nanochat-speedrun",
        help="Repository name on HuggingFace (default: nanochat-speedrun)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat")),
        help="Base directory for nanochat cache"
    )

    args = parser.parse_args()

    if not args.username:
        print("Error: HuggingFace username not provided.")
        print("Either pass --username or set HF_USERNAME environment variable")
        sys.exit(1)

    # Check for HF authentication
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Warning: No HuggingFace token found in environment.")
        print("Make sure you've run: huggingface-cli login")
        print("Or set HF_TOKEN environment variable")

    success = upload_model_to_hf(args.username, args.repo_name, args.base_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()