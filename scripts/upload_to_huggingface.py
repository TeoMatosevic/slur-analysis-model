#!/usr/bin/env python3
"""
Upload models to HuggingFace Hub.

Usage:
    1. Login first: huggingface-cli login
    2. Run: python scripts/upload_to_huggingface.py --username YOUR_HF_USERNAME
"""

import argparse
from huggingface_hub import HfApi, create_repo
from pathlib import Path


def upload_bertic_model(username: str):
    """Upload BERTić model to HuggingFace."""
    api = HfApi()

    repo_name = f"{username}/croatian-hate-speech-bertic"
    model_path = Path("checkpoints/bertic/best_model")

    print(f"Creating repository: {repo_name}")
    try:
        create_repo(repo_name, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Repository may already exist: {e}")

    print(f"Uploading model from {model_path}...")

    # Upload all files
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_name,
        repo_type="model"
    )

    print(f"\nSuccess! Model uploaded to: https://huggingface.co/{repo_name}")
    print(f"\nYour professor can now download it with:")
    print(f"  git clone https://huggingface.co/{repo_name}")
    print(f"\nOr in Python:")
    print(f"  from huggingface_hub import snapshot_download")
    print(f"  snapshot_download('{repo_name}', local_dir='checkpoints/bertic/best_model')")


def upload_baseline_model(username: str):
    """Upload baseline model to HuggingFace."""
    api = HfApi()

    repo_name = f"{username}/croatian-hate-speech-baseline"
    model_path = Path("checkpoints/baseline")

    if not model_path.exists():
        print("Baseline model not found at checkpoints/baseline/")
        return

    print(f"Creating repository: {repo_name}")
    try:
        create_repo(repo_name, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Repository may already exist: {e}")

    print(f"Uploading baseline model...")

    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_name,
        repo_type="model"
    )

    print(f"\nBaseline model uploaded to: https://huggingface.co/{repo_name}")


def upload_xlm_roberta_model(username: str):
    """Upload XLM-RoBERTa model to HuggingFace."""
    api = HfApi()

    repo_name = f"{username}/croatian-hate-speech-xlm-roberta"
    model_path = Path("checkpoints/xlm_roberta/best_model")

    if not model_path.exists():
        print("XLM-RoBERTa model not found at checkpoints/xlm_roberta/best_model/")
        return

    print(f"Creating repository: {repo_name}")
    try:
        create_repo(repo_name, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Repository may already exist: {e}")

    print(f"Uploading XLM-RoBERTa model from {model_path}...")

    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_name,
        repo_type="model"
    )

    print(f"\nXLM-RoBERTa model uploaded to: https://huggingface.co/{repo_name}")


def main():
    parser = argparse.ArgumentParser(description="Upload models to HuggingFace Hub")
    parser.add_argument("--username", "-u", required=True, help="Your HuggingFace username")
    parser.add_argument("--model", "-m", choices=["bertic", "baseline", "xlm_roberta", "all"],
                        default="all", help="Which model to upload")

    args = parser.parse_args()

    print("="*60)
    print("HuggingFace Model Upload")
    print("="*60)
    print(f"Username: {args.username}")
    print(f"Model: {args.model}")
    print()

    if args.model in ["bertic", "all"]:
        upload_bertic_model(args.username)

    if args.model in ["xlm_roberta", "all"]:
        upload_xlm_roberta_model(args.username)

    if args.model in ["baseline", "all"]:
        upload_baseline_model(args.username)

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
