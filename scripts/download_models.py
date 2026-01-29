#!/usr/bin/env python3
"""
Download pre-trained models from HuggingFace Hub.

Usage:
    python scripts/download_models.py

This will download:
    - BERTić hate speech model -> checkpoints/bertic/best_model/
    - Baseline model -> checkpoints/baseline/
"""

from huggingface_hub import snapshot_download
from pathlib import Path


# UPDATE THESE WITH YOUR ACTUAL HUGGINGFACE REPO NAMES
BERTIC_REPO = "TeoMatosevic/croatian-hate-speech-bertic"
BASELINE_REPO = "TeoMatosevic/croatian-hate-speech-baseline"


def download_bertic():
    """Download BERTić model."""
    print("Downloading BERTić model...")
    local_dir = Path("checkpoints/bertic/best_model")
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=BERTIC_REPO,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False
        )
        print(f"  Downloaded to: {local_dir}")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def download_baseline():
    """Download baseline model."""
    print("Downloading baseline model...")
    local_dir = Path("checkpoints/baseline")
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=BASELINE_REPO,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False
        )
        print(f"  Downloaded to: {local_dir}")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def main():
    print("="*60)
    print("Croatian Hate Speech Detection - Model Download")
    print("="*60)
    print()

    success_bertic = download_bertic()
    print()
    success_baseline = download_baseline()

    print()
    print("="*60)
    if success_bertic and success_baseline:
        print("All models downloaded successfully!")
        print("\nYou can now run:")
        print("  python src/demo.py --text 'Your text here'")
    else:
        print("Some downloads failed. Check your internet connection")
        print("and verify the repository names are correct.")
    print("="*60)


if __name__ == "__main__":
    main()
