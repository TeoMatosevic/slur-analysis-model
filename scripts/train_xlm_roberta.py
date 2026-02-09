#!/usr/bin/env python3
"""
Train XLM-RoBERTa model for Croatian hate speech detection.

INSTRUCTIONS FOR GPU TRAINING:
    1. Clone the repo:   git clone https://github.com/TeoMatosevic/slur-analysis-model.git
    2. Install deps:     pip install -r requirements.txt
    3. Run training:     python scripts/train_xlm_roberta.py

    With GPU (CUDA) this takes ~30-60 minutes.
    Without GPU (CPU only) this takes ~3-5 hours.

    The script auto-detects GPU and adjusts batch size accordingly.
    After training, the best model is saved to checkpoints/xlm_roberta/best_model/
    Push the checkpoint back to git or share the checkpoints/xlm_roberta/ folder.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
from src.models.xlm_roberta import XLMRobertaTrainer

def main():
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Config based on device
    if device == 'cuda':
        batch_size = 16
        max_length = 256
        num_epochs = 5
    else:
        batch_size = 8
        max_length = 128
        num_epochs = 3
        print("WARNING: Training on CPU will be slow (~3-5 hours).")

    # Load data
    data_dir = project_root / 'data' / 'processed'
    train_df = pd.read_json(data_dir / 'frenk_train.jsonl', lines=True)
    dev_df = pd.read_json(data_dir / 'frenk_dev.jsonl', lines=True)
    test_df = pd.read_json(data_dir / 'frenk_test.jsonl', lines=True)

    print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

    # Initialize trainer
    trainer = XLMRobertaTrainer(
        model_name='xlm-roberta-base',
        num_labels=2,
        learning_rate=2e-5,
        batch_size=batch_size,
        max_length=max_length,
        num_epochs=num_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        device=device,
    )

    # Train
    print("\nStarting training...")
    history = trainer.train(
        train_texts=train_df['text'].tolist(),
        train_labels=train_df['label'].tolist(),
        val_texts=dev_df['text'].tolist(),
        val_labels=dev_df['label'].tolist(),
        output_dir=str(project_root / 'checkpoints' / 'xlm_roberta'),
    )

    print(f"\nBest validation F1: {history['best_val_f1']:.4f} (epoch {history['best_epoch']})")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    trainer.load(str(project_root / 'checkpoints' / 'xlm_roberta' / 'best_model'))
    results = trainer.evaluate(test_df['text'].tolist(), test_df['label'].tolist())

    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(f"Accuracy:    {results['accuracy']:.4f}")
    print(f"F1-Macro:    {results['f1_macro']:.4f}")
    print(f"F1-Weighted: {results['f1_weighted']:.4f}")
    print(f"MCC:         {results['mcc']:.4f}")

    # Save test results
    results_path = project_root / 'checkpoints' / 'xlm_roberta' / 'test_results.json'
    results_to_save = {k: v for k, v in results.items() if k not in ['predictions', 'classification_report']}
    results_to_save['classification_report'] = results.get('classification_report', {})
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    print("Done! Push checkpoints/xlm_roberta/ back to git.")


if __name__ == '__main__':
    main()
