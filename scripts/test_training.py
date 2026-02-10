#!/usr/bin/env python3
"""
Quick smoke test for XLM-RoBERTa training.
Runs a few forward+backward passes on a tiny dataset to verify
training works before committing to the full 45-minute run.

Usage: python scripts/test_training.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd

print("=== Environment ===")
print(f"Python:       {sys.version}")
print(f"PyTorch:      {torch.__version__}")
print(f"NumPy:        {np.__version__}")
print(f"CUDA:         {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:          {torch.cuda.get_device_name(0)}")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except ImportError:
    print("Transformers: NOT INSTALLED")
    sys.exit(1)

from src.models.xlm_roberta import XLMRobertaTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load just 20 samples
data_dir = project_root / 'data' / 'processed'
train_df = pd.read_json(data_dir / 'frenk_train.jsonl', lines=True).head(20)

print(f"\n=== Test Data ===")
print(f"Samples: {len(train_df)}")
print(f"Labels:  {train_df['label'].value_counts().to_dict()}")

# Create trainer with small config
trainer = XLMRobertaTrainer(
    model_name='xlm-roberta-base',
    num_labels=2,
    learning_rate=2e-5,
    batch_size=4,
    max_length=64,
    num_epochs=1,
    device=device,
)

# Prepare a single batch manually
train_loader, _ = trainer.prepare_data(
    train_texts=train_df['text'].tolist(),
    train_labels=train_df['label'].tolist(),
)

print(f"\n=== Running 3 forward+backward passes ===")
trainer.model.train()
optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=2e-5)

for i, batch in enumerate(train_loader):
    if i >= 3:
        break

    optimizer.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    print(f"\nBatch {i+1}:")
    print(f"  input_ids shape:      {input_ids.shape}")
    print(f"  attention_mask shape:  {attention_mask.shape}")
    print(f"  labels shape:          {labels.shape}")
    print(f"  labels values:         {labels.tolist()}")

    outputs = trainer.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs['loss']

    print(f"  logits shape:          {outputs['logits'].shape}")
    print(f"  loss shape:            {loss.shape}")
    print(f"  loss value:            {loss.item():.4f}")
    print(f"  loss requires_grad:    {loss.requires_grad}")

    loss.backward()
    optimizer.step()
    print(f"  backward + step:       OK")

print("\n=== ALL PASSES SUCCEEDED ===")
print("Safe to run: python scripts/train_xlm_roberta.py")
