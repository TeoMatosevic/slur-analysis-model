"""
XLM-RoBERTa-based classifier for Croatian hate speech detection.
Fine-tunes the xlm-roberta-base model for binary classification.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

XLM_ROBERTA_MODEL = "xlm-roberta-base"


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * torch.pow(1.0 - pt, self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class HateSpeechDataset(Dataset):
    """PyTorch dataset for hate speech classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_length,
            padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(int(label), dtype=torch.long)
        }


class XLMRobertaClassifier(nn.Module):
    """XLM-RoBERTa-based classifier with custom classification head."""

    def __init__(self, model_name: str = XLM_ROBERTA_MODEL, num_labels: int = 2,
                 dropout: float = 0.1, freeze_bert: bool = False):
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers not available.")
        try:
            self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        if freeze_bert:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        result = {'logits': logits}
        if labels is not None:
            result['loss'] = self.loss_fn(logits, labels)
        return result


class XLMRobertaTrainer:
    """Trainer class for XLM-RoBERTa hate speech classifier."""

    def __init__(self, model_name: str = XLM_ROBERTA_MODEL, num_labels: int = 2,
                 learning_rate: float = 2e-5, batch_size: int = 16, max_length: int = 256,
                 num_epochs: int = 5, warmup_ratio: float = 0.1, weight_decay: float = 0.01,
                 device: Optional[str] = None):
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = XLMRobertaClassifier(
            model_name=model_name, num_labels=num_labels
        ).to(self.device)
        self.label_encoder = LabelEncoder()
        self.label_names = None

    def prepare_data(self, train_texts: List[str], train_labels: List[str],
                     val_texts: Optional[List[str]] = None,
                     val_labels: Optional[List[str]] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
        train_labels_encoded = self.label_encoder.fit_transform(train_labels)
        self.label_names = list(self.label_encoder.classes_)
        train_dataset = HateSpeechDataset(train_texts, train_labels_encoded, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_labels_encoded = self.label_encoder.transform(val_labels)
            val_dataset = HateSpeechDataset(val_texts, val_labels_encoded, self.tokenizer, self.max_length)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def train(self, train_texts: List[str], train_labels: List[str],
              val_texts: Optional[List[str]] = None, val_labels: Optional[List[str]] = None,
              output_dir: str = 'checkpoints/xlm_roberta') -> Dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_loader, val_loader = self.prepare_data(train_texts, train_labels, val_texts, val_labels)
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'best_val_f1': 0.0, 'best_epoch': 0}
        best_val_f1 = 0.0

        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            history['train_loss'].append(train_loss)
            logger.info(f"  Train Loss: {train_loss:.4f}")

            if val_loader is not None:
                val_loss, val_metrics = self._evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_f1'].append(val_metrics['f1_macro'])
                logger.info(f"  Val Loss: {val_loss:.4f}")
                logger.info(f"  Val F1 Macro: {val_metrics['f1_macro']:.4f}")
                if val_metrics['f1_macro'] > best_val_f1:
                    best_val_f1 = val_metrics['f1_macro']
                    history['best_val_f1'] = best_val_f1
                    history['best_epoch'] = epoch + 1
                    self.save(output_dir / 'best_model')
                    logger.info(f"  New best model saved! (F1: {best_val_f1:.4f})")

        self.save(output_dir / 'final_model')
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        return history

    def _train_epoch(self, train_loader: DataLoader, optimizer, scheduler) -> float:
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix({'loss': loss.item()})
        return total_loss / len(train_loader)

    def _evaluate(self, data_loader: DataLoader) -> Tuple[float, Dict]:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs['loss'].item()
                preds = torch.argmax(outputs['logits'], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
            'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
            'mcc': matthews_corrcoef(all_labels, all_preds),
        }
        return total_loss / len(data_loader), metrics

    def predict(self, texts: List[str]) -> np.ndarray:
        self.model.eval()
        predictions = []
        dataset = HateSpeechDataset(texts, [0] * len(texts), self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs['logits'], dim=1)
                predictions.extend(preds.cpu().numpy())
        return self.label_encoder.inverse_transform(predictions)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get prediction probabilities for ROC curves."""
        self.model.eval()
        all_probs = []
        dataset = HateSpeechDataset(texts, [0] * len(texts), self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs['logits'], dim=1)
                all_probs.extend(probs.cpu().numpy())
        return np.array(all_probs)

    def evaluate(self, texts: List[str], labels: List[str]) -> Dict:
        labels_encoded = self.label_encoder.transform(labels)
        dataset = HateSpeechDataset(texts, labels_encoded, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        _, metrics = self._evaluate(loader)
        predictions = self.predict(texts)
        preds_encoded = self.label_encoder.transform(predictions)
        report = classification_report(
            labels_encoded, preds_encoded,
            target_names=self.label_names, output_dict=True
        )
        metrics['classification_report'] = report
        metrics['predictions'] = predictions.tolist()
        return metrics

    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / 'model.pt')
        config = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'label_names': self.label_names,
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        self.tokenizer.save_pretrained(path / 'tokenizer')
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        path = Path(path)
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)
        self.model_name = config['model_name']
        self.num_labels = config['num_labels']
        self.max_length = config['max_length']
        self.label_names = config['label_names']
        self.label_encoder.fit(self.label_names)
        self.tokenizer = AutoTokenizer.from_pretrained(path / 'tokenizer')
        self.model = XLMRobertaClassifier(
            model_name=self.model_name, num_labels=self.num_labels
        ).to(self.device)
        self.model.load_state_dict(torch.load(path / 'model.pt', map_location=self.device, weights_only=False))
        logger.info(f"Model loaded from {path}")
