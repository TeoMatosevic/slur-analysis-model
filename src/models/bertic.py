"""
BERTić-based classifier for Croatian hate speech detection.
Fine-tunes the classla/bcms-bertic model for multi-label classification.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

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
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Default model names
BERTIC_MODEL = "classla/bcms-bertic"
BERTIC_HATE_MODEL = "classla/bcms-bertic-frenk-hate"


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (logits), shape (batch_size, num_classes)
            targets: Ground truth labels, shape (batch_size,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class HateSpeechDataset(Dataset):
    """PyTorch dataset for hate speech classification."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 256
    ):
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
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTicClassifier(nn.Module):
    """
    BERTić-based classifier with custom classification head.
    """

    def __init__(
        self,
        model_name: str = BERTIC_MODEL,
        num_labels: int = 5,
        dropout: float = 0.1,
        freeze_bert: bool = False
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers not available. Install with: pip install transformers")

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        result = {'logits': logits}

        if labels is not None:
            loss_fn = FocalLoss(gamma=2.0)
            result['loss'] = loss_fn(logits, labels)

        return result


class BERTicTrainer:
    """
    Trainer class for BERTić-based hate speech classifier.
    """

    def __init__(
        self,
        model_name: str = BERTIC_MODEL,
        num_labels: int = 5,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        max_length: int = 256,
        num_epochs: int = 5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        device: Optional[str] = None,
        use_focal_loss: bool = True
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.use_focal_loss = use_focal_loss

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize model
        self.model = BERTicClassifier(
            model_name=model_name,
            num_labels=num_labels
        ).to(self.device)

        # Label encoder
        self.label_encoder = LabelEncoder()
        self.label_names = None

    def prepare_data(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[str]] = None
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare DataLoaders for training.
        """
        # Encode labels
        train_labels_encoded = self.label_encoder.fit_transform(train_labels)
        self.label_names = list(self.label_encoder.classes_)

        # Create training dataset
        train_dataset = HateSpeechDataset(
            train_texts, train_labels_encoded,
            self.tokenizer, self.max_length
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Create validation dataset
        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_labels_encoded = self.label_encoder.transform(val_labels)
            val_dataset = HateSpeechDataset(
                val_texts, val_labels_encoded,
                self.tokenizer, self.max_length
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )

        return train_loader, val_loader

    def train(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[str]] = None,
        output_dir: str = 'checkpoints/bertic'
    ) -> Dict:
        """
        Train the model.

        Returns:
            Dictionary with training history
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data
        train_loader, val_loader = self.prepare_data(
            train_texts, train_labels, val_texts, val_labels
        )

        # Calculate total steps
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        # Initialize optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'best_val_f1': 0.0,
            'best_epoch': 0
        }

        best_val_f1 = 0.0

        # Training loop
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            history['train_loss'].append(train_loss)
            logger.info(f"  Train Loss: {train_loss:.4f}")

            # Validate
            if val_loader is not None:
                val_loss, val_metrics = self._evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_f1'].append(val_metrics['f1_macro'])

                logger.info(f"  Val Loss: {val_loss:.4f}")
                logger.info(f"  Val F1 Macro: {val_metrics['f1_macro']:.4f}")

                # Save best model
                if val_metrics['f1_macro'] > best_val_f1:
                    best_val_f1 = val_metrics['f1_macro']
                    history['best_val_f1'] = best_val_f1
                    history['best_epoch'] = epoch + 1
                    self.save(output_dir / 'best_model')
                    logger.info(f"  New best model saved! (F1: {best_val_f1:.4f})")

        # Save final model
        self.save(output_dir / 'final_model')

        # Save training history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        return history

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer,
        scheduler
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs['loss']
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / len(train_loader)

    def _evaluate(self, data_loader: DataLoader) -> Tuple[float, Dict]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs['loss'].item()
                preds = torch.argmax(outputs['logits'], dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
            'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
            'mcc': matthews_corrcoef(all_labels, all_preds),
        }

        avg_loss = total_loss / len(data_loader)
        return avg_loss, metrics

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict labels for texts.
        """
        self.model.eval()
        predictions = []

        dataset = HateSpeechDataset(
            texts, [0] * len(texts),  # Dummy labels
            self.tokenizer, self.max_length
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs['logits'], dim=1)
                predictions.extend(preds.cpu().numpy())

        return self.label_encoder.inverse_transform(predictions)

    def evaluate(self, texts: List[str], labels: List[str]) -> Dict:
        """
        Full evaluation with classification report.
        """
        labels_encoded = self.label_encoder.transform(labels)
        dataset = HateSpeechDataset(
            texts, labels_encoded,
            self.tokenizer, self.max_length
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        _, metrics = self._evaluate(loader)

        # Get predictions for detailed report
        predictions = self.predict(texts)
        preds_encoded = self.label_encoder.transform(predictions)

        # Classification report
        report = classification_report(
            labels_encoded, preds_encoded,
            target_names=self.label_names,
            output_dict=True
        )
        metrics['classification_report'] = report

        return metrics

    def save(self, path: str):
        """Save model and configuration."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(self.model.state_dict(), path / 'model.pt')

        # Save config
        config = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'label_names': self.label_names,
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Save tokenizer
        self.tokenizer.save_pretrained(path / 'tokenizer')

        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from checkpoint."""
        path = Path(path)

        # Load config
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)

        self.model_name = config['model_name']
        self.num_labels = config['num_labels']
        self.max_length = config['max_length']
        self.label_names = config['label_names']

        # Rebuild label encoder
        self.label_encoder.fit(self.label_names)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path / 'tokenizer')

        # Load model
        self.model = BERTicClassifier(
            model_name=self.model_name,
            num_labels=self.num_labels
        ).to(self.device)
        self.model.load_state_dict(torch.load(path / 'model.pt', map_location=self.device))

        logger.info(f"Model loaded from {path}")


def main():
    """Main entry point for BERTić training."""
    parser = argparse.ArgumentParser(description="Train BERTić hate speech classifier")
    parser.add_argument('--data', type=str, help="Path to training data (CSV or JSONL)")
    parser.add_argument('--text-column', type=str, default='text')
    parser.add_argument('--label-column', type=str, default='label')
    parser.add_argument('--model', type=str, default=BERTIC_MODEL,
                        help=f"Model name (default: {BERTIC_MODEL})")
    parser.add_argument('--output', type=str, default='checkpoints/bertic')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--test-split', type=float, default=0.15)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--test', action='store_true', help="Run test mode")

    args = parser.parse_args()

    if args.test:
        logger.info("Running in test mode")
        print("Test mode: BERTić module loaded successfully")
        print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if TRANSFORMERS_AVAILABLE:
            print(f"\nTesting tokenizer loading from {BERTIC_MODEL}...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(BERTIC_MODEL)
                test_text = "Ovo je testna rečenica na hrvatskom jeziku."
                tokens = tokenizer(test_text)
                print(f"Input: {test_text}")
                print(f"Tokens: {tokens['input_ids'][:10]}...")
                print("Tokenizer loaded successfully!")
            except Exception as e:
                print(f"Tokenizer loading failed: {e}")
                print("You may need to run: pip install transformers sentencepiece")

        return

    if not args.data:
        parser.error("--data is required (unless --test is specified)")

    # Load data
    data_path = Path(args.data)
    if data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
    else:
        df = pd.read_json(data_path, lines=data_path.suffix == '.jsonl')

    texts = df[args.text_column].tolist()
    labels = df[args.label_column].tolist()

    # Split data
    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=args.test_split, random_state=42, stratify=labels
    )

    # Initialize trainer
    trainer = BERTicTrainer(
        model_name=args.model,
        num_labels=len(set(labels)),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_epochs=args.epochs,
        device=args.device
    )

    # Train
    history = trainer.train(
        train_texts, train_labels,
        val_texts, val_labels,
        output_dir=args.output
    )

    # Final evaluation
    logger.info("\nFinal evaluation on validation set:")
    metrics = trainer.evaluate(val_texts, val_labels)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
