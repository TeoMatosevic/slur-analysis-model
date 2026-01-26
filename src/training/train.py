"""
Main training script for Croatian hate speech detection models.
Supports baseline (TF-IDF + ML) and BERTić models.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

# Import models
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.baseline import BaselineClassifier, train_baseline_models
from src.models.bertic import BERTicTrainer, BERTIC_MODEL
from src.utils.lexicon import CodedTermLexicon, create_lexicon_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(
    data_path: str,
    text_column: str = 'text',
    label_column: str = 'label'
) -> Tuple[List[str], List[str]]:
    """
    Load data from file.

    Args:
        data_path: Path to data file (CSV or JSONL)
        text_column: Name of text column
        label_column: Name of label column

    Returns:
        Tuple of (texts, labels)
    """
    path = Path(data_path)
    if path.suffix == '.csv':
        df = pd.read_csv(path)
    elif path.suffix in ['.jsonl', '.json']:
        df = pd.read_json(path, lines=path.suffix == '.jsonl')
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    texts = df[text_column].tolist()
    labels = df[label_column].tolist()

    logger.info(f"Loaded {len(texts)} samples from {path}")
    logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    return texts, labels


def split_data(
    texts: List[str],
    labels: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[List, List, List, List, List, List]:
    """
    Split data into train/val/test sets.

    Returns:
        Tuple of (train_texts, val_texts, test_texts, train_labels, val_labels, test_labels)
    """
    # First split: separate test set
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels,
        test_size=test_ratio,
        random_state=random_state,
        stratify=labels
    )

    # Second split: separate train and validation
    val_size = val_ratio / (train_ratio + val_ratio)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val_labels
    )

    logger.info(f"Data split: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels


def train_baseline(
    train_texts: List[str],
    train_labels: List[str],
    val_texts: List[str],
    val_labels: List[str],
    config: Dict,
    output_dir: str
) -> Dict:
    """Train baseline models."""
    logger.info("Training baseline models...")

    output_dir = Path(output_dir) / 'baseline'
    output_dir.mkdir(parents=True, exist_ok=True)

    results = train_baseline_models(
        train_texts, train_labels,
        val_texts, val_labels,
        output_dir=str(output_dir)
    )

    return results


def train_bertic(
    train_texts: List[str],
    train_labels: List[str],
    val_texts: List[str],
    val_labels: List[str],
    config: Dict,
    output_dir: str
) -> Dict:
    """Train BERTić model."""
    logger.info("Training BERTić model...")

    output_dir = Path(output_dir) / 'bertic'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get config settings
    bertic_config = config.get('models', {}).get('bertic', {})
    training_config = bertic_config.get('training', {})

    trainer = BERTicTrainer(
        model_name=bertic_config.get('model_name', BERTIC_MODEL),
        num_labels=len(set(train_labels)),
        learning_rate=training_config.get('learning_rate', 2e-5),
        batch_size=training_config.get('batch_size', 16),
        max_length=training_config.get('max_length', 256),
        num_epochs=training_config.get('num_epochs', 5),
        warmup_ratio=training_config.get('warmup_ratio', 0.1),
        weight_decay=training_config.get('weight_decay', 0.01),
    )

    history = trainer.train(
        train_texts, train_labels,
        val_texts, val_labels,
        output_dir=str(output_dir)
    )

    # Evaluate on validation set
    val_metrics = trainer.evaluate(val_texts, val_labels)
    history['val_metrics'] = val_metrics

    return history


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train hate speech detection models")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help="Path to configuration file")
    parser.add_argument('--data', type=str, required=True,
                        help="Path to training data")
    parser.add_argument('--text-column', type=str, default='text')
    parser.add_argument('--label-column', type=str, default='label')
    parser.add_argument('--model', type=str, choices=['baseline', 'bertic', 'all'],
                        default='all', help="Model to train")
    parser.add_argument('--output', type=str, default='checkpoints',
                        help="Output directory")
    parser.add_argument('--no-test-split', action='store_true',
                        help="Don't create test split (use all data for train/val)")

    args = parser.parse_args()

    # Load config
    config = {}
    if Path(args.config).exists():
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")

    # Load data
    texts, labels = load_data(
        args.data,
        text_column=args.text_column,
        label_column=args.label_column
    )

    # Split data
    if args.no_test_split:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.15, random_state=42, stratify=labels
        )
        test_texts, test_labels = [], []
    else:
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(
            texts, labels
        )

    # Train models
    all_results = {}

    if args.model in ['baseline', 'all']:
        baseline_results = train_baseline(
            train_texts, train_labels,
            val_texts, val_labels,
            config, args.output
        )
        all_results['baseline'] = baseline_results

    if args.model in ['bertic', 'all']:
        bertic_results = train_bertic(
            train_texts, train_labels,
            val_texts, val_labels,
            config, args.output
        )
        all_results['bertic'] = bertic_results

    # Save overall results
    results_path = Path(args.output) / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nTraining complete. Results saved to {results_path}")

    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)

    if 'baseline' in all_results:
        print("\nBaseline Models:")
        for clf_name, results in all_results['baseline'].items():
            if isinstance(results, dict) and 'f1_macro' in results:
                print(f"  {clf_name}: F1-macro = {results['f1_macro']:.4f}")

    if 'bertic' in all_results:
        bertic = all_results['bertic']
        if 'best_val_f1' in bertic:
            print(f"\nBERTić: Best Val F1-macro = {bertic['best_val_f1']:.4f} (epoch {bertic.get('best_epoch', '?')})")


if __name__ == "__main__":
    main()
