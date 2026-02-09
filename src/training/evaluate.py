"""
Evaluation script for Croatian hate speech detection models.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    matthews_corrcoef,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.baseline import BaselineClassifier
from src.models.bertic import BERTicTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(
    data_path: str,
    text_column: str = 'text',
    label_column: str = 'label'
) -> tuple:
    """Load evaluation data."""
    path = Path(data_path)
    if path.suffix == '.csv':
        df = pd.read_csv(path)
    else:
        df = pd.read_json(path, lines=path.suffix == '.jsonl')

    texts = df[text_column].tolist()
    labels = df[label_column].tolist()

    return texts, labels


def evaluate_baseline(
    model_path: str,
    texts: List[str],
    labels: List[str]
) -> Dict:
    """Evaluate baseline model."""
    logger.info(f"Loading baseline model from {model_path}")
    model = BaselineClassifier.load(model_path)

    results = model.evaluate(texts, labels, return_predictions=True)

    return results


def evaluate_bertic(
    model_path: str,
    texts: List[str],
    labels: List[str],
    device: Optional[str] = None
) -> Dict:
    """Evaluate BERTić model."""
    logger.info(f"Loading BERTić model from {model_path}")

    trainer = BERTicTrainer(device=device)
    trainer.load(model_path)

    results = trainer.evaluate(texts, labels)
    # predictions are now included in evaluate() results

    return results


def plot_confusion_matrix(
    y_true: List,
    y_pred: List,
    labels: List[str],
    output_path: str,
    title: str = "Confusion Matrix"
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels
    )
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info(f"Confusion matrix saved to {output_path}")


def plot_per_class_metrics(
    report: Dict,
    output_path: str,
    title: str = "Per-Class Metrics"
):
    """Plot per-class precision, recall, F1."""
    classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]

    metrics = ['precision', 'recall', 'f1-score']
    data = {
        'class': [],
        'metric': [],
        'value': []
    }

    for cls in classes:
        for metric in metrics:
            data['class'].append(cls)
            data['metric'].append(metric)
            data['value'].append(report[cls][metric])

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='class', y='value', hue='metric')
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(title='Metric')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info(f"Per-class metrics plot saved to {output_path}")


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> tuple:
    """
    Calculate bootstrap confidence interval for a metric.

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, n)
        score = metric_fn(y_true[indices], y_pred[indices])
        scores.append(score)

    alpha = (1 - confidence) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)

    return lower, upper


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar's test for comparing two classifiers.
    Tests if the disagreements between two classifiers are symmetric.
    """
    from statsmodels.stats.contingency_tables import mcnemar as mcnemar_stat

    y_true = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    n11 = np.sum(correct_a & correct_b)      # both correct
    n10 = np.sum(correct_a & ~correct_b)     # a correct, b wrong
    n01 = np.sum(~correct_a & correct_b)     # a wrong, b correct
    n00 = np.sum(~correct_a & ~correct_b)    # both wrong

    table = np.array([[n11, n10], [n01, n00]])
    result = mcnemar_stat(table, exact=True)

    return {
        'statistic': float(result.statistic),
        'p_value': float(result.pvalue),
        'significant': result.pvalue < 0.05,
        'contingency_table': table.tolist(),
    }


def plot_roc_curves(models_data: dict, output_path: str, title: str = "ROC Curves - Model Comparison"):
    """
    Plot ROC curves for multiple models.

    Args:
        models_data: {model_name: {'y_true': array, 'y_score': array}}
            y_score should be probability of the positive class
        output_path: path to save figure
    """
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(10, 8))

    for model_name, data in models_data.items():
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_score'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info(f"ROC curves saved to {output_path}")


def plot_all_confusion_matrices(models_data: dict, labels: List[str], output_path: str):
    """
    Plot confusion matrices for multiple models in a grid.

    Args:
        models_data: {model_name: {'y_true': list, 'y_pred': list}}
        labels: class names
        output_path: path to save figure
    """
    n_models = len(models_data)
    cols = min(n_models, 2)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (model_name, data) in enumerate(models_data.items()):
        cm = confusion_matrix(data['y_true'], data['y_pred'], labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=axes[idx])
        axes[idx].set_title(f'{model_name}', fontsize=13, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Confusion matrices saved to {output_path}")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate hate speech detection models")
    parser.add_argument('--data', type=str, required=True, help="Path to test data")
    parser.add_argument('--text-column', type=str, default='text')
    parser.add_argument('--label-column', type=str, default='label')
    parser.add_argument('--model', type=str, choices=['baseline', 'bertic', 'xlm_roberta'], required=True)
    parser.add_argument('--model-path', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--output', type=str, default='evaluation_results')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--bootstrap', action='store_true', help="Calculate bootstrap CI")

    args = parser.parse_args()

    # Load data
    texts, labels = load_data(
        args.data,
        text_column=args.text_column,
        label_column=args.label_column
    )

    # Evaluate model
    if args.model == 'baseline':
        results = evaluate_baseline(args.model_path, texts, labels)
    elif args.model == 'xlm_roberta':
        from src.models.xlm_roberta import XLMRobertaTrainer
        trainer = XLMRobertaTrainer(device=args.device)
        trainer.load(args.model_path)
        results = trainer.evaluate(texts, labels)
    else:
        results = evaluate_bertic(args.model_path, texts, labels, args.device)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"\nAccuracy: {results.get('accuracy', 'N/A'):.4f}")
    print(f"F1 Macro: {results.get('f1_macro', 'N/A'):.4f}")
    print(f"F1 Weighted: {results.get('f1_weighted', 'N/A'):.4f}")
    print(f"Precision Macro: {results.get('precision_macro', 'N/A'):.4f}")
    print(f"Recall Macro: {results.get('recall_macro', 'N/A'):.4f}")
    print(f"MCC: {results.get('mcc', 'N/A'):.4f}")

    # Bootstrap confidence intervals
    if args.bootstrap and 'predictions' in results:
        print("\nBootstrap 95% Confidence Intervals:")
        y_true = np.array([labels.index(l) if isinstance(labels[0], str) else l for l in labels])
        y_pred = np.array([labels.index(p) if isinstance(labels[0], str) else p for p in results['predictions']])

        for metric_name, metric_fn in [
            ('F1 Macro', lambda y, p: f1_score(y, p, average='macro')),
            ('Accuracy', accuracy_score),
        ]:
            lower, upper = bootstrap_ci(y_true, y_pred, metric_fn)
            print(f"  {metric_name}: [{lower:.4f}, {upper:.4f}]")

    # Save results
    results_path = output_dir / f"{args.model}_results.json"
    with open(results_path, 'w') as f:
        # Remove predictions for cleaner JSON (can be large)
        results_to_save = {k: v for k, v in results.items() if k != 'predictions'}
        json.dump(results_to_save, f, indent=2, default=str)

    logger.info(f"\nResults saved to {results_path}")

    # Generate plots
    report_key = 'classification_report' if 'classification_report' in results else 'per_class'
    if 'predictions' in results and report_key in results:
        report = results[report_key]
        class_names = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]

        # Confusion matrix
        plot_confusion_matrix(
            labels, results['predictions'],
            class_names,
            output_dir / f"{args.model}_confusion_matrix.png",
            title=f"{args.model.title()} Confusion Matrix"
        )

        # Per-class metrics
        plot_per_class_metrics(
            report,
            output_dir / f"{args.model}_per_class_metrics.png",
            title=f"{args.model.title()} Per-Class Metrics"
        )


if __name__ == "__main__":
    main()
