#!/usr/bin/env python3
"""
Run statistical analysis: bootstrap confidence intervals and McNemar tests.
Outputs results for inclusion in the paper.

Usage: python scripts/run_statistical_analysis.py
"""

import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef

from src.models.baseline import BaselineClassifier
from src.training.evaluate import bootstrap_ci, mcnemar_test

CACHE_DIR = project_root / 'checkpoints' / 'predictions_cache'


def get_predictions(texts, labels):
    """Load models and get predictions, using cache when available."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    predictions = {}

    # Logistic Regression
    cache_path = CACHE_DIR / 'lr_preds.npy'
    lr_path = project_root / 'checkpoints' / 'baseline' / 'baseline' / 'logistic_regression_model.pkl'
    if cache_path.exists():
        predictions['Logistic Regression'] = np.load(cache_path, allow_pickle=True)
        print("  Logistic Regression loaded from cache")
    elif lr_path.exists():
        lr = BaselineClassifier.load(str(lr_path))
        preds = np.array(lr.predict(texts))
        np.save(cache_path, preds)
        predictions['Logistic Regression'] = preds
        print("  Logistic Regression loaded")

    # SVM
    cache_path = CACHE_DIR / 'svm_preds.npy'
    svm_path = project_root / 'checkpoints' / 'baseline' / 'baseline' / 'svm_model.pkl'
    if cache_path.exists():
        predictions['SVM (Linear)'] = np.load(cache_path, allow_pickle=True)
        print("  SVM loaded from cache")
    elif svm_path.exists():
        svm = BaselineClassifier.load(str(svm_path))
        preds = np.array(svm.predict(texts))
        np.save(cache_path, preds)
        predictions['SVM (Linear)'] = preds
        print("  SVM loaded")

    # BERTić
    cache_path = CACHE_DIR / 'bertic_preds.npy'
    bertic_path = project_root / 'checkpoints' / 'bertic' / 'best_model'
    if cache_path.exists():
        predictions['BERTić'] = np.load(cache_path, allow_pickle=True)
        print("  BERTić loaded from cache")
    elif bertic_path.exists():
        try:
            from src.models.bertic import BERTicTrainer
            bertic = BERTicTrainer(device='cpu')
            bertic.load(str(bertic_path))
            preds = np.array(bertic.predict(texts))
            np.save(cache_path, preds)
            predictions['BERTić'] = preds
            print("  BERTić loaded (predictions cached for future runs)")
        except Exception as e:
            print(f"  BERTić failed: {e}")

    # XLM-RoBERTa (if available)
    cache_path = CACHE_DIR / 'xlm_preds.npy'
    xlm_path = project_root / 'checkpoints' / 'xlm_roberta' / 'best_model'
    if cache_path.exists():
        predictions['XLM-RoBERTa'] = np.load(cache_path, allow_pickle=True)
        print("  XLM-RoBERTa loaded from cache")
    elif xlm_path.exists():
        try:
            from src.models.xlm_roberta import XLMRobertaTrainer
            xlm = XLMRobertaTrainer(device='cpu')
            xlm.load(str(xlm_path))
            preds = np.array(xlm.predict(texts))
            np.save(cache_path, preds)
            predictions['XLM-RoBERTa'] = preds
            print("  XLM-RoBERTa loaded")
        except Exception as e:
            print(f"  XLM-RoBERTa failed: {e}")

    return predictions


def main():
    # Load test data
    test_df = pd.read_json(project_root / 'data' / 'processed' / 'frenk_test.jsonl', lines=True)
    texts = test_df['text'].tolist()
    labels = test_df['label'].tolist()
    y_true = np.array(labels)

    print("Loading models and generating predictions...\n")
    predictions = get_predictions(texts, labels)

    if not predictions:
        print("No models found!")
        return

    # --- Bootstrap Confidence Intervals ---
    print("\n" + "=" * 70)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS (n=1000)")
    print("=" * 70)

    ci_results = {}
    for model_name, y_pred in predictions.items():
        cis = {}
        for metric_name, metric_fn in [
            ('F1-Macro', lambda yt, yp: f1_score(yt, yp, average='macro')),
            ('Accuracy', lambda yt, yp: accuracy_score(yt, yp)),
            ('MCC', lambda yt, yp: matthews_corrcoef(yt, yp)),
        ]:
            lower, upper = bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000)
            point = metric_fn(y_true, y_pred)
            cis[metric_name] = {'point': float(point), 'lower': float(lower), 'upper': float(upper)}

        ci_results[model_name] = cis
        print(f"\n{model_name}:")
        for metric, vals in cis.items():
            print(f"  {metric}: {vals['point']:.3f} [{vals['lower']:.3f}, {vals['upper']:.3f}]")

    # --- McNemar Tests ---
    print("\n" + "=" * 70)
    print("McNEMAR TESTS (pairwise)")
    print("=" * 70)

    mcnemar_results = {}
    model_names = list(predictions.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a = model_names[i]
            name_b = model_names[j]
            comparison = f"{name_a} vs {name_b}"
            result = mcnemar_test(y_true, predictions[name_a], predictions[name_b])
            mcnemar_results[comparison] = result
            sig = "YES" if result['significant'] else "no"
            print(f"\n{comparison}:")
            print(f"  Statistic: {result['statistic']:.1f}")
            print(f"  p-value: {result['p_value']:.6f}")
            print(f"  Significant (p<0.05): {sig}")

    # --- Save results ---
    output = {
        'bootstrap_ci': ci_results,
        'mcnemar_tests': mcnemar_results,
    }
    output_path = project_root / 'docs' / 'statistical_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n\nResults saved to {output_path}")

    # --- Print paper-ready tables ---
    print("\n" + "=" * 70)
    print("PAPER-READY TABLE: Bootstrap 95% CIs")
    print("=" * 70)
    print(f"| {'Model':<22} | {'F1-Macro':<22} | {'Accuracy':<22} | {'MCC':<22} |")
    print(f"|{'-'*24}|{'-'*24}|{'-'*24}|{'-'*24}|")
    for model_name, cis in ci_results.items():
        f1 = cis['F1-Macro']
        acc = cis['Accuracy']
        mcc = cis['MCC']
        print(f"| {model_name:<22} | {f1['point']:.3f} [{f1['lower']:.3f}, {f1['upper']:.3f}] | "
              f"{acc['point']:.3f} [{acc['lower']:.3f}, {acc['upper']:.3f}] | "
              f"{mcc['point']:.3f} [{mcc['lower']:.3f}, {mcc['upper']:.3f}] |")

    print("\n" + "=" * 70)
    print("PAPER-READY TABLE: McNemar Tests")
    print("=" * 70)
    print(f"| {'Comparison':<35} | {'p-value':<12} | {'Significant':<12} |")
    print(f"|{'-'*37}|{'-'*14}|{'-'*14}|")
    for comparison, result in mcnemar_results.items():
        p = result['p_value']
        p_str = f"{p:.6f}" if p >= 0.001 else "< 0.001"
        sig = "Yes (p<0.05)" if result['significant'] else "No"
        print(f"| {comparison:<35} | {p_str:<12} | {sig:<12} |")


if __name__ == '__main__':
    main()
