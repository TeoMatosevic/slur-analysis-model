#!/usr/bin/env python3
"""
Generate confusion matrices and ROC curves for all models.
Saves figures to docs/figures/.

Usage: python scripts/generate_figures.py
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
from sklearn.metrics import roc_auc_score

from src.models.baseline import BaselineClassifier
from src.training.evaluate import (
    plot_all_confusion_matrices, plot_roc_curves
)

CACHE_DIR = project_root / 'checkpoints' / 'predictions_cache'


def main():
    # Output directory
    figures_dir = project_root / 'docs' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load test data
    test_df = pd.read_json(project_root / 'data' / 'processed' / 'frenk_test.jsonl', lines=True)
    texts = test_df['text'].tolist()
    labels = test_df['label'].tolist()
    label_names = ['ACC', 'OFF']

    # Encode labels for ROC (OFF=1 as positive class)
    label_map = {'ACC': 0, 'OFF': 1}
    y_true_binary = np.array([label_map[l] for l in labels])

    models_cm_data = {}
    models_roc_data = {}

    # --- Logistic Regression ---
    print("Loading Logistic Regression...")
    lr_path = project_root / 'checkpoints' / 'baseline' / 'baseline' / 'logistic_regression_model.pkl'
    lr_pred_cache = CACHE_DIR / 'lr_preds.npy'
    lr_proba_cache = CACHE_DIR / 'lr_proba.npy'
    if lr_path.exists():
        # Get predictions (from cache or model)
        if lr_pred_cache.exists():
            lr_preds = np.load(lr_pred_cache, allow_pickle=True).tolist()
            print("  LR predictions from cache")
        else:
            lr = BaselineClassifier.load(str(lr_path))
            lr_preds = lr.predict(texts)
            np.save(lr_pred_cache, np.array(lr_preds))
        models_cm_data['Logistic Regression'] = {'y_true': labels, 'y_pred': lr_preds}

        # Get probabilities for ROC
        if lr_proba_cache.exists():
            lr_proba = np.load(lr_proba_cache)
            models_roc_data['Logistic Regression'] = {
                'y_true': y_true_binary,
                'y_score': lr_proba[:, 1] if lr_proba.shape[1] > 1 else lr_proba[:, 0]
            }
        else:
            lr = BaselineClassifier.load(str(lr_path)) if 'lr' not in dir() else lr
            lr_proba = lr.predict_proba(texts)
            if lr_proba is not None:
                np.save(lr_proba_cache, lr_proba)
                models_roc_data['Logistic Regression'] = {
                    'y_true': y_true_binary,
                    'y_score': lr_proba[:, 1] if lr_proba.shape[1] > 1 else lr_proba[:, 0]
                }
        print(f"  LR loaded, {len(lr_preds)} predictions")
    else:
        print(f"  LR model not found at {lr_path}")

    # --- SVM ---
    print("Loading SVM...")
    svm_path = project_root / 'checkpoints' / 'baseline' / 'baseline' / 'svm_model.pkl'
    svm_pred_cache = CACHE_DIR / 'svm_preds.npy'
    svm_score_cache = CACHE_DIR / 'svm_scores.npy'
    if svm_path.exists():
        if svm_pred_cache.exists():
            svm_preds = np.load(svm_pred_cache, allow_pickle=True).tolist()
            print("  SVM predictions from cache")
        else:
            svm = BaselineClassifier.load(str(svm_path))
            svm_preds = svm.predict(texts)
            np.save(svm_pred_cache, np.array(svm_preds))
        models_cm_data['SVM (Linear)'] = {'y_true': labels, 'y_pred': svm_preds}

        # SVM: use decision_function for ROC
        if svm_score_cache.exists():
            svm_scores = np.load(svm_score_cache)
            models_roc_data['SVM (Linear)'] = {
                'y_true': y_true_binary,
                'y_score': svm_scores
            }
        else:
            try:
                svm = BaselineClassifier.load(str(svm_path))
                if hasattr(svm.classifier, 'decision_function'):
                    transformed = svm.vectorizer.transform(texts)
                    svm_scores = svm.classifier.decision_function(transformed)
                    np.save(svm_score_cache, svm_scores)
                    models_roc_data['SVM (Linear)'] = {
                        'y_true': y_true_binary,
                        'y_score': svm_scores
                    }
            except Exception as e:
                print(f"  SVM ROC skipped: {e}")
        print(f"  SVM loaded, {len(svm_preds)} predictions")
    else:
        print(f"  SVM model not found at {svm_path}")

    # --- BERTić ---
    print("Loading BERTić...")
    bertic_path = project_root / 'checkpoints' / 'bertic' / 'best_model'
    bertic_pred_cache = CACHE_DIR / 'bertic_preds.npy'
    bertic_proba_cache = CACHE_DIR / 'bertic_proba.npy'
    if bertic_path.exists():
        try:
            if bertic_pred_cache.exists():
                bertic_preds = np.load(bertic_pred_cache, allow_pickle=True).tolist()
                print("  BERTić predictions from cache")
            else:
                from src.models.bertic import BERTicTrainer
                bertic = BERTicTrainer(device='cpu')
                bertic.load(str(bertic_path))
                bertic_preds = bertic.predict(texts).tolist()
                np.save(bertic_pred_cache, np.array(bertic_preds))
            models_cm_data['BERTić'] = {'y_true': labels, 'y_pred': bertic_preds}

            if bertic_proba_cache.exists():
                bertic_proba = np.load(bertic_proba_cache)
                # OFF class is index 1 (alphabetical: ACC=0, OFF=1)
                models_roc_data['BERTić'] = {
                    'y_true': y_true_binary,
                    'y_score': bertic_proba[:, 1]
                }
            elif 'bertic' in dir():
                bertic_proba = bertic.predict_proba(texts)
                off_idx = bertic.label_encoder.transform(['OFF'])[0]
                np.save(bertic_proba_cache, bertic_proba)
                models_roc_data['BERTić'] = {
                    'y_true': y_true_binary,
                    'y_score': bertic_proba[:, off_idx]
                }
            print(f"  BERTić loaded, {len(bertic_preds)} predictions")
        except Exception as e:
            print(f"  BERTić failed: {e}")
    else:
        print(f"  BERTić model not found at {bertic_path}")

    # --- XLM-RoBERTa (if trained) ---
    xlm_path = project_root / 'checkpoints' / 'xlm_roberta' / 'best_model'
    xlm_pred_cache = CACHE_DIR / 'xlm_preds.npy'
    xlm_proba_cache = CACHE_DIR / 'xlm_proba.npy'
    if xlm_path.exists() or xlm_pred_cache.exists():
        print("Loading XLM-RoBERTa...")
        try:
            if xlm_pred_cache.exists():
                xlm_preds = np.load(xlm_pred_cache, allow_pickle=True).tolist()
                print("  XLM-RoBERTa predictions from cache")
            else:
                from src.models.xlm_roberta import XLMRobertaTrainer
                xlm = XLMRobertaTrainer(device='cpu')
                xlm.load(str(xlm_path))
                xlm_preds = xlm.predict(texts).tolist()
                np.save(xlm_pred_cache, np.array(xlm_preds))
            models_cm_data['XLM-RoBERTa'] = {'y_true': labels, 'y_pred': xlm_preds}

            if xlm_proba_cache.exists():
                xlm_proba = np.load(xlm_proba_cache)
                models_roc_data['XLM-RoBERTa'] = {
                    'y_true': y_true_binary,
                    'y_score': xlm_proba[:, 1]
                }
            elif 'xlm' in dir():
                xlm_proba = xlm.predict_proba(texts)
                off_idx = xlm.label_encoder.transform(['OFF'])[0]
                np.save(xlm_proba_cache, xlm_proba)
                models_roc_data['XLM-RoBERTa'] = {
                    'y_true': y_true_binary,
                    'y_score': xlm_proba[:, off_idx]
                }
            print(f"  XLM-RoBERTa loaded, {len(xlm_preds)} predictions")
        except Exception as e:
            print(f"  XLM-RoBERTa failed: {e}")
    else:
        print("  XLM-RoBERTa not yet trained, skipping.")

    # --- Generate figures ---
    if models_cm_data:
        print("\nGenerating confusion matrices...")
        plot_all_confusion_matrices(
            models_cm_data, label_names,
            str(figures_dir / 'confusion_matrices.png')
        )

    if models_roc_data:
        print("Generating ROC curves...")
        plot_roc_curves(
            models_roc_data,
            str(figures_dir / 'roc_curves.png')
        )
        # Print AUC values
        print("\nAUC-ROC Values:")
        auc_results = {}
        for name, data in models_roc_data.items():
            auc_val = roc_auc_score(data['y_true'], data['y_score'])
            print(f"  {name}: {auc_val:.3f}")
            auc_results[name] = auc_val
        with open(figures_dir / 'auc_results.json', 'w') as f:
            json.dump(auc_results, f, indent=2)

    print("\nDone! Figures saved to docs/figures/")


if __name__ == '__main__':
    main()
