"""
Baseline models for Croatian hate speech detection.
Implements TF-IDF + classical ML classifiers (Logistic Regression, SVM).
"""

import json
import pickle
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    matthews_corrcoef
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineClassifier:
    """
    Baseline text classifier using TF-IDF and classical ML algorithms.
    """

    CLASSIFIERS = {
        'logistic_regression': LogisticRegression,
        'svm': LinearSVC,
        'random_forest': RandomForestClassifier,
    }

    def __init__(
        self,
        classifier_type: str = 'logistic_regression',
        vectorizer_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None
    ):
        """
        Initialize the baseline classifier.

        Args:
            classifier_type: Type of classifier ('logistic_regression', 'svm', 'random_forest')
            vectorizer_config: Configuration for TF-IDF vectorizer
            classifier_config: Configuration for the classifier
        """
        self.classifier_type = classifier_type

        # Default vectorizer config
        default_vectorizer_config = {
            'max_features': 10000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95,
            'sublinear_tf': True,
        }
        vectorizer_config = vectorizer_config or {}
        self.vectorizer_config = {**default_vectorizer_config, **vectorizer_config}

        # Default classifier configs
        default_classifier_configs = {
            'logistic_regression': {
                'C': 1.0,
                'class_weight': 'balanced',
                'max_iter': 1000,
                'random_state': 42,
            },
            'svm': {
                'C': 1.0,
                'class_weight': 'balanced',
                'max_iter': 1000,
                'random_state': 42,
            },
            'random_forest': {
                'n_estimators': 100,
                'class_weight': 'balanced',
                'random_state': 42,
            }
        }
        classifier_config = classifier_config or {}
        self.classifier_config = {
            **default_classifier_configs.get(classifier_type, {}),
            **classifier_config
        }

        # Initialize components
        self.vectorizer = TfidfVectorizer(**self.vectorizer_config)

        classifier_class = self.CLASSIFIERS.get(classifier_type)
        if classifier_class is None:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        self.classifier = classifier_class(**self.classifier_config)

        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]):
        """
        Fit the classifier on training data.

        Args:
            texts: List of text documents
            labels: List of labels
        """
        logger.info(f"Fitting {self.classifier_type} classifier on {len(texts)} samples...")

        # Encode labels
        y = self.label_encoder.fit_transform(labels)

        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)

        # Train classifier
        self.classifier.fit(X, y)
        self.is_fitted = True

        logger.info("Training complete")
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        logger.info(f"Labels: {list(self.label_encoder.classes_)}")

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict labels for texts.

        Args:
            texts: List of text documents

        Returns:
            Array of predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        X = self.vectorizer.transform(texts)
        y_pred = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Predict class probabilities (if supported by classifier).

        Args:
            texts: List of text documents

        Returns:
            Array of class probabilities or None
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        if not hasattr(self.classifier, 'predict_proba'):
            return None

        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def evaluate(
        self,
        texts: List[str],
        labels: List[str],
        return_predictions: bool = False
    ) -> Dict:
        """
        Evaluate the classifier on test data.

        Args:
            texts: List of text documents
            labels: List of true labels
            return_predictions: Whether to include predictions in output

        Returns:
            Dictionary with evaluation metrics
        """
        y_true = self.label_encoder.transform(labels)
        y_pred_labels = self.predict(texts)
        y_pred = self.label_encoder.transform(y_pred_labels)

        # Calculate metrics
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred),
        }

        # Per-class metrics
        report = classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        results['per_class'] = report

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()

        if return_predictions:
            results['predictions'] = y_pred_labels

        return results

    def cross_validate(
        self,
        texts: List[str],
        labels: List[str],
        n_folds: int = 5
    ) -> Dict:
        """
        Perform cross-validation.

        Args:
            texts: List of text documents
            labels: List of labels
            n_folds: Number of folds

        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Running {n_folds}-fold cross-validation...")

        # Encode labels
        y = self.label_encoder.fit_transform(labels)

        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)

        # Create pipeline for cross-validation
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Calculate scores
        scoring = ['accuracy', 'f1_macro', 'f1_weighted']
        results = {}

        for metric in scoring:
            scores = cross_val_score(
                self.classifier, X, y,
                cv=cv, scoring=metric
            )
            results[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }

        logger.info(f"CV Results: F1-macro = {results['f1_macro']['mean']:.4f} (+/- {results['f1_macro']['std']:.4f})")

        return results

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get most important features per class.

        Args:
            top_n: Number of top features to return

        Returns:
            Dictionary mapping class names to list of (feature, weight) tuples
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        feature_names = self.vectorizer.get_feature_names_out()

        # Get coefficients (works for LR and SVM)
        if hasattr(self.classifier, 'coef_'):
            coefs = self.classifier.coef_
            if coefs.ndim == 1:
                coefs = coefs.reshape(1, -1)

            importance = {}
            if coefs.shape[0] == 1:
                # Binary classification case
                # coef_[0] corresponds to class 1 (alphabetically second)
                class_0 = self.label_encoder.classes_[0]
                class_1 = self.label_encoder.classes_[1]
                
                # Class 1: Top positive coefficients
                indices_1 = np.argsort(coefs[0])[-top_n:][::-1]
                importance[class_1] = [
                    (feature_names[idx], float(coefs[0, idx]))
                    for idx in indices_1
                ]
                
                # Class 0: Top negative coefficients
                indices_0 = np.argsort(coefs[0])[:top_n]
                importance[class_0] = [
                    (feature_names[idx], -float(coefs[0, idx]))
                    for idx in indices_0
                ]
            else:
                # Multi-class case
                for i, class_name in enumerate(self.label_encoder.classes_):
                    if i < coefs.shape[0]:
                        # Get top positive coefficients for this class
                        indices = np.argsort(coefs[i])[-top_n:][::-1]
                        importance[class_name] = [
                            (feature_names[idx], float(coefs[i, idx]))
                            for idx in indices
                        ]

            return importance

        return {}

    def save(self, path: str):
        """Save the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'classifier_type': self.classifier_type,
            'vectorizer_config': self.vectorizer_config,
            'classifier_config': self.classifier_config,
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'BaselineClassifier':
        """Load a model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        instance = cls(
            classifier_type=model_data['classifier_type'],
            vectorizer_config=model_data['vectorizer_config'],
            classifier_config=model_data['classifier_config']
        )
        instance.vectorizer = model_data['vectorizer']
        instance.classifier = model_data['classifier']
        instance.label_encoder = model_data['label_encoder']
        instance.is_fitted = True

        return instance


def train_baseline_models(
    train_texts: List[str],
    train_labels: List[str],
    test_texts: List[str],
    test_labels: List[str],
    output_dir: str = 'checkpoints/baseline'
) -> Dict:
    """
    Train and evaluate multiple baseline models.

    Args:
        train_texts: Training texts
        train_labels: Training labels
        test_texts: Test texts
        test_labels: Test labels
        output_dir: Directory to save models

    Returns:
        Dictionary with results for all models
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for clf_type in ['logistic_regression', 'svm']:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {clf_type}...")
        logger.info(f"{'='*50}")

        # Create and train classifier
        clf = BaselineClassifier(classifier_type=clf_type)
        clf.fit(train_texts, train_labels)

        # Evaluate
        eval_results = clf.evaluate(test_texts, test_labels)
        results[clf_type] = eval_results

        # Log results
        logger.info(f"\nResults for {clf_type}:")
        logger.info(f"  Accuracy: {eval_results['accuracy']:.4f}")
        logger.info(f"  F1 Macro: {eval_results['f1_macro']:.4f}")
        logger.info(f"  F1 Weighted: {eval_results['f1_weighted']:.4f}")
        logger.info(f"  MCC: {eval_results['mcc']:.4f}")

        # Get feature importance
        importance = clf.get_feature_importance(top_n=10)
        results[clf_type]['feature_importance'] = importance

        # Save model
        model_path = output_dir / f"{clf_type}_model.pkl"
        clf.save(model_path)

    # Save results
    results_path = output_dir / "baseline_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        # Convert numpy arrays to lists for JSON serialization
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\nResults saved to {results_path}")

    return results


def main():
    """Main entry point for baseline models."""
    parser = argparse.ArgumentParser(description="Train baseline hate speech classifier")
    parser.add_argument('--data', type=str, help="Path to training data (CSV or JSONL)")
    parser.add_argument('--text-column', type=str, default='text', help="Name of text column")
    parser.add_argument('--label-column', type=str, default='label', help="Name of label column")
    parser.add_argument('--classifier', type=str, default='logistic_regression',
                        choices=['logistic_regression', 'svm', 'random_forest'])
    parser.add_argument('--output', type=str, default='checkpoints/baseline')
    parser.add_argument('--test-split', type=float, default=0.2, help="Test split ratio")
    parser.add_argument('--cv', action='store_true', help="Run cross-validation")
    parser.add_argument('--test', action='store_true', help="Run test mode")

    args = parser.parse_args()

    if args.test:
        logger.info("Running in test mode")
        print("Test mode: Baseline model module loaded successfully")

        # Create a simple test
        test_texts = [
            "Ovo je normalan komentar",
            "Mrzim sve inženjere koji dolaze u našu zemlju",
            "Globalisti kontroliraju sve",
            "Lijepo je danas vrijeme",
            "Svi su ovce, vjeruju u laži",
        ]
        test_labels = ["ACC", "IHS", "CON", "ACC", "CON"]

        clf = BaselineClassifier(classifier_type='logistic_regression')
        clf.fit(test_texts, test_labels)

        predictions = clf.predict(test_texts)
        print(f"\nTest predictions: {predictions}")
        print(f"True labels: {test_labels}")

        results = clf.evaluate(test_texts, test_labels)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Macro: {results['f1_macro']:.4f}")

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

    if args.cv:
        # Cross-validation mode
        clf = BaselineClassifier(classifier_type=args.classifier)
        cv_results = clf.cross_validate(texts, labels)
        print(json.dumps(cv_results, indent=2))
    else:
        # Train-test split mode
        from sklearn.model_selection import train_test_split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=args.test_split, random_state=42, stratify=labels
        )

        results = train_baseline_models(
            train_texts, train_labels,
            test_texts, test_labels,
            output_dir=args.output
        )


if __name__ == "__main__":
    main()
