#!/usr/bin/env python3
"""
Interactive demo script to test hate speech detection models.
Shows predictions and explains which words/features contributed.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress verbose logging
import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("src.models.bertic").setLevel(logging.WARNING)
logging.getLogger("src.utils.lexicon").setLevel(logging.WARNING)
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import argparse
import numpy as np
from typing import Optional

from src.models.baseline import BaselineClassifier
from src.models.bertic import BERTicTrainer
from src.utils.lexicon import CodedTermLexicon


def load_models(baseline_path: Optional[str], bertic_path: Optional[str]):
    """Load available models."""
    models = {}

    if baseline_path and Path(baseline_path).exists():
        print(f"Loading baseline model from {baseline_path}...")
        models['baseline'] = BaselineClassifier.load(baseline_path)
        print("  Baseline loaded.")

    if bertic_path and Path(bertic_path).exists():
        print(f"Loading BERTić model from {bertic_path}...")
        trainer = BERTicTrainer()
        trainer.load(bertic_path)
        models['bertic'] = trainer
        print("  BERTić loaded.")

    return models


def load_lexicon(lexicon_path: str) -> Optional[CodedTermLexicon]:
    """Load coded terms lexicon."""
    if Path(lexicon_path).exists():
        return CodedTermLexicon(lexicon_path)
    return None


def analyze_text(text: str, models: dict, lexicon: Optional[CodedTermLexicon]):
    """Analyze a single text with all available models."""
    print("\n" + "="*60)
    print("INPUT TEXT:")
    print(f"  \"{text}\"")
    print("="*60)

    # Check lexicon for coded terms
    if lexicon:
        matches = lexicon.find_matches(text)
        print("\n[LEXICON] Coded terms found:")
        if matches:
            for match in matches:
                print(f"  - \"{match['term']}\" -> {match['coded_meaning']} (target: {match['target_group']})")
        else:
            print("  None detected")

    # Baseline prediction
    if 'baseline' in models:
        baseline = models['baseline']
        pred = baseline.predict([text])[0]
        proba = baseline.predict_proba([text])

        print(f"\n[BASELINE] Prediction: {pred}")
        if proba is not None:
            print(f"  Confidence: {proba[0].max():.1%}")

        # Get feature importance for this prediction
        if hasattr(baseline, 'vectorizer') and hasattr(baseline, 'model'):
            try:
                # Transform text and get feature names
                X = baseline.vectorizer.transform([text])
                feature_names = baseline.vectorizer.get_feature_names_out()

                # Get non-zero features in this text
                nonzero_idx = X.nonzero()[1]
                text_features = [(feature_names[i], X[0, i]) for i in nonzero_idx]

                # Get model coefficients
                if hasattr(baseline.model, 'coef_'):
                    coef = baseline.model.coef_[0] if len(baseline.model.coef_.shape) > 1 else baseline.model.coef_

                    # Calculate contribution of each word
                    contributions = []
                    for feat, tfidf in text_features:
                        idx = list(feature_names).index(feat)
                        contrib = tfidf * coef[idx]
                        contributions.append((feat, contrib))

                    # Sort by absolute contribution
                    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

                    print("  Top contributing words:")
                    for word, contrib in contributions[:10]:
                        direction = "OFF" if contrib > 0 else "ACC"
                        print(f"    \"{word}\": {contrib:+.3f} -> {direction}")
            except Exception as e:
                pass

    # BERTić prediction
    if 'bertic' in models:
        bertic = models['bertic']
        pred = bertic.predict([text])[0]

        # Get probabilities
        bertic.model.eval()
        import torch
        with torch.no_grad():
            encoding = bertic.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=bertic.max_length,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(bertic.device)
            attention_mask = encoding['attention_mask'].to(bertic.device)

            outputs = bertic.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        print(f"\n[BERTić] Prediction: {pred}")
        print(f"  Probabilities: ACC={probs[0]:.1%}, OFF={probs[1]:.1%}")

        # Show tokens (what BERTić sees)
        tokens = bertic.tokenizer.tokenize(text)
        print(f"  Tokens: {' '.join(tokens[:20])}{'...' if len(tokens) > 20 else ''}")

    print()


def interactive_mode(models: dict, lexicon: Optional[CodedTermLexicon]):
    """Run interactive testing loop."""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("Enter text to analyze (or 'quit' to exit)")
    print("="*60)

    while True:
        try:
            text = input("\n> ").strip()
            if text.lower() in ('quit', 'exit', 'q'):
                break
            if not text:
                continue
            analyze_text(text, models, lexicon)
        except KeyboardInterrupt:
            break
        except EOFError:
            break

    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="Test hate speech detection models")
    parser.add_argument('--text', '-t', type=str, help="Single text to analyze")
    parser.add_argument('--baseline', type=str,
                        default='checkpoints/baseline/logistic_regression_model.pkl',
                        help="Path to baseline model")
    parser.add_argument('--bertic', type=str,
                        default='checkpoints/bertic/best_model',
                        help="Path to BERTić model")
    parser.add_argument('--lexicon', type=str,
                        default='data/lexicon/coded_terms.json',
                        help="Path to coded terms lexicon")
    parser.add_argument('--no-baseline', action='store_true', help="Skip baseline model")
    parser.add_argument('--no-bertic', action='store_true', help="Skip BERTić model")

    args = parser.parse_args()

    # Load models
    baseline_path = None if args.no_baseline else args.baseline
    bertic_path = None if args.no_bertic else args.bertic

    models = load_models(baseline_path, bertic_path)

    if not models:
        print("No models loaded! Check paths.")
        print(f"  Baseline: {args.baseline}")
        print(f"  BERTić: {args.bertic}")
        return

    # Load lexicon
    lexicon = load_lexicon(args.lexicon)
    if lexicon:
        print(f"Lexicon loaded: {len(lexicon.terms)} coded terms")

    # Single text or interactive mode
    if args.text:
        analyze_text(args.text, models, lexicon)
    else:
        interactive_mode(models, lexicon)


if __name__ == "__main__":
    main()
