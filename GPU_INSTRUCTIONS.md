# GPU Training Instructions

## Setup

```bash
git clone https://github.com/TeoMatosevic/slur-analysis-model.git
cd slur-analysis-model
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 1: Train XLM-RoBERTa (~30-60 min on GPU)

```bash
python scripts/train_xlm_roberta.py
```

This auto-detects GPU and saves the model to `checkpoints/xlm_roberta/best_model/`.

## Step 2: Generate predictions + statistical analysis

```bash
python scripts/run_statistical_analysis.py
```

This runs bootstrap confidence intervals and McNemar tests for all 4 models.
Results are saved to `docs/statistical_analysis.json`.

## Step 3: Generate figures

```bash
python scripts/generate_figures.py
```

This creates confusion matrices and ROC curves in `docs/figures/`.

## What to send back

Send these files/folders:
- `checkpoints/xlm_roberta/` (the trained model)
- `docs/statistical_analysis.json` (updated with all 4 models + McNemar tests)
- `docs/figures/` (confusion_matrices.png, roc_curves.png, auc_results.json)
- `checkpoints/predictions_cache/` (cached predictions for all models)

Or just commit and push everything:
```bash
git add docs/statistical_analysis.json docs/figures/ checkpoints/predictions_cache/
git commit -m "Add XLM-RoBERTa results, figures, and statistical analysis"
git push
```

Note: the `checkpoints/` folder is in .gitignore, so the trained model itself
won't be pushed via git. Either share `checkpoints/xlm_roberta/` separately
or remove that line from .gitignore before pushing.
