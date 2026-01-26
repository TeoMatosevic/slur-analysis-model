# Detection of Hidden Hate Speech and Conspiracy Theory Narratives on Croatian Fringe Portals using NLP Methods

Detekcija prikrivenog govora mržnje i narativa teorija zavjere na rubnim portalima korištenjem NLP metoda

## Authors

- Duje Jurić (0036539312)
- Teo Matošević (0036542778)
- Teo Radolović (0036544270)

## Project Overview

This project focuses on detecting **implicit/coded hate speech** (dog whistles) and **conspiracy theory narratives** in Croatian online discourse using Natural Language Processing (NLP) methods. The key challenge is identifying words used with hidden pejorative meanings (e.g., "inženjeri" sarcastically used for immigrants).

## Repository Structure

```
opj/
├── data/
│   ├── raw/              # Scraped comments
│   ├── processed/        # Cleaned and preprocessed data
│   ├── annotations/      # Label files
│   └── lexicon/          # Coded language lexicon
│       └── coded_terms.json
├── src/
│   ├── scraping/         # Data collection
│   │   └── scraper.py
│   ├── preprocessing/    # Text cleaning with CLASSLA
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── baseline.py   # TF-IDF + Logistic Regression/SVM
│   │   └── bertic.py     # Fine-tuned BERTić classifier
│   ├── training/
│   │   ├── train.py      # Main training script
│   │   └── evaluate.py   # Evaluation and visualization
│   └── utils/
│       └── lexicon.py    # Coded term matching utilities
├── notebooks/            # Jupyter notebooks for EDA
├── configs/
│   └── config.yaml       # Configuration file
├── docs/                 # Documentation and paper
├── requirements.txt      # Python dependencies
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/croatian-hate-speech-detection.git
cd croatian-hate-speech-detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download CLASSLA models for Croatian
python -c "import classla; classla.download('hr', type='nonstandard')"
```

## Quick Start

### 1. Test Installation

```bash
# Test preprocessing module
python src/preprocessing/preprocessor.py --test

# Test baseline model
python src/models/baseline.py --test

# Test BERTić model
python src/models/bertic.py --test

# Test lexicon
python src/utils/lexicon.py --lexicon data/lexicon/coded_terms.json --stats
```

### 2. Scrape Comments (Optional)

```bash
# Scrape from Croatian news portals
python src/scraping/scraper.py --portal index.hr --sections vijesti hrvatska --max-articles 50

# Or test the scraper
python src/scraping/scraper.py --test
```

### 3. Preprocess Data

```bash
python src/preprocessing/preprocessor.py \
    --input data/raw/comments.jsonl \
    --output data/processed/comments_processed.jsonl \
    --classla-type nonstandard
```

### 4. Train Models

```bash
# Train all models (baseline + BERTić)
python src/training/train.py \
    --data data/processed/comments_processed.jsonl \
    --model all \
    --config configs/config.yaml \
    --output checkpoints

# Train only baseline
python src/training/train.py --data your_data.csv --model baseline

# Train only BERTić
python src/training/train.py --data your_data.csv --model bertic
```

### 5. Evaluate Models

```bash
# Evaluate baseline model
python src/training/evaluate.py \
    --data data/processed/test.jsonl \
    --model baseline \
    --model-path checkpoints/baseline/logistic_regression_model.pkl \
    --output evaluation_results

# Evaluate BERTić model
python src/training/evaluate.py \
    --data data/processed/test.jsonl \
    --model bertic \
    --model-path checkpoints/bertic/best_model \
    --output evaluation_results \
    --bootstrap
```

## Data Format

### Input Data

The training data should be a CSV or JSONL file with at least two columns:

```csv
text,label
"Ovo je normalan komentar",ACC
"Inženjeri opet prave probleme",IHS
"Globalisti kontroliraju sve",CON
```

### Labels

| Code | Label | Description |
|------|-------|-------------|
| `ACC` | Acceptable | No problematic content |
| `EHS` | Explicit Hate Speech | Direct slurs, threats, dehumanization |
| `IHS` | Implicit/Coded Hate Speech | Dog whistles, sarcasm, coded terms |
| `CON` | Conspiracy Theory | Secret plots, cover-ups, hidden agents |
| `OFF` | Offensive | Rude but not targeting protected groups |

### Target Groups

- `TGT_MIG` - Migrants/Refugees
- `TGT_SRB` - Ethnic Serbs
- `TGT_LGBT` - LGBTQ+ community
- `TGT_ROM` - Roma
- `TGT_ELT` - Media/Elites/"Globalists"

## Models

### Baseline (TF-IDF + Classical ML)

- TF-IDF vectorization with n-grams (1,2)
- Logistic Regression with balanced class weights
- Linear SVM

### BERTić (Transformer)

- Pre-trained on 8 billion Croatian tokens
- Fine-tuned for multi-class classification
- Focal loss for class imbalance

Model: [classla/bcms-bertic](https://huggingface.co/classla/bcms-bertic)

## Coded Language Lexicon

The lexicon contains Croatian dog whistle terms. **See [docs/WORKFLOW.md](docs/WORKFLOW.md) for complete workflow documentation.**

### Adding New Terms (Easy Way)

```bash
# 1. Edit the simple text file
nano data/lexicon/quick_add.txt

# Add lines like:
# new_term | meaning | TARGET
# Example: plandemija | COVID hoax | VAX

# 2. Import to main lexicon
python src/utils/add_term.py --import

# 3. Verify
python src/utils/add_term.py --list
```

### Target Codes

| Code | Description |
|------|-------------|
| MIG | Migrants/Refugees |
| SRB | Ethnic Serbs |
| LGBT | LGBTQ+ community |
| ROM | Roma |
| ELT | Elites/Globalists |
| VAX | Anti-vax/COVID |
| POL | Political groups |
| OTH | Other |

### Using the Lexicon in Python

```python
from src.utils.lexicon import CodedTermLexicon

lexicon = CodedTermLexicon('data/lexicon/coded_terms.json')
matches = lexicon.find_matches("Inženjeri opet prave nered u gradu")
print(matches)
```

### Do I Need to Re-train After Adding Terms?

**NO** - The lexicon is loaded fresh every time. No re-training needed for term additions.

## GPU Training

For faster training on AMD or NVIDIA GPUs, see **[docs/GPU_TRAINING.md](docs/GPU_TRAINING.md)**.

Quick setup for GPU:
```bash
# AMD ROCm
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# NVIDIA CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Key Resources

- **BERTić**: [https://huggingface.co/classla/bcms-bertic](https://huggingface.co/classla/bcms-bertic)
- **CLASSLA**: [https://github.com/clarinsi/classla](https://github.com/clarinsi/classla)
- **CLARIN.SI Datasets**: [https://www.clarin.si/repository/](https://www.clarin.si/repository/)
- **FRENK Hate Speech**: [https://huggingface.co/datasets/classla/FRENK-hate-hr](https://huggingface.co/datasets/classla/FRENK-hate-hr)

## References

- Ljubešić, N., & Lauc, D. (2021). BERTić - The Transformer Language Model for Bosnian, Croatian, Montenegrin and Serbian. ACL BSNLP Workshop.
- Shekhar, R., Karan, M., & Purver, M. (2022). CoRAL: a Context-aware Croatian Abusive Language Dataset. ACL Findings.
- Ljubešić, N., Erjavec, T., & Fišer, D. (2018). Datasets of Slovene and Croatian Moderated News Comments.

## Results

### Baseline Models (FRENK Dataset)

| Model | Accuracy | F1-Macro | F1-Weighted | MCC |
|-------|----------|----------|-------------|-----|
| Logistic Regression | 69.0% | 0.684 | 0.689 | 0.371 |
| SVM (Linear) | 68.5% | 0.680 | 0.684 | 0.361 |

### BERTić Model

Training in progress. Expected F1-Macro: 0.70-0.80

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note:** The FRENK dataset used in this project is subject to its own license. See [FRENK on HuggingFace](https://huggingface.co/datasets/classla/FRENK-hate-hr).

---

*University of Zagreb, Faculty of Electrical Engineering and Computing*
*Course: Obrada prirodnog jezika (Natural Language Processing)*
