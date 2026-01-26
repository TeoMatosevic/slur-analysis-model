# Project Workflow Guide

Quick reference for running the Croatian Hate Speech Detection project.

---

## Adding New Coded Terms

### Where Do Terms Live?

```
data/lexicon/
├── quick_add.txt       <- YOU EDIT THIS (simple format)
├── coded_terms.json    <- Main lexicon (auto-generated)
```

### Step-by-Step

```bash
# 1. Edit quick_add.txt
nano data/lexicon/quick_add.txt

# Add lines like:
# new_term | meaning | TARGET
# Example: četnički | Serb nationalist slur | SRB

# 2. Import to main lexicon
python src/utils/add_term.py --import

# 3. Verify it worked
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

---

## Do I Need to Re-train After Adding Terms?

| What You Did | Re-train? |
|--------------|-----------|
| Added new terms to lexicon | **NO** |
| Got new annotated data | **YES** |
| Changed model parameters | **YES** |

**The lexicon is loaded fresh every time the code runs.** No re-training needed for term additions.

---

## Complete Pipeline

### 1. One-Time Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download Croatian NLP models
python -c "import classla; classla.download('hr', type='nonstandard')"

# Test everything
python src/utils/add_term.py --list
```

### 2. Add Coded Terms (Ongoing)

```bash
nano data/lexicon/quick_add.txt   # Add terms
python src/utils/add_term.py --import   # Import
python src/utils/add_term.py --list     # Verify
```

### 3. Scrape Data (Optional)

```bash
python src/scraping/scraper.py --portal index.hr --sections vijesti --max-articles 50
```

### 4. Preprocess Data

```bash
python src/preprocessing/preprocessor.py \
    --input data/raw/comments.jsonl \
    --output data/processed/comments.jsonl
```

### 5. Train Models

```bash
# Train all models
python src/training/train.py \
    --data data/processed/comments.jsonl \
    --model all

# Or just baseline
python src/training/train.py --data data/processed/comments.jsonl --model baseline

# Or just BERTić
python src/training/train.py --data data/processed/comments.jsonl --model bertic
```

### 6. Evaluate

```bash
python src/training/evaluate.py \
    --data data/processed/test.jsonl \
    --model bertic \
    --model-path checkpoints/bertic/best_model
```

---

## Using the Lexicon in Python

```python
from src.utils.lexicon import CodedTermLexicon

# Load (always gets latest terms)
lexicon = CodedTermLexicon('data/lexicon/coded_terms.json')

# Find matches
text = "Inženjeri opet prave probleme"
matches = lexicon.find_matches(text)
print(matches)
# [{'term': 'inženjeri', 'coded_meaning': '...', 'target_group': 'TGT_MIG'}]

# Check if contains coded terms
has_coded = lexicon.contains_coded_term(text)
print(has_coded)  # True

# Get statistics
stats = lexicon.get_statistics()
print(stats)
```

---

## Quick Commands Reference

| Task | Command |
|------|---------|
| List all terms | `python src/utils/add_term.py --list` |
| Import new terms | `python src/utils/add_term.py --import` |
| Test lexicon | `python src/utils/lexicon.py --text "your text"` |
| Test scraper | `python src/scraping/scraper.py --test` |
| Test preprocessor | `python src/preprocessing/preprocessor.py --test` |
| Test baseline | `python src/models/baseline.py --test` |
| Test BERTić | `python src/models/bertic.py --test` |

---

## Current Lexicon Stats (21 terms)

| Target | Count | Examples |
|--------|-------|----------|
| MIG | 5 | inženjeri, doktori, specijalisti |
| ELT | 6 | globalisti, soroševci, NWO |
| VAX | 6 | ovce, plandemija, čipiranje |
| LGBT | 3 | lobi, gay agenda, ideologija |
| POL | 1 | jugokomunisti |
