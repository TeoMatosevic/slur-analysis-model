# Contributing Guide

Thank you for your interest in contributing to the Croatian Hate Speech Detection project.

## Getting Started

### Prerequisites

- Python 3.9+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/TeoMatosevic/slur-analysis-model.git
cd slur-analysis-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Croatian NLP models
python -c "import classla; classla.download('hr', type='nonstandard')"

# Verify installation
python src/models/baseline.py --test
```

## Project Structure

```
opj/
├── data/
│   ├── processed/     # Training data (FRENK dataset)
│   └── lexicon/       # Coded terms lexicon
├── src/
│   ├── models/        # ML models (baseline, BERTić, XLM-RoBERTa)
│   ├── training/      # Training and evaluation scripts
│   ├── preprocessing/ # Text preprocessing with CLASSLA
│   ├── scraping/      # Web scraping utilities
│   └── utils/         # Lexicon management
├── configs/           # Configuration files
└── docs/              # Documentation
```

## How to Contribute

### Adding Coded Terms (Dog Whistles)

The most valuable contribution is expanding the coded language lexicon:

1. Edit `data/lexicon/quick_add.txt`:
   ```
   term | meaning | TARGET_CODE
   ```

2. Import to main lexicon:
   ```bash
   python src/utils/add_term.py --import
   ```

3. Verify:
   ```bash
   python src/utils/add_term.py --list
   ```

**Target codes:** MIG (migrants), SRB (Serbs), LGBT, ROM (Roma), ELT (elites), VAX (anti-vax), POL (political), OTH (other)

### Code Contributions

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Test your changes: `python -m pytest` (if tests exist)
5. Commit: `git commit -m "Add your feature"`
6. Push: `git push origin feature/your-feature`
7. Open a Pull Request

### Reporting Issues

- Use GitHub Issues
- Include Python version and OS
- Provide minimal reproducible example
- Include full error traceback

## Code Style

- Follow PEP 8
- Use type hints where practical
- Add docstrings to functions
- Keep functions focused and small

## Data Guidelines

- **Never commit sensitive data** (personal information, credentials)
- Raw scraped data stays in `data/raw/` (gitignored)
- Processed/anonymized data goes in `data/processed/`

## Questions?

Open an issue or contact the maintainers.
