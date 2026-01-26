"""
Text preprocessor for Croatian hate speech detection.
Uses CLASSLA (Stanford NLP fork) for Croatian language processing.
"""

import re
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field

import pandas as pd
from tqdm import tqdm

try:
    import classla
    CLASSLA_AVAILABLE = True
except ImportError:
    CLASSLA_AVAILABLE = False
    logging.warning("CLASSLA not available. Install with: pip install classla")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedText:
    """Represents preprocessed text with NLP annotations."""
    original: str
    cleaned: str
    tokens: List[str] = field(default_factory=list)
    lemmas: List[str] = field(default_factory=list)
    pos_tags: List[str] = field(default_factory=list)
    is_valid: bool = True
    word_count: int = 0
    char_count: int = 0


class TextCleaner:
    """Clean and normalize text for NLP processing."""

    # URL pattern
    URL_PATTERN = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )

    # Mention pattern (@username)
    MENTION_PATTERN = re.compile(r'@[\w]+')

    # Hashtag pattern
    HASHTAG_PATTERN = re.compile(r'#[\w]+')

    # Multiple spaces
    MULTI_SPACE_PATTERN = re.compile(r'\s+')

    # Multiple punctuation
    MULTI_PUNCT_PATTERN = re.compile(r'([!?.]){2,}')

    # Email pattern
    EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')

    # HTML tags
    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')

    def __init__(
        self,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        remove_emails: bool = True,
        remove_html: bool = True,
        lowercase: bool = False,
        normalize_whitespace: bool = True,
        normalize_punctuation: bool = True,
        min_length: int = 10,
        max_length: int = 512
    ):
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_emails = remove_emails
        self.remove_html = remove_html
        self.lowercase = lowercase
        self.normalize_whitespace = normalize_whitespace
        self.normalize_punctuation = normalize_punctuation
        self.min_length = min_length
        self.max_length = max_length

    def clean(self, text: str) -> Tuple[str, bool]:
        """
        Clean the text and return (cleaned_text, is_valid).

        Args:
            text: Input text to clean

        Returns:
            Tuple of (cleaned_text, is_valid)
        """
        if not text or not isinstance(text, str):
            return "", False

        # Remove HTML tags
        if self.remove_html:
            text = self.HTML_TAG_PATTERN.sub(' ', text)

        # Remove URLs
        if self.remove_urls:
            text = self.URL_PATTERN.sub(' ', text)

        # Remove emails
        if self.remove_emails:
            text = self.EMAIL_PATTERN.sub(' ', text)

        # Remove mentions
        if self.remove_mentions:
            text = self.MENTION_PATTERN.sub(' ', text)

        # Process hashtags (optionally keep the text without #)
        if self.remove_hashtags:
            text = self.HASHTAG_PATTERN.sub(' ', text)
        else:
            # Keep hashtag text without the # symbol
            text = re.sub(r'#([\w]+)', r'\1', text)

        # Normalize punctuation
        if self.normalize_punctuation:
            text = self.MULTI_PUNCT_PATTERN.sub(r'\1', text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = self.MULTI_SPACE_PATTERN.sub(' ', text)
            text = text.strip()

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Check validity
        is_valid = self.min_length <= len(text) <= self.max_length

        return text, is_valid


class CroatianNLPProcessor:
    """NLP processor for Croatian using CLASSLA."""

    def __init__(
        self,
        language: str = 'hr',
        processor_type: str = 'nonstandard',
        processors: str = 'tokenize,pos,lemma'
    ):
        """
        Initialize Croatian NLP processor.

        Args:
            language: Language code ('hr' for Croatian)
            processor_type: 'standard' or 'nonstandard' (for internet text)
            processors: Comma-separated list of processors to use
        """
        self.language = language
        self.processor_type = processor_type
        self.processors = processors
        self.nlp = None

    def initialize(self):
        """Download models and initialize the pipeline."""
        if not CLASSLA_AVAILABLE:
            raise RuntimeError("CLASSLA not available. Install with: pip install classla")

        logger.info(f"Downloading CLASSLA models for {self.language} ({self.processor_type})...")
        classla.download(self.language, type=self.processor_type)

        logger.info("Initializing CLASSLA pipeline...")
        self.nlp = classla.Pipeline(
            self.language,
            type=self.processor_type,
            processors=self.processors
        )
        logger.info("CLASSLA pipeline initialized")

    def process(self, text: str) -> Dict:
        """
        Process text through the NLP pipeline.

        Args:
            text: Input text

        Returns:
            Dictionary with tokens, lemmas, and POS tags
        """
        if not self.nlp:
            self.initialize()

        doc = self.nlp(text)

        tokens = []
        lemmas = []
        pos_tags = []

        for sentence in doc.sentences:
            for word in sentence.words:
                tokens.append(word.text)
                lemmas.append(word.lemma)
                pos_tags.append(word.upos)

        return {
            'tokens': tokens,
            'lemmas': lemmas,
            'pos_tags': pos_tags
        }


class TextPreprocessor:
    """Main preprocessor class that combines cleaning and NLP processing."""

    def __init__(
        self,
        use_classla: bool = True,
        classla_type: str = 'nonstandard',
        clean_config: Optional[Dict] = None
    ):
        """
        Initialize the text preprocessor.

        Args:
            use_classla: Whether to use CLASSLA for NLP processing
            classla_type: 'standard' or 'nonstandard' for CLASSLA
            clean_config: Configuration dict for TextCleaner
        """
        # Initialize cleaner
        clean_config = clean_config or {}
        self.cleaner = TextCleaner(**clean_config)

        # Initialize NLP processor
        self.use_classla = use_classla and CLASSLA_AVAILABLE
        self.nlp_processor = None
        if self.use_classla:
            self.nlp_processor = CroatianNLPProcessor(
                processor_type=classla_type
            )

    def preprocess(self, text: str, apply_nlp: bool = True) -> ProcessedText:
        """
        Preprocess a single text.

        Args:
            text: Input text
            apply_nlp: Whether to apply NLP processing

        Returns:
            ProcessedText object with all annotations
        """
        # Clean the text
        cleaned, is_valid = self.cleaner.clean(text)

        result = ProcessedText(
            original=text,
            cleaned=cleaned,
            is_valid=is_valid,
            word_count=len(cleaned.split()) if cleaned else 0,
            char_count=len(cleaned) if cleaned else 0
        )

        # Apply NLP processing
        if is_valid and apply_nlp and self.use_classla:
            try:
                nlp_result = self.nlp_processor.process(cleaned)
                result.tokens = nlp_result['tokens']
                result.lemmas = nlp_result['lemmas']
                result.pos_tags = nlp_result['pos_tags']
            except Exception as e:
                logger.warning(f"NLP processing failed: {e}")

        return result

    def preprocess_batch(
        self,
        texts: List[str],
        apply_nlp: bool = True,
        show_progress: bool = True
    ) -> List[ProcessedText]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of input texts
            apply_nlp: Whether to apply NLP processing
            show_progress: Whether to show progress bar

        Returns:
            List of ProcessedText objects
        """
        results = []
        iterator = tqdm(texts, desc="Preprocessing") if show_progress else texts

        for text in iterator:
            result = self.preprocess(text, apply_nlp=apply_nlp)
            results.append(result)

        return results


def preprocess_dataset(
    input_path: str,
    output_path: str,
    text_column: str = 'text',
    use_classla: bool = True,
    classla_type: str = 'nonstandard'
):
    """
    Preprocess a dataset file (JSONL or CSV).

    Args:
        input_path: Path to input file
        output_path: Path to output file
        text_column: Name of the column containing text
        use_classla: Whether to use CLASSLA
        classla_type: 'standard' or 'nonstandard'
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Load data
    logger.info(f"Loading data from {input_path}")
    if input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
    elif input_path.suffix in ['.jsonl', '.json']:
        df = pd.read_json(input_path, lines=input_path.suffix == '.jsonl')
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        use_classla=use_classla,
        classla_type=classla_type
    )

    # Process texts
    texts = df[text_column].tolist()
    results = preprocessor.preprocess_batch(texts)

    # Add processed columns to dataframe
    df['cleaned_text'] = [r.cleaned for r in results]
    df['tokens'] = [r.tokens for r in results]
    df['lemmas'] = [r.lemmas for r in results]
    df['pos_tags'] = [r.pos_tags for r in results]
    df['is_valid'] = [r.is_valid for r in results]
    df['word_count'] = [r.word_count for r in results]

    # Filter invalid texts
    valid_count = df['is_valid'].sum()
    logger.info(f"Valid texts: {valid_count}/{len(df)} ({100*valid_count/len(df):.1f}%)")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == '.csv':
        df.to_csv(output_path, index=False)
    else:
        df.to_json(output_path, orient='records', lines=True, force_ascii=False)

    logger.info(f"Saved preprocessed data to {output_path}")


def main():
    """Main entry point for the preprocessor."""
    parser = argparse.ArgumentParser(description="Preprocess Croatian text data")
    parser.add_argument('--input', type=str, help="Input file path (CSV or JSONL)")
    parser.add_argument('--output', type=str, help="Output file path")
    parser.add_argument('--text-column', type=str, default='text', help="Name of text column")
    parser.add_argument('--no-classla', action='store_true', help="Disable CLASSLA processing")
    parser.add_argument('--classla-type', type=str, default='nonstandard',
                        choices=['standard', 'nonstandard'])
    parser.add_argument('--test', action='store_true', help="Run test mode")

    args = parser.parse_args()

    if args.test:
        logger.info("Running in test mode")
        print("Test mode: Preprocessor module loaded successfully")
        print(f"CLASSLA available: {CLASSLA_AVAILABLE}")

        # Test basic cleaning
        cleaner = TextCleaner()
        test_text = "Ovo je test https://example.com @user #hashtag!!!"
        cleaned, valid = cleaner.clean(test_text)
        print(f"Original: {test_text}")
        print(f"Cleaned: {cleaned}")
        print(f"Valid: {valid}")

        # Test CLASSLA if available
        if CLASSLA_AVAILABLE:
            print("\nTesting CLASSLA (this may take a moment on first run)...")
            try:
                preprocessor = TextPreprocessor(use_classla=True, classla_type='nonstandard')
                result = preprocessor.preprocess("Ovo je testna rečenica na hrvatskom jeziku.")
                print(f"Tokens: {result.tokens}")
                print(f"Lemmas: {result.lemmas}")
                print(f"POS tags: {result.pos_tags}")
            except Exception as e:
                print(f"CLASSLA test failed: {e}")
                print("You may need to run: python -c \"import classla; classla.download('hr', type='nonstandard')\"")

        return

    if not args.input or not args.output:
        parser.error("--input and --output are required (unless --test is specified)")

    preprocess_dataset(
        input_path=args.input,
        output_path=args.output,
        text_column=args.text_column,
        use_classla=not args.no_classla,
        classla_type=args.classla_type
    )


if __name__ == "__main__":
    main()
