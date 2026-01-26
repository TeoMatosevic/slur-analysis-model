"""
Lexicon utilities for coded/dog whistle term detection.
Manages the Croatian coded language lexicon and provides matching functions.
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CodedTerm:
    """Represents a coded/dog whistle term."""
    id: str
    term: str
    literal_meaning: str
    coded_meaning: str
    target_group: str
    implicitness: str = "IMP_HR"
    sentiment: str = "negative"
    frequency: str = "medium"
    related_terms: List[str] = field(default_factory=list)
    example_usage: str = ""
    notes: str = ""


class CodedTermLexicon:
    """
    Manages a lexicon of coded/dog whistle terms for hate speech detection.
    """

    def __init__(self, lexicon_path: Optional[str] = None):
        """
        Initialize the lexicon.

        Args:
            lexicon_path: Path to the lexicon JSON file
        """
        self.terms: Dict[str, CodedTerm] = {}  # term -> CodedTerm
        self.terms_by_target: Dict[str, List[CodedTerm]] = defaultdict(list)
        self.all_terms: Set[str] = set()  # For quick lookup
        self.term_patterns: Dict[str, re.Pattern] = {}  # Compiled regex patterns

        if lexicon_path:
            self.load(lexicon_path)

    def load(self, lexicon_path: str):
        """Load lexicon from JSON file."""
        path = Path(lexicon_path)
        if not path.exists():
            logger.warning(f"Lexicon file not found: {lexicon_path}")
            return

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Load coded terms
        for term_data in data.get('coded_terms', []):
            term = CodedTerm(
                id=term_data.get('id', ''),
                term=term_data['term'],
                literal_meaning=term_data.get('literal_meaning', ''),
                coded_meaning=term_data.get('coded_meaning', ''),
                target_group=term_data.get('target_group', 'TGT_OTH'),
                implicitness=term_data.get('implicitness', 'IMP_HR'),
                sentiment=term_data.get('sentiment', 'negative'),
                frequency=term_data.get('frequency', 'medium'),
                related_terms=term_data.get('related_terms', []),
                example_usage=term_data.get('example_usage', ''),
                notes=term_data.get('notes', '')
            )
            self.add_term(term)

        # Also add user-provided terms
        for term_data in data.get('user_provided_terms', []):
            if isinstance(term_data, dict) and 'term' in term_data:
                term = CodedTerm(
                    id=term_data.get('id', f"USR{len(self.terms)}"),
                    term=term_data['term'],
                    literal_meaning=term_data.get('literal_meaning', ''),
                    coded_meaning=term_data.get('coded_meaning', ''),
                    target_group=term_data.get('target_group', 'TGT_OTH'),
                )
                self.add_term(term)

        logger.info(f"Loaded {len(self.terms)} coded terms from lexicon")

    def add_term(self, term: CodedTerm):
        """Add a term to the lexicon."""
        self.terms[term.term.lower()] = term
        self.terms_by_target[term.target_group].append(term)
        self.all_terms.add(term.term.lower())

        # Also add related terms for quick lookup
        for related in term.related_terms:
            self.all_terms.add(related.lower())

        # Compile regex pattern for the term (word boundary matching)
        pattern = re.compile(
            r'\b' + re.escape(term.term) + r'\b',
            re.IGNORECASE
        )
        self.term_patterns[term.term.lower()] = pattern

    def find_matches(self, text: str) -> List[Dict]:
        """
        Find all coded term matches in text.

        Args:
            text: Input text to search

        Returns:
            List of match dictionaries with term info and positions
        """
        matches = []
        text_lower = text.lower()

        for term_str, term_obj in self.terms.items():
            pattern = self.term_patterns.get(term_str)
            if pattern:
                for match in pattern.finditer(text):
                    matches.append({
                        'term': term_obj.term,
                        'coded_meaning': term_obj.coded_meaning,
                        'target_group': term_obj.target_group,
                        'implicitness': term_obj.implicitness,
                        'start': match.start(),
                        'end': match.end(),
                        'matched_text': match.group()
                    })

        # Sort by position
        matches.sort(key=lambda x: x['start'])
        return matches

    def contains_coded_term(self, text: str) -> bool:
        """Check if text contains any coded term."""
        text_lower = text.lower()
        for term in self.all_terms:
            if term in text_lower:
                return True
        return False

    def get_feature_vector(self, text: str) -> Dict[str, int]:
        """
        Get a feature vector for text based on lexicon matches.

        Args:
            text: Input text

        Returns:
            Dictionary with feature counts
        """
        matches = self.find_matches(text)

        features = {
            'coded_term_count': len(matches),
            'unique_terms': len(set(m['term'] for m in matches)),
            'TGT_MIG': 0,
            'TGT_SRB': 0,
            'TGT_LGBT': 0,
            'TGT_ROM': 0,
            'TGT_ELT': 0,
            'TGT_VAX': 0,
            'TGT_OTH': 0,
            'IMP_LI': 0,
            'IMP_HR': 0,
            'IMP_CTX': 0,
        }

        for match in matches:
            target = match.get('target_group', 'TGT_OTH')
            if target in features:
                features[target] += 1

            impl = match.get('implicitness', 'IMP_HR')
            if impl in features:
                features[impl] += 1

        return features

    def get_terms_for_target(self, target_group: str) -> List[CodedTerm]:
        """Get all terms targeting a specific group."""
        return self.terms_by_target.get(target_group, [])

    def export_term_list(self) -> List[str]:
        """Export a simple list of all terms."""
        return list(self.all_terms)

    def get_statistics(self) -> Dict:
        """Get lexicon statistics."""
        stats = {
            'total_terms': len(self.terms),
            'total_variants': len(self.all_terms),
            'by_target': {},
            'by_implicitness': defaultdict(int),
            'by_frequency': defaultdict(int),
        }

        for target, terms in self.terms_by_target.items():
            stats['by_target'][target] = len(terms)

        for term in self.terms.values():
            stats['by_implicitness'][term.implicitness] += 1
            stats['by_frequency'][term.frequency] += 1

        return stats


def create_lexicon_features(texts: List[str], lexicon: CodedTermLexicon) -> List[Dict]:
    """
    Create lexicon-based features for a list of texts.

    Args:
        texts: List of input texts
        lexicon: CodedTermLexicon instance

    Returns:
        List of feature dictionaries
    """
    return [lexicon.get_feature_vector(text) for text in texts]


def main():
    """Test the lexicon functionality."""
    import argparse

    parser = argparse.ArgumentParser(description="Test coded term lexicon")
    parser.add_argument('--lexicon', type=str, default='data/lexicon/coded_terms.json')
    parser.add_argument('--text', type=str, help="Text to analyze")
    parser.add_argument('--stats', action='store_true', help="Show lexicon statistics")

    args = parser.parse_args()

    lexicon = CodedTermLexicon(args.lexicon)

    if args.stats:
        stats = lexicon.get_statistics()
        print("Lexicon Statistics:")
        print(f"  Total terms: {stats['total_terms']}")
        print(f"  Total variants: {stats['total_variants']}")
        print("\n  By target group:")
        for target, count in stats['by_target'].items():
            print(f"    {target}: {count}")
        print("\n  By implicitness:")
        for impl, count in stats['by_implicitness'].items():
            print(f"    {impl}: {count}")

    if args.text:
        print(f"\nAnalyzing: '{args.text}'")
        matches = lexicon.find_matches(args.text)
        if matches:
            print("Matches found:")
            for m in matches:
                print(f"  - '{m['matched_text']}': {m['coded_meaning']} (target: {m['target_group']})")
        else:
            print("No coded terms found.")

        features = lexicon.get_feature_vector(args.text)
        print(f"\nFeature vector: {features}")


if __name__ == "__main__":
    main()
