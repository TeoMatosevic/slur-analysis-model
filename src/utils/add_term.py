#!/usr/bin/env python3
"""
Easy lexicon management tool.
Add coded terms quickly using a simple text format.
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_LEXICON = "data/lexicon/coded_terms.json"
DEFAULT_QUICK_ADD = "data/lexicon/quick_add.txt"

# Target code mapping (short -> full)
TARGET_CODES = {
    'MIG': 'TGT_MIG',
    'SRB': 'TGT_SRB',
    'LGBT': 'TGT_LGBT',
    'ROM': 'TGT_ROM',
    'ELT': 'TGT_ELT',
    'VAX': 'TGT_VAX',
    'POL': 'TGT_POL',
    'OTH': 'TGT_OTH',
}

# Reverse mapping (full -> short)
TARGET_CODES_REV = {v: k for k, v in TARGET_CODES.items()}


def load_lexicon(path: str) -> Dict:
    """Load the main lexicon JSON file."""
    path = Path(path)
    if not path.exists():
        return {
            "metadata": {
                "description": "Croatian coded language lexicon",
                "version": "1.0.0",
                "last_updated": datetime.now().strftime("%Y-%m-%d")
            },
            "coded_terms": [],
            "user_provided_terms": []
        }

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_lexicon(data: Dict, path: str):
    """Save the lexicon to JSON file."""
    # Update timestamp
    if 'metadata' in data:
        data['metadata']['last_updated'] = datetime.now().strftime("%Y-%m-%d")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved lexicon to {path}")


def parse_quick_add_line(line: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse a line from quick_add.txt.

    Format: term | meaning | target

    Returns:
        Tuple of (term, meaning, target_code) or None if invalid
    """
    line = line.strip()

    # Skip empty lines and comments
    if not line or line.startswith('#'):
        return None

    parts = [p.strip() for p in line.split('|')]

    if len(parts) < 3:
        logger.warning(f"Invalid line (need 3 parts): {line}")
        return None

    term = parts[0]
    meaning = parts[1]
    target_short = parts[2].upper()

    # Convert short code to full code
    target = TARGET_CODES.get(target_short, 'TGT_OTH')

    if target_short not in TARGET_CODES:
        logger.warning(f"Unknown target code '{target_short}' for term '{term}', using OTH")

    return (term, meaning, target)


def import_from_quick_add(quick_add_path: str, lexicon_path: str) -> int:
    """
    Import terms from quick_add.txt to the main lexicon.

    Returns:
        Number of terms imported
    """
    quick_path = Path(quick_add_path)
    if not quick_path.exists():
        logger.error(f"Quick add file not found: {quick_add_path}")
        return 0

    # Load current lexicon
    lexicon = load_lexicon(lexicon_path)

    # Get existing terms (lowercase for comparison)
    existing_terms = set()
    for term in lexicon.get('coded_terms', []):
        existing_terms.add(term['term'].lower())
    for term in lexicon.get('user_provided_terms', []):
        if isinstance(term, dict):
            existing_terms.add(term.get('term', '').lower())

    # Read and parse quick_add.txt
    new_terms = []
    with open(quick_path, 'r', encoding='utf-8') as f:
        for line in f:
            parsed = parse_quick_add_line(line)
            if parsed:
                term, meaning, target = parsed
                if term.lower() not in existing_terms:
                    new_terms.append({
                        'term': term,
                        'coded_meaning': meaning,
                        'target_group': target
                    })
                    existing_terms.add(term.lower())

    if not new_terms:
        logger.info("No new terms to import")
        return 0

    # Add new terms to user_provided_terms
    if 'user_provided_terms' not in lexicon:
        lexicon['user_provided_terms'] = []

    # Generate IDs for new terms
    max_id = len(lexicon.get('coded_terms', [])) + len(lexicon.get('user_provided_terms', []))
    for i, term in enumerate(new_terms):
        term['id'] = f"USR{max_id + i + 1:03d}"
        lexicon['user_provided_terms'].append(term)

    # Save updated lexicon
    save_lexicon(lexicon, lexicon_path)

    logger.info(f"Imported {len(new_terms)} new terms:")
    for term in new_terms:
        logger.info(f"  + {term['term']} -> {term['coded_meaning']} ({TARGET_CODES_REV.get(term['target_group'], term['target_group'])})")

    return len(new_terms)


def export_to_quick_add(lexicon_path: str, quick_add_path: str):
    """Export all terms from lexicon to quick_add.txt format."""
    lexicon = load_lexicon(lexicon_path)

    lines = [
        "# Croatian Coded Terms - Quick Add Format",
        "# Format: term | meaning | target",
        "# Target codes: MIG, SRB, LGBT, ROM, ELT, VAX, POL, OTH",
        "",
    ]

    # Export coded_terms
    for term in lexicon.get('coded_terms', []):
        target_short = TARGET_CODES_REV.get(term.get('target_group', 'TGT_OTH'), 'OTH')
        meaning = term.get('coded_meaning', '')
        lines.append(f"{term['term']} | {meaning} | {target_short}")

    # Export user_provided_terms
    for term in lexicon.get('user_provided_terms', []):
        if isinstance(term, dict):
            target_short = TARGET_CODES_REV.get(term.get('target_group', 'TGT_OTH'), 'OTH')
            meaning = term.get('coded_meaning', '')
            lines.append(f"{term['term']} | {meaning} | {target_short}")

    quick_path = Path(quick_add_path)
    quick_path.parent.mkdir(parents=True, exist_ok=True)

    with open(quick_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    logger.info(f"Exported {len(lines) - 4} terms to {quick_add_path}")


def add_single_term(term: str, meaning: str, target: str, lexicon_path: str):
    """Add a single term directly."""
    lexicon = load_lexicon(lexicon_path)

    # Check if exists
    existing = set()
    for t in lexicon.get('coded_terms', []):
        existing.add(t['term'].lower())
    for t in lexicon.get('user_provided_terms', []):
        if isinstance(t, dict):
            existing.add(t.get('term', '').lower())

    if term.lower() in existing:
        logger.warning(f"Term '{term}' already exists in lexicon")
        return False

    # Convert target code
    target_full = TARGET_CODES.get(target.upper(), 'TGT_OTH')

    # Generate ID
    max_id = len(lexicon.get('coded_terms', [])) + len(lexicon.get('user_provided_terms', []))

    new_term = {
        'id': f"USR{max_id + 1:03d}",
        'term': term,
        'coded_meaning': meaning,
        'target_group': target_full
    }

    if 'user_provided_terms' not in lexicon:
        lexicon['user_provided_terms'] = []

    lexicon['user_provided_terms'].append(new_term)
    save_lexicon(lexicon, lexicon_path)

    logger.info(f"Added: {term} -> {meaning} ({target.upper()})")
    return True


def list_terms(lexicon_path: str):
    """List all terms in the lexicon."""
    lexicon = load_lexicon(lexicon_path)

    print("\n=== CODED TERMS LEXICON ===\n")

    # Group by target
    by_target: Dict[str, List] = {}

    for term in lexicon.get('coded_terms', []):
        target = term.get('target_group', 'TGT_OTH')
        if target not in by_target:
            by_target[target] = []
        by_target[target].append(term)

    for term in lexicon.get('user_provided_terms', []):
        if isinstance(term, dict):
            target = term.get('target_group', 'TGT_OTH')
            if target not in by_target:
                by_target[target] = []
            by_target[target].append(term)

    # Print grouped
    for target in sorted(by_target.keys()):
        target_short = TARGET_CODES_REV.get(target, target)
        print(f"\n[{target_short}] {target}:")
        for term in by_target[target]:
            meaning = term.get('coded_meaning', term.get('meaning', ''))
            print(f"  - {term['term']}: {meaning}")

    total = sum(len(v) for v in by_target.values())
    print(f"\n--- Total: {total} terms ---\n")


def main():
    parser = argparse.ArgumentParser(
        description="Easy lexicon management for Croatian coded terms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import terms from quick_add.txt
  python add_term.py --import

  # Add a single term
  python add_term.py --term "specijalisti" --meaning "immigrants" --target MIG

  # List all terms
  python add_term.py --list

  # Export to quick_add.txt format
  python add_term.py --export
        """
    )

    parser.add_argument('--import', dest='import_file', nargs='?', const=DEFAULT_QUICK_ADD,
                        metavar='FILE', help=f"Import from quick_add file (default: {DEFAULT_QUICK_ADD})")
    parser.add_argument('--export', dest='export_file', nargs='?', const=DEFAULT_QUICK_ADD,
                        metavar='FILE', help=f"Export to quick_add format (default: {DEFAULT_QUICK_ADD})")
    parser.add_argument('--term', type=str, help="Term to add")
    parser.add_argument('--meaning', type=str, help="Coded meaning")
    parser.add_argument('--target', type=str, choices=list(TARGET_CODES.keys()),
                        help="Target group (MIG, SRB, LGBT, ROM, ELT, VAX, POL, OTH)")
    parser.add_argument('--list', action='store_true', help="List all terms")
    parser.add_argument('--lexicon', type=str, default=DEFAULT_LEXICON,
                        help=f"Path to lexicon JSON (default: {DEFAULT_LEXICON})")

    args = parser.parse_args()

    # Handle import
    if args.import_file:
        import_from_quick_add(args.import_file, args.lexicon)
        return

    # Handle export
    if args.export_file:
        export_to_quick_add(args.lexicon, args.export_file)
        return

    # Handle list
    if args.list:
        list_terms(args.lexicon)
        return

    # Handle single term addition
    if args.term:
        if not args.meaning or not args.target:
            parser.error("--term requires --meaning and --target")
        add_single_term(args.term, args.meaning, args.target, args.lexicon)
        return

    # No action specified - show help
    parser.print_help()


if __name__ == "__main__":
    main()
