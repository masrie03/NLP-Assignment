"""
preprocessing.py

Standalone preprocessing pipeline for the arXiv Paper Abstracts corpus.
Builds the reference vocabulary (Unigrams) and the bigram transition table
used by the spelling-correction engine (Non-word and Real-word detection).

This script is intentionally decoupled from the GUI application: it is run
once (offline) to produce two pickle files that the GUI loads instantly at
startup.

Usage:
    python preprocessing.py

Inputs:
    ../data/raw/arxiv_data.csv   (columns: titles, summaries, terms)

Outputs:
    ../models/unigrams.pkl   -> dict[str, int]              word -> frequency
    ../models/bigrams.pkl    -> dict[str, dict[str, int]]    w1 -> {w2: freq}
"""

import os
import re
import pickle
from collections import Counter, defaultdict

import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
RAW_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "arxiv_data.csv")
UNIGRAMS_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "unigrams.pkl")
BIGRAMS_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "bigrams.pkl")

# Minimum frequency threshold: words appearing fewer times than this are
# considered noise (typos from the original researchers, OCR artifacts, etc.)
MIN_UNIGRAM_FREQUENCY = 3

# Short English words that MUST be preserved even though they are very short
# (a naive "remove tokens of length <= 1" filter would wrongly delete them).
ESSENTIAL_SHORT_WORDS = {
    "a", "i", "an", "as", "at", "be", "by", "do", "go", "he", "if",
    "in", "is", "it", "me", "my", "no", "of", "on", "or", "so", "to",
    "up", "us", "we",
}

# Matches LaTeX math blocks: $$ ... $$ (display mode) and $ ... $ (inline mode).
# The display-mode pattern is applied first so that a stray inline "$" inside
# a "$$" block is not mistaken for a lone inline delimiter.
LATEX_DISPLAY_MATH_RE = re.compile(r"\$\$.*?\$\$", flags=re.DOTALL)
LATEX_INLINE_MATH_RE = re.compile(r"\$.*?\$", flags=re.DOTALL)

# After lowercasing and removing LaTeX, strip anything that isn't a letter or
# whitespace (this removes punctuation AND digits in one pass).
NON_ALPHA_RE = re.compile(r"[^a-z\s]")

WHITESPACE_RE = re.compile(r"\s+")


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def remove_latex_math(text: str) -> str:
    """Remove LaTeX equation blocks ($...$ and $$...$$) from a string.

    Scientific abstracts are full of inline/display equations that would
    otherwise pollute the vocabulary with garbage tokens (e.g. "x_i", "o(n)").
    """
    text = LATEX_DISPLAY_MATH_RE.sub(" ", text)
    text = LATEX_INLINE_MATH_RE.sub(" ", text)
    return text


def clean_text(text: str) -> str:
    """Full cleaning pipeline applied to a single title or abstract.

    Steps (in order):
      1. Remove LaTeX math blocks.
      2. Lowercase everything.
      3. Remove punctuation and digits.
      4. Collapse repeated whitespace.
    """
    text = remove_latex_math(text)
    text = text.lower()
    text = NON_ALPHA_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Split cleaned text into tokens, dropping isolated noise letters.

    A token of length 1 is kept only if it belongs to ESSENTIAL_SHORT_WORDS
    (e.g. "a", "i"); other single letters are leftover noise (e.g. stray
    variable names like "x", "n" that survived LaTeX removal) and are
    discarded.
    """
    tokens = text.split(" ")
    cleaned_tokens = []
    for tok in tokens:
        if not tok:
            continue
        if len(tok) == 1 and tok not in ESSENTIAL_SHORT_WORDS:
            continue
        cleaned_tokens.append(tok)
    return cleaned_tokens


# ----------------------------------------------------------------------------
# Corpus building
# ----------------------------------------------------------------------------
def load_corpus(csv_path: str) -> pd.DataFrame:
    """Load the arXiv abstracts CSV (columns: titles, summaries, terms)."""
    df = pd.read_csv(csv_path)
    required_cols = {"titles", "summaries"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing expected columns: {missing}")
    return df


def build_unigrams_and_bigrams(df: pd.DataFrame) -> tuple[Counter, dict]:
    """Iterate over titles + summaries and build raw (unfiltered) unigram
    and bigram counts.

    Bigrams are only counted WITHIN a single title or a single abstract
    (we never count a transition that spans two unrelated documents).
    """
    unigram_counts: Counter = Counter()
    bigram_counts: dict = defaultdict(Counter)

    total_documents = 0
    for _, row in df.iterrows():
        for raw_field in (row["titles"], row["summaries"]):
            if not isinstance(raw_field, str):
                continue
            cleaned = clean_text(raw_field)
            tokens = tokenize(cleaned)
            if not tokens:
                continue

            unigram_counts.update(tokens)

            for w1, w2 in zip(tokens, tokens[1:]):
                bigram_counts[w1][w2] += 1

            total_documents += 1

    print(f"Processed {total_documents} text fields (titles + summaries).")
    return unigram_counts, bigram_counts


def filter_unigrams(unigram_counts: Counter, min_freq: int) -> dict:
    """Apply the minimum-frequency threshold to eliminate noisy/rare tokens."""
    return {word: count for word, count in unigram_counts.items() if count >= min_freq}


def filter_bigrams(bigram_counts: dict, valid_vocabulary: set) -> dict:
    """Keep only bigram transitions between two words that both survived the
    unigram frequency filter, then convert to a plain (picklable) dict.
    """
    filtered = {}
    for w1, following_counter in bigram_counts.items():
        if w1 not in valid_vocabulary:
            continue
        kept = {
            w2: count
            for w2, count in following_counter.items()
            if w2 in valid_vocabulary
        }
        if kept:
            filtered[w1] = kept
    return filtered


def save_pickle(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 1 - Corpus preprocessing and dictionary construction")
    print("=" * 70)

    print(f"\nLoading corpus from: {RAW_CSV_PATH}")
    df = load_corpus(RAW_CSV_PATH)
    print(f"Corpus loaded: {len(df)} rows (arXiv papers).")

    print("\nCleaning text and building raw unigram/bigram counts...")
    raw_unigrams, raw_bigrams = build_unigrams_and_bigrams(df)

    total_word_occurrences = sum(raw_unigrams.values())
    print(f"Total word occurrences (before filtering): {total_word_occurrences:,}")
    print(f"Unique words (before filtering): {len(raw_unigrams):,}")

    print(f"\nApplying minimum frequency filter (Count >= {MIN_UNIGRAM_FREQUENCY})...")
    unigrams = filter_unigrams(raw_unigrams, MIN_UNIGRAM_FREQUENCY)
    print(f"Unique words after filtering: {len(unigrams):,}")

    kept_word_occurrences = sum(unigrams.values())
    print(f"Total word occurrences after filtering: {kept_word_occurrences:,}")

    print("\nFiltering bigrams to the retained vocabulary...")
    bigrams = filter_bigrams(raw_bigrams, set(unigrams.keys()))
    total_bigram_pairs = sum(len(v) for v in bigrams.values())
    print(f"Distinct bigram transitions kept: {total_bigram_pairs:,}")

    print(f"\nSaving unigrams to: {UNIGRAMS_OUTPUT_PATH}")
    save_pickle(unigrams, UNIGRAMS_OUTPUT_PATH)

    print(f"Saving bigrams to: {BIGRAMS_OUTPUT_PATH}")
    save_pickle(bigrams, BIGRAMS_OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total word occurrences in cleaned corpus : {kept_word_occurrences:,}")
    print(f"Unique vocabulary size (Unigrams)        : {len(unigrams):,}")
    print(f"Assignment requirement (> 100,000 words) : "
          f"{'SATISFIED' if kept_word_occurrences > 100_000 else 'NOT SATISFIED'}")
    print("Done.")


if __name__ == "__main__":
    main()
