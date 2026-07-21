"""
nlp_engine.py

Core algorithmic engine for the spelling-correction system.

This module is completely independent from the GUI: it can be imported,
unit-tested, or driven from a notebook. It loads the pickled Unigram /
Bigram tables produced by preprocessing.py and exposes:

  - Minimum Edit Distance (Levenshtein) via Dynamic Programming.
  - Non-word candidate generation (words absent from the dictionary).
  - Real-word (contextual) error detection using bigram probabilities
    with Add-k (Laplace) smoothing.
  - Real-word candidate generation based on the bigram transition table.

All public functions/docstrings/messages are in English, as required.
"""

import os
import re
import pickle
from dataclasses import dataclass, field


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
UNIGRAMS_PATH = os.path.join(MODELS_DIR, "unigrams.pkl")
BIGRAMS_PATH = os.path.join(MODELS_DIR, "bigrams.pkl")

# A "word" for detection purposes is a maximal run of alphabetic characters.
# Using finditer (rather than split) lets us keep the exact start/end
# character offsets needed by the GUI to highlight text in the Tkinter widget.
WORD_TOKEN_RE = re.compile(r"[A-Za-z]+")

# Default thresholds (tunable, documented for the report's algorithmic section)
DEFAULT_MAX_EDIT_DISTANCE = 2      # candidates beyond this distance are not proposed
DEFAULT_TOP_N_SUGGESTIONS = 5
DEFAULT_ADD_K = 1.0                # Laplace/Add-k smoothing constant
DEFAULT_REALWORD_PROB_THRESHOLD = 5e-5  # below this smoothed probability -> suspect
# NOTE (for the report): this threshold was calibrated empirically on the
# arXiv vocabulary (~32k words). Because Add-k smoothing spreads probability
# mass over the whole vocabulary, the "floor" probability for an unseen
# bigram is on the order of 1/V; the threshold must sit somewhat above that
# floor to catch unseen-but-plausible-looking transitions, while staying low
# enough to avoid flagging rare-but-valid scientific collocations.


# ----------------------------------------------------------------------------
# Minimum Edit Distance (Levenshtein) - Dynamic Programming
# ----------------------------------------------------------------------------
def levenshtein_matrix(source: str, target: str,
                        insertion_cost: int = 1,
                        deletion_cost: int = 1,
                        substitution_cost: int = 1) -> list[list[int]]:
    """Build the full Dynamic Programming matrix for the Levenshtein distance.

    matrix[i][j] = minimum edit distance between source[:i] and target[:j].

    Returning the whole matrix (not just the final distance) is useful for
    the report: it lets us show, step by step, how the optimal alignment
    is derived (backtracking through insertions/deletions/substitutions).
    """
    n, m = len(source), len(target)
    matrix = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        matrix[i][0] = i * deletion_cost
    for j in range(1, m + 1):
        matrix[0][j] = j * insertion_cost

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if source[i - 1] == target[j - 1]:
                substitution = matrix[i - 1][j - 1]  # no cost, characters match
            else:
                substitution = matrix[i - 1][j - 1] + substitution_cost

            deletion = matrix[i - 1][j] + deletion_cost      # remove from source
            insertion = matrix[i][j - 1] + insertion_cost    # insert into source

            matrix[i][j] = min(substitution, deletion, insertion)

    return matrix


def levenshtein_distance(source: str, target: str) -> int:
    """Return only the final Minimum Edit Distance between two strings."""
    matrix = levenshtein_matrix(source, target)
    return matrix[-1][-1]


# ----------------------------------------------------------------------------
# Add-k (Laplace) smoothing for bigram probabilities
# ----------------------------------------------------------------------------
def smoothed_bigram_probability(w1: str, w2: str,
                                 unigrams: dict, bigrams: dict,
                                 vocabulary_size: int,
                                 k: float = DEFAULT_ADD_K) -> float:
    """Compute P(w2 | w1) with Add-k smoothing.

    P(w2|w1) = (count(w1, w2) + k) / (count(w1) + k * V)

    Add-k smoothing guarantees a non-zero probability even for bigrams that
    were never observed in the training corpus, which is essential since
    natural language bigram distributions are extremely sparse.
    """
    bigram_count = bigrams.get(w1, {}).get(w2, 0)
    unigram_count = unigrams.get(w1, 0)
    return (bigram_count + k) / (unigram_count + k * vocabulary_size)


# ----------------------------------------------------------------------------
# Spell-check engine
# ----------------------------------------------------------------------------
@dataclass
class DetectedError:
    word: str
    start: int
    end: int
    error_type: str            # "nonword" or "realword"
    suggestions: list = field(default_factory=list)


class SpellCheckEngine:
    """Loads the Unigram/Bigram dictionaries and performs detection +
    candidate generation for both Non-word and Real-word errors.
    """

    def __init__(self, unigrams_path: str = UNIGRAMS_PATH,
                 bigrams_path: str = BIGRAMS_PATH):
        with open(unigrams_path, "rb") as f:
            self.unigrams: dict = pickle.load(f)
        with open(bigrams_path, "rb") as f:
            self.bigrams: dict = pickle.load(f)

        self.vocabulary_size = len(self.unigrams)

        # Index words by length to avoid comparing a misspelled word against
        # the entire ~32k-word vocabulary: two strings whose length differs
        # by more than max_edit_distance can never be within that distance.
        self._words_by_length: dict[int, list[str]] = {}
        for word in self.unigrams:
            self._words_by_length.setdefault(len(word), []).append(word)

    # -- Non-word errors ------------------------------------------------
    def is_known_word(self, word: str) -> bool:
        return word.lower() in self.unigrams

    def get_nonword_candidates(self, word: str,
                                max_distance: int = DEFAULT_MAX_EDIT_DISTANCE,
                                top_n: int = DEFAULT_TOP_N_SUGGESTIONS) -> list[str]:
        """Generate correction candidates for a word absent from the dictionary.

        Candidates are ranked by (1) smallest edit distance, (2) highest
        corpus frequency as a tie-breaker.
        """
        word = word.lower()
        candidates = []

        for length in range(len(word) - max_distance, len(word) + max_distance + 1):
            for vocab_word in self._words_by_length.get(length, []):
                distance = levenshtein_distance(word, vocab_word)
                if distance <= max_distance:
                    candidates.append((vocab_word, distance, self.unigrams[vocab_word]))

        candidates.sort(key=lambda item: (item[1], -item[2]))
        return [c[0] for c in candidates[:top_n]]

    # -- Real-word (contextual) errors ----------------------------------
    def check_realword(self, prev_word: str, word: str,
                        k: float = DEFAULT_ADD_K,
                        threshold: float = DEFAULT_REALWORD_PROB_THRESHOLD) -> bool:
        """Return True if (prev_word, word) looks like a contextual error.

        The bigram is flagged as suspicious only when BOTH conditions hold:
          1. The transition (prev_word -> word) was never observed in the
             bigram table.
          2. Its Add-k-smoothed probability falls below `threshold`.
        """
        prev_word, word = prev_word.lower(), word.lower()
        bigram_exists = word in self.bigrams.get(prev_word, {})
        if bigram_exists:
            return False

        probability = smoothed_bigram_probability(
            prev_word, word, self.unigrams, self.bigrams, self.vocabulary_size, k
        )
        return probability < threshold

    def get_realword_candidates(self, prev_word: str,
                                 top_n: int = DEFAULT_TOP_N_SUGGESTIONS) -> list[str]:
        """Suggest the words most frequently observed after `prev_word`."""
        prev_word = prev_word.lower()
        following = self.bigrams.get(prev_word, {})
        ranked = sorted(following.items(), key=lambda item: -item[1])
        return [w for w, _ in ranked[:top_n]]

    # -- Full-text check --------------------------------------------------
    def check_text(self, text: str) -> list[DetectedError]:
        """Run both Non-word and Real-word detection over a full text.

        Returns a list of DetectedError, each carrying the character
        offsets needed by the GUI to highlight the exact substring.
        """
        errors: list[DetectedError] = []
        matches = list(WORD_TOKEN_RE.finditer(text))

        previous_known_word = None
        for match in matches:
            raw_word = match.group()
            word = raw_word.lower()

            if not self.is_known_word(word):
                suggestions = self.get_nonword_candidates(word)
                errors.append(DetectedError(
                    word=raw_word, start=match.start(), end=match.end(),
                    error_type="nonword", suggestions=suggestions,
                ))
                # A non-word does not participate in the bigram context chain
                previous_known_word = None
                continue

            if previous_known_word is not None:
                if self.check_realword(previous_known_word, word):
                    suggestions = self.get_realword_candidates(previous_known_word)
                    errors.append(DetectedError(
                        word=raw_word, start=match.start(), end=match.end(),
                        error_type="realword", suggestions=suggestions,
                    ))

            previous_known_word = word

        return errors

    # -- Dictionary explorer helpers ---------------------------------------
    def search_dictionary(self, prefix: str, limit: int = 200) -> list[tuple[str, int]]:
        """Return (word, frequency) pairs whose word starts with `prefix`,
        sorted by descending frequency. Used by the GUI's live search box.
        """
        prefix = prefix.lower()
        matches = [(w, f) for w, f in self.unigrams.items() if w.startswith(prefix)]
        matches.sort(key=lambda item: -item[1])
        return matches[:limit]


# ----------------------------------------------------------------------------
# Manual sanity checks (run this file directly to see example outputs)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    engine = SpellCheckEngine()

    print("Vocabulary size:", engine.vocabulary_size)

    print("\n--- Levenshtein distance demo ---")
    print("distance('netwrk', 'network') =", levenshtein_distance("netwrk", "network"))

    print("\n--- Non-word candidates ---")
    for w in ["netwrk", "graffe", "algoritm"]:
        print(f"{w!r} known? {engine.is_known_word(w)} -> "
              f"candidates: {engine.get_nonword_candidates(w)}")

    print("\n--- Real-word contextual check ---")
    test_text = "This paper proposes a novel computer see architecture for deep learning."
    for err in engine.check_text(test_text):
        print(err)
