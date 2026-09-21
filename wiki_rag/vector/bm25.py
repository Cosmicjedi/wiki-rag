#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""BM25 sparse-vector encoder for the Qdrant backend.

Qdrant stores and scores sparse vectors natively but does not analyse text
on the server, so the lexical channel of the hybrid search has to be produced
client side. This module turns text into Qdrant sparse vectors the same way
the Milvus ``english`` analyser + BM25 function did: lowercase, split into
words, drop English stop words, Porter-stem, then weight each surviving term.

Only the term-frequency half of BM25 is computed here. The inverse document
frequency half depends on corpus statistics that the client does not have, so
the collection's sparse vector field is declared with ``Modifier.IDF`` and
Qdrant multiplies every matched term by its IDF at query time. The score
Qdrant returns is therefore a genuine BM25 score, with the corpus statistics
always current, and the encoder stays stateless.

Term indices are stable 32-bit hashes of the stemmed token so that the same
word always lands on the same sparse dimension, in every process and across
reindexes.
"""

import hashlib
import math
import re

from collections import Counter

from qdrant_client import models

#: BM25 term-frequency saturation. Standard Robertson/Lucene default.
K1: float = 1.2
#: BM25 document-length normalisation. Standard Robertson/Lucene default.
B: float = 0.75
#: Assumed average document length (in tokens) used for length normalisation.
#: The encoder is stateless, so it cannot compute the real average; a fixed
#: value keeps the weights consistent between indexing runs.
AVG_LEN: float = 256.0

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

#: English stop words (the Lucene / Snowball list, plus a few contractions).
STOP_WORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "s", "t", "d", "ll", "m", "re", "ve",
})


# ---------------------------------------------------------------------------
# Porter stemmer (M.F. Porter, "An algorithm for suffix stripping", 1980)
# ---------------------------------------------------------------------------

_VOWELS = frozenset("aeiou")


def _is_consonant(word: str, i: int) -> bool:
    """Return True when ``word[i]`` is a consonant in the Porter sense.

    ``y`` counts as a consonant at the start of the word or after a vowel,
    and as a vowel after a consonant.
    """
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Return the Porter measure *m* of ``stem``: the number of VC sequences."""
    m = 0
    i = 0
    n = len(stem)
    # Skip the initial consonant run.
    while i < n and _is_consonant(stem, i):
        i += 1
    while i < n:
        # Vowel run.
        while i < n and not _is_consonant(stem, i):
            i += 1
        if i >= n:
            break
        # Consonant run: a complete VC sequence.
        while i < n and _is_consonant(stem, i):
            i += 1
        m += 1
    return m


def _contains_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    """Return True when ``word`` ends consonant-vowel-consonant, last not w/x/y."""
    if len(word) < 3:
        return False
    n = len(word)
    return (
        _is_consonant(word, n - 1)
        and not _is_consonant(word, n - 2)
        and _is_consonant(word, n - 3)
        and word[-1] not in "wxy"
    )


def _replace_suffix(word: str, suffix: str, replacement: str, min_measure: int) -> str | None:
    """Replace ``suffix`` with ``replacement`` when the stem measure is > ``min_measure``.

    Returns the new word, or None when the suffix does not apply (either it is
    absent, or the measure condition fails), so callers can tell "no suffix"
    from "suffix present but condition failed" - the Porter rules stop at the
    first matching suffix in each step, whether or not it is replaced.
    """
    if not word.endswith(suffix):
        return None
    stem = word[: len(word) - len(suffix)]
    if _measure(stem) > min_measure:
        return stem + replacement
    return word


_STEP2 = (
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("bli", "ble"), ("alli", "al"), ("entli", "ent"), ("eli", "e"),
    ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
    ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"),
    ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"), ("logi", "log"),
)
_STEP3 = (
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
)
_STEP4 = (
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
    "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
)


def porter_stem(word: str) -> str:
    """Return the Porter stem of a lowercase ``word``.

    Words of one or two letters are returned unchanged, as in the original
    algorithm.
    """
    if len(word) <= 2:
        return word

    # Step 1a: plurals.
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]

    # Step 1b: -ed / -ing.
    if word.endswith("eed"):
        stem = word[:-3]
        if _measure(stem) > 0:
            word = word[:-1]
    else:
        stripped = None
        if word.endswith("ed") and _contains_vowel(word[:-2]):
            stripped = word[:-2]
        elif word.endswith("ing") and _contains_vowel(word[:-3]):
            stripped = word[:-3]
        if stripped is not None:
            word = stripped
            if word.endswith(("at", "bl", "iz")):
                word += "e"
            elif _ends_double_consonant(word) and word[-1] not in "lsz":
                word = word[:-1]
            elif _measure(word) == 1 and _ends_cvc(word):
                word += "e"

    # Step 1c: -y -> -i when the stem has a vowel.
    if word.endswith("y") and _contains_vowel(word[:-1]):
        word = word[:-1] + "i"

    # Step 2.
    for suffix, replacement in _STEP2:
        result = _replace_suffix(word, suffix, replacement, 0)
        if result is not None:
            word = result
            break

    # Step 3.
    for suffix, replacement in _STEP3:
        result = _replace_suffix(word, suffix, replacement, 0)
        if result is not None:
            word = result
            break

    # Step 4: strip a final suffix when m > 1 ("ion" only after s or t).
    for suffix in _STEP4:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if _measure(stem) > 1 and (suffix != "ion" or stem.endswith(("s", "t"))):
                word = stem
            break

    # Step 5a: drop a final e.
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            word = stem

    # Step 5b: -ll -> -l when m > 1.
    if word.endswith("ll") and _measure(word[:-1]) > 1:
        word = word[:-1]

    return word


# ---------------------------------------------------------------------------
# Tokenisation and encoding
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Split ``text`` into lowercase, stop-word-free, stemmed tokens.

    Words are runs of Unicode letters/digits (underscores and punctuation are
    separators, so ``wiki_link`` yields two tokens). Stop words are removed
    before stemming, matching the analyser order used by the Milvus backend.

    Args:
        text: Raw text to analyse.

    Returns:
        The list of terms in document order (duplicates preserved).

    """
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text.lower()):
        word = match.group(0)
        if word in STOP_WORDS:
            continue
        tokens.append(porter_stem(word))
    return tokens


def term_index(term: str) -> int:
    """Return the stable sparse dimension (uint32) assigned to ``term``.

    A 4-byte BLAKE2b digest keeps the index inside Qdrant's ``u32`` sparse
    index range and is deterministic across processes (unlike ``hash()``).
    """
    return int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest(), "big")


def encode_document(text: str) -> models.SparseVector:
    """Encode a document as a BM25 term-frequency sparse vector.

    Each term's value is the saturated, length-normalised term frequency::

        tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))

    The IDF factor is applied by Qdrant at query time (``Modifier.IDF``).
    An empty document (nothing left after stop-word removal) yields an empty
    sparse vector, which simply never matches.

    Args:
        text: Document text.

    Returns:
        Sparse vector ready to be stored in Qdrant.

    """
    tokens = tokenize(text)
    if not tokens:
        return models.SparseVector(indices=[], values=[])
    doc_len = len(tokens)
    norm = K1 * (1.0 - B + B * doc_len / AVG_LEN)
    weights: dict[int, float] = {}
    for term, tf in Counter(tokens).items():
        # Two different terms can (very rarely) hash to the same index; keep
        # the larger weight rather than silently overwriting it.
        weight = tf * (K1 + 1.0) / (tf + norm)
        index = term_index(term)
        weights[index] = max(weight, weights.get(index, 0.0))
    indices = sorted(weights)
    return models.SparseVector(indices=indices, values=[weights[i] for i in indices])


def encode_query(text: str) -> models.SparseVector:
    """Encode a query as a BM25 sparse vector.

    Query terms carry unit weight: in BM25 the query side contributes only
    IDF, which Qdrant applies server side, so every distinct term is listed
    once with value ``1.0``.

    Args:
        text: Query text.

    Returns:
        Sparse vector to search the collection with.

    """
    indices = sorted({term_index(term) for term in tokenize(text)})
    return models.SparseVector(indices=indices, values=[1.0] * len(indices))


def normalise_score(score: float) -> float:
    """Map an unbounded BM25 score onto ``[0, 1)`` with arctan.

    Used to make BM25 scores comparable with cosine similarities when fusing
    the two search channels: arctan is monotonic, bounded and keeps the
    useful resolution in the typical BM25 range of roughly 1 to 20.
    """
    return 2.0 / math.pi * math.atan(max(score, 0.0))
