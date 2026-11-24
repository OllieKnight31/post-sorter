"""Shared utility functions for the briefpost-management pipeline."""
import logging
import re
from core.constants import COMPANY_SUFFIXES

logger = logging.getLogger(__name__)

_UMLAUT_MAP = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}


def normalize_creditor(name: str) -> str:
    """Normalize creditor name for comparison.

    Single canonical implementation: used by page_matcher,
    vorgang_matcher, cross_intelligence, and bank_reconciler.
    """
    if not name:
        return ""
    normalized = name.lower().strip()
    for char, rep in _UMLAUT_MAP.items():
        normalized = normalized.replace(char, rep)
    for suffix in COMPANY_SUFFIXES:
        if re.fullmatch(r"\w+", suffix):
            normalized = re.sub(r"\b" + re.escape(suffix) + r"\b", "", normalized).strip()
        else:
            normalized = re.sub(r"(?:^|\s)" + re.escape(suffix) + r"(?:\s|$)", " ", normalized).strip()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def jaro_winkler_similarity(s1: str, s2: str, winkler_prefix_weight: float = 0.1) -> float:
    """Compute Jaro-Winkler similarity between two strings.

    Pure Python implementation (no external dependencies).

    Args:
        s1: First string.
        s2: Second string.
        winkler_prefix_weight: Scaling factor for common prefix bonus (default 0.1).

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (
        matches / len1
        + matches / len2
        + (matches - transpositions / 2) / matches
    ) / 3.0

    # Winkler modification: boost for common prefix (up to 4 chars)
    prefix_len = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    return jaro + prefix_len * winkler_prefix_weight * (1.0 - jaro)


def fuzzy_creditor_match(name_a: str, name_b: str, threshold: float = 0.85) -> bool:
    """Check if two creditor names are fuzzy matches using Jaro-Winkler.

    Normalizes both names before comparison.

    Args:
        name_a: First creditor name.
        name_b: Second creditor name.
        threshold: Minimum similarity score to consider a match.

    Returns:
        True if normalized names have Jaro-Winkler similarity >= threshold.
    """
    norm_a = normalize_creditor(name_a)
    norm_b = normalize_creditor(name_b)
    if not norm_a or not norm_b:
        return False
    return jaro_winkler_similarity(norm_a, norm_b) >= threshold
