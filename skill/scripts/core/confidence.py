"""Confidence computation module for the briefpost pipeline.

Replaces all hardcoded confidence values with computed, propagating scores.
Each pipeline stage produces a confidence value that flows into downstream stages.

Confidence chain:
  extraction_confidence = weighted mean of field_confidences
  verification_confidence = pass_2 * 0.6 + pass_3_agreement * 0.4
  letter_confidence = classification_confidence * verification_confidence
  match_confidence = algorithmic OR llm (per-method)
  recon_confidence = algorithmic * date_proximity
  pipeline_confidence = letter * match * recon
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# --- Extraction confidence ---

# CQ-051: Weights for field_confidence aggregation (higher = more important).
# "payment_info" is the legacy key (v4 and earlier, covers iban + payment_other).
# v5 splits it into "iban" and "payment_other". Both are kept for backward
# compatibility: if a letter uses the old key, it still gets a weight.
FIELD_WEIGHTS: dict[str, float] = {
    "creditor": 2.0,
    "person": 1.5,
    "document_type": 1.5,
    "amounts": 2.0,
    "dates": 1.0,
    "iban": 2.5,
    "payment_other": 1.0,
    "payment_info": 1.5,  # Legacy key: split into iban + payment_other for v5
    "identifiers": 1.5,
}

DEFAULT_FIELD_WEIGHT = 1.0


def compute_extraction_confidence(field_confidences: dict[str, float]) -> float:
    """Compute weighted mean of per-field confidence scores.

    Args:
        field_confidences: Dict mapping field names to confidence (0.0-1.0).

    Returns:
        Weighted mean confidence (0.0-1.0). Returns 0.5 if no fields provided.
    """
    if not field_confidences:
        return 0.50

    total_weight = 0.0
    weighted_sum = 0.0

    for field, conf in field_confidences.items():
        weight = FIELD_WEIGHTS.get(field, DEFAULT_FIELD_WEIGHT)
        # CQ-009: Normalize confidence to 0-1 range with logging
        if conf > 1.0:
            if conf <= 2.0:
                logger.warning(
                    "Confidence value %.4f > 1.0 for field '%s', clamping to 1.0",
                    conf, field,
                )
                conf = 1.0
            else:
                logger.warning(
                    "Confidence value %.4f > 1.0 for field '%s', treating as percentage",
                    conf, field,
                )
                conf = conf / 100.0
        weighted_sum += conf * weight
        total_weight += weight

    if total_weight == 0:
        return 0.50

    return round(min(1.0, max(0.0, weighted_sum / total_weight)), 4)


# --- Verification confidence ---

PASS2_WEIGHT = 0.6
PASS3_WEIGHT = 0.4
AGREEMENT_BOOST = 0.10
DISAGREEMENT_PENALTY = -0.15


def compute_verification_confidence(
    pass2_confidence: float,
    agreement_rate: float,
) -> float:
    """Compute verification confidence from Pass 2 + Pass 3 agreement.

    Formula: pass2 * 0.6 + agreement_rate * 0.4

    Args:
        pass2_confidence: Confidence from Pass 2 extraction (0.0-1.0 or 0-100).
        agreement_rate: Fraction of fields confirmed by Pass 3 (0.0-1.0).

    Returns:
        Verified confidence on 0.0-1.0 scale.
    """
    # CQ-009: Normalize to 0-1 with logging
    if pass2_confidence > 1.0:
        if pass2_confidence <= 2.0:
            logger.warning(
                "Confidence value %.4f > 1.0, clamping to 1.0", pass2_confidence,
            )
            pass2_confidence = 1.0
        else:
            logger.warning(
                "Confidence value %.4f > 1.0, treating as percentage", pass2_confidence,
            )
            pass2_confidence = pass2_confidence / 100.0

    result = pass2_confidence * PASS2_WEIGHT + agreement_rate * PASS3_WEIGHT
    return round(min(1.0, max(0.0, result)), 4)


# --- Letter confidence ---

def compute_letter_confidence(
    classification_confidence: float,
    verification_confidence: float,
) -> float:
    """Compute letter-level confidence from classification and verification.

    Formula: classification_confidence * verification_confidence

    Args:
        classification_confidence: Pass 1 classification confidence (0.0-1.0).
        verification_confidence: Combined Pass 2 + Pass 3 confidence (0.0-1.0).

    Returns:
        Letter confidence on 0.0-1.0 scale.
    """
    # CQ-009: Normalize with logging
    if classification_confidence > 1.0:
        if classification_confidence <= 2.0:
            logger.warning(
                "Confidence value %.4f > 1.0, clamping to 1.0", classification_confidence,
            )
            classification_confidence = 1.0
        else:
            logger.warning(
                "Confidence value %.4f > 1.0, treating as percentage", classification_confidence,
            )
            classification_confidence = classification_confidence / 100.0
    if verification_confidence > 1.0:
        if verification_confidence <= 2.0:
            logger.warning(
                "Confidence value %.4f > 1.0, clamping to 1.0", verification_confidence,
            )
            verification_confidence = 1.0
        else:
            logger.warning(
                "Confidence value %.4f > 1.0, treating as percentage", verification_confidence,
            )
            verification_confidence = verification_confidence / 100.0

    return round(min(1.0, max(0.0, classification_confidence * verification_confidence)), 4)


# --- Match confidence (per-method constants from config) ---

MATCH_CONFIDENCE: dict[str, float] = {
    "aktenzeichen_exact": 1.0,
    "creditor_amount": 0.90,
    "single_creditor": 0.85,
    "llm": 0.75,
    "single_unknown": 0.50,
    "orphan": 0.30,
}


def get_match_confidence(method: str) -> float:
    """Get match confidence for a given matching method.

    Args:
        method: Matching method string (e.g. "aktenzeichen_exact").

    Returns:
        Match confidence (0.0-1.0).
    """
    return MATCH_CONFIDENCE.get(method.lower(), 0.50)


# --- Reconciliation confidence ---

def compute_recon_confidence(
    base_confidence: float,
    date_proximity: float,
) -> float:
    """Compute reconciliation confidence with date proximity factor.

    Formula: base_confidence * date_proximity

    Args:
        base_confidence: Algorithmic match confidence (0.0-1.0).
        date_proximity: Temporal proximity factor (0.70-1.0).

    Returns:
        Reconciliation confidence on 0.0-1.0 scale.
    """
    return round(min(1.0, max(0.0, base_confidence * date_proximity)), 4)


# --- Pipeline confidence (end-to-end) ---

def compute_pipeline_confidence(
    letter_confidence: float,
    match_confidence: float,
    recon_confidence: float | None = None,
) -> float:
    """Compute end-to-end pipeline confidence.

    Formula: letter * match * recon
    If recon_confidence is None (pre-reconciliation), uses letter * match.

    CQ-039: Note that multiplication compresses confidence (0.85^3 = 0.61).
    This is intentional: pipeline confidence should be conservative and
    reflect compound uncertainty across stages.

    Args:
        letter_confidence: Letter-level confidence (0.0-1.0).
        match_confidence: Vorgang matching confidence (0.0-1.0).
        recon_confidence: Bank reconciliation confidence (0.0-1.0), or None.

    Returns:
        Pipeline confidence on 0.0-1.0 scale.
    """
    result = letter_confidence * match_confidence
    if recon_confidence is not None:
        result *= recon_confidence
    return round(min(1.0, max(0.0, result)), 4)


# --- Cross-intelligence confidence ---

# Per-phase confidence for gap-filling operations
CROSS_INTEL_CONFIDENCE: dict[str, float] = {
    "person_inference": 0.60,  # Minimum threshold (majority vote confidence used directly)
    "iban_lookup": 0.85,
    "betrag_from_related": 0.80,
    "aktenzeichen_from_bank": 0.75,
    "unfillable_categorization": 0.90,
}


def get_cross_intel_confidence(phase: str) -> float:
    """Get confidence for a cross-intelligence phase.

    Args:
        phase: Phase name (e.g. "iban_lookup").

    Returns:
        Phase confidence (0.0-1.0).
    """
    return CROSS_INTEL_CONFIDENCE.get(phase, 0.75)


# --- Confidence from letter data ---

def extract_letter_confidence(letter: dict[str, Any]) -> float:
    """Extract the best available confidence from a letter JSON.

    Checks in order: letter_confidence, verified_confidence,
    confidence.overall, ocr_confidence. Falls back to 0.50.

    Args:
        letter: Letter JSON data dict.

    Returns:
        Confidence on 0.0-1.0 scale.
    """
    # Try v4 letter_confidence first
    lc = letter.get("letter_confidence")
    if lc is not None:
        return lc if lc <= 1.0 else lc / 100.0

    # Try verified_confidence (from Pass 3)
    vc = letter.get("verified_confidence")
    if vc is not None:
        return vc if vc <= 1.0 else vc / 100.0

    # Try nested confidence block
    conf_block = letter.get("confidence", {})
    if isinstance(conf_block, dict):
        overall = conf_block.get("overall")
        if overall is not None:
            return overall if overall <= 1.0 else overall / 100.0

    # Fallback to ocr_confidence
    ocr = letter.get("ocr_confidence")
    if ocr is not None:
        return ocr if ocr <= 1.0 else ocr / 100.0

    # CQ-040: Log when falling back to default
    letter_id = letter.get("letter_id", letter.get("doc_id", "unknown"))
    logger.debug(
        "extract_letter_confidence: no confidence key found for %s, using 0.50",
        letter_id,
    )
    return 0.50


def compute_pipeline_confidence_from_letters(
    letters: list[dict[str, Any]],
    match_confidence: float,
) -> float:
    """Compute pipeline confidence from letter data and match confidence.

    Replaces the old formula (avg_ocr * avg_grouping * match) with
    the new formula (avg_letter_confidence * match_confidence).

    Args:
        letters: List of letter JSON data dicts.
        match_confidence: Confidence of the matching step.

    Returns:
        Pipeline confidence (0.0-1.0).
    """
    if not letters:
        return round(0.50 * match_confidence, 4)

    letter_confs = [extract_letter_confidence(letter) for letter in letters]
    avg_letter = sum(letter_confs) / len(letter_confs)

    return compute_pipeline_confidence(avg_letter, match_confidence)
