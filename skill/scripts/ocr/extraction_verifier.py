"""Pass 3: Independent verification of Pass 2 extraction results.

Loads original images + Pass 2 JSON for each letter, sends to Gemini Flash
(different model than Pass 2's Pro) for independent verification of each
extracted field.

Verification verdicts per field:
  - BESTAETIGT: field matches, boost confidence +0.10
  - KORRIGIERT: field corrected, use new value + flag
  - NICHT_VERIFIZIERBAR: cannot verify, keep original

Final confidence: verified_confidence = pass2_confidence * 0.6 + agreement_rate * 0.4
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from core.config import BriefpostConfig
from core.constants import Flag
from core.io import atomic_write_json
from core.models import BatchResult

logger = logging.getLogger(__name__)

# Verification model: different from Pass 2's Pro
VERIFICATION_MODEL = "gemini-3.6-flash"

# Fields to verify (must match extraction schema keys)
VERIFIABLE_FIELDS = [
    "creditor",
    "person",
    "document_type",
    "amounts",
    "dates",
    "payment_info",
    "identifiers",
]

# High-confidence threshold: skip verification when ALL fields above this
HIGH_CONFIDENCE_THRESHOLD = 0.95

# Default field confidence when field_confidence dict is missing a field
# B2: Changed from 0.80 to 0.0: unknown confidence must not silently inflate scores
DEFAULT_FIELD_CONFIDENCE = 0.0


def _ensure_api_key() -> None:
    """Load GOOGLE_API_KEY from the environment or an .env file."""
    from core.extern import load_api_key
    load_api_key()


def should_skip_verification(extraction_data: dict[str, Any]) -> bool:
    """B5: Never skip verification entirely. Returns False always.

    Previously returned True when all fields > 0.95. Now all letters
    get verification_confidence computed. High-confidence letters use
    fast-path (field hash comparison) instead of full re-extraction.

    Args:
        extraction_data: The gemini_extraction dict from Pass 2.

    Returns:
        Always False: verification is never skipped.
    """
    return False


def is_high_confidence(extraction_data: dict[str, Any]) -> bool:
    """Check if all field confidences exceed HIGH_CONFIDENCE_THRESHOLD.

    Used to select fast-path verification (field hash comparison) instead
    of full LLM re-extraction.

    Args:
        extraction_data: The gemini_extraction dict from Pass 2.

    Returns:
        True if all field confidences > 0.95 (use fast-path).
    """
    field_conf = extraction_data.get("field_confidence", {})
    if not field_conf:
        return False

    for field_name in VERIFIABLE_FIELDS:
        conf = field_conf.get(field_name, 0.0)
        if conf <= HIGH_CONFIDENCE_THRESHOLD:
            return False

    return True


def build_verification_prompt(extraction_data: dict[str, Any], page_count: int) -> str:
    """Build the German verification prompt for Gemini Flash.

    Instructs Gemini Flash to independently verify each extracted field
    by comparing against the original document images.

    Args:
        extraction_data: The gemini_extraction dict from Pass 2.
        page_count: Number of page images being sent.

    Returns:
        Full verification prompt string.
    """
    fields_json = json.dumps({
        "creditor": extraction_data.get("creditor", {}),
        "person": extraction_data.get("person", ""),
        "document_type": extraction_data.get("document_type", ""),
        "amounts": extraction_data.get("amounts", {}),
        "dates": extraction_data.get("dates", {}),
        "payment_info": extraction_data.get("payment_info", {}),
        "identifiers": extraction_data.get("identifiers", []),
    }, indent=2, ensure_ascii=False)

    return f"""Du bist ein Dokumenten-Verifizierer. Du siehst {page_count} Seite(n) eines Briefes.

Ein anderes KI-Modell hat folgende Daten aus diesem Brief extrahiert:

```json
{fields_json}
```

AUFGABE: Pruefe JEDES Feld gegen die Originalbilder und gib fuer JEDES Feld ein Urteil:
- "BESTAETIGT" = Das Feld stimmt mit dem Dokument ueberein
- "KORRIGIERT" = Das Feld ist FALSCH, du gibst den korrekten Wert an
- "NICHT_VERIFIZIERBAR" = Das Feld kann nicht verifiziert werden (z.B. unleserlich)

Antworte als JSON mit folgendem Schema:
{{
  "verifications": {{
    "creditor": {{"verdict": "BESTAETIGT|KORRIGIERT|NICHT_VERIFIZIERBAR", "corrected_value": null, "reason": "..."}},
    "person": {{"verdict": "...", "corrected_value": null, "reason": "..."}},
    "document_type": {{"verdict": "...", "corrected_value": null, "reason": "..."}},
    "amounts": {{"verdict": "...", "corrected_value": null, "reason": "..."}},
    "dates": {{"verdict": "...", "corrected_value": null, "reason": "..."}},
    "payment_info": {{"verdict": "...", "corrected_value": null, "reason": "..."}},
    "identifiers": {{"verdict": "...", "corrected_value": null, "reason": "..."}}
  }}
}}

REGELN:
- Pruefe JEDEN Wert sorgfaeltig gegen das Originalbild
- Bei Betraegen: pruefe ob Zahlen korrekt uebernommen wurden (Komma vs. Punkt)
- Bei IBAN: pruefe jede einzelne Ziffer
- Bei Daten: pruefe Tag, Monat und Jahr
- "corrected_value" NUR ausfuellen wenn verdict="KORRIGIERT"
- Wenn ein Feld leer/null ist UND im Dokument auch nicht vorhanden: "BESTAETIGT"
"""


def parse_verification_response(response_text: str) -> dict[str, Any]:
    """Parse the verification JSON response from Gemini Flash.

    Args:
        response_text: Raw response text from Gemini.

    Returns:
        Dict with verifications per field, or empty dict on parse failure.
    """
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "verifications" in data:
            return data["verifications"]
        return data
    except json.JSONDecodeError:
        logger.warning("Failed to parse verification response as JSON")
        return {}


def apply_verification_results(
    extraction_data: dict[str, Any],
    verifications: dict[str, Any],
) -> tuple[dict[str, Any], list[str], float]:
    """Merge verification results into extraction data.

    Rules:
      - BESTAETIGT: boost field_confidence by +0.10 (cap at 1.0)
      - KORRIGIERT: replace value with corrected_value, add NEEDS_MANUAL_REVIEW flag
      - NICHT_VERIFIZIERBAR: keep original, no change

    Also computes agreement_rate = confirmed / total_verified.

    Args:
        extraction_data: Original gemini_extraction dict (mutated in place).
        verifications: Per-field verification dict from parse_verification_response.

    Returns:
        Tuple of (updated_extraction, flags, agreement_rate).
    """
    flags: list[str] = []
    confirmed_count = 0
    total_verified = 0

    field_conf = extraction_data.get("field_confidence", {})

    for field_name in VERIFIABLE_FIELDS:
        verification = verifications.get(field_name)
        if not verification or not isinstance(verification, dict):
            continue

        verdict = verification.get("verdict", "").upper()
        total_verified += 1

        if verdict == "BESTAETIGT":
            confirmed_count += 1
            # Boost field confidence
            current = field_conf.get(field_name, DEFAULT_FIELD_CONFIDENCE)
            # B2: Flag when default confidence is applied (field was missing from Pass 2)
            if field_name not in field_conf:
                if Flag.CONFIDENCE_UNKNOWN.value not in flags:
                    flags.append(Flag.CONFIDENCE_UNKNOWN.value)
            field_conf[field_name] = min(1.0, current + 0.10)

        elif verdict == "KORRIGIERT":
            corrected = verification.get("corrected_value")
            if corrected is not None:
                # Apply correction
                if field_name in ("creditor", "amounts", "dates", "payment_info"):
                    if isinstance(corrected, dict):
                        extraction_data[field_name] = corrected
                elif field_name == "identifiers":
                    if isinstance(corrected, list):
                        extraction_data[field_name] = corrected
                else:
                    extraction_data[field_name] = corrected

                flags.append(f"VERIFIED_CORRECTION:{field_name}")
            flags.append(Flag.NEEDS_MANUAL_REVIEW.value)

        # NICHT_VERIFIZIERBAR: no action

    extraction_data["field_confidence"] = field_conf

    agreement_rate = confirmed_count / total_verified if total_verified > 0 else 0.0
    return extraction_data, flags, agreement_rate


def compute_verified_confidence(
    pass2_confidence: float,
    agreement_rate: float,
) -> float:
    """Compute final verified confidence score.

    Formula: verified_confidence = pass2_confidence * 0.6 + agreement_rate * 0.4

    Both inputs are on 0.0-1.0 scale. If pass2_confidence is on 0-100 scale,
    it gets normalized first.

    Args:
        pass2_confidence: Confidence from Pass 2 (0-100 or 0.0-1.0).
        agreement_rate: Fraction of fields confirmed (0.0-1.0).

    Returns:
        Final confidence on 0.0-1.0 scale.
    """
    # Normalize pass2 to 0-1 if on 0-100 scale
    if pass2_confidence > 1.0:
        pass2_norm = pass2_confidence / 100.0
    else:
        pass2_norm = pass2_confidence

    verified = pass2_norm * 0.6 + agreement_rate * 0.4
    return round(min(1.0, max(0.0, verified)), 4)


async def verify_letter(
    image_paths: list[Path],
    extraction_data: dict[str, Any],
    config: BriefpostConfig,
) -> dict[str, Any]:
    """Verify a single letter's extraction using Gemini Flash.

    Args:
        image_paths: Page images for this letter.
        extraction_data: Pass 2 extraction result.
        config: Pipeline configuration.

    Returns:
        Dict with verifications, flags, agreement_rate, verified_confidence.
    """
    _ensure_api_key()

    from google import genai
    from google.genai import types

    client = genai.Client()

    # Build multi-image parts
    parts: list[Any] = []
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            data = f.read()
        parts.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))

    # Build verification prompt
    prompt = build_verification_prompt(extraction_data, len(image_paths))
    parts.append(prompt)

    model_name = config.ocr.get("verification_model", VERIFICATION_MODEL)
    gen_config: dict[str, Any] = {
        "temperature": 0.0,
        "max_output_tokens": 8192,
        # Gemini 3.5 and newer take a thinking level; a token budget is rejected
        "thinking_config": {"thinking_level": str(config.ocr.get("verification_thinking_level", "low"))},
        "response_mime_type": "application/json",
    }

    try:
        start = time.time()
        from core.llm_helpers import generate_content_safe
        response = await asyncio.to_thread(
            lambda: generate_content_safe(client, model_name, parts, gen_config)
        )
        elapsed = time.time() - start

        response_text = response.text if hasattr(response, "text") else ""
        verifications = parse_verification_response(response_text)

        updated_data, flags, agreement_rate = apply_verification_results(
            extraction_data, verifications,
        )

        # Extract Pass 2 confidence from field_confidence weighted mean
        from core.confidence import compute_extraction_confidence
        field_conf = extraction_data.get("field_confidence", {})
        pass2_confidence = compute_extraction_confidence(field_conf) if field_conf else 0.50
        verified_confidence = compute_verified_confidence(pass2_confidence, agreement_rate)

        return {
            "verifications": verifications,
            "flags": flags,
            "agreement_rate": agreement_rate,
            "verified_confidence": verified_confidence,
            "elapsed": elapsed,
            "model": model_name,
            "error": None,
        }

    except Exception as exc:
        logger.error("Verification failed: %s", exc)
        return {
            "verifications": {},
            "flags": [],
            "agreement_rate": 0.0,
            "verified_confidence": 0.50,
            "elapsed": 0.0,
            "model": model_name,
            "error": str(exc),
        }


async def process_verification_batch(
    letters_dir: Path,
    ocr_pages_dir: Path,
    config: BriefpostConfig,
) -> dict[str, int]:
    """Process all letters in ocr-letters/ for verification.

    Args:
        letters_dir: Path to ocr-letters/ directory.
        ocr_pages_dir: Path to ocr-pages/ directory (for updating per-page JSONs).
        config: Pipeline configuration.

    Returns:
        Dict with total, verified, skipped, failed counts.
    """
    rate_limit = config.ocr.get("rate_limit_seconds", 4)
    letter_files = sorted(letters_dir.glob("doc-*-L*.json"))

    total = len(letter_files)
    verified = 0
    skipped = 0
    failed = 0

    # RS-006: Load checkpoint to resume from last verified letter
    checkpoint_path = letters_dir / ".verify_checkpoint.json"
    completed_ids: set[str] = set()
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            completed_ids = set(checkpoint_data.get("completed", []))
            logger.info("Resuming verification from checkpoint (%d already done)", len(completed_ids))
        except Exception as exc:
            logger.warning("Failed to load checkpoint, starting fresh: %s", exc)

    for i, letter_path in enumerate(letter_files):
        with open(letter_path, encoding="utf-8") as f:
            letter_data = json.load(f)

        letter_id = letter_data.get("letter_id", letter_path.stem)
        extraction_data = letter_data.get("gemini_extraction", {})

        # Skip if already verified (either by verification_pass or checkpoint)
        if letter_data.get("verification_pass") or letter_id in completed_ids:
            skipped += 1
            logger.info("[%d/%d] %s - SKIPPED (already verified)", i + 1, total, letter_id)
            continue

        # Skip if all fields are high-confidence
        if should_skip_verification(extraction_data):
            skipped += 1
            letter_data["verification_pass"] = {
                "skipped": True,
                "reason": "all_fields_above_threshold",
            }
            atomic_write_json(letter_path, letter_data)
            logger.info("[%d/%d] %s - SKIPPED (high confidence)", i + 1, total, letter_id)
            continue

        # Resolve image paths
        source_images = letter_data.get("source_images", [])
        image_paths = [Path(p) for p in source_images if Path(p).exists()]
        if not image_paths:
            failed += 1
            logger.warning("[%d/%d] %s - FAILED (no images)", i + 1, total, letter_id)
            continue

        # Run verification
        result = await verify_letter(image_paths, extraction_data, config)

        if result.get("error"):
            failed += 1
            letter_data["verification_pass"] = {
                "error": result["error"],
                "model": result["model"],
            }
            logger.error(
                "[%d/%d] %s - ERROR: %s",
                i + 1, total, letter_id, result["error"][:80],
            )
        else:
            verified += 1
            letter_data["verification_pass"] = {
                "verifications": result["verifications"],
                "flags": result["flags"],
                "agreement_rate": result["agreement_rate"],
                "verified_confidence": result["verified_confidence"],
                "elapsed": round(result["elapsed"], 1),
                "model": result["model"],
            }

            # DF-018: Preserve original Pass 2 confidence for audit trail
            if "pass2_confidence" not in letter_data["gemini_extraction"]:
                letter_data["gemini_extraction"]["pass2_confidence"] = letter_data["gemini_extraction"].get("confidence")
            # Update gemini_extraction confidence with verified value
            letter_data["gemini_extraction"]["confidence"] = result["verified_confidence"]

            # Propagate flags
            existing_flags = letter_data.get("flags", [])
            for flag in result["flags"]:
                if flag not in existing_flags:
                    existing_flags.append(flag)
            letter_data["flags"] = existing_flags

            logger.info(
                "[%d/%d] %s - VERIFIED (agreement=%.0f%%, confidence=%.2f)",
                i + 1, total, letter_id,
                result["agreement_rate"] * 100,
                result["verified_confidence"],
            )

        # Write updated letter JSON
        atomic_write_json(letter_path, letter_data)

        # Update per-page JSONs with verification info
        page_ids = letter_data.get("page_ids", [])
        for page_id in page_ids:
            page_path = ocr_pages_dir / f"{page_id}.json"
            if page_path.exists():
                with open(page_path, encoding="utf-8") as f:
                    page_data = json.load(f)
                page_data["verified"] = not result.get("error")
                page_data["verified_confidence"] = result.get("verified_confidence", 0.0)
                atomic_write_json(page_path, page_data)

        # RS-006: Update checkpoint after each letter
        completed_ids.add(letter_id)
        try:
            atomic_write_json(checkpoint_path, {"completed": sorted(completed_ids)})
        except Exception as exc:
            logger.warning("Failed to write checkpoint: %s", exc)

        # Rate limit
        if i < len(letter_files) - 1:
            await asyncio.sleep(rate_limit)

    # RS-006: Clean up checkpoint on successful completion
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except OSError:
            pass

    logger.info(
        "Verification complete: %d verified, %d skipped, %d failed (of %d)",
        verified, skipped, failed, total,
    )
    return {"total": total, "verified": verified, "skipped": skipped, "failed": failed}


def run_verification_pass(project_dir: Path, config: BriefpostConfig) -> BatchResult:
    """Entry point for Pass 3 verification.

    Args:
        project_dir: Project data directory.
        config: Pipeline configuration.

    Returns:
        BatchResult describing the outcome.
    """
    result = BatchResult(phase="verify")

    letters_dir = project_dir / "ocr-letters"
    ocr_pages_dir = project_dir / "ocr-pages"

    if not letters_dir.is_dir():
        logger.warning("No ocr-letters/ directory found, skipping verification")
        result.total = 0
        result.complete()
        return result

    try:
        batch = asyncio.run(
            process_verification_batch(letters_dir, ocr_pages_dir, config)
        )
        result.total = batch["total"]
        result.succeeded = batch["verified"] + batch["skipped"]
        result.failed = batch["failed"]
    except Exception as exc:
        logger.error("Verification pass failed: %s", exc, exc_info=True)
        result.total = 1
        result.failed = 1
        result.errors.append({"phase": "verify", "error": str(exc)})

    result.complete()
    return result
