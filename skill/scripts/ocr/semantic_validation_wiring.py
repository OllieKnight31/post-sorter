"""Wire the bundled semantic validator into the briefpost pipeline.

Loads letter-level JSONs from ocr-letters/, converts gemini_extraction to
GermanExtractionResult, runs SemanticValidator.validate() + adjust_confidence_for_validation(),
writes validation results back into the letter JSON.

Integration point: after verify pass, before postprocess (in pipeline.py as "validate" phase).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.config import BriefpostConfig
from core.io import atomic_write_json
from core.models import BatchResult

logger = logging.getLogger(__name__)

def _import_extraction_types() -> Any:
    """Lazy-load the extraction_types module from the bundled OCR helpers."""
    from core.extern import load_ocr_module
    return load_ocr_module("extraction_types")


def _import_semantic_validator() -> Any:
    """Lazy-load the semantic_validator module from the bundled OCR helpers."""
    from core.extern import load_ocr_module
    return load_ocr_module("semantic_validator")


def validate_letter_extraction(
    extraction_data: dict[str, Any],
) -> dict[str, Any]:
    """Run semantic validation on a single letter's extraction data.

    Converts raw dict to GermanExtractionResult, runs all 8 validation rules,
    applies confidence penalties, and returns validation summary.

    Args:
        extraction_data: The gemini_extraction dict from a letter JSON.

    Returns:
        Dict with validation_result, adjusted_confidence, and issue count.
    """
    types_mod = _import_extraction_types()
    validator_mod = _import_semantic_validator()

    # Convert raw dict to typed extraction result
    result = types_mod.GermanExtractionResult.from_dict(extraction_data)

    # Run 8-rule semantic validator
    validator = validator_mod.SemanticValidator()
    validation = validator.validate(result)

    # Adjust confidence with VALIDATION_PENALTIES
    base_confidence = extraction_data.get("confidence", 0.0)
    issues_dicts = [issue.to_dict() for issue in validation.issues]
    adjusted_confidence = types_mod.adjust_confidence_for_validation(
        base_confidence, issues_dicts,
    )

    return {
        "issues": issues_dicts,
        "high_count": validation.high_severity_count,
        "medium_count": validation.medium_severity_count,
        "passed": validation.passed,
        "needs_review": validation.needs_review,
        "base_confidence": base_confidence,
        "adjusted_confidence": adjusted_confidence,
    }


def run_semantic_validation(project_dir: Path, config: BriefpostConfig) -> BatchResult:
    """Run semantic validation on all letter JSONs in ocr-letters/.

    Reads each letter JSON, validates extraction data, writes validation
    results and adjusted confidence back into the letter JSON.

    Args:
        project_dir: Project data directory.
        config: Pipeline configuration.

    Returns:
        BatchResult describing the outcome.
    """
    result = BatchResult(phase="validate")

    letters_dir = project_dir / "ocr-letters"
    if not letters_dir.is_dir():
        logger.warning("No ocr-letters/ directory, skipping semantic validation")
        result.total = 0
        result.complete()
        return result

    letter_files = sorted(letters_dir.glob("doc-*-L*.json"))
    result.total = len(letter_files)

    validated = 0
    failed = 0
    total_issues = 0

    # RS-007: Load checkpoint to resume from last validated letter
    checkpoint_path = letters_dir / ".validate_checkpoint.json"
    completed_ids: set[str] = set()
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            completed_ids = set(checkpoint_data.get("completed", []))
            logger.info("Resuming validation from checkpoint (%d already done)", len(completed_ids))
        except Exception as exc:
            logger.warning("Failed to load validation checkpoint, starting fresh: %s", exc)

    for letter_path in letter_files:
        try:
            with open(letter_path, encoding="utf-8") as f:
                letter_data = json.load(f)

            letter_id = letter_data.get("letter_id", letter_path.stem)
            extraction_data = letter_data.get("gemini_extraction", {})

            # RS-007: Skip if already validated via checkpoint or existing data
            if letter_id in completed_ids or letter_data.get("semantic_validation"):
                validated += 1
                continue

            if not extraction_data.get("extraction_successful", True):
                # Skip failed extractions
                validated += 1
                completed_ids.add(letter_id)
                continue

            validation = validate_letter_extraction(extraction_data)

            # Write validation results into letter JSON
            letter_data["semantic_validation"] = {
                "issues": validation["issues"],
                "high_count": validation["high_count"],
                "medium_count": validation["medium_count"],
                "passed": validation["passed"],
                "needs_review": validation["needs_review"],
            }

            # DF-019: Check verification_pass block for verified_confidence,
            # not gemini_extraction (where it was never stored).
            has_verified = bool(
                letter_data.get("verification_pass", {}).get("verified_confidence")
            )
            if has_verified:
                letter_data["gemini_extraction"]["semantic_confidence"] = validation["adjusted_confidence"]
            else:
                letter_data["gemini_extraction"]["confidence"] = validation["adjusted_confidence"]

            atomic_write_json(letter_path, letter_data)

            issue_count = len(validation["issues"])
            total_issues += issue_count
            validated += 1
            completed_ids.add(letter_id)

            # RS-007: Update checkpoint after each letter
            try:
                atomic_write_json(checkpoint_path, {"completed": sorted(completed_ids)})
            except Exception as ckpt_exc:
                logger.warning("Failed to write validation checkpoint: %s", ckpt_exc)

            if issue_count > 0:
                logger.info(
                    "%s: %d issues (%d high, %d medium), confidence %.1f -> %.1f",
                    letter_id,
                    issue_count,
                    validation["high_count"],
                    validation["medium_count"],
                    validation["base_confidence"],
                    validation["adjusted_confidence"],
                )

        except Exception as exc:
            failed += 1
            logger.error("Validation failed for %s: %s", letter_path.stem, exc)
            result.errors.append({
                "phase": "validate",
                "file": str(letter_path),
                "error": str(exc),
            })

    result.succeeded = validated
    result.failed = failed
    result.complete()

    # RS-007: Clean up checkpoint on successful completion
    if checkpoint_path.exists() and failed == 0:
        try:
            checkpoint_path.unlink()
        except OSError:
            pass

    logger.info(
        "Semantic validation complete: %d/%d letters, %d issues found, %d failures",
        validated, result.total, total_issues, failed,
    )
    return result
