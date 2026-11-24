"""Phase B2: Post-process and normalize OCR JSON outputs.

Reads all ocr-pages/*.json, applies corrections in-place:
1. Confidence normalization (0-100 -> 0.0-1.0)
2. Signals regeneration with extract_page_signals()
3. IBAN validation via the bundled iban_validator
4. Missing urgency_signals filled with []
5. Null-Amounts flagging (MISSING_AMOUNT)
6. postprocessed_at timestamp
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import BriefpostConfig
from core.constants import Flag
from core.interfaces import OCRPostprocessor
from core.models import BatchResult

logger = logging.getLogger(__name__)


def _get_iban_validator() -> Any:
    """Lazy-load the IBAN validator from the bundled OCR helpers."""
    from core.extern import load_ocr_attr
    return load_ocr_attr("iban_validator", "validate_iban")


def _get_signal_extractor() -> Any:
    """Lazy-load extract_page_signals from OCR orchestrator."""
    from ocr.orchestrator import extract_page_signals
    return extract_page_signals


def _get_german_number_verifier() -> Any:
    """Lazy-load verify_german_number_format from the bundled OCR helpers."""
    from core.extern import load_ocr_attr
    return load_ocr_attr("extraction_types", "verify_german_number_format")


def _get_confidence_adjuster() -> Any:
    """Lazy-load adjust_confidence_for_validation from the bundled OCR helpers."""
    from core.extern import load_ocr_attr
    return load_ocr_attr("extraction_types", "adjust_confidence_for_validation")


class OCRFixPostprocessor(OCRPostprocessor):
    """Concrete implementation of Phase B2 post-processing."""

    def __init__(self, config: BriefpostConfig) -> None:
        """Initialize post-processor.

        Args:
            config: Pipeline configuration.
        """
        self.config = config

    def postprocess(
        self,
        ocr_pages_dir: Path,
        config: dict[str, Any],
    ) -> BatchResult:
        """Apply post-processing fixes to all OCR JSON files.

        Args:
            ocr_pages_dir: Directory containing per-page .json files.
            config: OCR configuration.

        Returns:
            BatchResult with counts of processed/fixed files.
        """
        return _process_all(ocr_pages_dir, self.config, dry_run=False)


def normalize_confidence(data: dict[str, Any]) -> list[str]:
    """Fix 1: Normalize confidence values from 0-100 to 0.0-1.0.

    Args:
        data: Page JSON data to fix in-place.

    Returns:
        List of fix descriptions applied.
    """
    fixes: list[str] = []
    gemini = data.get("gemini_extraction", {})

    conf = gemini.get("confidence")
    if conf is not None and conf > 1.0:
        gemini["confidence"] = round(conf / 100.0, 4)
        fixes.append(f"confidence: {conf} -> {gemini['confidence']}")

    field_conf = gemini.get("field_confidence", {})
    for key, val in field_conf.items():
        if isinstance(val, (int, float)) and val > 1.0:
            field_conf[key] = round(val / 100.0, 4)
            fixes.append(f"field_confidence.{key}: {val} -> {field_conf[key]}")

    return fixes


def regenerate_signals(data: dict[str, Any], ocr_pages_dir: Path) -> list[str]:
    """Fix 2: Regenerate signals using extract_page_signals().

    Args:
        data: Page JSON data to fix in-place.
        ocr_pages_dir: Directory containing .md files.

    Returns:
        List of change descriptions.
    """
    page_id = data.get("page_id", "unknown")
    md_path = ocr_pages_dir / f"{page_id}.md"

    if not md_path.exists():
        return [f"WARN: {page_id}.md not found, signals not regenerated"]

    with open(md_path) as f:
        raw_text = f.read()

    # Strip the markdown header (lines before ---)
    parts = raw_text.split("---\n\n", 1)
    text_content = parts[1] if len(parts) == 2 else raw_text

    if not text_content or text_content.strip() == "[NO TEXT EXTRACTED]":
        data["signals"] = {}
        return ["signals: cleared (no text)"]

    extract_page_signals = _get_signal_extractor()
    new_signals = extract_page_signals(text_content)
    old_signals = data.get("signals", {})
    data["signals"] = new_signals

    changes: list[str] = []
    all_keys = set(list(new_signals.keys()) + list(old_signals.keys()))
    for key in all_keys:
        old_val = old_signals.get(key)
        new_val = new_signals.get(key)
        if old_val != new_val:
            changes.append(f"signals.{key}: {old_val} -> {new_val}")

    return changes


def validate_ibans(data: dict[str, Any]) -> list[str]:
    """Fix 3: Validate all IBANs and add iban_valid field.

    Args:
        data: Page JSON data to fix in-place.

    Returns:
        List of validation descriptions.
    """
    fixes: list[str] = []
    validate_iban = _get_iban_validator()

    gemini = data.get("gemini_extraction", {})
    payment_info = gemini.get("payment_info", {})

    # Validate payment_info IBAN
    iban = payment_info.get("iban")
    if iban and isinstance(iban, str) and len(iban) >= 5:
        result = validate_iban(iban)
        payment_info["iban_valid"] = result.valid
        if not result.valid:
            fixes.append(f"IBAN INVALID: {iban} ({result.error})")
        else:
            fixes.append(f"IBAN valid: {iban}")
    elif iban:
        payment_info["iban_valid"] = False
        fixes.append(f"IBAN too short: {iban}")

    # Validate IBANs found in signals
    signals = data.get("signals", {})
    ibans_found = signals.get("ibans_found", [])
    validated_ibans: list[dict[str, Any]] = []
    for iban_str in ibans_found:
        result = validate_iban(iban_str)
        validated_ibans.append({
            "iban": iban_str,
            "valid": result.valid,
            "error": result.error,
        })
    if validated_ibans:
        signals["ibans_validated"] = validated_ibans

    return fixes


def validate_bics(data: dict[str, Any]) -> list[str]:
    """Fix 3a: Validate BIC length (must be 8 or 11 characters).

    Args:
        data: Page JSON data to fix in-place.

    Returns:
        List of validation descriptions.
    """
    fixes: list[str] = []
    gemini = data.get("gemini_extraction", {})
    payment_info = gemini.get("payment_info", {})

    bic = payment_info.get("bic")
    if bic and isinstance(bic, str):
        clean_bic = bic.strip().replace(" ", "")
        bic_len = len(clean_bic)
        if bic_len not in (8, 11):
            flag_value = f"{Flag.INVALID_BIC_LENGTH.value}:{bic_len}_chars"
            if "flags" not in data:
                data["flags"] = []
            data["flags"].append(flag_value)
            fixes.append(f"BIC INVALID LENGTH: {clean_bic} ({bic_len} chars, expected 8 or 11)")
            logger.warning("BIC invalid length: %s (%d chars)", clean_bic, bic_len)
        else:
            fixes.append(f"BIC valid length: {clean_bic} ({bic_len} chars)")

    return fixes


def verify_amount_formats(data: dict[str, Any], ocr_pages_dir: Path | None = None) -> list[str]:
    """Fix 3b: Verify extracted amounts against German number format in OCR text.

    Invokes verify_german_number_format() from the bundled OCR helpers to detect 100x and
    1000x extraction errors. Applies corrections and flags with AMOUNT_FORMAT_CORRECTED.

    Args:
        data: Page JSON data to fix in-place.
        ocr_pages_dir: Directory containing .md files with OCR text.

    Returns:
        List of fix descriptions.
    """
    fixes: list[str] = []
    gemini = data.get("gemini_extraction", {})

    # Read OCR text from the .md file (same pattern as regenerate_signals).
    # Falls back to data["ocr_text"] if .md file is unavailable.
    raw_text = ""
    if ocr_pages_dir is not None:
        page_id = data.get("page_id", "unknown")
        md_path = ocr_pages_dir / f"{page_id}.md"
        if md_path.exists():
            with open(md_path) as f:
                md_content = f.read()
            parts = md_content.split("---\n\n", 1)
            raw_text = parts[1] if len(parts) == 2 else md_content
    if not raw_text:
        raw_text = data.get("ocr_text", "")

    if not raw_text or not gemini.get("extraction_successful"):
        return fixes

    amounts = gemini.get("amounts", {})
    if not amounts:
        return fixes

    extracted_amounts = {}
    for field_name in ("original", "current", "fees"):
        val = amounts.get(field_name)
        if val is not None:
            extracted_amounts[field_name] = val

    if not extracted_amounts:
        return fixes

    try:
        verify_german_number_format = _get_german_number_verifier()
        result = verify_german_number_format(raw_text, extracted_amounts)

        if not result["valid"]:
            for field_name, corrected in result["suggested_corrections"].items():
                old_val = amounts[field_name]
                amounts[field_name] = corrected
                flag_value = f"{Flag.AMOUNT_FORMAT_CORRECTED.value}:{old_val}_{corrected}"
                if "flags" not in data:
                    data["flags"] = []
                data["flags"].append(flag_value)
                fixes.append(f"amount_format: {field_name} {old_val} -> {corrected}")
                logger.warning("Amount format corrected: %s %s -> %s", field_name, old_val, corrected)
    except Exception as exc:
        logger.debug("verify_german_number_format unavailable: %s", exc)

    return fixes


def adjust_confidence(data: dict[str, Any]) -> list[str]:
    """Fix 3c: Adjust confidence based on validation issues.

    Invokes adjust_confidence_for_validation() from the bundled OCR helpers to apply
    penalties for validation failures (current >= original, missing AZ, etc.).

    Args:
        data: Page JSON data to fix in-place.

    Returns:
        List of fix descriptions.
    """
    fixes: list[str] = []
    gemini = data.get("gemini_extraction", {})
    confidence = gemini.get("confidence")

    if confidence is None:
        return fixes

    # Build validation issues list from data
    validation_issues: list[dict[str, Any]] = []

    amounts = gemini.get("amounts", {})
    current = amounts.get("current")
    original = amounts.get("original")
    if current is not None and original is not None and current >= original and original > 0:
        validation_issues.append({"rule": "current_gte_original", "severity": "high"})

    doc_type = gemini.get("document_type", "")
    if doc_type in ("mahnbescheid", "vollstreckung", "pfaendung"):
        aktenzeichen = gemini.get("aktenzeichen") or gemini.get("reference_number")
        if not aktenzeichen:
            validation_issues.append({"rule": "court_needs_aktenzeichen", "severity": "high"})

    payment_info = gemini.get("payment_info", {})
    if payment_info.get("iban_valid") is False:
        validation_issues.append({"rule": "iban_format", "severity": "high"})

    dates = gemini.get("dates", {})
    due_date_str = dates.get("due_date") or dates.get("faellig_am")
    if due_date_str:
        try:
            from datetime import date as date_type
            due_date = date_type.fromisoformat(due_date_str)
            if due_date > date_type.today():
                pass  # Future due dates are expected
        except (ValueError, TypeError):
            pass

    doc_date_str = dates.get("document_date") or dates.get("datum")
    if doc_date_str:
        try:
            from datetime import date as date_type
            doc_date = date_type.fromisoformat(doc_date_str)
            if doc_date > date_type.today():
                validation_issues.append({"rule": "date_in_future", "severity": "high"})
        except (ValueError, TypeError):
            pass

    if not validation_issues:
        return fixes

    try:
        adjust_confidence_for_validation = _get_confidence_adjuster()
        adjusted = adjust_confidence_for_validation(confidence, validation_issues)

        if adjusted != confidence:
            old_conf = confidence
            gemini["confidence"] = adjusted
            fixes.append(f"confidence_adjusted: {old_conf} -> {adjusted} ({len(validation_issues)} issues)")
            logger.info(
                "Confidence adjusted: %.3f -> %.3f (%d validation issues)",
                old_conf, adjusted, len(validation_issues),
            )
    except Exception as exc:
        logger.debug("adjust_confidence_for_validation unavailable: %s", exc)

    return fixes


def check_cross_page_consistency(
    data: dict[str, Any],
    all_pages_in_doc: list[dict[str, Any]],
) -> list[str]:
    """Fix 3d: Check cross-page consistency within a document group.

    Compares this page's extracted data with other pages in the same document
    group to detect inconsistencies in creditor, person, and amounts.

    Args:
        data: Page JSON data to fix in-place.
        all_pages_in_doc: List of all page JSON dicts in the same doc group.

    Returns:
        List of inconsistency descriptions.
    """
    fixes: list[str] = []
    page_id = data.get("page_id", "unknown")
    gemini = data.get("gemini_extraction", {})

    if not gemini.get("extraction_successful"):
        return fixes

    this_creditor = gemini.get("absender") or gemini.get("sender_name", "")
    this_person = gemini.get("person", "")
    this_current = (gemini.get("amounts") or {}).get("current")

    for other_data in all_pages_in_doc:
        other_id = other_data.get("page_id", "")
        if other_id == page_id:
            continue  # Skip self-comparison

        other_gemini = other_data.get("gemini_extraction", {})
        if not other_gemini.get("extraction_successful"):
            continue

        other_creditor = other_gemini.get("absender") or other_gemini.get("sender_name", "")
        other_person = other_gemini.get("person", "")
        other_current = (other_gemini.get("amounts") or {}).get("current")

        # Check creditor consistency
        if this_creditor and other_creditor:
            if this_creditor.lower() != other_creditor.lower():
                flag = f"{Flag.CROSS_PAGE_INCONSISTENCY.value}:creditor:{page_id}_vs_{other_id}"
                if "flags" not in data:
                    data["flags"] = []
                data["flags"].append(flag)
                fixes.append(f"cross_page: creditor mismatch {page_id} ({this_creditor}) vs {other_id} ({other_creditor})")

        # Check person consistency
        if this_person and other_person:
            if this_person != other_person:
                flag = f"{Flag.CROSS_PAGE_INCONSISTENCY.value}:person:{page_id}_vs_{other_id}"
                if "flags" not in data:
                    data["flags"] = []
                data["flags"].append(flag)
                fixes.append(f"cross_page: person mismatch {page_id} ({this_person}) vs {other_id} ({other_person})")

        # Check amount consistency (only if both have current amounts)
        if this_current is not None and other_current is not None:
            if this_current != other_current and this_current > 0 and other_current > 0:
                diff_pct = abs(this_current - other_current) / max(this_current, other_current) * 100
                if diff_pct > 50:
                    flag = f"{Flag.CROSS_PAGE_INCONSISTENCY.value}:amount:{page_id}_vs_{other_id}"
                    if "flags" not in data:
                        data["flags"] = []
                    data["flags"].append(flag)
                    fixes.append(
                        f"cross_page: amount mismatch {page_id} ({this_current}) vs {other_id} ({other_current})"
                    )

    return fixes


def fill_missing_fields(data: dict[str, Any]) -> list[str]:
    """Fix 4: Fill missing urgency_signals with [].

    Args:
        data: Page JSON data to fix in-place.

    Returns:
        List of fix descriptions.
    """
    fixes: list[str] = []
    gemini = data.get("gemini_extraction", {})

    if "urgency_signals" not in gemini:
        gemini["urgency_signals"] = []
        fixes.append("urgency_signals: added []")

    return fixes


def flag_null_amounts(data: dict[str, Any]) -> list[str]:
    """Fix 5: Flag MISSING_AMOUNT when extraction_successful but no amounts.

    Args:
        data: Page JSON data to fix in-place.

    Returns:
        List of fix descriptions.
    """
    fixes: list[str] = []
    gemini = data.get("gemini_extraction", {})

    if gemini.get("extraction_successful") and not gemini.get("amounts"):
        if "flags" not in data:
            data["flags"] = []
        if Flag.MISSING_AMOUNT.value not in data["flags"]:
            data["flags"].append(Flag.MISSING_AMOUNT.value)
            fixes.append("flag: MISSING_AMOUNT (extraction OK but no amounts)")

    amounts = gemini.get("amounts", {})
    if amounts and amounts.get("current") is None and amounts.get("original") is None:
        if "flags" not in data:
            data["flags"] = []
        if Flag.MISSING_AMOUNT.value not in data["flags"]:
            data["flags"].append(Flag.MISSING_AMOUNT.value)
            fixes.append("flag: MISSING_AMOUNT (amounts block exists but all null)")

    return fixes


def add_timestamp(data: dict[str, Any]) -> list[str]:
    """Fix 6: Add postprocessed_at timestamp.

    Args:
        data: Page JSON data to fix in-place.

    Returns:
        Empty list (no fix descriptions needed).
    """
    data["postprocessed_at"] = datetime.now().isoformat()
    return []


def _process_all(
    ocr_pages_dir: Path,
    config: BriefpostConfig,
    dry_run: bool = False,
) -> BatchResult:
    """Process all OCR JSON files with fixes.

    Args:
        ocr_pages_dir: Directory containing per-page .json files.
        config: Pipeline configuration.
        dry_run: If True, show changes without writing.

    Returns:
        BatchResult with processing stats.
    """
    result = BatchResult(phase="postprocess")
    json_files = sorted(ocr_pages_dir.glob("doc-*.json"))

    logger.info("Post-processing %d JSON files (dry_run=%s)", len(json_files), dry_run)

    stats = {
        "confidence_fixed": 0,
        "signals_changed": 0,
        "ibans_validated": 0,
        "ibans_invalid": 0,
        "bics_invalid": 0,
        "amounts_format_corrected": 0,
        "confidence_adjusted": 0,
        "cross_page_issues": 0,
        "missing_amounts": 0,
        "urgency_filled": 0,
    }

    result.total = len(json_files)

    # Pre-load all page data grouped by doc_id for cross-page consistency
    doc_groups: dict[str, list[dict[str, Any]]] = {}
    all_page_data: dict[str, dict[str, Any]] = {}
    for json_path in json_files:
        try:
            with open(json_path) as f:
                data = json.load(f)
            page_id = json_path.stem
            all_page_data[page_id] = data
            doc_id = data.get("doc_id", "")
            if not doc_id:
                # Infer doc_id from page_id (e.g., "doc-001-page-01" -> "doc-001")
                parts = page_id.split("-page-")
                doc_id = parts[0] if parts else page_id
            doc_groups.setdefault(doc_id, []).append(data)
        except Exception as exc:
            logger.warning("Failed to pre-load %s: %s", json_path.stem, exc)

    for json_path in json_files:
        page_id = json_path.stem
        try:
            data = all_page_data.get(page_id)
            if data is None:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)

            # RS-008: Skip already-processed pages (idempotent re-runs)
            if data.get("postprocessed_at") and not dry_run:
                result.succeeded += 1
                logger.debug("SKIP: %s (already postprocessed)", page_id)
                continue

            page_fixes: list[str] = []

            # Apply all fixes
            page_fixes.extend(normalize_confidence(data))
            page_fixes.extend(regenerate_signals(data, ocr_pages_dir))
            iban_fixes = validate_ibans(data)
            page_fixes.extend(iban_fixes)
            page_fixes.extend(validate_bics(data))
            page_fixes.extend(verify_amount_formats(data, ocr_pages_dir))
            page_fixes.extend(adjust_confidence(data))

            # Cross-page consistency check within doc group
            doc_id = data.get("doc_id", "")
            if not doc_id:
                parts = page_id.split("-page-")
                doc_id = parts[0] if parts else page_id
            group_pages = doc_groups.get(doc_id, [])
            if len(group_pages) > 1:
                page_fixes.extend(check_cross_page_consistency(data, group_pages))

            page_fixes.extend(fill_missing_fields(data))
            page_fixes.extend(flag_null_amounts(data))
            page_fixes.extend(add_timestamp(data))

            # Update stats
            result.succeeded += 1
            if any("confidence:" in f for f in page_fixes):
                stats["confidence_fixed"] += 1
            if any("signals." in f for f in page_fixes):
                stats["signals_changed"] += 1
            if any("IBAN valid" in f for f in page_fixes):
                stats["ibans_validated"] += 1
            if any("IBAN INVALID" in f for f in page_fixes):
                stats["ibans_invalid"] += 1
            if any("BIC INVALID" in f for f in page_fixes):
                stats["bics_invalid"] += 1
            if any("amount_format:" in f for f in page_fixes):
                stats["amounts_format_corrected"] += 1
            if any("confidence_adjusted:" in f for f in page_fixes):
                stats["confidence_adjusted"] += 1
            if any("cross_page:" in f for f in page_fixes):
                stats["cross_page_issues"] += 1
            if any("MISSING_AMOUNT" in f for f in page_fixes):
                stats["missing_amounts"] += 1
            if any("urgency_signals" in f for f in page_fixes):
                stats["urgency_filled"] += 1

            # Write back
            if not dry_run:
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            if page_fixes:
                fix_summary = "; ".join(page_fixes[:3])
                if len(page_fixes) > 3:
                    fix_summary += f" (+{len(page_fixes) - 3} more)"
                logger.debug("[%d/%d] %s: %s", result.succeeded, result.total, page_id, fix_summary)

        except Exception as exc:
            result.failed += 1
            result.errors.append({"page_id": page_id, "error": str(exc)})
            logger.error("[ERROR] %s: %s", page_id, exc)

    # DF-004: Log warning about confidence divergence between ocr-pages and ocr-letters
    letters_dir = ocr_pages_dir.parent / "ocr-letters"
    if letters_dir.is_dir() and any(letters_dir.glob("doc-*-L*.json")):
        logger.info(
            "Note: postprocessor updated ocr-pages/ confidence values. "
            "ocr-letters/ may have stale confidence values until the next "
            "verification pass reconciles them."
        )

    # Verification: all confidence values should be 0.0-1.0
    if not dry_run:
        bad_confidence = 0
        for json_path in json_files:
            with open(json_path) as f:
                data = json.load(f)
            conf = data.get("gemini_extraction", {}).get("confidence", 0)
            if conf > 1.0:
                bad_confidence += 1
                logger.error("FAIL: %s still has confidence=%s", json_path.stem, conf)

        if bad_confidence == 0:
            logger.info("VERIFICATION: All confidence values in 0.0-1.0 range")
        else:
            logger.error("VERIFICATION FAILED: %d files still have confidence > 1.0", bad_confidence)

    logger.info(
        "Post-processing complete: %d processed, confidence_fixed=%d, signals_changed=%d, "
        "ibans_validated=%d, ibans_invalid=%d, bics_invalid=%d, amounts_format_corrected=%d, "
        "confidence_adjusted=%d, cross_page_issues=%d, missing_amounts=%d, "
        "urgency_filled=%d, errors=%d",
        result.succeeded, stats["confidence_fixed"], stats["signals_changed"],
        stats["ibans_validated"], stats["ibans_invalid"], stats["bics_invalid"],
        stats["amounts_format_corrected"], stats["confidence_adjusted"],
        stats["cross_page_issues"], stats["missing_amounts"],
        stats["urgency_filled"], result.failed,
    )

    result.complete()
    return result


def run_postprocess(project_dir: Path, config: BriefpostConfig) -> BatchResult:
    """Entry point called by pipeline.py.

    Args:
        project_dir: Project data directory.
        config: Pipeline configuration.

    Returns:
        BatchResult describing the outcome.
    """
    ocr_pages_dir = project_dir / "ocr-pages"

    if not ocr_pages_dir.exists():
        logger.error("ocr-pages/ directory not found: %s", ocr_pages_dir)
        result = BatchResult(phase="postprocess")
        result.total = 0
        result.complete()
        return result

    return _process_all(ocr_pages_dir, config, dry_run=False)
