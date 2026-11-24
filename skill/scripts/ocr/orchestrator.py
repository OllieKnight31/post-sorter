"""Phase B1: OCR extraction for page images.

Two modes:
  - Per-page OCR (legacy): processes each page-N.jpg individually via run_ocr()
  - Multi-image letter OCR (v4): sends ALL pages of a letter in one Gemini Pro
    call via run_multi_image_ocr(). Requires classification manifest from Pass 1.

Stores per-page results as ocr-pages/doc-XXX-page-NN.md/.json and letter-level
results as ocr-letters/LTR-XXX.json.

Uses the bundled Gemini provider (scripts/vendor/ocr_core) for extraction and
regex-based signal extraction for page-to-letter matching.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core.config import BriefpostConfig
from core.constants import Flag
from core.io import atomic_write_json
from core.interfaces import OCRExtractor
from core.models import BatchResult, Page

logger = logging.getLogger(__name__)


def _import_gemini_ocr() -> Any:
    """Lazy-load the GeminiOCR class from the bundled OCR helpers."""
    from core.extern import load_ocr_attr
    return load_ocr_attr("gemini", "GeminiOCR")


def _import_extraction_config() -> tuple[Any, Any]:
    """Lazy-load GeminiExtractionConfig and ThinkingLevel from the bundled OCR helpers."""
    from core.extern import load_ocr_module
    mod = load_ocr_module("extraction_types")
    return mod.GeminiExtractionConfig, mod.ThinkingLevel


def ensure_api_key() -> None:
    """Load GOOGLE_API_KEY from the environment or an .env file."""
    from core.extern import load_api_key
    load_api_key()


def extract_page_signals(text: str) -> dict[str, Any]:
    """Extract structural signals from raw OCR text for page-to-letter matching.

    These signals are used alongside Gemini's structured extraction to match
    pages into actual letters. The regex signals catch things Gemini might miss
    and provide redundancy for critical matching decisions.

    Args:
        text: Raw OCR text from a page.

    Returns:
        Dictionary of detected signals.
    """
    signals: dict[str, Any] = {}

    # Page numbering (critical for letter boundary detection)
    page_match = re.search(r"[Ss]eite\s+(\d+)\s+(?:von|/)\s+(\d+)", text)
    if page_match:
        signals["page_indicator"] = f"Seite {page_match.group(1)} von {page_match.group(2)}"
        signals["page_current"] = int(page_match.group(1))
        signals["page_total"] = int(page_match.group(2))
        signals["is_first_page"] = int(page_match.group(1)) == 1

    # Abbreviated page format: "1/4", "2/3" (standalone, last 200 chars)
    if not page_match:
        page_match = re.search(r"(?<!\d)(\d{1,2})\s*/\s*(\d{1,2})(?!\d)", text[-200:])
        if page_match:
            signals["page_indicator"] = f"{page_match.group(1)}/{page_match.group(2)}"
            signals["page_current"] = int(page_match.group(1))
            signals["page_total"] = int(page_match.group(2))
            signals["is_first_page"] = int(page_match.group(1)) == 1

    # Address block detection (strong new-letter signal)
    from core import persons as _persons
    signals["has_address_block"] = _persons.has_address_block(text)

    # Letterhead detection in top portion (strong new-letter signal)
    has_letterhead = bool(re.search(
        r"(?:GmbH|AG|A\u00f6R|e\.V\.|Inkasso|Amtsgericht|Telekom|Sparkasse|Notar|SCHUFA|Finanzamt|Jobcenter"
        r"|\bLtd\b|\bLimited\b|\bPLC\b|\bLLP\b|\bInc\b|\bLLC\b|\bCorp\b|County Court|Council|HMRC|\bBank\b|Solicitors|Insurance)",
        text[:500], re.IGNORECASE,
    ))
    signals["has_letterhead"] = has_letterhead

    # Dates found
    date_matches = re.findall(r"\d{2}\.\d{2}\.\d{4}", text)
    if date_matches:
        signals["dates_found"] = date_matches[:5]

    # Amounts with EUR
    amount_matches = re.findall(r"(\d{1,3}(?:\.\d{3})*,\d{2}|\d{4,},\d{2})\s*(?:EUR|\u20ac)", text)
    if amount_matches:
        signals["amounts_found"] = amount_matches[:10]

    # IBANs
    iban_matches = re.findall(r"[A-Z]{2}\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2,4}", text)
    if iban_matches:
        signals["ibans_found"] = [iban.replace(" ", "") for iban in iban_matches]

    # Reference numbers / Aktenzeichen
    az_matches = re.findall(
        r"(?:Aktenzeichen|Gesch\u00e4ftsnummer|Gesch\u00e4ftszeichen|Kassenzeichen|Kunden-?Nr|Versichertennr|Rechnungsnr|Rechnungs-Nr|Rechn\.-Nr)[.:]*\s*([^\n,]{3,40})",
        text, re.IGNORECASE,
    )
    if az_matches:
        signals["references_found"] = [r.strip() for r in az_matches]

    # Document type keywords
    type_keywords = {
        "rechnung": bool(re.search(r"\bRechnung\b", text)),
        "mahnung": bool(re.search(r"\bMahnung\b", text)),
        "mahnbescheid": bool(re.search(r"\bMahnbescheid\b", text)),
        "vollstreckung": bool(re.search(r"\bVollstreckungsbescheid\b", text)),
        "pfaendung": bool(re.search(r"\bPf\u00e4ndung", text)),
        "bescheid": bool(re.search(r"\bBescheid\b", text)),
        "versicherung": bool(re.search(r"\bVersicherung", text)),
        "kuendigung": bool(re.search(r"\bK\u00fcndigung\b", text)),
        "erbschaft": bool(re.search(r"\bErb(?:schein|schaft|folge)\b", text, re.IGNORECASE)),
        "widerspruch": bool(re.search(r"\bWiderspruch\b", text)),
        "kremation": bool(re.search(r"\b(?:Kremation|Ein\u00e4scherung|Urne)\b", text, re.IGNORECASE)),
    }
    detected = [k for k, v in type_keywords.items() if v]
    if detected:
        signals["type_keywords"] = detected

    # Person signals (critical for person assignment); names come from config
    person_signals: list[str] = _persons.detect_persons(text)
    signals["persons_detected"] = person_signals

    # Sender extraction from first 500 chars
    sender_match = re.search(
        r"((?:[\w\-]+\s+){1,4}(?:GmbH|AG|A\u00f6R|e\.V\.|KG|SE|mbH)(?:\s*&\s*Co\.\s*KG)?)",
        text[:500],
    )
    if sender_match:
        signals["sender_name"] = sender_match.group(1).strip()

    return signals


class GeminiOCRExtractor(OCRExtractor):
    """Concrete implementation of Phase B1 OCR using Gemini."""

    def __init__(self, config: BriefpostConfig) -> None:
        """Initialize OCR extractor with configuration.

        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.rate_limit = config.ocr.get("rate_limit_seconds", 4)
        self.max_retries = config.ocr.get("max_retries", 3)
        self.retry_backoff = config.ocr.get("retry_backoff", [10, 20, 30])

    def extract_page(
        self,
        image_path: Path,
        config: dict[str, Any],
    ) -> Page:
        """Run OCR on a single page image (sync wrapper).

        Args:
            image_path: Path to the page image (JPG).
            config: OCR configuration.

        Returns:
            Page with OCR text, confidence, signals, and extraction data.
        """
        result = asyncio.run(self._extract_page_async(image_path, config))
        return result

    async def _extract_page_async(
        self,
        image_path: Path,
        config: dict[str, Any],
    ) -> Page:
        """Async OCR extraction for a single page."""
        ensure_api_key()

        GeminiOCR = _import_gemini_ocr()

        GeminiExtractionConfig, ThinkingLevel = _import_extraction_config()
        extraction_config = GeminiExtractionConfig(
            temperature=0.0,
            thinking_level=ThinkingLevel.HIGH,
            use_structured_output=True,
            german_mode=True,
        )

        gemini_client = GeminiOCR()
        gemini_client.MODEL = self.config.ocr.get("model", gemini_client.MODEL)
        start = time.time()
        result = await gemini_client.extract_german_financial(
            str(image_path), config=extraction_config,
        )
        elapsed = time.time() - start

        raw_text = result.raw_text or ""
        confidence = result.confidence
        gemini_data = result.to_dict()
        gemini_data.pop("raw_text", None)

        signals = extract_page_signals(raw_text) if raw_text else {}

        return Page(
            page_id=image_path.stem,
            doc_group="",
            page_number=0,
            ocr_text=raw_text,
            ocr_confidence=confidence / 100.0 if confidence > 1.0 else confidence,
            ocr_provider="gemini",
            signals=signals,
            gemini_extraction=gemini_data,
            source_image=str(image_path),
        )


def discover_all_pages(grouped_dir: Path) -> list[dict[str, Any]]:
    """Find all page images across all document groups.

    Args:
        grouped_dir: Path to grouped/ directory.

    Returns:
        List of page info dicts with page_id, doc_id, page_num, image_path.
    """
    pages: list[dict[str, Any]] = []
    for doc_dir in sorted(grouped_dir.iterdir()):
        if not doc_dir.is_dir() or not doc_dir.name.startswith("doc-"):
            continue
        doc_id = doc_dir.name
        page_images = []
        for f in doc_dir.glob("page-*.jpg"):
            m = re.search(r"page-(\d+)", f.name)
            if m is None:
                logger.warning("Skipping unexpected filename: %s", f.name)
                continue
            page_images.append((int(m.group(1)), f))
        page_images.sort(key=lambda x: x[0])
        for page_num, img in page_images:
            page_id = f"{doc_id}-page-{page_num:02d}"
            pages.append({
                "page_id": page_id,
                "doc_id": doc_id,
                "page_num": page_num,
                "image_path": str(img),
            })
    return pages


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load processing checkpoint.

    Args:
        checkpoint_path: Path to .checkpoint.json file.

    Returns:
        Checkpoint data with completed and failed lists.
    """
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "last_updated": None}


def save_checkpoint(checkpoint: dict[str, Any], checkpoint_path: Path) -> None:
    """Save processing checkpoint.

    Args:
        checkpoint: Checkpoint data to save.
        checkpoint_path: Path to write checkpoint.
    """
    checkpoint["last_updated"] = datetime.now().isoformat()
    atomic_write_json(checkpoint_path, checkpoint)


async def ocr_single_page(
    gemini_client: Any,
    page_info: dict[str, Any],
    ocr_pages_dir: Path,
    config: BriefpostConfig,
) -> dict[str, Any]:
    """OCR a single page using Gemini extract_german_financial.

    Args:
        gemini_client: Initialized GeminiOCR client.
        page_info: Page info dict from discover_all_pages.
        ocr_pages_dir: Output directory for OCR results.
        config: Pipeline configuration.

    Returns:
        Result dict with status, page_id, and metadata.
    """
    page_id = page_info["page_id"]
    image_path = page_info["image_path"]
    doc_id = page_info["doc_id"]
    page_num = page_info["page_num"]

    md_path = ocr_pages_dir / f"{page_id}.md"
    json_path = ocr_pages_dir / f"{page_id}.json"

    if md_path.exists() and json_path.exists():
        return {"status": "skipped", "page_id": page_id}

    max_retries = config.ocr.get("max_retries", 3)
    retry_backoff = config.ocr.get("retry_backoff", [10, 20, 30])
    last_error = None

    GeminiExtractionConfig, ThinkingLevel = _import_extraction_config()
    extraction_config = GeminiExtractionConfig(
        temperature=0.0,
        thinking_level=ThinkingLevel.HIGH,
        use_structured_output=True,
        german_mode=True,
    )

    for attempt in range(max_retries):
        try:
            start = time.time()
            result = await gemini_client.extract_german_financial(
                image_path, config=extraction_config,
            )
            elapsed = time.time() - start

            raw_text = result.raw_text or ""
            confidence = result.confidence
            extraction_ok = result.extraction_successful

            # Check for soft failure (503/UNAVAILABLE inside result)
            if not extraction_ok and result.error_reason and (
                "503" in str(result.error_reason) or "UNAVAILABLE" in str(result.error_reason)
            ):
                if attempt < max_retries - 1:
                    wait = retry_backoff[attempt] if attempt < len(retry_backoff) else 30
                    logger.warning(
                        "[%s] Gemini 503 in result, retry %d/%d after %ds",
                        page_id, attempt + 1, max_retries, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                else:
                    last_error = result.error_reason
                    break

            # Write raw OCR markdown
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# {page_id}\n\n")
                f.write(f"**Source:** {image_path}\n")
                f.write(f"**Provider:** gemini\n")
                f.write(f"**Confidence:** {confidence}\n")
                f.write(f"**Extraction OK:** {extraction_ok}\n")
                f.write(f"**Time:** {elapsed:.1f}s\n\n")
                f.write("---\n\n")
                f.write(raw_text if raw_text else "[NO TEXT EXTRACTED]")

            # Build per-page JSON with all data
            page_meta: dict[str, Any] = {
                "page_id": page_id,
                "doc_id": doc_id,
                "page_num": page_num,
                "image_path": image_path,
                "ocr_provider": "gemini",
                "ocr_time_seconds": round(elapsed, 1),
                "raw_text_length": len(raw_text),
                "raw_ocr_file": f"ocr-pages/{page_id}.md",
            }

            gemini_data = result.to_dict()
            gemini_data.pop("raw_text", None)
            page_meta["gemini_extraction"] = gemini_data

            if raw_text:
                page_meta["signals"] = extract_page_signals(raw_text)

            atomic_write_json(json_path, page_meta)

            return {
                "status": "success",
                "page_id": page_id,
                "confidence": confidence,
                "time": elapsed,
                "text_length": len(raw_text),
            }

        except Exception as exc:
            last_error = str(exc)
            is_transient = "503" in last_error or "UNAVAILABLE" in last_error or "quota" in last_error.lower()

            if is_transient and attempt < max_retries - 1:
                wait = retry_backoff[attempt] if attempt < len(retry_backoff) else 30
                logger.warning(
                    "[%s] Retry %d/%d after %ds: %s",
                    page_id, attempt + 1, max_retries, wait, last_error[:60],
                )
                await asyncio.sleep(wait)
                continue
            break

    # All retries failed - write error JSON
    atomic_write_json(json_path, {
        "page_id": page_id,
        "doc_id": doc_id,
        "page_num": page_num,
        "image_path": image_path,
        "ocr_provider": "gemini",
        "error": last_error,
        "gemini_extraction": {
            "extraction_successful": False,
            "confidence": 0.0,
        },
    })

    return {"status": "error", "page_id": page_id, "error": last_error}


async def process_batch(
    pages: list[dict[str, Any]],
    ocr_pages_dir: Path,
    config: BriefpostConfig,
) -> dict[str, int]:
    """Process pages sequentially with rate limiting and checkpointing.

    Args:
        pages: List of page info dicts from discover_all_pages.
        ocr_pages_dir: Output directory for OCR results.
        config: Pipeline configuration.

    Returns:
        Dict with total, done, failed counts.
    """
    ensure_api_key()

    GeminiOCR = _import_gemini_ocr()

    gemini_client = GeminiOCR()
    gemini_client.MODEL = config.ocr.get("model", gemini_client.MODEL)
    rate_limit = config.ocr.get("rate_limit_seconds", 4)
    checkpoint_frequency = config.pipeline.get("checkpoint_frequency", 5)

    checkpoint_path = ocr_pages_dir / ".checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)
    completed_set = set(checkpoint["completed"])
    failed_set: set[str] = set()
    for f in checkpoint.get("failed", []):
        if isinstance(f, dict):
            failed_set.add(f.get("page_id", ""))
        else:
            failed_set.add(str(f))
    skip_set = completed_set | failed_set

    remaining = [p for p in pages if p["page_id"] not in skip_set]
    if failed_set:
        logger.info("Skipping %d previously failed pages", len(failed_set))

    total = len(pages)
    done = len(completed_set & {p["page_id"] for p in pages})
    failed = 0

    logger.info(
        "OCR batch: %d total, %d done, %d remaining, rate=%ds",
        total, done, len(remaining), rate_limit,
    )

    for i, page in enumerate(remaining):
        result = await ocr_single_page(gemini_client, page, ocr_pages_dir, config)

        if result["status"] == "success":
            done += 1
            checkpoint["completed"].append(result["page_id"])
            logger.info(
                "[%d/%d] %s - OK (%.0f%%, %.1fs, %d chars)",
                done, total, result["page_id"],
                result.get("confidence", 0), result.get("time", 0),
                result.get("text_length", 0),
            )
        elif result["status"] == "skipped":
            done += 1
            if result["page_id"] not in checkpoint["completed"]:
                checkpoint["completed"].append(result["page_id"])
            logger.info("[%d/%d] %s - SKIPPED (already exists)", done, total, result["page_id"])
        else:
            failed += 1
            checkpoint["failed"].append({
                "page_id": result["page_id"],
                "error": result.get("error", "unknown"),
            })
            logger.error(
                "[%d/%d] %s - ERROR: %s",
                done, total, result["page_id"], result.get("error", "unknown")[:80],
            )

        # Checkpoint every N pages
        if (i + 1) % checkpoint_frequency == 0:
            save_checkpoint(checkpoint, checkpoint_path)
            logger.info("[CHECKPOINT] %d/%d done, %d failed", done, total, failed)

        # Rate limit between API calls (skip for skipped pages)
        if result["status"] != "skipped" and i < len(remaining) - 1:
            await asyncio.sleep(rate_limit)

    # Final checkpoint
    save_checkpoint(checkpoint, checkpoint_path)

    logger.info("OCR batch complete: %d/%d pages, %d failures", done, total, failed)
    return {"total": total, "done": done, "failed": failed}


def run_ocr(project_dir: Path, config: BriefpostConfig) -> BatchResult:
    """Entry point called by pipeline.py.

    Args:
        project_dir: Project data directory.
        config: Pipeline configuration.

    Returns:
        BatchResult describing the outcome.
    """
    result = BatchResult(phase="ocr")
    from core.persons import configure_persons
    configure_persons(config)

    grouped_dir = project_dir / "grouped"
    ocr_pages_dir = project_dir / "ocr-pages"
    ocr_pages_dir.mkdir(exist_ok=True)

    pages = discover_all_pages(grouped_dir)
    logger.info("Discovered %d pages across %d groups",
                len(pages), len({p["doc_id"] for p in pages}))

    if not pages:
        logger.error("No pages found in %s", grouped_dir)
        result.total = 0
        result.complete()
        return result

    try:
        batch_result = asyncio.run(process_batch(pages, ocr_pages_dir, config))
        result.total = batch_result["total"]
        result.succeeded = batch_result["done"]
        result.failed = batch_result["failed"]
    except Exception as exc:
        logger.error("OCR batch failed: %s", exc, exc_info=True)
        result.total = len(pages)
        result.failed = len(pages)
        result.errors.append({"phase": "ocr", "error": str(exc)})

    result.complete()
    return result


# ---------------------------------------------------------------------------
# v4: Multi-image letter-level OCR (Pass 2)
# ---------------------------------------------------------------------------

MULTI_IMAGE_MODEL = "gemini-3.8-flash"


def _import_extraction_prompt() -> Any:
    """Lazy-load the extraction prompt builder from the bundled OCR helpers."""
    from core.extern import load_ocr_module
    return load_ocr_module("extraction_prompts")


def _import_extraction_schemas() -> Any:
    """Lazy-load the extraction schemas from the bundled OCR helpers."""
    from core.extern import load_ocr_module
    return load_ocr_module("extraction_schemas")


def load_classification_manifest(project_dir: Path) -> Optional[dict[str, Any]]:
    """Load the classification manifest written by Pass 1.

    Args:
        project_dir: Project data directory.

    Returns:
        Manifest dict with classifications, or None if not found.
    """
    manifest_path = project_dir / "classification" / "classification-manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def validate_letter_assignments(manifest: dict[str, Any], grouped_dir: Path) -> list[str]:
    """B3: Pre-extraction gate: validate every page is assigned exactly once.

    Loads the classification manifest and checks for gaps (unassigned pages)
    and duplicates (pages assigned to multiple letters).

    Args:
        manifest: Classification manifest from Pass 1.
        grouped_dir: Path to grouped/ directory for page count validation.

    Returns:
        List of error strings. Empty list means validation passed.
    """
    errors: list[str] = []
    import re as _re

    for group in manifest.get("classifications", []):
        doc_id = group.get("doc_id", "")
        if not group.get("valid"):
            continue

        doc_dir = grouped_dir / doc_id
        if not doc_dir.is_dir():
            continue

        # Count actual pages in directory
        _page_tuples = []
        for _f in doc_dir.glob("page-*.jpg"):
            _m = _re.search(r"page-(\d+)", _f.name)
            if _m is None:
                continue
            _page_tuples.append((int(_m.group(1)), _f))
        _page_tuples.sort(key=lambda x: x[0])
        actual_pages = [t[1] for t in _page_tuples]
        expected_page_nums = set(range(1, len(actual_pages) + 1))

        # Collect assigned page numbers
        assigned_pages: list[int] = []
        for letter in group.get("letters", []):
            assigned_pages.extend(letter.get("pages", []))

        assigned_set = set(assigned_pages)

        # Check for gaps
        missing = expected_page_nums - assigned_set
        if missing:
            errors.append(f"{doc_id}: missing pages {sorted(missing)}")

        # Check for duplicates
        if len(assigned_pages) != len(assigned_set):
            seen: set[int] = set()
            dupes: list[int] = []
            for p in assigned_pages:
                if p in seen:
                    dupes.append(p)
                seen.add(p)
            errors.append(f"{doc_id}: duplicate pages {dupes}")

    return errors


def build_letter_assignments(manifest: dict[str, Any], grouped_dir: Path) -> list[dict[str, Any]]:
    """Convert classification manifest into letter assignments with image paths.

    Each letter assignment contains all page image paths for that letter,
    ready for multi-image Gemini extraction.

    Args:
        manifest: Classification manifest from Pass 1.
        grouped_dir: Path to grouped/ directory for resolving image paths.

    Returns:
        List of letter assignment dicts, each with:
          - letter_idx: sequential index within doc group
          - doc_id: document group ID
          - page_ids: list of page IDs (e.g., "doc-001-page-01")
          - image_paths: list of Path objects for page images
          - absender: sender from classification (may be None)
          - document_type: type from classification
          - person: person from classification (may be None)
          - classification_confidence: confidence from Pass 1
    """
    assignments: list[dict[str, Any]] = []

    for group in manifest.get("classifications", []):
        doc_id = group.get("doc_id", "")
        if not group.get("valid"):
            continue

        doc_dir = grouped_dir / doc_id
        if not doc_dir.is_dir():
            continue

        for letter_idx, letter in enumerate(group.get("letters", [])):
            page_nums = letter.get("pages", [])
            if not page_nums:
                continue

            page_ids: list[str] = []
            image_paths: list[Path] = []

            for page_num in page_nums:
                page_path = doc_dir / f"page-{page_num}.jpg"
                if not page_path.exists():
                    logger.warning("Missing image: %s", page_path)
                    continue
                page_ids.append(f"{doc_id}-page-{page_num:02d}")
                image_paths.append(page_path)

            if not image_paths:
                continue

            assignments.append({
                "letter_idx": letter_idx,
                "doc_id": doc_id,
                "page_ids": page_ids,
                "image_paths": image_paths,
                "absender": letter.get("absender"),
                "document_type": letter.get("document_type", "unknown"),
                "person": letter.get("person"),
                "classification_confidence": letter.get("confidence", 0.0),
            })

    return assignments


async def ocr_letter_multi_image(
    image_paths: list[Path],
    config: BriefpostConfig,
    letter_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Extract structured data from ALL pages of a letter in one Gemini Pro call.

    Sends all page images as a multi-image request to Gemini 2.5 Pro with
    the v4 extraction prompt, temperature 0.0, and Thinking HIGH (32K tokens).

    Args:
        image_paths: Ordered list of page image paths for this letter.
        config: Pipeline configuration.
        letter_context: Optional classification context (absender, type, person).

    Returns:
        Extraction result dict with gemini_extraction, raw_text, confidence, etc.
    """
    ensure_api_key()

    from google import genai
    from google.genai import types

    client = genai.Client()

    # Build multi-image parts
    parts: list[Any] = []
    for image_path in image_paths:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        parts.append(types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        ))

    # Build extraction prompt with multi-image prefix
    prompts_mod = _import_extraction_prompt()
    base_prompt = prompts_mod.get_extraction_prompt(
        mode=prompts_mod.ExtractionMode.GERMAN_FINANCIAL,
        include_few_shot=True,
    )

    # Multi-image context prefix
    page_count = len(image_paths)
    context_lines = [
        f"Du siehst {page_count} Seite(n) eines einzelnen Briefes.",
        "Alle Bilder gehoeren zum GLEICHEN Brief/Dokument.",
        "Extrahiere die Daten aus ALLEN Seiten zusammen als ein Dokument.",
    ]
    if letter_context:
        if letter_context.get("absender"):
            context_lines.append(
                f"Vermuteter Absender (aus Klassifikation): {letter_context['absender']}"
            )
        if letter_context.get("document_type") and letter_context["document_type"] != "unknown":
            context_lines.append(
                f"Vermuteter Dokumenttyp: {letter_context['document_type']}"
            )
        if letter_context.get("person"):
            context_lines.append(
                f"Vermuteter Empfaenger: {letter_context['person']}"
            )

    multi_image_prefix = "\n".join(context_lines) + "\n\n"
    full_prompt = multi_image_prefix + base_prompt
    parts.append(full_prompt)

    # Build generation config: temp 0.0, thinking level high
    schemas_mod = _import_extraction_schemas()
    gen_config: dict[str, Any] = {
        "temperature": 0.0,
        "max_output_tokens": 8192,
        # Gemini 3.5 and newer take a thinking level; a token budget is rejected
        "thinking_config": {"thinking_level": str(config.ocr.get("thinking_level", "high"))},
        "response_mime_type": "application/json",
        "response_schema": schemas_mod.GERMAN_EXTRACTION_SCHEMA,
    }

    model_name = config.ocr.get("multi_image_model", MULTI_IMAGE_MODEL)
    max_retries = config.ocr.get("max_retries", 3)
    retry_backoff = config.ocr.get("retry_backoff", [10, 20, 30])
    last_error = None

    for attempt in range(max_retries):
        try:
            start = time.time()
            from core.llm_helpers import generate_content_safe
            response = await asyncio.to_thread(
                lambda: generate_content_safe(client, model_name, parts, gen_config)
            )
            elapsed = time.time() - start

            response_text = response.text if hasattr(response, "text") else ""

            # Parse JSON response
            json_text = response_text
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()

            try:
                extraction_data = json.loads(json_text)
                if isinstance(extraction_data, list):
                    extraction_data = extraction_data[0] if extraction_data else {}
            except json.JSONDecodeError:
                extraction_data = {
                    "extraction_successful": False,
                    "raw_text": response_text,
                    "confidence": 0.0,
                    "flags": [Flag.EXTRACTION_PARSE_FAILURE.value],
                }

            raw_text = extraction_data.get("raw_text", "")
            confidence = extraction_data.get("confidence", 0.0)

            return {
                "extraction_data": extraction_data,
                "raw_text": raw_text,
                "confidence": confidence,
                "elapsed": elapsed,
                "model": model_name,
                "page_count": page_count,
                "error": None,
            }

        except Exception as exc:
            last_error = str(exc)
            is_transient = "503" in last_error or "UNAVAILABLE" in last_error or "quota" in last_error.lower()
            if is_transient and attempt < max_retries - 1:
                wait = retry_backoff[attempt] if attempt < len(retry_backoff) else 30
                logger.warning(
                    "Multi-image OCR retry %d/%d after %ds: %s",
                    attempt + 1, max_retries, wait, last_error[:60],
                )
                await asyncio.sleep(wait)
                continue
            break

    return {
        "extraction_data": {"extraction_successful": False, "confidence": 0.0},
        "raw_text": "",
        "confidence": 0.0,
        "elapsed": 0.0,
        "model": model_name,
        "page_count": page_count,
        "error": last_error,
    }


def write_letter_ocr_output(
    assignment: dict[str, Any],
    extraction_result: dict[str, Any],
    ocr_letters_dir: Path,
    ocr_pages_dir: Path,
) -> None:
    """Write letter-level and per-page OCR output files.

    Letter-level JSON goes to ocr-letters/. Per-page JSONs are also written
    to ocr-pages/ for QC traceability.

    Args:
        assignment: Letter assignment dict from build_letter_assignments.
        extraction_result: Result from ocr_letter_multi_image.
        ocr_letters_dir: Output directory for letter-level JSONs.
        ocr_pages_dir: Output directory for per-page JSONs (QC).
    """
    doc_id = assignment["doc_id"]
    letter_idx = assignment["letter_idx"]
    page_ids = assignment["page_ids"]
    image_paths = assignment["image_paths"]

    extraction_data = extraction_result.get("extraction_data", {})
    raw_text = extraction_result.get("raw_text", "")

    # Generate a letter ID: doc-001-L0, doc-001-L1, etc.
    letter_id = f"{doc_id}-L{letter_idx}"

    # Write letter-level JSON
    letter_json = {
        "letter_id": letter_id,
        "doc_id": doc_id,
        "page_ids": page_ids,
        "page_count": len(page_ids),
        "source_images": [str(p) for p in image_paths],
        "ocr_provider": "gemini",
        "ocr_model": extraction_result.get("model", MULTI_IMAGE_MODEL),
        "ocr_mode": "multi_image",
        "ocr_time_seconds": round(extraction_result.get("elapsed", 0), 1),
        "classification_context": {
            "absender": assignment.get("absender"),
            "document_type": assignment.get("document_type"),
            "person": assignment.get("person"),
            "classification_confidence": assignment.get("classification_confidence", 0.0),
        },
        "gemini_extraction": extraction_data,
    }
    if extraction_result.get("error"):
        letter_json["error"] = extraction_result["error"]

    letter_path = ocr_letters_dir / f"{letter_id}.json"
    atomic_write_json(letter_path, letter_json)

    # Write per-page JSONs for QC traceability
    for i, page_id in enumerate(page_ids):
        _page_m = re.search(r"page-(\d+)", page_id)
        _page_num = int(_page_m.group(1)) if _page_m else i + 1
        page_meta: dict[str, Any] = {
            "page_id": page_id,
            "doc_id": doc_id,
            "page_num": _page_num,
            "image_path": str(image_paths[i]),
            "ocr_provider": "gemini",
            "ocr_mode": "multi_image",
            "letter_id": letter_id,
            "raw_text_length": len(raw_text),
            "raw_ocr_file": f"ocr-letters/{letter_id}.json",
            "gemini_extraction": extraction_data,
        }
        if raw_text:
            page_meta["signals"] = extract_page_signals(raw_text)

        json_path = ocr_pages_dir / f"{page_id}.json"
        atomic_write_json(json_path, page_meta)

        # Write per-page markdown for compatibility
        md_path = ocr_pages_dir / f"{page_id}.md"
        confidence = extraction_result.get("confidence", 0.0)
        elapsed = extraction_result.get("elapsed", 0.0)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {page_id}\n\n")
            f.write(f"**Source:** {image_paths[i]}\n")
            f.write(f"**Provider:** gemini (multi-image, letter {letter_id})\n")
            f.write(f"**Confidence:** {confidence}\n")
            f.write(f"**Time:** {elapsed:.1f}s (letter total)\n\n")
            f.write("---\n\n")
            f.write(raw_text if raw_text else "[NO TEXT EXTRACTED]")


async def process_letters_multi_image(
    assignments: list[dict[str, Any]],
    ocr_letters_dir: Path,
    ocr_pages_dir: Path,
    config: BriefpostConfig,
) -> dict[str, int]:
    """Process all letters using multi-image extraction.

    Args:
        assignments: Letter assignments from build_letter_assignments.
        ocr_letters_dir: Output directory for letter-level JSONs.
        ocr_pages_dir: Output directory for per-page JSONs.
        config: Pipeline configuration.

    Returns:
        Dict with total, done, failed counts.
    """
    rate_limit = config.ocr.get("rate_limit_seconds", 4)
    checkpoint_frequency = config.pipeline.get("checkpoint_frequency", 5)

    checkpoint_path = ocr_letters_dir / ".checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)
    completed_set = set(checkpoint["completed"])

    total = len(assignments)
    done = 0
    failed = 0

    for i, assignment in enumerate(assignments):
        letter_id = f"{assignment['doc_id']}-L{assignment['letter_idx']}"

        # Skip if already completed
        if letter_id in completed_set:
            done += 1
            logger.info("[%d/%d] %s - SKIPPED (checkpoint)", done, total, letter_id)
            continue

        # Skip if output already exists
        letter_path = ocr_letters_dir / f"{letter_id}.json"
        previous_failed = False
        if letter_path.exists():
            # A file written by a failed attempt (quota, bad key, retired model)
            # carries an "error" entry and must be read again, not skipped.
            try:
                with open(letter_path, encoding="utf-8") as _prev:
                    previous_failed = bool(json.load(_prev).get("error"))
            except (OSError, json.JSONDecodeError, AttributeError):
                previous_failed = True
        if letter_path.exists() and not previous_failed:
            done += 1
            checkpoint["completed"].append(letter_id)
            logger.info("[%d/%d] %s - SKIPPED (exists)", done, total, letter_id)
            continue

        try:
            result = await ocr_letter_multi_image(
                assignment["image_paths"],
                config,
                letter_context={
                    "absender": assignment.get("absender"),
                    "document_type": assignment.get("document_type"),
                    "person": assignment.get("person"),
                },
            )

            write_letter_ocr_output(assignment, result, ocr_letters_dir, ocr_pages_dir)

            if result.get("error"):
                failed += 1
                checkpoint["failed"].append({
                    "letter_id": letter_id,
                    "error": result["error"],
                })
                logger.error(
                    "[%d/%d] %s - ERROR: %s",
                    done + failed, total, letter_id, result["error"][:80],
                )
            else:
                done += 1
                checkpoint["completed"].append(letter_id)
                logger.info(
                    "[%d/%d] %s - OK (%d pages, %.0f%%, %.1fs)",
                    done, total, letter_id,
                    result.get("page_count", 0),
                    result.get("confidence", 0),
                    result.get("elapsed", 0),
                )

        except Exception as exc:
            failed += 1
            checkpoint["failed"].append({
                "letter_id": letter_id,
                "error": str(exc),
            })
            logger.error("[%d/%d] %s - EXCEPTION: %s", done + failed, total, letter_id, exc)

        # Checkpoint every N letters
        if (i + 1) % checkpoint_frequency == 0:
            save_checkpoint(checkpoint, checkpoint_path)

        # Rate limit between API calls
        if i < len(assignments) - 1:
            await asyncio.sleep(rate_limit)

    save_checkpoint(checkpoint, checkpoint_path)
    logger.info("Multi-image OCR complete: %d/%d letters, %d failures", done, total, failed)
    return {
        "total": total, "done": done, "failed": failed,
        "errors": [f for f in checkpoint.get("failed", []) if isinstance(f, dict)][-3:],
    }


def run_multi_image_ocr(project_dir: Path, config: BriefpostConfig) -> BatchResult:
    """Entry point for multi-image letter-level OCR (v4 Pass 2).

    Reads the classification manifest from Pass 1, groups pages by letter,
    and sends all pages of each letter to Gemini Pro in a single API call.

    Falls back to per-page OCR (run_ocr) if no classification manifest exists.

    Args:
        project_dir: Project data directory.
        config: Pipeline configuration.

    Returns:
        BatchResult describing the outcome.
    """
    result = BatchResult(phase="ocr")
    from core.persons import configure_persons
    configure_persons(config)

    grouped_dir = project_dir / "grouped"
    ocr_pages_dir = project_dir / "ocr-pages"
    ocr_pages_dir.mkdir(exist_ok=True)
    ocr_letters_dir = project_dir / "ocr-letters"
    ocr_letters_dir.mkdir(exist_ok=True)

    # Load classification manifest from Pass 1
    manifest = load_classification_manifest(project_dir)
    if manifest is None:
        logger.warning("No classification manifest found, falling back to per-page OCR")
        return run_ocr(project_dir, config)

    # B3: Pre-extraction gate: validate page assignments
    assignment_errors = validate_letter_assignments(manifest, grouped_dir)
    if assignment_errors:
        for err in assignment_errors:
            logger.error("B3: Letter assignment validation failed: %s", err)
        raise ValueError(
            f"B3: Classification manifest has {len(assignment_errors)} assignment error(s): "
            + "; ".join(assignment_errors[:3])
        )

    # Build letter assignments from manifest
    assignments = build_letter_assignments(manifest, grouped_dir)
    logger.info(
        "Multi-image OCR: %d letters from %d groups",
        len(assignments),
        manifest.get("total_groups", 0),
    )

    if not assignments:
        logger.error("No valid letter assignments from classification manifest")
        result.total = 0
        result.complete()
        return result

    try:
        batch_result = asyncio.run(
            process_letters_multi_image(assignments, ocr_letters_dir, ocr_pages_dir, config)
        )
        result.total = batch_result["total"]
        result.succeeded = batch_result["done"]
        result.failed = batch_result["failed"]
        if result.failed:
            # Surface the reason (wrong key, retired model, quota) in the run summary
            for failure in batch_result.get("errors", []):
                result.errors.append({
                    "phase": "ocr",
                    "error": f"{failure.get('letter_id', '?')}: {str(failure.get('error'))[:300]}",
                })
    except Exception as exc:
        logger.error("Multi-image OCR failed: %s", exc, exc_info=True)
        result.total = len(assignments)
        result.failed = len(assignments)
        result.errors.append({"phase": "ocr", "error": str(exc)})

    result.complete()
    return result
