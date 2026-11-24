"""Pass 1: Multi-image letter boundary classification.

Sends ALL page images in a document group as a single multi-image request
to Gemini Flash. The LLM identifies letter boundaries visually by analyzing
address blocks, letterheads, page numbering, and sender changes across pages.

Falls back to algorithmic boundary scoring if the API call fails or
classification confidence < 0.60.

For oversized groups (>15 pages), batches 12 pages at a time with 2-page
overlap to maintain context across batch boundaries.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core.config import BriefpostConfig
from core.constants import Flag
from core.io import atomic_write_json
from core.models import BatchResult

logger = logging.getLogger(__name__)


# Default batch size for oversized groups (>15 pages)
DEFAULT_BATCH_SIZE = 12
DEFAULT_OVERLAP = 2
MAX_PAGES_SINGLE_CALL = 15
MIN_CLASSIFICATION_CONFIDENCE = 0.60

# 25 document types from extraction_schemas.py
DOCUMENT_TYPES = (
    "rechnung, mahnung_1, mahnung_2, mahnung_3, zahlungserinnerung, "
    "kontoauszug, gehaltsabrechnung, inkasso, mahnbescheid, "
    "vollstreckungsbescheid, pfaendung, raeumungsklage, "
    "gerichtsvollzieher, insolvenzverfahren, kostenfestsetzungsantrag, "
    "schuldtitel, verjaehrungsbescheid, restschuldbefreiung, "
    "vertrag, kuendigung, bescheid, steuerbescheid, bussgeldbescheid, "
    "allgemeine_korrespondenz, unknown"
)

CLASSIFICATION_PROMPT = """Du siehst {page_count} fotografierte Seiten aus einem Stapel Briefpost (Deutsch, Englisch oder eine andere Sprache).
Diese Seiten wurden innerhalb weniger Minuten nacheinander fotografiert.

AUFGABE: Bestimme, welche Seiten zu welchem Brief gehoeren.

SIGNALE fuer GLEICHEN Brief:
- Gleicher Briefkopf/Logo
- Fortlaufende Seitennummern ("Seite 2 von 3")
- Gleicher Absender
- Text setzt sich fort (Satz beginnt auf einer Seite, endet auf der naechsten)
- Gleiches Aktenzeichen/Referenznummer

SIGNALE fuer NEUEN Brief:
- Neuer Briefkopf/Logo
- Neues Adressfeld
- Neuer Absender
- "Seite 1" nach "Seite X von Y"
- Komplett anderer Inhalt/Thema

DOKUMENTTYPEN (waehle einen pro Brief):
{document_types}

Fuer JEDEN identifizierten Brief gib an:
- pages: Welche Seitennummern gehoeren dazu (1-indexed)
- absender: Absender (wenn erkennbar)
- document_type: Dokumenttyp (einer der obigen Typen)
- person: Person/Empfaenger, genau einer dieser Werte: {person_values}
- confidence: Konfidenz der Gruppierung (0.0-1.0)

Antworte im folgenden JSON-Format:
{{
  "letters": [
    {{
      "pages": [1, 2],
      "absender": "Muster Inkasso GmbH",
      "document_type": "inkasso",
      "person": null,
      "confidence": 0.95
    }},
    {{
      "pages": [3],
      "absender": "Energy Supplier Ltd",
      "document_type": "rechnung",
      "person": null,
      "confidence": 0.90
    }}
  ]
}}

Die Seitennummern beziehen sich auf die Reihenfolge der Bilder (1 = erstes Bild, 2 = zweites Bild, usw.).
Jede Seite muss genau einem Brief zugeordnet werden. Keine Seite darf fehlen."""


def _get_genai_client() -> Any:
    """Get google.genai Client (same pattern as the bundled Gemini provider)."""
    _ensure_api_key()
    try:
        from google import genai
        return genai.Client()
    except ImportError:
        raise ImportError("google-genai package required for multi-image classification")


def _ensure_api_key() -> None:
    """Load GOOGLE_API_KEY from the environment or an .env file."""
    from core.extern import load_api_key
    load_api_key()


def load_doc_group_images(grouped_dir: Path, doc_id: str) -> list[Path]:
    """Load all page images for a document group in order.

    Args:
        grouped_dir: Path to grouped/ directory.
        doc_id: Document group ID (e.g., "doc-001").

    Returns:
        Sorted list of page image paths.
    """
    doc_dir = grouped_dir / doc_id
    if not doc_dir.is_dir():
        return []
    page_tuples = []
    for f in doc_dir.glob("page-*.jpg"):
        m = re.search(r"page-(\d+)", f.name)
        if m is None:
            logger.warning("Skipping unexpected filename: %s", f.name)
            continue
        page_tuples.append((int(m.group(1)), f))
    page_tuples.sort(key=lambda x: x[0])
    return [t[1] for t in page_tuples]


def parse_classification_response(response_text: str, page_count: int) -> dict[str, Any]:
    """Parse the LLM classification response into structured assignments.

    Args:
        response_text: Raw LLM response text.
        page_count: Expected total number of pages.

    Returns:
        Dict with "letters" (list of page assignments), "confidence", and "valid" flag.
    """
    # Extract JSON from response (may be wrapped in markdown code blocks)
    json_match = re.search(r"\{[\s\S]*\}", response_text)
    if not json_match:
        return {"letters": [], "confidence": 0.0, "valid": False, "error": "No JSON found"}

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as exc:
        return {"letters": [], "confidence": 0.0, "valid": False, "error": f"JSON parse error: {exc}"}

    letters = data.get("letters", [])

    # Compute overall confidence as average of per-letter confidences
    letter_confidences = [lt.get("confidence", 0.0) for lt in letters if isinstance(lt, dict)]
    confidence = sum(letter_confidences) / len(letter_confidences) if letter_confidences else 0.0

    # Validate: every page 1..page_count must be assigned exactly once
    assigned_pages: set[int] = set()
    for letter in letters:
        pages = letter.get("pages", [])
        for page_num in pages:
            if not isinstance(page_num, int) or page_num < 1 or page_num > page_count:
                return {
                    "letters": letters,
                    "confidence": confidence,
                    "valid": False,
                    "error": f"Invalid page number: {page_num}",
                }
            if page_num in assigned_pages:
                return {
                    "letters": letters,
                    "confidence": confidence,
                    "valid": False,
                    "error": f"Duplicate page assignment: {page_num}",
                }
            assigned_pages.add(page_num)

    expected_pages = set(range(1, page_count + 1))
    missing = expected_pages - assigned_pages
    if missing:
        return {
            "letters": letters,
            "confidence": confidence,
            "valid": False,
            "error": f"Missing pages: {sorted(missing)}",
        }

    # Validate page ordering within each letter (must be ascending)
    for letter in letters:
        pages = letter.get("pages", [])
        if pages != sorted(pages):
            return {
                "letters": letters,
                "confidence": confidence,
                "valid": False,
                "error": f"Non-ascending pages in letter: {pages}",
            }

    # B4: Compute per-page confidences from letter assignments
    page_confidences: dict[int, float] = {}
    for letter in letters:
        letter_conf = letter.get("confidence", 0.0)
        for page_num in letter.get("pages", []):
            page_confidences[page_num] = letter_conf

    return {
        "letters": letters,
        "confidence": confidence,
        "valid": True,
        "page_confidences": page_confidences,
    }


async def classify_single_batch(
    image_paths: list[Path],
    config: BriefpostConfig,
    page_offset: int = 0,
) -> dict[str, Any]:
    """Send a batch of images to Gemini Flash for letter boundary classification.

    Args:
        image_paths: List of page image paths (in order).
        config: Pipeline configuration.
        page_offset: Offset to add to page numbers (for batched groups).

    Returns:
        Classification result dict with letters, confidence, and validity.
    """
    client = _get_genai_client()
    from google.genai import types

    model_name = config.ocr.get("model", "gemini-3.6-flash")

    # Build multi-image parts
    parts = []
    for image_path in image_paths:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        parts.append(types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        ))

    from core import persons as _persons
    _persons.configure_persons(config)
    prompt = CLASSIFICATION_PROMPT.format(
        page_count=len(image_paths),
        document_types=DOCUMENT_TYPES,
        person_values=_persons.person_prompt_hint(),
    )
    parts.append(prompt)

    try:
        response = await asyncio.to_thread(
            lambda: client.models.generate_content(
                model=model_name,
                contents=parts,
                config={
                    "temperature": 0.0,
                    "max_output_tokens": 8192,
                    "response_mime_type": "application/json",
                },
            )
        )
        response_text = response.text if hasattr(response, "text") else ""
    except Exception as exc:
        logger.error("Gemini classification API call failed: %s", exc)
        return {"letters": [], "confidence": 0.0, "valid": False, "error": str(exc)}

    result = parse_classification_response(response_text, len(image_paths))

    # Apply page offset for batched groups
    if page_offset > 0 and result.get("valid"):
        for letter in result.get("letters", []):
            letter["pages"] = [p + page_offset for p in letter["pages"]]

    return result


def create_batches(
    image_paths: list[Path],
    batch_size: int = DEFAULT_BATCH_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[tuple[list[Path], int]]:
    """Split oversized image lists into overlapping batches.

    Args:
        image_paths: All page image paths in order.
        batch_size: Maximum images per batch.
        overlap: Number of overlapping pages between batches.

    Returns:
        List of (batch_images, page_offset) tuples.
    """
    if len(image_paths) <= batch_size:
        return [(image_paths, 0)]

    batches: list[tuple[list[Path], int]] = []
    stride = batch_size - overlap
    start = 0

    while start < len(image_paths):
        end = min(start + batch_size, len(image_paths))
        batch = image_paths[start:end]
        batches.append((batch, start))
        if end >= len(image_paths):
            break
        start += stride

    return batches


def merge_batch_results(
    batch_results: list[dict[str, Any]],
    total_pages: int,
) -> dict[str, Any]:
    """Merge overlapping batch classification results into a single result.

    For overlap pages, the batch with higher per-letter confidence wins.

    Args:
        batch_results: List of classification results from each batch.
        total_pages: Total number of pages across all batches.

    Returns:
        Merged classification result.
    """
    if len(batch_results) == 1:
        return batch_results[0]

    # Build page-to-letter assignment from each batch
    page_assignments: dict[int, dict[str, Any]] = {}
    page_confidence: dict[int, float] = {}

    for batch_result in batch_results:
        if not batch_result.get("valid"):
            continue
        for letter in batch_result.get("letters", []):
            letter_conf = letter.get("confidence", batch_result.get("confidence", 0.0))
            absender = letter.get("absender", "")
            doc_type = letter.get("document_type", "unknown")
            person = letter.get("person")
            for page_num in letter.get("pages", []):
                existing_conf = page_confidence.get(page_num, -1)
                if letter_conf > existing_conf:
                    page_assignments[page_num] = {
                        "page": page_num,
                        "absender": absender,
                        "document_type": doc_type,
                        "person": person,
                        "confidence": letter_conf,
                    }
                    page_confidence[page_num] = letter_conf

    if not page_assignments:
        return {"letters": [], "confidence": 0.0, "valid": False, "error": "All batches failed"}

    # Reconstruct letter groups from sequential same-absender pages
    sorted_pages = sorted(page_assignments.keys())
    letters: list[dict[str, Any]] = []
    current_letter: dict[str, Any] = {
        "pages": [sorted_pages[0]],
        "absender": page_assignments[sorted_pages[0]]["absender"],
        "document_type": page_assignments[sorted_pages[0]]["document_type"],
        "person": page_assignments[sorted_pages[0]]["person"],
        "confidence": page_assignments[sorted_pages[0]]["confidence"],
    }

    for page_num in sorted_pages[1:]:
        assignment = page_assignments[page_num]
        if (assignment["absender"] == current_letter["absender"]
                and page_num == current_letter["pages"][-1] + 1):
            current_letter["pages"].append(page_num)
            # Use minimum confidence across pages in the letter
            current_letter["confidence"] = min(
                current_letter["confidence"], assignment["confidence"],
            )
        else:
            letters.append(current_letter)
            current_letter = {
                "pages": [page_num],
                "absender": assignment["absender"],
                "document_type": assignment["document_type"],
                "person": assignment["person"],
                "confidence": assignment["confidence"],
            }
    letters.append(current_letter)

    # Check completeness
    assigned = set()
    for letter in letters:
        assigned.update(letter["pages"])
    expected = set(range(1, total_pages + 1))
    missing = expected - assigned

    avg_confidence = sum(page_confidence.values()) / len(page_confidence) if page_confidence else 0.0

    return {
        "letters": letters,
        "confidence": round(avg_confidence, 4),
        "valid": len(missing) == 0,
        "error": f"Missing pages after merge: {sorted(missing)}" if missing else None,
    }


def algorithmic_boundary_fallback(
    image_paths: list[Path],
    ocr_pages_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Fallback boundary detection using existing OCR signal data.

    Reads pre-extracted signals from ocr-pages/*.json (if available) and
    applies the same boundary scoring logic as page_matcher: creditor change,
    address block, page number reset, etc.

    Args:
        image_paths: List of page image paths in order.
        ocr_pages_dir: Path to ocr-pages/ with per-page JSON files.

    Returns:
        Classification result dict compatible with LLM output format.
    """
    if not ocr_pages_dir or not ocr_pages_dir.is_dir():
        # No OCR data: treat each page as its own letter
        letters = [
            {
                "pages": [i + 1],
                "absender": None,
                "document_type": "unknown",
                "person": None,
                "confidence": 0.30,
            }
            for i in range(len(image_paths))
        ]
        return {"letters": letters, "confidence": 0.30, "valid": True, "method": "fallback_no_data"}

    # Load page signals
    page_signals: list[dict[str, Any]] = []
    for image_path in image_paths:
        page_id = _image_path_to_page_id(image_path)
        json_path = ocr_pages_dir / f"{page_id}.json"
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            page_signals.append(data.get("signals", {}))
        else:
            page_signals.append({})

    # Score boundaries between consecutive pages
    boundaries: list[bool] = [True]  # First page always starts a new letter
    for i in range(1, len(page_signals)):
        prev = page_signals[i - 1]
        curr = page_signals[i]
        score = _compute_boundary_score(prev, curr)
        boundaries.append(score >= 4)  # split_threshold from constants

    # Build letter groups from boundaries
    letters: list[dict[str, Any]] = []
    current_pages: list[int] = [1]
    current_sender = page_signals[0].get("sender_name", "") if page_signals else ""

    for i in range(1, len(boundaries)):
        if boundaries[i]:
            letters.append({
                "pages": list(current_pages),
                "absender": current_sender or None,
                "document_type": "unknown",
                "person": None,
                "confidence": 0.70,
            })
            current_pages = [i + 1]
            current_sender = page_signals[i].get("sender_name", "") if i < len(page_signals) else ""
        else:
            current_pages.append(i + 1)

    if current_pages:
        letters.append({
            "pages": list(current_pages),
            "absender": current_sender or None,
            "document_type": "unknown",
            "person": None,
            "confidence": 0.70,
        })

    return {
        "letters": letters,
        "confidence": 0.70,
        "valid": True,
        "method": "fallback_algorithmic",
    }


def _compute_boundary_score(prev_signals: dict[str, Any], curr_signals: dict[str, Any]) -> int:
    """Compute boundary score between two consecutive pages.

    Positive score = more likely new letter boundary.
    Uses same signal weights as page_matcher constants.

    Args:
        prev_signals: Signals dict from previous page.
        curr_signals: Signals dict from current page.

    Returns:
        Integer boundary score (>= 4 = split).
    """
    score = 0

    # Address block on current page = strong new letter signal
    if curr_signals.get("has_address_block"):
        score += 4

    # Letterhead on current page
    if curr_signals.get("has_letterhead"):
        score += 3

    # Page number reset (current page says "Seite 1 von X")
    if curr_signals.get("is_first_page") and curr_signals.get("page_current") == 1:
        if prev_signals.get("page_current") and prev_signals.get("page_current") > 1:
            score += 4

    # Sender name change
    prev_sender = prev_signals.get("sender_name", "")
    curr_sender = curr_signals.get("sender_name", "")
    if prev_sender and curr_sender and prev_sender.lower() != curr_sender.lower():
        score += 4

    # Person change
    prev_persons = set(prev_signals.get("persons_detected", []))
    curr_persons = set(curr_signals.get("persons_detected", []))
    if prev_persons and curr_persons and prev_persons != curr_persons:
        score += 3

    # Sequential page numbering = continuity (negative score)
    prev_page = prev_signals.get("page_current")
    curr_page = curr_signals.get("page_current")
    prev_total = prev_signals.get("page_total")
    curr_total = curr_signals.get("page_total")
    if prev_page and curr_page and prev_total and curr_total:
        if curr_page == prev_page + 1 and prev_total == curr_total:
            score -= 5

    # Shared identifiers = continuity
    prev_refs = set(prev_signals.get("references_found", []))
    curr_refs = set(curr_signals.get("references_found", []))
    if prev_refs and curr_refs and prev_refs & curr_refs:
        score -= 3

    return score


def _load_classification_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load classification checkpoint.

    Args:
        checkpoint_path: Path to .checkpoint.json file.

    Returns:
        Checkpoint data with completed doc_ids and last_updated.
    """
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "last_updated": None}


def _save_classification_checkpoint(
    checkpoint: dict[str, Any], checkpoint_path: Path
) -> None:
    """Save classification checkpoint atomically.

    Args:
        checkpoint: Checkpoint data to save.
        checkpoint_path: Path to write checkpoint.
    """
    checkpoint["last_updated"] = datetime.now().isoformat()
    atomic_write_json(checkpoint_path, checkpoint)


def _image_path_to_page_id(image_path: Path) -> str:
    """Convert image path to page_id format.

    e.g., grouped/doc-001/page-1.jpg -> doc-001-page-01

    Args:
        image_path: Path to page image.

    Returns:
        Page ID string.
    """
    page_match = re.search(r"page-(\d+)", image_path.name)
    doc_match = re.search(r"(doc-\d+)", str(image_path))
    if page_match and doc_match:
        return f"{doc_match.group(1)}-page-{int(page_match.group(1)):02d}"
    return image_path.stem


def classify_document_group(
    group_dir: Path,
    config: BriefpostConfig,
    ocr_pages_dir: Optional[Path] = None,
) -> Optional[list[dict]]:
    """Classify pages in a document group into letters.

    Sends all page images as multi-image to Gemini Flash. Falls back to
    algorithmic boundary detection if the API call fails or confidence
    is below MIN_CLASSIFICATION_CONFIDENCE.

    Args:
        group_dir: Path to grouped/doc-XXX/ directory.
        config: Pipeline configuration.
        ocr_pages_dir: Optional path to ocr-pages/ for fallback signals.

    Returns:
        List of letter dicts with page assignments, or None if classification fails.
        Each dict: {"pages": [1,2,3], "absender": str|None, "document_type": str,
                    "person": str|None, "confidence": float}
    """
    doc_id = group_dir.name
    grouped_dir = group_dir.parent

    image_paths = load_doc_group_images(grouped_dir, doc_id)
    if not image_paths:
        return None

    # Single page = single letter, no classification needed
    if len(image_paths) == 1:
        return [{
            "pages": [1],
            "absender": None,
            "document_type": "unknown",
            "person": None,
            "confidence": 1.0,
        }]

    try:
        if len(image_paths) <= MAX_PAGES_SINGLE_CALL:
            result = asyncio.run(classify_single_batch(image_paths, config))
        else:
            # Oversized group: batch with overlap
            logger.info(
                "Oversized group %s (%d pages), batching with overlap",
                doc_id, len(image_paths),
            )
            batches = create_batches(image_paths)
            batch_results = []
            for batch_images, offset in batches:
                batch_result = asyncio.run(
                    classify_single_batch(batch_images, config, page_offset=offset)
                )
                batch_results.append(batch_result)
            result = merge_batch_results(batch_results, len(image_paths))

        # Check if LLM result is usable
        if result.get("valid") and result.get("confidence", 0) >= MIN_CLASSIFICATION_CONFIDENCE:
            return result["letters"]
        else:
            logger.info(
                "LLM classification for %s unusable (valid=%s, confidence=%.2f), using fallback",
                doc_id, result.get("valid"), result.get("confidence", 0),
            )

    except Exception as exc:
        logger.warning("LLM classification failed for %s: %s, using fallback", doc_id, exc)

    # Fallback to algorithmic boundary detection
    fallback_result = algorithmic_boundary_fallback(image_paths, ocr_pages_dir)
    if fallback_result.get("valid"):
        return fallback_result["letters"]

    return None


def _classify_for_pipeline(
    doc_id: str,
    grouped_dir: Path,
    config: BriefpostConfig,
    ocr_pages_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Internal wrapper for pipeline integration.

    Returns a full result dict with doc_id, method, etc. for the manifest.

    Args:
        doc_id: Document group ID (e.g., "doc-001").
        grouped_dir: Path to grouped/ directory.
        config: Pipeline configuration.
        ocr_pages_dir: Optional path to ocr-pages/ for fallback signals.

    Returns:
        Classification result dict with letters, confidence, method, and doc_id.
    """
    group_dir = grouped_dir / doc_id
    image_paths = load_doc_group_images(grouped_dir, doc_id)

    if not image_paths:
        return {
            "doc_id": doc_id,
            "letters": [],
            "confidence": 0.0,
            "valid": False,
            "method": "no_images",
            "error": f"No page images found for {doc_id}",
        }

    # Single page = single letter, no classification needed
    if len(image_paths) == 1:
        return {
            "doc_id": doc_id,
            "letters": [{
                "pages": [1],
                "absender": None,
                "document_type": "unknown",
                "person": None,
                "confidence": 1.0,
            }],
            "confidence": 1.0,
            "valid": True,
            "method": "single_page",
        }

    letters = classify_document_group(group_dir, config, ocr_pages_dir)
    if letters is not None:
        # Determine method from letter content
        avg_conf = sum(lt.get("confidence", 0) for lt in letters) / len(letters) if letters else 0
        method = "llm_multi_image" if avg_conf >= MIN_CLASSIFICATION_CONFIDENCE else "fallback_algorithmic"
        return {
            "doc_id": doc_id,
            "letters": letters,
            "confidence": round(avg_conf, 4),
            "valid": True,
            "method": method,
            "page_count": len(image_paths),
        }
    else:
        return {
            "doc_id": doc_id,
            "letters": [],
            "confidence": 0.0,
            "valid": False,
            "method": "failed",
            "error": "Classification returned None",
        }


def run_classification(project_dir: Path, config: BriefpostConfig) -> BatchResult:
    """Entry point called by pipeline.py.

    Runs multi-image classification for all document groups.

    Args:
        project_dir: Project data directory.
        config: Pipeline configuration.

    Returns:
        BatchResult describing the outcome.
    """
    result = BatchResult(phase="classify")
    grouped_dir = project_dir / "grouped"
    classification_dir = project_dir / "classification"
    classification_dir.mkdir(exist_ok=True)

    ocr_pages_dir = project_dir / "ocr-pages"
    if not ocr_pages_dir.is_dir():
        ocr_pages_dir = None

    # Discover all document groups
    if not grouped_dir.is_dir():
        result.total = 0
        result.complete()
        return result

    doc_dirs = sorted(
        d for d in grouped_dir.iterdir()
        if d.is_dir() and d.name.startswith("doc-")
    )

    if not doc_dirs:
        result.total = 0
        result.complete()
        return result

    result.total = len(doc_dirs)
    all_classifications: list[dict[str, Any]] = []

    # RS-003: Load checkpoint to resume interrupted classification runs
    checkpoint_path = classification_dir / ".checkpoint.json"
    checkpoint = _load_classification_checkpoint(checkpoint_path)
    completed_set = set(checkpoint["completed"])
    checkpoint_frequency = config.pipeline.get("checkpoint_frequency", 5)

    for i, doc_dir in enumerate(doc_dirs):
        doc_id = doc_dir.name

        # Skip already-completed groups from checkpoint
        if doc_id in completed_set:
            # Reload cached classification from existing manifest if available
            result.succeeded += 1
            logger.info("[%d/%d] %s: SKIPPED (checkpoint)", result.succeeded + result.failed, result.total, doc_id)
            continue

        try:
            classification = _classify_for_pipeline(
                doc_id, grouped_dir, config, ocr_pages_dir,
            )
            all_classifications.append(classification)

            if classification.get("valid"):
                result.succeeded += 1
                checkpoint["completed"].append(doc_id)
                logger.info(
                    "[%d/%d] %s: %d letters (method=%s, confidence=%.2f)",
                    result.succeeded + result.failed, result.total,
                    doc_id, len(classification.get("letters", [])),
                    classification.get("method", "?"),
                    classification.get("confidence", 0),
                )
            else:
                result.failed += 1
                checkpoint["failed"].append({"doc_id": doc_id, "error": classification.get("error", "unknown")})
                logger.warning(
                    "[%d/%d] %s: FAILED - %s",
                    result.succeeded + result.failed, result.total,
                    doc_id, classification.get("error", "unknown"),
                )

        except Exception as exc:
            result.failed += 1
            result.errors.append({"doc_id": doc_id, "error": str(exc)})
            checkpoint["failed"].append({"doc_id": doc_id, "error": str(exc)})
            logger.error("[%s] Classification failed: %s", doc_id, exc)

        # Checkpoint every N groups
        if (i + 1) % checkpoint_frequency == 0:
            _save_classification_checkpoint(checkpoint, checkpoint_path)

    # Final checkpoint
    _save_classification_checkpoint(checkpoint, checkpoint_path)

    # Write classification manifest
    manifest_path = classification_dir / "classification-manifest.json"
    atomic_write_json(manifest_path, {
        "total_groups": len(doc_dirs),
        "classifications": all_classifications,
    })
    logger.info("Written: %s", manifest_path)

    result.complete()
    return result
