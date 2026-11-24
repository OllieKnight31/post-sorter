"""Phase B3: Forward-Scan Boundary Detection for page-to-letter matching.

Groups per-page OCR results into actual letters using signal-weighted
boundary scoring. Each transition between consecutive pages gets a score;
score >= split_threshold triggers a new letter boundary.

Split signals (positive): creditor change, address block, page reset,
IBAN change, identifier change, person change, document type change.
Continuity signals (negative): sequential pages, shared identifiers,
same creditor, same IBAN.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core.config import BriefpostConfig
from core.constants import (
    get_escalation,
    CONTINUITY_SAME_CREDITOR,
    CONTINUITY_SAME_IBAN,
    CONTINUITY_SAME_IDENTIFIER,
    CONTINUITY_SEQUENTIAL_PAGES,
    SPLIT_SIGNAL_ADDRESS_BLOCK,
    SPLIT_SIGNAL_CREDITOR_CHANGE,
    SPLIT_SIGNAL_DOCTYPE_CHANGE,
    SPLIT_SIGNAL_IBAN_CHANGE,
    SPLIT_SIGNAL_IDENTIFIER_CHANGE,
    SPLIT_SIGNAL_PAGE_RESET,
    SPLIT_SIGNAL_PERSON_CHANGE,
    Flag,
)
from core.interfaces import PageMatcher
from core.io import atomic_write_json
from core.models import BatchResult, Brief, DocumentGroup, Page
from core.utils import normalize_creditor

logger = logging.getLogger(__name__)

# CQ-028: Named constants for scores not yet in core/constants.py
SPLIT_SIGNAL_LETTERHEAD = 1
CONTINUITY_SAME_AMOUNTS = -2


def _parse_date_for_sort(date_str: str) -> datetime:
    """Parse date string for chronological sorting.

    Supports YYYY-MM-DD and DD.MM.YYYY formats. Falls back to epoch
    for unparseable strings so they sort first (lowest).

    Args:
        date_str: Date string.

    Returns:
        Parsed datetime for sorting.
    """
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return datetime(1970, 1, 1)


def normalize_identifier(identifier: str) -> str:
    """Normalize identifier for comparison.

    Args:
        identifier: Raw identifier string.

    Returns:
        Normalized identifier with whitespace and punctuation removed.
    """
    if not identifier:
        return ""
    return re.sub(r"[\s\-/.]", "", str(identifier).lower().strip())


def get_creditor_name(page_data: dict[str, Any]) -> str:
    """Extract creditor name from page data."""
    gemini = page_data.get("gemini_extraction", {})
    creditor = gemini.get("creditor", {})
    if isinstance(creditor, dict):
        return creditor.get("name", "")
    return ""


def get_original_creditor(page_data: dict[str, Any]) -> str:
    """Extract original creditor (for inkasso cases)."""
    gemini = page_data.get("gemini_extraction", {})
    creditor = gemini.get("creditor", {})
    if isinstance(creditor, dict):
        return creditor.get("original_creditor", "")
    return ""


def get_creditor_confidence(page_data: dict[str, Any]) -> float:
    """Get creditor field confidence."""
    gemini = page_data.get("gemini_extraction", {})
    field_conf = gemini.get("field_confidence", {})
    return field_conf.get("creditor", 0.0)


def get_iban(page_data: dict[str, Any]) -> str:
    """Extract IBAN from page data."""
    gemini = page_data.get("gemini_extraction", {})
    payment_info = gemini.get("payment_info", {})
    if isinstance(payment_info, dict):
        return payment_info.get("iban", "")
    return ""


def get_identifiers(page_data: dict[str, Any]) -> list[str]:
    """Extract normalized identifiers from page data."""
    gemini = page_data.get("gemini_extraction", {})
    ids = gemini.get("identifiers", [])
    if isinstance(ids, list):
        return [normalize_identifier(i) for i in ids if i]
    return []


def get_page_info(page_data: dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    """Extract page numbering info (current, total)."""
    gemini = page_data.get("gemini_extraction", {})
    page_info = gemini.get("page_info", {})
    signals = page_data.get("signals", {})

    current = None
    total = None

    if isinstance(page_info, dict):
        current = page_info.get("current_page")
        total = page_info.get("total_pages")

    if current is None:
        current = signals.get("page_current")
    if total is None:
        total = signals.get("page_total")

    return current, total


def get_person(page_data: dict[str, Any]) -> Optional[str]:
    """Determine primary person from page data."""
    signals = page_data.get("signals", {})
    persons = signals.get("persons_detected", [])
    if persons:
        return persons[0]
    return None


def get_all_persons(page_data: dict[str, Any]) -> list[str]:
    """Get all detected persons from page data.

    CQ-044: Uses all persons detected on a page, not just the first.

    Args:
        page_data: Page JSON data dict.

    Returns:
        List of person names detected on this page.
    """
    signals = page_data.get("signals", {})
    return signals.get("persons_detected", [])


def compute_boundary_score(
    prev_page: dict[str, Any],
    curr_page: dict[str, Any],
    letter_context: dict[str, Any],
    config: dict[str, Any],
) -> tuple[int, list[str]]:
    """Compute boundary score between two consecutive pages.

    Positive score = evidence for SPLIT (new letter).
    Negative score = evidence for CONTINUATION (same letter).

    Args:
        prev_page: Previous page data.
        curr_page: Current page data.
        letter_context: Context about the current letter being built.
        config: Matching configuration.

    Returns:
        Tuple of (score, list of signal descriptions).
    """
    score = 0
    evidence: list[str] = []
    creditor_confidence_threshold = config.get("creditor_confidence", 0.7)

    # === SPLIT SIGNALS (positive) ===

    # Creditor change
    prev_cred = normalize_creditor(get_creditor_name(prev_page))
    curr_cred = normalize_creditor(get_creditor_name(curr_page))
    prev_conf = get_creditor_confidence(prev_page)
    curr_conf = get_creditor_confidence(curr_page)

    if prev_cred and curr_cred and prev_cred != curr_cred:
        if prev_conf >= creditor_confidence_threshold and curr_conf >= creditor_confidence_threshold:
            curr_orig = normalize_creditor(get_original_creditor(curr_page))
            all_creditors = letter_context.get("all_creditors", set())
            if curr_cred in all_creditors or (curr_orig and curr_orig in all_creditors):
                score -= 2
                evidence.append("inkasso_crossref:-2")
            else:
                score += SPLIT_SIGNAL_CREDITOR_CHANGE
                evidence.append(f"creditor_change:+{SPLIT_SIGNAL_CREDITOR_CHANGE}")

    # Address block appears
    curr_signals = curr_page.get("signals", {})
    prev_signals = prev_page.get("signals", {})
    if curr_signals.get("has_address_block"):
        prev_page_cur, prev_page_tot = get_page_info(prev_page)
        if not prev_signals.get("has_address_block") or (
            prev_page_cur and prev_page_tot and prev_page_cur == prev_page_tot
        ):
            score += SPLIT_SIGNAL_ADDRESS_BLOCK
            evidence.append(f"address_block:+{SPLIT_SIGNAL_ADDRESS_BLOCK}")

    # Page number reset to 1
    curr_page_cur, curr_page_tot = get_page_info(curr_page)
    prev_page_cur, prev_page_tot = get_page_info(prev_page)
    if curr_page_cur == 1 and prev_page_cur and prev_page_cur > 1:
        score += SPLIT_SIGNAL_PAGE_RESET
        evidence.append(f"page_reset:+{SPLIT_SIGNAL_PAGE_RESET}")
    elif curr_page_cur == 1 and prev_page_cur and prev_page_cur == prev_page_tot:
        score += SPLIT_SIGNAL_PAGE_RESET
        evidence.append(f"page_reset:+{SPLIT_SIGNAL_PAGE_RESET}")

    # IBAN change
    prev_iban = get_iban(prev_page)
    curr_iban = get_iban(curr_page)
    if prev_iban and curr_iban and prev_iban != curr_iban:
        score += SPLIT_SIGNAL_IBAN_CHANGE
        evidence.append(f"iban_change:+{SPLIT_SIGNAL_IBAN_CHANGE}")

    # Identifier family change
    prev_ids = set(get_identifiers(prev_page))
    curr_ids = set(get_identifiers(curr_page))
    if prev_ids and curr_ids and not prev_ids.intersection(curr_ids):
        score += SPLIT_SIGNAL_IDENTIFIER_CHANGE
        evidence.append(f"identifier_change:+{SPLIT_SIGNAL_IDENTIFIER_CHANGE}")

    # Person change
    prev_person = get_person(prev_page)
    curr_person = get_person(curr_page)
    if prev_person and curr_person and prev_person != curr_person:
        score += SPLIT_SIGNAL_PERSON_CHANGE
        evidence.append(f"person_change:+{SPLIT_SIGNAL_PERSON_CHANGE}")

    # Document type change
    prev_type = prev_page.get("gemini_extraction", {}).get("document_type", "")
    curr_type = curr_page.get("gemini_extraction", {}).get("document_type", "")
    if prev_type and curr_type and prev_type != curr_type:
        score += SPLIT_SIGNAL_DOCTYPE_CHANGE
        evidence.append(f"type_change:+{SPLIT_SIGNAL_DOCTYPE_CHANGE}")

    # Letterhead appears
    if curr_signals.get("has_letterhead") and not prev_signals.get("has_letterhead"):
        score += SPLIT_SIGNAL_LETTERHEAD
        evidence.append(f"letterhead:+{SPLIT_SIGNAL_LETTERHEAD}")

    # === CONTINUITY SIGNALS (negative) ===

    # Sequential page numbers
    if curr_page_cur and prev_page_cur and curr_page_tot and prev_page_tot:
        if curr_page_cur == prev_page_cur + 1 and curr_page_tot == prev_page_tot:
            score += CONTINUITY_SEQUENTIAL_PAGES
            evidence.append(f"sequential_pages:{CONTINUITY_SEQUENTIAL_PAGES}")

    # Shared identifiers
    shared_ids = prev_ids.intersection(curr_ids)
    if shared_ids:
        score += CONTINUITY_SAME_IDENTIFIER
        evidence.append(f"shared_identifiers:{CONTINUITY_SAME_IDENTIFIER}")

    # Same creditor
    if prev_cred and curr_cred and prev_cred == curr_cred:
        score += CONTINUITY_SAME_CREDITOR
        evidence.append(f"same_creditor:{CONTINUITY_SAME_CREDITOR}")

    # Same IBAN
    if prev_iban and curr_iban and prev_iban == curr_iban:
        score += CONTINUITY_SAME_IBAN
        evidence.append(f"same_iban:{CONTINUITY_SAME_IBAN}")

    # Same amounts
    prev_amounts = prev_page.get("gemini_extraction", {}).get("amounts", {})
    curr_amounts = curr_page.get("gemini_extraction", {}).get("amounts", {})
    if isinstance(prev_amounts, dict) and isinstance(curr_amounts, dict):
        prev_current = prev_amounts.get("current")
        curr_current = curr_amounts.get("current")
        if prev_current and curr_current and prev_current == curr_current:
            score += CONTINUITY_SAME_AMOUNTS
            evidence.append(f"same_amounts:{CONTINUITY_SAME_AMOUNTS}")

    return score, evidence


def aggregate_letter_fields(pages_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate fields across pages into a single letter record.

    Args:
        pages_data: List of page JSON data dicts.

    Returns:
        Aggregated field dict for letter construction.
    """
    if not pages_data:
        return {}

    # Creditor: page with highest field_confidence.creditor
    best_creditor_page = max(pages_data, key=lambda p: get_creditor_confidence(p))
    creditor_data = best_creditor_page.get("gemini_extraction", {}).get("creditor", {})

    # Person: majority vote (CQ-044: count ALL detected persons, not just first)
    person_votes: dict[str, int] = defaultdict(int)
    for page in pages_data:
        for person_name in get_all_persons(page):
            if person_name:
                person_votes[person_name] += 1
    person = max(person_votes, key=person_votes.get) if person_votes else None

    # Document type: highest escalation
    best_type = None
    best_escalation = -1
    for page in pages_data:
        doc_type = page.get("gemini_extraction", {}).get("document_type", "")
        esc = get_escalation(doc_type)
        if esc > best_escalation:
            best_escalation = esc
            best_type = doc_type

    # Identifiers: union, deduplicated
    all_identifiers: list[str] = []
    seen_ids: set[str] = set()
    for page in pages_data:
        gemini = page.get("gemini_extraction", {})
        for ident in gemini.get("identifiers", []):
            if ident and normalize_identifier(ident) not in seen_ids:
                all_identifiers.append(ident)
                seen_ids.add(normalize_identifier(ident))

    # Amounts: page with highest amounts confidence AND highest value
    best_amounts = None
    best_amounts_score = -1.0
    for page in pages_data:
        amounts = page.get("gemini_extraction", {}).get("amounts", {})
        if isinstance(amounts, dict) and amounts.get("current") is not None:
            conf = page.get("gemini_extraction", {}).get("field_confidence", {}).get("amounts", 0)
            val = amounts.get("current", 0) or 0
            score = conf * 1000 + val
            if score > best_amounts_score:
                best_amounts_score = score
                best_amounts = amounts

    # Dates: latest document_date, earliest upcoming due_date
    doc_dates: list[str] = []
    due_dates: list[str] = []
    for page in pages_data:
        dates = page.get("gemini_extraction", {}).get("dates", {})
        if isinstance(dates, dict):
            dd = dates.get("document_date")
            if dd:
                doc_dates.append(dd)
            dud = dates.get("due_date")
            if dud:
                due_dates.append(dud)

    # Payment info: page with highest payment_info confidence
    best_payment: Optional[dict[str, Any]] = None
    best_payment_conf = -1.0
    for page in pages_data:
        gemini = page.get("gemini_extraction", {})
        payment_info = gemini.get("payment_info", {})
        if isinstance(payment_info, dict) and payment_info.get("iban"):
            conf = gemini.get("field_confidence", {}).get("payment_info", 0)
            if conf > best_payment_conf:
                best_payment_conf = conf
                best_payment = payment_info

    # Urgency signals: union
    urgency: list[str] = []
    for page in pages_data:
        for sig in page.get("gemini_extraction", {}).get("urgency_signals", []):
            if sig and sig not in urgency:
                urgency.append(sig)

    # Confidence: aggregate (DF-032: exclude pages with failed extraction)
    confidences: list[float] = []
    for page in pages_data:
        gemini = page.get("gemini_extraction", {})
        if not gemini.get("extraction_successful", True):
            continue
        conf = gemini.get("confidence", 0)
        if conf:
            confidences.append(conf)

    return {
        "creditor": creditor_data if isinstance(creditor_data, dict) else {"name": str(creditor_data)},
        "person": person,
        "document_type": best_type or "unknown",
        "eskalation": best_escalation if best_escalation >= 0 else 0,
        "identifiers": all_identifiers,
        "primary_reference": all_identifiers[0] if all_identifiers else None,
        "amounts": best_amounts or {"original": None, "current": None, "fees": None},
        "dates": {
            "document_date": sorted(doc_dates, key=_parse_date_for_sort)[-1] if doc_dates else None,
            "due_date": sorted(due_dates, key=_parse_date_for_sort)[0] if due_dates else None,
        },
        "payment_info": best_payment or {},
        "urgency_signals": urgency,
        "confidence": {
            "overall": round(sum(confidences) / len(confidences), 4) if confidences else 0,
            "page_confidences": confidences,
            "min_confidence": min(confidences) if confidences else 0,
        },
    }


def build_letter(
    counter: int,
    doc_id: str,
    pages_data: list[dict[str, Any]],
    split_evidence: list[str],
    split_score: int,
) -> dict[str, Any]:
    """Build a letter JSON from a list of page data.

    Args:
        counter: Letter counter for ID generation.
        doc_id: Document group ID.
        pages_data: List of page JSON data dicts.
        split_evidence: Evidence signals for the split decision.
        split_score: Boundary score that triggered the split.

    Returns:
        Letter dict ready for JSON serialization.
    """
    letter_id = f"LTR-{counter:03d}"
    agg = aggregate_letter_fields(pages_data)

    pages_list: list[dict[str, Any]] = []
    for i, page in enumerate(pages_data):
        role = "first_page" if i == 0 else "continuation"
        pages_list.append({
            "page_id": page["page_id"],
            "page_num": page.get("page_num", i + 1),
            "role": role,
        })

    flags: list[str] = []
    for page in pages_data:
        if not page.get("gemini_extraction", {}).get("extraction_successful", True):
            flags.append(Flag.HAS_UNREADABLE_PAGE.value)
            break
    for page in pages_data:
        page_flags = page.get("flags", [])
        for flag in page_flags:
            if flag not in flags:
                flags.append(flag)

    source_images = [page.get("image_path", "") for page in pages_data]
    raw_ocr_files = [f"ocr-pages/{page['page_id']}.md" for page in pages_data]

    confidence_block = agg["confidence"]
    ocr_confidence = (
        confidence_block.get("overall", 0.0)
        if isinstance(confidence_block, dict)
        else confidence_block if confidence_block else 0.0
    )

    # CQ-027/DF-029: Output includes all Brief model fields plus pipeline metadata
    return {
        # Brief model fields
        "letter_id": letter_id,
        "pages": pages_list,
        "person": agg["person"],
        "creditor": agg["creditor"],
        "document_type": agg["document_type"],
        "eskalation": agg["eskalation"],
        "kategorie": "SONSTIGES",
        "dates": agg["dates"],
        "betreff": "",
        "identifiers": agg["identifiers"],
        "amounts": agg["amounts"],
        "payment_info": agg["payment_info"],
        "urgency_signals": agg["urgency_signals"],
        "flags": flags,
        "notizen": "",
        "ocr_confidence": ocr_confidence,
        "ocr_provider": "gemini",
        "source_images": source_images,
        "raw_ocr_files": raw_ocr_files,
        "match_confidence": 0.0,
        "match_method": "",
        # Pipeline metadata (not in Brief model, used by downstream phases)
        "doc_group": doc_id,
        "primary_reference": agg["primary_reference"],
        "confidence": confidence_block,
        "grouping_confidence": 0.95 if not split_evidence else min(0.99, 0.80 + abs(split_score) * 0.02),
        "split_evidence": {
            "before": {"score": split_score, "signals": split_evidence},
        } if split_evidence else {},
    }


class ForwardScanPageMatcher(PageMatcher):
    """Concrete implementation of Phase B3 page-to-letter matching."""

    def __init__(self, config: BriefpostConfig) -> None:
        """Initialize page matcher.

        Args:
            config: Pipeline configuration.
        """
        self.config = config

    def match_pages_to_letters(
        self,
        pages: list[Page],
        groups: list[DocumentGroup],
        config: dict[str, Any],
    ) -> list[Brief]:
        """Group pages into letters using signal-weighted boundary scoring.

        Args:
            pages: All OCR-processed pages (not used directly, reads from disk).
            groups: Document groups from Phase A (not used directly, reads manifest).
            config: Matching configuration.

        Returns:
            List of Brief records.
        """
        # This is the ABC interface; actual work is in run_page_matching
        raise NotImplementedError("Use run_page_matching() directly")


def run_page_matching(project_dir: Path, config: BriefpostConfig) -> BatchResult:
    """Entry point called by pipeline.py.

    Args:
        project_dir: Project data directory.
        config: Pipeline configuration.

    Returns:
        BatchResult describing the outcome.
    """
    result = BatchResult(phase="match_pages")
    ocr_pages_dir = project_dir / "ocr-pages"
    grouped_dir = project_dir / "grouped"
    letters_dir = project_dir / "letters"
    matching_config = config.matching

    split_threshold = matching_config.get("split_threshold", 4)

    # Load manifest
    manifest_path = grouped_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error("grouped/manifest.json not found")
        result.total = 1
        result.failed = 1
        result.errors.append({"error": "manifest.json not found"})
        result.complete()
        return result

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Load all page JSONs
    all_pages: dict[str, dict[str, Any]] = {}
    for json_path in sorted(ocr_pages_dir.glob("doc-*.json")):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        all_pages[data["page_id"]] = data

    logger.info(
        "Page matching: %d pages, %d groups, threshold=%d",
        len(all_pages), len(manifest.get("groups", [])), split_threshold,
    )

    letters: list[dict[str, Any]] = []
    letter_counter = 0
    total_pages_assigned = 0
    assigned_page_ids: set[str] = set()

    for group in sorted(manifest.get("groups", []), key=lambda g: g["doc_id"]):
        doc_id = group["doc_id"]

        group_pages: list[dict[str, Any]] = []
        for page_file in sorted(
            group["pages"],
            key=lambda f: int(m.group(1)) if (m := re.search(r"page-(\d+)", f)) else 0,
        ):
            m = re.search(r"page-(\d+)", page_file)
            if m is None:
                logger.warning("Skipping unexpected filename in sort: %s", page_file)
                continue
            page_num = int(m.group(1))
            page_id = f"{doc_id}-page-{page_num:02d}"
            if page_id in all_pages:
                group_pages.append(all_pages[page_id])
            else:
                logger.warning("%s not found in ocr-pages/", page_id)

        if not group_pages:
            logger.warning("SKIP: %s - no pages found", doc_id)
            continue

        # Forward-scan boundary detection
        current_letter_pages = [group_pages[0]]
        letter_context: dict[str, Any] = {
            "all_creditors": {normalize_creditor(get_creditor_name(group_pages[0]))},
            "all_original_creditors": {normalize_creditor(get_original_creditor(group_pages[0]))},
        }

        for i in range(1, len(group_pages)):
            curr_page = group_pages[i]

            # Failed page = potential new document boundary
            # Start a new letter rather than merging into current
            gemini = curr_page.get("gemini_extraction", {})
            if not gemini.get("extraction_successful", True):
                if current_letter_pages:
                    letter_counter += 1
                    letter = build_letter(letter_counter, doc_id, current_letter_pages, [], 0)
                    letters.append(letter)
                    total_pages_assigned += len(current_letter_pages)
                    for page in current_letter_pages:
                        assigned_page_ids.add(page["page_id"])
                current_letter_pages = [curr_page]
                continue

            score, evidence = compute_boundary_score(
                group_pages[i - 1], curr_page, letter_context, matching_config,
            )

            if score >= split_threshold:
                # SPLIT: Finalize current letter
                letter_counter += 1
                letter = build_letter(letter_counter, doc_id, current_letter_pages, evidence, score)
                letters.append(letter)
                total_pages_assigned += len(current_letter_pages)
                for page in current_letter_pages:
                    assigned_page_ids.add(page["page_id"])

                current_letter_pages = [curr_page]
                letter_context = {
                    "all_creditors": {normalize_creditor(get_creditor_name(curr_page))},
                    "all_original_creditors": {normalize_creditor(get_original_creditor(curr_page))},
                }
            else:
                current_letter_pages.append(curr_page)
                letter_context["all_creditors"].add(normalize_creditor(get_creditor_name(curr_page)))
                letter_context["all_original_creditors"].add(normalize_creditor(get_original_creditor(curr_page)))

        # Finalize last letter in group
        letter_counter += 1
        letter = build_letter(letter_counter, doc_id, current_letter_pages, [], 0)
        letters.append(letter)
        total_pages_assigned += len(current_letter_pages)
        for page in current_letter_pages:
            assigned_page_ids.add(page["page_id"])

    # CQ-031m: Detect duplicate letter_ids
    letter_id_set: set[str] = set()
    duplicate_letter_ids: list[str] = []
    for letter in letters:
        lid = letter["letter_id"]
        if lid in letter_id_set:
            duplicate_letter_ids.append(lid)
        letter_id_set.add(lid)
    if duplicate_letter_ids:
        logger.error(
            "DUPLICATE LETTER IDS: %d duplicates detected: %s",
            len(duplicate_letter_ids), duplicate_letter_ids[:5],
        )

    # Verification
    all_page_ids = set(all_pages.keys())
    orphans = all_page_ids - assigned_page_ids
    if orphans:
        logger.warning("ORPHAN PAGES: %d", len(orphans))

    page_id_counts: dict[str, int] = defaultdict(int)
    for letter in letters:
        for page in letter["pages"]:
            page_id_counts[page["page_id"]] += 1
    duplicates = {pid: cnt for pid, cnt in page_id_counts.items() if cnt > 1}
    if duplicates:
        logger.warning("DUPLICATE PAGES: %d", len(duplicates))

    # Write output
    letters_dir.mkdir(exist_ok=True)

    for letter in letters:
        letter_path = letters_dir / f"{letter['letter_id']}.json"
        atomic_write_json(letter_path, letter)

    letters_manifest = {
        "created": datetime.now().isoformat(),
        "total_letters": len(letters),
        "total_pages": total_pages_assigned,
        "orphan_pages": len(orphans),
        "duplicate_pages": len(duplicates),
        "cross_person_mixes": 0,
        "split_threshold": split_threshold,
        "letters": [
            {
                "letter_id": letter["letter_id"],
                "doc_group": letter["doc_group"],
                "page_count": len(letter["pages"]),
                "creditor": letter.get("creditor", {}).get("name", ""),
                "person": letter.get("person"),
                "document_type": letter.get("document_type"),
            }
            for letter in letters
        ],
    }
    manifest_out_path = letters_dir / "manifest.json"
    atomic_write_json(manifest_out_path, letters_manifest)

    logger.info(
        "Page matching complete: %d letters, %d pages assigned, %d orphans, %d duplicates",
        len(letters), total_pages_assigned, len(orphans), len(duplicates),
    )

    result.total = len(letters)
    result.succeeded = len(letters)
    result.complete()
    return result
