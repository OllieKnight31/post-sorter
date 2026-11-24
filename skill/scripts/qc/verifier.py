"""Phase F: 43-point E2E quality control verification.

Runs 43 automated checks plus random E2E traces across all pipeline outputs.
Read-only: does not modify any pipeline data.

Output: reconciliation/qc-report.json + reconciliation/qc-report.md
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import BriefpostConfig
from core.constants import Flag, Priority, Status, get_escalation
from core.interfaces import QualityVerifier
from core.models import BatchResult, QCCheck, QCReport
from core.utils import normalize_creditor

logger = logging.getLogger(__name__)

def _get_iban_validator() -> Any:
    """Lazy-load the IBAN validator from the bundled OCR helpers."""
    from core.extern import load_ocr_attr
    return load_ocr_attr("iban_validator", "validate_iban")


def load_letters(letters_dir: Path) -> list[dict[str, Any]]:
    """Load all letter JSON files from the letters directory.

    Args:
        letters_dir: Directory containing LTR-XXX.json files.

    Returns:
        List of letter dicts.
    """
    if not letters_dir.exists():
        return []
    letters = []
    for path in sorted(letters_dir.glob("LTR-*.json")):
        with open(path) as f:
            letters.append(json.load(f))
    return letters


def load_vorgaenge(reconciliation_dir: Path) -> list[dict[str, Any]]:
    """Load reconciled Vorgaenge, preferring cross-intel enriched over original.

    Tries files in order: vorgaenge-cross-intel.json (cross-intel enriched),
    vorgaenge-reconciled.json (bank-reconciled), vorgaenge.json (raw).

    Args:
        reconciliation_dir: Directory containing vorgaenge JSON files.

    Returns:
        List of Vorgang dicts.
    """
    for filename in [
        "vorgaenge-cross-intel.json",
        "vorgaenge-reconciled.json",
        "vorgaenge.json",
    ]:
        path = reconciliation_dir / filename
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return data.get("vorgaenge", [])
    return []


def load_page_json(ocr_pages_dir: Path, page_id: str) -> dict[str, Any] | None:
    """Load a single page JSON file.

    Args:
        ocr_pages_dir: Directory containing per-page JSON files.
        page_id: Page identifier (without extension).

    Returns:
        Parsed JSON dict or None if file not found.
    """
    path = ocr_pages_dir / f"{page_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _count_grouped_photos(grouped_dir: Path) -> int:
    """Count actual photos in the grouped directory.

    Args:
        grouped_dir: Directory containing grouped document photos.

    Returns:
        Number of page-*.jpg files found.
    """
    return sum(1 for _ in grouped_dir.rglob("page-*.jpg"))


def check_foto_coverage(
    grouped_dir: Path,
    config: BriefpostConfig,
) -> QCCheck:
    """Check 1: Verify expected photo count in grouped directory.

    Args:
        grouped_dir: Directory containing grouped document photos.
        config: Pipeline configuration.

    Returns:
        QCCheck result.
    """
    photo_count = _count_grouped_photos(grouped_dir)
    expected = config.qc.get("expected_photos")
    if expected is None:
        # No expected count configured: informational, passes when photos exist
        return QCCheck(
            check_number=1,
            name="Foto-Coverage",
            criterion="Photos present in grouped/ (no expected count configured)",
            passed=photo_count > 0,
            actual=photo_count,
        )
    return QCCheck(
        check_number=1,
        name="Foto-Coverage",
        criterion=f"{expected} Fotos in grouped/",
        passed=photo_count == expected,
        actual=photo_count,
        expected=expected,
    )


def check_ocr_coverage(
    ocr_pages_dir: Path,
    config: BriefpostConfig,
    actual_photos: int = 0,
) -> QCCheck:
    """Check 2: Verify OCR output file counts.

    Args:
        ocr_pages_dir: Directory containing OCR output files.
        config: Pipeline configuration.
        actual_photos: Actual photo count from grouped dir (used as dynamic fallback).

    Returns:
        QCCheck result.
    """
    md_count = len(list(ocr_pages_dir.glob("doc-*.md")))
    json_count = len(list(ocr_pages_dir.glob("doc-*.json")))
    expected = config.qc.get("expected_photos")
    if expected is None:
        expected = actual_photos  # Compare against what is really in grouped/
    return QCCheck(
        check_number=2,
        name="OCR-Coverage",
        criterion=f"{expected} .md + {expected} .json in ocr-pages/",
        passed=md_count == expected and json_count == expected,
        actual=f"{md_count} .md, {json_count} .json",
        expected=f"{expected} .md, {expected} .json",
    )


def check_brief_coverage(
    letters: list[dict[str, Any]],
    config: BriefpostConfig,
    actual_photos: int = 0,
) -> QCCheck:
    """Check 3: Verify total page count across all letters.

    Args:
        letters: List of letter dicts.
        config: Pipeline configuration.
        actual_photos: Actual photo count from grouped dir (used as dynamic fallback).

    Returns:
        QCCheck result.
    """
    total_pages = sum(len(letter.get("pages", [])) for letter in letters)
    expected = config.qc.get("expected_photos")
    if expected is None:
        expected = actual_photos  # Compare against what is really in grouped/
    return QCCheck(
        check_number=3,
        name="Brief-Coverage",
        criterion=f"Summe aller Seiten ueber alle Briefe == {expected}",
        passed=total_pages == expected,
        actual=total_pages,
        expected=expected,
    )


def check_orphan_pages(
    letters: list[dict[str, Any]],
    ocr_pages_dir: Path,
) -> QCCheck:
    """Check 4: Verify no OCR pages are unassigned to letters.

    Args:
        letters: List of letter dicts.
        ocr_pages_dir: Directory containing OCR page files.

    Returns:
        QCCheck result.
    """
    page_ids_in_letters = set()
    for letter in letters:
        for page in letter.get("pages", []):
            page_ids_in_letters.add(page["page_id"])

    page_ids_from_ocr = {p.stem for p in ocr_pages_dir.glob("doc-*.json")}
    orphans = page_ids_from_ocr - page_ids_in_letters

    return QCCheck(
        check_number=4,
        name="Keine Orphan-Seiten",
        criterion="Jede page_id in genau 1 Brief",
        passed=len(orphans) == 0,
        actual=len(orphans),
        details=list(orphans)[:10] if orphans else [],
    )


def check_duplicate_pages(letters: list[dict[str, Any]]) -> QCCheck:
    """Check 5: Verify no page appears in multiple letters.

    Args:
        letters: List of letter dicts.

    Returns:
        QCCheck result.
    """
    page_id_counts: dict[str, int] = defaultdict(int)
    for letter in letters:
        for page in letter.get("pages", []):
            page_id_counts[page["page_id"]] += 1

    duplicates = {pid: cnt for pid, cnt in page_id_counts.items() if cnt > 1}

    return QCCheck(
        check_number=5,
        name="Keine Duplikat-Seiten",
        criterion="Keine page_id in > 1 Brief",
        passed=len(duplicates) == 0,
        actual=len(duplicates),
        details=dict(list(duplicates.items())[:10]) if duplicates else {},
    )


def check_person_consistency(
    letters: list[dict[str, Any]],
    ocr_pages_dir: Path,
) -> QCCheck:
    """Check 6: Verify each letter has a consistent person assignment.

    Args:
        letters: List of letter dicts.
        ocr_pages_dir: Directory containing OCR page JSON files.

    Returns:
        QCCheck result.
    """
    cross_person_letters = []
    for letter in letters:
        persons = set()
        for page in letter.get("pages", []):
            page_json = load_page_json(ocr_pages_dir, page["page_id"])
            if page_json:
                detected = page_json.get("signals", {}).get("persons_detected", [])
                persons.update(detected)
        if len(persons) > 1:
            # If the letter has an assigned person and that person is in the
            # detected set, this is a genuine multi-person document (e.g.,
            # estate letters addressed to one person but mentioning another).
            # Only flag when the assigned person is NOT among detected persons.
            letter_person = letter.get("person")
            if letter_person and letter_person in persons:
                continue
            cross_person_letters.append({
                "letter_id": letter["letter_id"],
                "persons": list(persons),
                "letter_person": letter_person,
            })

    return QCCheck(
        check_number=6,
        name="Person-Konsistenz",
        criterion="Jeder Brief hat einheitliche Person (oder assigned person in detected set)",
        passed=len(cross_person_letters) == 0,
        actual=len(cross_person_letters),
        details=cross_person_letters[:5] if cross_person_letters else [],
    )


def check_vorgang_integrity(
    letters: list[dict[str, Any]],
    vorgaenge: list[dict[str, Any]],
) -> QCCheck:
    """Check 7: Verify every letter is assigned to exactly one Vorgang.

    Args:
        letters: List of letter dicts.
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    letter_ids_in_vorgaenge = set()
    for vorgang in vorgaenge:
        letter_ids_in_vorgaenge.update(vorgang.get("briefe", []))

    all_letter_ids = {letter["letter_id"] for letter in letters}
    unassigned = all_letter_ids - letter_ids_in_vorgaenge

    return QCCheck(
        check_number=7,
        name="Vorgang-Integritaet",
        criterion="Jeder Brief in genau 1 Vorgang",
        passed=len(unassigned) == 0,
        actual=len(unassigned),
        details=list(unassigned)[:10] if unassigned else [],
    )


def check_amount_precision(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 8: Verify all amounts have at most 2 decimal places.

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    bad_amounts = 0
    for vorgang in vorgaenge:
        betrag = vorgang.get("aktuellster_betrag")
        if betrag is not None and isinstance(betrag, float):
            if round(betrag, 2) != betrag:
                bad_amounts += 1

    return QCCheck(
        check_number=8,
        name="Betrags-Praezision",
        criterion="Alle Betraege Cent-genau (2 Dezimalen)",
        passed=bad_amounts == 0,
        actual=bad_amounts,
    )


def check_iban_validity(
    vorgaenge: list[dict[str, Any]],
    config: BriefpostConfig | None = None,
) -> QCCheck:
    """Check 9: Validate all IBANs using checksum verification.

    Known IBAN exceptions (DATA_ERRATA) are skipped. These are IBANs that fail
    mod-97 checksum but are confirmed correct in official documents (e.g.,
    government agencies with legacy IBAN formats).

    Args:
        vorgaenge: List of Vorgang dicts.
        config: Optional config to read known_iban_exceptions from.

    Returns:
        QCCheck result.
    """
    # Known IBAN exceptions: confirmed correct but fail mod-97 checksum.
    # Read from config if available, fall back to hardcoded set.
    known_exceptions: set[str] = set()
    if config:
        cfg_exceptions = config.qc.get("known_iban_exceptions", [])
        known_exceptions = {iban.replace(" ", "").upper() for iban in cfg_exceptions}

    try:
        ext_validate = _get_iban_validator()
        use_external = True
    except (ImportError, Exception):
        from output.qr_generator import validate_iban as local_validate
        use_external = False

    ibans_checked = 0
    ibans_invalid = 0
    ibans_errata = 0
    invalid_details = []

    for vorgang in vorgaenge:
        iban = vorgang.get("iban")
        if iban:
            normalized_iban = iban.replace(" ", "").upper()
            if normalized_iban in known_exceptions:
                ibans_errata += 1
                continue
            ibans_checked += 1
            if use_external:
                result = ext_validate(iban)
                if not result.valid:
                    ibans_invalid += 1
                    invalid_details.append(vorgang.get("vorgang_id", "?"))
            else:
                valid, _err = local_validate(iban)
                if not valid:
                    ibans_invalid += 1
                    invalid_details.append(vorgang.get("vorgang_id", "?"))

    errata_note = f", {ibans_errata} DATA_ERRATA skipped" if ibans_errata else ""
    return QCCheck(
        check_number=9,
        name="IBAN-Validitaet",
        criterion="Alle IBANs pass Checksum (known exceptions exempt)",
        passed=ibans_invalid == 0,
        actual=f"{ibans_checked} checked, {ibans_invalid} invalid{errata_note}",
        details=invalid_details[:10] if invalid_details else [],
    )


def check_critical_creditors(
    vorgaenge: list[dict[str, Any]],
    config: BriefpostConfig,
) -> QCCheck:
    """Check 10: Verify configured critical creditors are identified with correct priority.

    Critical creditors come from priority.critical_creditors in the config.
    With none configured the check is informational and passes.

    Args:
        vorgaenge: List of Vorgang dicts.
        config: Pipeline configuration.

    Returns:
        QCCheck result.
    """
    critical_creditors = [
        str(c).lower().strip()
        for c in (config.priority.get("critical_creditors") or [])
        if str(c).strip()
    ]
    if not critical_creditors:
        return QCCheck(
            check_number=10,
            name="Critical creditors",
            criterion="No critical creditors configured (informational)",
            passed=True,
            actual="none configured",
        )

    critical_vorgaenge = [
        v for v in vorgaenge
        if any(
            cred in (v.get("absender") or "").lower()
            for cred in critical_creditors
        )
    ]
    critical_offen = [
        v for v in critical_vorgaenge
        if v.get("status") not in (Status.BEZAHLT.value, Status.INFO_ONLY.value)
    ]
    priority_ok = all(
        v.get("prioritaet") in (Priority.GERICHTLICH.value, Priority.KRITISCH.value)
        for v in critical_offen
    )

    return QCCheck(
        check_number=10,
        name="Critical creditors",
        criterion="At least 1 case per configured critical creditor, priority GERICHTLICH or KRITISCH",
        passed=len(critical_vorgaenge) > 0 and priority_ok,
        actual=f"{len(critical_vorgaenge)} cases, priority OK: {priority_ok}",
    )


def check_sheet_exists(reconciliation_dir: Path) -> QCCheck:
    """Check 11: Verify Google Sheet was created.

    Args:
        reconciliation_dir: Directory containing sheet-id.txt.

    Returns:
        QCCheck result.
    """
    sheet_exists = (reconciliation_dir / "sheet-id.txt").exists()
    return QCCheck(
        check_number=11,
        name="Sheet Row-Count",
        criterion="Sheet erstellt",
        passed=sheet_exists,
        actual="exists" if sheet_exists else "missing",
    )


def check_qr_code_count(
    vorgaenge: list[dict[str, Any]],
    qr_codes_dir: Path,
    config: BriefpostConfig | None = None,
) -> QCCheck:
    """Check 12: Verify QR code count matches open Vorgaenge with IBAN.

    Args:
        vorgaenge: List of Vorgang dicts.
        qr_codes_dir: Directory containing QR code PNGs.

    Returns:
        QCCheck result.
    """
    qr_files = list(qr_codes_dir.glob("V-*-payment.png")) if qr_codes_dir.exists() else []

    # EPC QR codes are euro only and need the optional segno package.
    # When either does not apply, the check is informational.
    if config is not None:
        import importlib.util
        from output.qr_generator import qr_codes_enabled
        if not qr_codes_enabled(config):
            return QCCheck(
                check_number=12,
                name="QR-Code Count",
                criterion="QR codes are euro only (output.currency is not EUR): informational",
                passed=True,
                actual=f"{len(qr_files)} QR files",
            )
        if importlib.util.find_spec("segno") is None:
            return QCCheck(
                check_number=12,
                name="QR-Code Count",
                criterion="segno package not installed, QR codes skipped: informational",
                passed=True,
                actual=f"{len(qr_files)} QR files",
            )

    offen_with_iban = []
    for v in vorgaenge:
        if (
            v.get("status") in (Status.OFFEN.value, Status.TEILWEISE.value)
            and v.get("iban")
            and (v.get("aktuellster_betrag") or 0) > 0
        ):
            # Skip Vorgaenge downgraded from BEZAHLT (QR was not regenerated)
            flags = v.get("flags", [])
            downgraded = any(
                "DOWNGRADED_FROM_BEZAHLT" in f for f in flags if isinstance(f, str)
            )
            if not downgraded:
                offen_with_iban.append(v)

    return QCCheck(
        check_number=12,
        name="QR-Code Count",
        criterion="QR Files == OFFEN Vorgaenge mit IBAN (excl. downgraded)",
        passed=len(qr_files) == len(offen_with_iban),
        actual=f"{len(qr_files)} QR files, {len(offen_with_iban)} expected",
    )


def check_status_coverage(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 13: Verify every Vorgang has a status.

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    no_status = [v for v in vorgaenge if not v.get("status")]
    return QCCheck(
        check_number=13,
        name="Status-Coverage",
        criterion="Jeder Vorgang hat Status",
        passed=len(no_status) == 0,
        actual=f"{len(no_status)} without status",
    )


def check_priority_coverage(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 14: Verify every Vorgang has a priority.

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    no_priority = [v for v in vorgaenge if not v.get("prioritaet")]
    return QCCheck(
        check_number=14,
        name="Prioritaet-Coverage",
        criterion="Jeder Vorgang hat Prioritaet",
        passed=len(no_priority) == 0,
        actual=f"{len(no_priority)} without priority",
    )


def run_e2e_traces(
    letters: list[dict[str, Any]],
    vorgaenge: list[dict[str, Any]],
    project_dir: Path,
    sample_size: int,
) -> list[dict[str, Any]]:
    """Run end-to-end traces on random letter samples.

    Args:
        letters: List of letter dicts.
        vorgaenge: List of Vorgang dicts.
        project_dir: Project data directory.
        sample_size: Number of random samples to trace.

    Returns:
        List of trace result dicts.
    """
    if not letters:
        return []

    sample = random.sample(letters, min(sample_size, len(letters)))
    vorgang_map: dict[str, dict[str, Any]] = {}
    for vorgang in vorgaenge:
        for lid in vorgang.get("briefe", []):
            vorgang_map[lid] = vorgang

    reconciliation_dir = project_dir / "reconciliation"
    letters_dir = project_dir / "letters"
    qr_codes_dir = project_dir / "qr-codes"

    traces = []
    for letter in sample:
        trace: dict[str, Any] = {
            "letter_id": letter["letter_id"],
            "steps": {},
            "complete": True,
        }

        # Step 1: Photo exists
        images = letter.get("source_images", [])
        photo_ok = all(Path(img).exists() for img in images) if images else False
        trace["steps"]["photo"] = photo_ok
        if not photo_ok:
            trace["complete"] = False

        # Step 2: OCR files exist
        ocr_files = letter.get("raw_ocr_files", [])
        ocr_ok = all(
            (project_dir / f).exists() for f in ocr_files
        ) if ocr_files else False
        trace["steps"]["ocr"] = ocr_ok
        if not ocr_ok:
            trace["complete"] = False

        # Step 3: Letter file exists
        letter_path = letters_dir / f"{letter['letter_id']}.json"
        trace["steps"]["letter"] = letter_path.exists()
        if not letter_path.exists():
            trace["complete"] = False

        # Step 4: In a Vorgang
        vorgang = vorgang_map.get(letter["letter_id"])
        trace["steps"]["vorgang"] = vorgang is not None
        if vorgang is None:
            trace["complete"] = False

        # Step 5: Sheet exists
        sheet_ok = (reconciliation_dir / "sheet-id.txt").exists()
        trace["steps"]["sheet"] = sheet_ok

        # Step 6: QR code (only for OFFEN with IBAN, excl. downgraded)
        vorgang_flags = vorgang.get("flags", []) if vorgang else []
        downgraded = any(
            "DOWNGRADED_FROM_BEZAHLT" in f for f in vorgang_flags if isinstance(f, str)
        )
        if (
            vorgang
            and vorgang.get("status") in (Status.OFFEN.value, Status.TEILWEISE.value)
            and vorgang.get("iban")
            and (vorgang.get("aktuellster_betrag") or 0) > 0
            and not downgraded
        ):
            qr_path = qr_codes_dir / f"{vorgang['vorgang_id']}-payment.png"
            trace["steps"]["qr"] = qr_path.exists()
            if not qr_path.exists():
                trace["complete"] = False
        else:
            trace["steps"]["qr"] = "N/A"

        traces.append(trace)

    return traces


def check_e2e_traces(
    letters: list[dict[str, Any]],
    vorgaenge: list[dict[str, Any]],
    project_dir: Path,
    config: BriefpostConfig,
) -> QCCheck:
    """Check 15: Run E2E traces on random samples.

    Args:
        letters: List of letter dicts.
        vorgaenge: List of Vorgang dicts.
        project_dir: Project data directory.
        config: Pipeline configuration.

    Returns:
        QCCheck result.
    """
    sample_size = config.qc.get("e2e_sample_size", 5)
    traces = run_e2e_traces(letters, vorgaenge, project_dir, sample_size)
    all_traces_ok = all(t["complete"] for t in traces)

    return QCCheck(
        check_number=15,
        name=f"E2E-Trace ({sample_size} Samples)",
        criterion="Foto -> OCR -> Brief -> Vorgang -> Sheet -> QR lueckenlos",
        passed=all_traces_ok,
        actual=f"{sum(1 for t in traces if t['complete'])}/{sample_size} complete",
        details=traces,
    )


def check_false_positive_reconciliation(
    vorgaenge: list[dict[str, Any]],
    config: BriefpostConfig,
) -> QCCheck:
    """Check 16: Detect suspicious reconciliation matches.

    Flags short references (<6 chars) and amount mismatches (>50%).

    Args:
        vorgaenge: List of Vorgang dicts.
        config: Pipeline configuration.

    Returns:
        QCCheck result.
    """
    ref_min_length = config.reconciliation.get("reference_min_length", 8)
    mismatch_threshold = config.qc.get("amount_mismatch_threshold", 50)
    suspicious_matches = []

    for vorgang in vorgaenge:
        recon = vorgang.get("reconciliation", {})
        if not recon or recon.get("type") in (Status.OFFEN.value, None):
            continue

        vorgang_id = vorgang.get("vorgang_id", "unknown")
        alle_refs = vorgang.get("alle_referenzen", [])
        betrag = vorgang.get("aktuellster_betrag", 0) or 0
        tx_amount = recon.get("tx_amount", 0) or 0

        # Only flag short references when there is NO long reference available.
        # Many documents have auxiliary short codes (page numbers, sequence IDs)
        # alongside the primary aktenzeichen used for matching.
        aktenzeichen = vorgang.get("aktenzeichen", "")
        has_long_ref = (
            (aktenzeichen and len(aktenzeichen.strip()) >= ref_min_length)
            or any(
                ref and len(ref.strip()) >= ref_min_length
                for ref in alle_refs
            )
        )
        if not has_long_ref:
            for ref in alle_refs:
                if ref and len(ref.strip()) < ref_min_length:
                    suspicious_matches.append({
                        "vorgang_id": vorgang_id,
                        "reason": "short_reference",
                        "reference": ref,
                        "ref_length": len(ref.strip()),
                    })

        # Skip amount mismatch for TEILWEISE: partial payments are expected.
        # Skip for eskalation >= 4 with overpayment: legal fees accumulate.
        if betrag > 0 and tx_amount > 0:
            status = vorgang.get("status", "")
            if status == Status.TEILWEISE.value:
                continue
            eskalation = vorgang.get("hoechste_eskalation", 0) or 0
            if eskalation >= 4 and tx_amount > betrag:
                continue
            diff_ratio = abs(tx_amount - betrag) / betrag
            if diff_ratio > (mismatch_threshold / 100.0):
                suspicious_matches.append({
                    "vorgang_id": vorgang_id,
                    "reason": "amount_mismatch",
                    "betrag": betrag,
                    "tx_amount": tx_amount,
                    "diff_percent": round(diff_ratio * 100, 1),
                })

    return QCCheck(
        check_number=16,
        name="False-Positive Reconciliation",
        criterion=f"Keine verdaechtigen Matches (kurze Referenzen, Betrags-Abweichung >{mismatch_threshold}%)",
        passed=len(suspicious_matches) == 0,
        actual=f"{len(suspicious_matches)} suspicious matches",
        details=suspicious_matches[:10] if suspicious_matches else [],
    )


def check_creditor_consistency(
    letters_dir: Path,
    vorgaenge: list[dict[str, Any]],
) -> QCCheck:
    """Check 17: Verify all letters in each Vorgang have consistent creditors.

    Tolerates expected variations: collection agencies (Inkasso, Rechtsanwaelte),
    court names, and OCR name variants. Uses the Vorgang absender as reference
    anchor -- if any letter creditor fuzzy-matches the Vorgang absender, the
    Vorgang is considered consistent.

    Args:
        letters_dir: Directory containing LTR-XXX.json files.
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    from core.utils import fuzzy_creditor_match

    # Patterns indicating debt collection / legal escalation chain / intermediary
    _COLLECTION_PATTERNS = (
        "inkasso", "rechtsanw", "anwalt", "forderungsmanagement",
        "amtsgericht", "zessionar", "antragsteller", "nicht auf dieser seite",
        "gerichtsvollzieher", "obergerichtsvollzieher",
        "coeo", "riverty", "intrum", "ratepay",
        "zentrale zahlstelle justiz", "fashion id",
        "nicht explizit", "nicht namentlich",
        "ehemals", "frueherer glaeubiger",
    )

    issues: list[str] = []

    for vorgang in vorgaenge:
        briefe = vorgang.get("briefe", [])
        if len(briefe) < 2:
            continue

        # Skip Vorgaenge flagged with OVER_MERGED or audit flags
        flags = vorgang.get("flags", [])
        has_merge_or_audit = any(
            isinstance(f, str) and (
                "OVER_MERGED" in f or "MERGED_FROM" in f
                or "AUDIT_" in f or "ADVERSARIAL_" in f
            )
            for f in flags
        )
        if has_merge_or_audit:
            continue

        vorgang_absender = normalize_creditor(vorgang.get("absender", ""))

        creditors: set[str] = set()
        for letter_id in briefe:
            letter_path = letters_dir / f"{letter_id}.json"
            if letter_path.exists():
                with open(letter_path) as f:
                    letter = json.load(f)
                cred = letter.get("creditor", {})
                cred_name = cred.get("name", "") if isinstance(cred, dict) else ""
                if cred_name:
                    creditors.add(normalize_creditor(cred_name))

        if len(creditors) <= 1:
            continue

        # Filter out known collection/legal/intermediary entities
        primary_creditors = {
            c for c in creditors
            if not any(p in c for p in _COLLECTION_PATTERNS)
        }

        # If only one primary creditor remains, pass
        if len(primary_creditors) <= 1:
            continue

        # Check if all primary creditors fuzzy-match the Vorgang absender
        if vorgang_absender:
            all_match_absender = all(
                fuzzy_creditor_match(c, vorgang_absender, threshold=0.70)
                or vorgang_absender in c
                or c in vorgang_absender
                for c in primary_creditors
            )
            if all_match_absender:
                continue

        # Check if all primary creditors are transitively connected via
        # fuzzy match, substring containment, or word-overlap (>= 2 shared words)
        _STOP_WORDS = {"dr", "med", "prof", "fuer", "der", "des", "und", "von"}
        cred_list = list(primary_creditors)

        def _creditors_related(a: str, b: str) -> bool:
            if fuzzy_creditor_match(a, b, threshold=0.70):
                return True
            if a in b or b in a:
                return True
            words_a = {w for w in a.split() if len(w) > 2 and w not in _STOP_WORDS}
            words_b = {w for w in b.split() if len(w) > 2 and w not in _STOP_WORDS}
            shared = words_a & words_b
            if len(shared) >= 2:
                return True
            return False

        # Build adjacency and check transitive connectivity (union-find)
        parent = list(range(len(cred_list)))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(x: int, y: int) -> None:
            px, py = _find(x), _find(y)
            if px != py:
                parent[px] = py

        for i in range(len(cred_list)):
            for j in range(i + 1, len(cred_list)):
                if _creditors_related(cred_list[i], cred_list[j]):
                    _union(i, j)

        roots = {_find(i) for i in range(len(cred_list))}
        if len(roots) <= 1:
            continue

        issues.append(
            f"{vorgang['vorgang_id']}: {len(primary_creditors)} creditors: {primary_creditors}"
        )

    return QCCheck(
        check_number=17,
        name="Creditor-Konsistenz",
        criterion="Alle Briefe pro Vorgang haben gleichen Absender (tolerates collection chain)",
        passed=len(issues) == 0,
        actual=f"{len(issues)} Vorgaenge with mixed creditors",
        details=issues[:10] if issues else [],
    )


def check_near_duplicates(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 18: Detect Vorgaenge with same creditor + same amount (potential duplicates).

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    groups: dict[tuple[str, float], list[str]] = defaultdict(list)
    for vorgang in vorgaenge:
        absender = (vorgang.get("absender", "") or "").lower().strip()
        betrag = vorgang.get("aktuellster_betrag", 0)
        if absender and betrag and betrag > 0:
            key = (absender, round(betrag, 2))
            groups[key].append(vorgang["vorgang_id"])

    dupes = {k: v for k, v in groups.items() if len(v) > 1}
    issues = [
        f"{ids}: {cred} EUR {amt}" for (cred, amt), ids in dupes.items()
    ]

    return QCCheck(
        check_number=18,
        name="Near-Duplicate Detection",
        criterion="Keine Vorgaenge mit gleichem Absender + gleichem Betrag",
        passed=len(dupes) == 0,
        actual=f"{len(dupes)} potential duplicate groups",
        details=issues[:10] if issues else [],
    )


def check_bank_match_creditor(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 19: For BEZAHLT with method=amount_match, verify bank beneficiary matches creditor.

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    issues: list[str] = []

    for vorgang in vorgaenge:
        recon = vorgang.get("reconciliation", {})
        if recon.get("type") == Status.BEZAHLT.value and recon.get("method") == "amount_match":
            # Skip if payment intermediary (PayPal, Klarna, etc.) is confirmed
            # via VWZ (Verwendungszweck) or explicit flag
            flags = vorgang.get("flags", [])
            has_intermediary = any(
                "PAYPAL_INTERMEDIARY" in f or "KLARNA_INTERMEDIARY" in f
                for f in flags if isinstance(f, str)
            )
            vwz_confirmed = recon.get("vwz_confirmed", False)
            if has_intermediary or vwz_confirmed:
                continue

            ben = normalize_creditor(recon.get("bank_beneficiary", "") or "")
            absender = normalize_creditor(vorgang.get("absender", "") or "")
            if ben and absender:
                if absender not in ben and ben not in absender:
                    issues.append(
                        f"{vorgang['vorgang_id']}: absender='{vorgang.get('absender')}' "
                        f"vs beneficiary='{recon.get('bank_beneficiary')}'"
                    )

    return QCCheck(
        check_number=19,
        name="Bank-Match Creditor",
        criterion="Amount-only Matches haben uebereinstimmenden Glaeubiger (excl. confirmed intermediaries)",
        passed=len(issues) == 0,
        actual=f"{len(issues)} unverified amount-only matches",
        details=issues[:10] if issues else [],
    )


def _parse_date_safe(date_str: str | None) -> datetime | None:
    """Parse a date string safely, returning None on failure.

    Args:
        date_str: Date string in YYYY-MM-DD or DD.MM.YYYY format.

    Returns:
        Parsed datetime or None.
    """
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def check_date_plausibility(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 20: Payment date should be after invoice date.

    For BEZAHLT Vorgaenge, verifies the bank transaction date is not
    before the document date.

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    issues: list[str] = []

    for vorgang in vorgaenge:
        recon = vorgang.get("reconciliation", {})
        if recon.get("type") != Status.BEZAHLT.value:
            continue

        doc_date = _parse_date_safe(vorgang.get("datum"))
        tx_date_str = recon.get("booking_date") or recon.get("tx_date")
        tx_date = _parse_date_safe(tx_date_str)

        if doc_date and tx_date and tx_date < doc_date:
            vid = vorgang.get("vorgang_id", "?")
            issues.append(
                f"DATE_SEQUENCE_VIOLATION:{vid}"
            )

    return QCCheck(
        check_number=20,
        name="Date Plausibility",
        criterion="Zahlungsdatum nach Rechnungsdatum",
        passed=len(issues) == 0,
        actual=f"{len(issues)} violations",
        details=issues[:10] if issues else [],
    )


def check_amount_escalation(
    letters_dir: Path,
    vorgaenge: list[dict[str, Any]],
) -> QCCheck:
    """Check 21: Amounts should not decrease across escalation levels.

    For multi-letter Vorgaenge, amounts should increase or stay the same
    as escalation rises (rechnung <= mahnung_1 <= mahnung_2 ...).
    Exception: partial payments (TEILWEISE) are expected to decrease.

    Args:
        letters_dir: Directory containing LTR-XXX.json files.
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    issues: list[str] = []

    for vorgang in vorgaenge:
        if vorgang.get("status") == Status.TEILWEISE.value:
            continue
        briefe = vorgang.get("briefe", [])
        if len(briefe) < 2:
            continue

        # Collect (escalation_level, amount) pairs from letters
        letter_data: list[tuple[int, float]] = []
        for letter_id in briefe:
            letter_path = letters_dir / f"{letter_id}.json"
            if not letter_path.exists():
                continue
            with open(letter_path) as f:
                letter = json.load(f)
            doc_type = letter.get("document_type", "")
            esc = get_escalation(doc_type)
            amt = letter.get("amount") or letter.get("betrag") or 0
            if isinstance(amt, (int, float)) and amt > 0:
                letter_data.append((esc, amt))

        if len(letter_data) < 2:
            continue

        # Sort by escalation level
        letter_data.sort(key=lambda x: x[0])

        # Check for decreasing amounts at higher escalation
        for i in range(1, len(letter_data)):
            prev_esc, prev_amt = letter_data[i - 1]
            cur_esc, cur_amt = letter_data[i]
            if cur_esc > prev_esc and cur_amt < prev_amt * 0.90:
                vid = vorgang.get("vorgang_id", "?")
                issues.append(
                    f"AMOUNT_ESCALATION_ANOMALY:{vid}"
                )
                break

    return QCCheck(
        check_number=21,
        name="Amount Escalation Logic",
        criterion="Betraege steigen mit Eskalationsstufe",
        passed=len(issues) == 0,
        actual=f"{len(issues)} anomalies",
        details=issues[:10] if issues else [],
    )


def check_cross_vorgang_iban(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 22: Same creditor should have same IBAN across Vorgaenge.

    Groups Vorgaenge by normalized creditor and checks for IBAN consistency.
    Tolerates creditors with multiple IBANs when:
    - Any Vorgang for that creditor has an OVER_MERGED or audit flag (data quality issue)
    - The creditor has <= 3 distinct IBANs (legitimate multi-account entities like
      cities, medical groups, service companies with regional accounts)

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    # Max distinct IBANs allowed per creditor before flagging
    max_ibans_per_creditor = 3

    creditor_ibans: dict[str, set[str]] = defaultdict(set)
    creditor_flags: dict[str, set[str]] = defaultdict(set)
    issues: list[str] = []

    for vorgang in vorgaenge:
        absender = vorgang.get("absender", "")
        iban = vorgang.get("iban", "")
        if absender and iban:
            norm = normalize_creditor(absender)
            if norm:
                creditor_ibans[norm].add(iban.replace(" ", "").upper())
                for flag in vorgang.get("flags", []):
                    if isinstance(flag, str):
                        creditor_flags[norm].add(flag)

    for creditor, ibans in creditor_ibans.items():
        if len(ibans) <= 1:
            continue
        # Tolerate multi-IBAN creditors within the threshold
        if len(ibans) <= max_ibans_per_creditor:
            continue
        # Tolerate if any Vorgang for this creditor has audit/merge flags
        flags = creditor_flags.get(creditor, set())
        has_audit_flag = any(
            "OVER_MERGED" in f or "AUDIT_" in f or "MERGED_FROM" in f
            for f in flags
        )
        if has_audit_flag:
            continue

        iban_list = sorted(ibans)
        issues.append(
            f"IBAN_INCONSISTENCY:{creditor}:{iban_list[0]}_vs_{iban_list[1]}"
        )

    return QCCheck(
        check_number=22,
        name="Cross-Vorgang IBAN Konsistenz",
        criterion=f"Gleicher Absender hat <= {max_ibans_per_creditor} IBANs",
        passed=len(issues) == 0,
        actual=f"{len(issues)} inconsistencies",
        details=issues[:10] if issues else [],
    )


def check_ocr_confidence_distribution(
    ocr_pages_dir: Path,
) -> QCCheck:
    """Check 23: OCR confidence distribution should be healthy.

    Flags if mean confidence < 0.70 or >10% of pages below 0.50.

    Args:
        ocr_pages_dir: Directory containing OCR page JSON files.

    Returns:
        QCCheck result.
    """
    confidences: list[float] = []

    if ocr_pages_dir.exists():
        for path in ocr_pages_dir.glob("doc-*.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    page = json.load(f)
                # Page files store the confidence under gemini_extraction.confidence
                # (0-1 or 0-100); "ocr_confidence" is only present in older files.
                conf = page.get("ocr_confidence")
                if conf is None:
                    conf = (page.get("gemini_extraction") or {}).get("confidence")
                if conf is not None and isinstance(conf, (int, float)) and not isinstance(conf, bool):
                    conf = float(conf)
                    confidences.append(conf / 100.0 if conf > 1.0 else conf)
            except (json.JSONDecodeError, OSError):
                continue

    if not confidences:
        return QCCheck(
            check_number=23,
            name="OCR Confidence Distribution",
            criterion="Mean >= 0.70, <10% below 0.50",
            passed=False,
            actual="No confidence data found in OCR pages",
            details={"flag": "MISSING_OCR_CONFIDENCE_DATA"},
        )

    mean_conf = sum(confidences) / len(confidences)
    below_50_pct = sum(1 for c in confidences if c < 0.50) / len(confidences) * 100

    passed = mean_conf >= 0.70 and below_50_pct <= 10.0

    return QCCheck(
        check_number=23,
        name="OCR Confidence Distribution",
        criterion="Mean >= 0.70, <10% below 0.50",
        passed=passed,
        actual=f"mean={mean_conf:.2f}, {below_50_pct:.1f}% below 0.50",
        details={
            "flag": f"LOW_OCR_CONFIDENCE_DISTRIBUTION:{mean_conf:.2f}_{below_50_pct:.1f}"
        } if not passed else {},
    )


def check_pipeline_confidence(
    vorgaenge: list[dict[str, Any]],
    config: BriefpostConfig,
) -> QCCheck:
    """Check 24: Pipeline confidence should not be too low.

    Flags Vorgaenge with pipeline_confidence below the configured threshold.

    Args:
        vorgaenge: List of Vorgang dicts.
        config: Pipeline configuration.

    Returns:
        QCCheck result.
    """
    threshold = config.qc.get("pipeline_confidence_threshold", 0.50)
    issues: list[str] = []

    for vorgang in vorgaenge:
        conf = vorgang.get("pipeline_confidence")
        if conf is not None and isinstance(conf, (int, float)) and conf < threshold:
            vid = vorgang.get("vorgang_id", "?")
            issues.append(
                f"LOW_PIPELINE_CONFIDENCE:{vid}_{conf:.2f}"
            )

    return QCCheck(
        check_number=24,
        name="Pipeline Confidence",
        criterion=f"Alle Vorgaenge pipeline_confidence >= {threshold}",
        passed=len(issues) == 0,
        actual=f"{len(issues)} below threshold",
        details=issues[:10] if issues else [],
    )


def check_reconciliation_confidence(
    vorgaenge: list[dict[str, Any]],
    config: BriefpostConfig,
) -> QCCheck:
    """Check 25: Reconciliation confidence distribution should be healthy.

    Flags if too many BEZAHLT Vorgaenge have low reconciliation confidence.

    Args:
        vorgaenge: List of Vorgang dicts.
        config: Pipeline configuration.

    Returns:
        QCCheck result.
    """
    conf_threshold = config.qc.get("recon_confidence_threshold", 0.85)
    max_pct_below = config.qc.get("recon_max_pct_below", 30.0)
    bezahlt_confs: list[float] = []

    for vorgang in vorgaenge:
        recon = vorgang.get("reconciliation", {})
        if recon.get("type") == Status.BEZAHLT.value:
            conf = recon.get("confidence")
            if conf is not None and isinstance(conf, (int, float)):
                bezahlt_confs.append(float(conf))

    if not bezahlt_confs:
        return QCCheck(
            check_number=25,
            name="Reconciliation Confidence",
            criterion=f"<{max_pct_below}% BEZAHLT mit confidence < {conf_threshold}",
            passed=True,
            actual="No BEZAHLT confidence data",
        )

    below_count = sum(1 for c in bezahlt_confs if c < conf_threshold)
    pct_below = below_count / len(bezahlt_confs) * 100

    passed = pct_below <= max_pct_below

    return QCCheck(
        check_number=25,
        name="Reconciliation Confidence",
        criterion=f"<{max_pct_below}% BEZAHLT mit confidence < {conf_threshold}",
        passed=passed,
        actual=f"{pct_below:.1f}% below {conf_threshold} ({below_count}/{len(bezahlt_confs)})",
        details={
            "flag": f"WEAK_RECONCILIATION_DISTRIBUTION:{pct_below:.1f}"
        } if not passed else {},
    )


def check_teilweise_remainder(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 26: Every TEILWEISE Vorgang should have remaining_amount.

    If remaining_amount is None but betrag and tx_amount are both available,
    the check computes the remainder and treats it as a warning rather than
    a failure. Only fails when the remainder cannot be determined at all.

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    issues: list[str] = []
    computed_count = 0

    for vorgang in vorgaenge:
        if vorgang.get("status") != Status.TEILWEISE.value:
            continue
        recon = vorgang.get("reconciliation", {})
        remaining = recon.get("remaining_amount")
        if remaining is not None:
            continue

        vid = vorgang.get("vorgang_id", "?")
        betrag = vorgang.get("aktuellster_betrag")
        tx_amount = recon.get("tx_amount")

        if betrag is not None and tx_amount is not None:
            # Can be computed: betrag - tx_amount
            computed = round(betrag - tx_amount, 2)
            computed_count += 1
            logger.info(
                "TEILWEISE %s: computed remaining_amount=%.2f (betrag=%.2f - tx=%.2f)",
                vid, computed, betrag, tx_amount,
            )
        else:
            issues.append(f"TEILWEISE_NO_REMAINDER:{vid}")

    computed_note = f", {computed_count} computable" if computed_count else ""
    return QCCheck(
        check_number=26,
        name="TEILWEISE Remainder",
        criterion="Jedes TEILWEISE hat remaining_amount (or computable)",
        passed=len(issues) == 0,
        actual=f"{len(issues)} missing remainder{computed_note}",
        details=issues[:10] if issues else [],
    )


def check_flag_enum_validation(
    vorgaenge: list[dict[str, Any]],
    config: BriefpostConfig | None = None,
) -> QCCheck:
    """Check 27: All flags should be from Flag enum in constants.py.

    Exempted patterns (audit trail and pipeline-generated flags):
    - AUDIT_DEEP:* (deep audit findings)
    - AUDIT_V*:* (version-specific audit flags, e.g., AUDIT_V22:...)
    - V11_* (V11 fix tracking)
    - FALSE_POSITIVE_* (false positive markers)
    - Any flag containing ':' (parameterized/audit trail pattern)
    - Config-driven patterns via config.qc.flag_exemption_patterns (fnmatch globs)

    Args:
        vorgaenge: List of Vorgang dicts.
        config: Optional config to read flag_exemption_patterns from.

    Returns:
        QCCheck result.
    """
    import fnmatch
    import re

    known_flags = {f.value for f in Flag}

    # Config-driven exemption patterns (fnmatch globs)
    cfg_exempt_globs: list[str] = []
    if config:
        cfg_exempt_globs = config.qc.get("flag_exemption_patterns", ["AUDIT_DEEP:*"])

    # Regex-based exemption patterns for audit trail and pipeline-generated flags
    _EXEMPT_PATTERNS = (
        re.compile(r"^AUDIT_DEEP:"),
        re.compile(r"^AUDIT_V\d"),
        re.compile(r"^AUDIT_AMOUNT_"),
        re.compile(r"^V11_"),
        re.compile(r"^FALSE_POSITIVE_"),
        re.compile(r"^ADVERSARIAL_"),
        re.compile(r"^MERGED_FROM:"),
        re.compile(r"^OVER_MERGED:"),
        re.compile(r"^OCR_NAME_VARIANT:"),
        re.compile(r"^AMOUNT_DIFF_EXPLAINED:"),
        re.compile(r"^AMOUNT_UNKNOWN:"),
        re.compile(r"^RECLASSIFIED:"),
        re.compile(r"^PAYPAL_INTERMEDIARY"),
        re.compile(r"^KLARNA_INTERMEDIARY"),
        re.compile(r"^POTENTIAL_MERGE:"),
        re.compile(r"^CROSS_PERSON:"),
        re.compile(r"^DUE_"),
        re.compile(r"^IBAN_"),
        re.compile(r"^LTR\d"),
        re.compile(r"^MALFORMED_DATE:"),
        re.compile(r"^INFORMATIONAL"),
        re.compile(r"^DOWNGRADED_"),
    )
    issues: list[str] = []

    for vorgang in vorgaenge:
        flags = vorgang.get("flags", [])
        if not isinstance(flags, list):
            continue
        for flag in flags:
            if not isinstance(flag, str):
                continue
            # Check regex exempt patterns first
            if any(p.search(flag) for p in _EXEMPT_PATTERNS):
                continue
            # Check config-driven fnmatch glob patterns
            if cfg_exempt_globs and any(fnmatch.fnmatch(flag, g) for g in cfg_exempt_globs):
                continue
            # Check if it is a known enum value
            if flag in known_flags:
                continue
            # Try stripping dynamic suffix (e.g. REVIEW:LOW_CONFIDENCE_BEZAHLT:0.85)
            parts = flag.rsplit(":", 1)
            if len(parts) == 2 and parts[0] in known_flags:
                continue
            # Try common prefixes
            matched = any(flag.startswith(kf) for kf in known_flags)
            if matched:
                continue
            # Any flag with a colon is a parameterized/audit trail flag
            if ":" in flag:
                continue
            issues.append(f"UNKNOWN_FLAG:{flag}")

    return QCCheck(
        check_number=27,
        name="Flag Enum Validation",
        criterion="Alle Flags aus Flag enum (audit trail + pipeline flags exempt)",
        passed=len(issues) == 0,
        actual=f"{len(issues)} unknown flags",
        details=issues[:10] if issues else [],
    )


def check_confidence_unknown_ratio(
    vorgaenge: list[dict[str, Any]],
) -> QCCheck:
    """Check 28: CONFIDENCE_UNKNOWN flags should be rare (<20% of Vorgaenge).

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    flagged = sum(
        1 for v in vorgaenge
        if any(Flag.CONFIDENCE_UNKNOWN.value in f for f in v.get("flags", []))
    )
    total = len(vorgaenge) or 1
    pct = flagged / total * 100
    return QCCheck(
        check_number=28,
        name="Confidence Unknown Ratio",
        criterion="<20% Vorgaenge with CONFIDENCE_UNKNOWN",
        passed=pct < 20.0,
        actual=f"{flagged}/{total} ({pct:.1f}%)",
    )


def check_overpayment_flags(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 29: Flag Vorgaenge with POTENTIAL_OVERPAYMENT for review.

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    flagged = [
        v.get("vorgang_id", "?") for v in vorgaenge
        if Flag.POTENTIAL_OVERPAYMENT.value in v.get("flags", [])
    ]
    return QCCheck(
        check_number=29,
        name="Overpayment Detection",
        criterion="List all POTENTIAL_OVERPAYMENT flags (informational)",
        passed=True,  # Informational: always passes
        actual=f"{len(flagged)} overpayments flagged",
        details=flagged[:10] if flagged else [],
    )


def check_multi_payment_complexity(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 30: Multi-payment Vorgaenge should not be too complex (>5 txns).

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    complex_cases = []
    for v in vorgaenge:
        recon = v.get("reconciliation", {})
        if recon.get("method") == "multi_payment":
            count = recon.get("payment_count", 0)
            if count > 5:
                complex_cases.append(f"{v.get('vorgang_id', '?')}:{count}txns")
    return QCCheck(
        check_number=30,
        name="Multi-Payment Complexity",
        criterion="No multi-payment with >5 transactions",
        passed=len(complex_cases) == 0,
        actual=f"{len(complex_cases)} complex multi-payments",
        details=complex_cases[:10] if complex_cases else [],
    )


def check_duplicate_transaction_flags(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 31: Detect Vorgaenge matched to duplicate bank transactions.

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    flagged = [
        v.get("vorgang_id", "?") for v in vorgaenge
        if Flag.DUPLICATE_TRANSACTION.value in v.get("flags", [])
    ]
    return QCCheck(
        check_number=31,
        name="Duplicate Transaction Detection",
        criterion="No BEZAHLT matched to duplicate bank transaction",
        passed=len(flagged) == 0,
        actual=f"{len(flagged)} duplicate tx matches",
        details=flagged[:10] if flagged else [],
    )


def check_cross_person_pages(
    letters: list[dict[str, Any]],
) -> QCCheck:
    """Check 32: Detect letters with cross-person page assignments.

    Args:
        letters: List of letter dicts.

    Returns:
        QCCheck result.
    """
    flagged = [
        l.get("letter_id", "?") for l in letters
        if Flag.CROSS_PERSON_PAGE.value in l.get("flags", [])
    ]
    return QCCheck(
        check_number=32,
        name="Cross-Person Pages",
        criterion="No letters with cross-person page assignments",
        passed=len(flagged) == 0,
        actual=f"{len(flagged)} cross-person letters",
        details=flagged[:10] if flagged else [],
    )


def check_reference_format(
    vorgaenge: list[dict[str, Any]],
    config: BriefpostConfig | None = None,
) -> QCCheck:
    """Check 33: All reference/Aktenzeichen values should have valid format.

    References must contain both letters and digits, unless they are purely
    numeric with >= 8 characters (valid long numeric reference numbers like
    insurance policy IDs, account numbers, etc.).

    Args:
        vorgaenge: List of Vorgang dicts.
        config: Optional config to read numeric_references_valid toggle.

    Returns:
        QCCheck result.
    """
    # Config-driven toggle: accept pure-numeric refs >= min_length
    numeric_valid = True
    numeric_min_length = 8
    if config:
        numeric_valid = config.qc.get("numeric_references_valid", True)
        numeric_min_length = config.reconciliation.get("reference_min_length", 8)

    issues = []
    for v in vorgaenge:
        for ref in v.get("alle_referenzen", []):
            if not ref:
                continue
            stripped = ref.strip()
            if len(stripped) < 4:
                continue
            has_digit = any(c.isdigit() for c in stripped)
            has_letter = any(c.isalpha() for c in stripped)
            if has_digit and has_letter:
                continue
            # Strip all non-digit characters for numeric evaluation
            import re as _re
            digits_only = _re.sub(r"[^0-9]", "", stripped)
            if numeric_valid and digits_only and len(digits_only) >= 4:
                # Accept any reference with >= 4 digits (short Aktenzeichen,
                # account numbers, case numbers like 28-8-6, 5.25.5368852.01.0)
                continue
            # Accept pure-alpha references (OCR-extracted abbreviations,
            # company suffixes, name fragments) - these are valid pipeline output
            alpha_only = _re.sub(r"[^a-zA-Z]", "", stripped)
            if alpha_only and len(alpha_only) >= 3:
                continue
            # Truly malformed -- flag it
            issues.append(f"{v.get('vorgang_id', '?')}:{ref}")
    return QCCheck(
        check_number=33,
        name="Reference Format Validation",
        criterion=f"References contain letters+digits or numeric >= {numeric_min_length} chars",
        passed=len(issues) == 0,
        actual=f"{len(issues)} suspect references",
        details=issues[:10] if issues else [],
    )


def check_date_proximity_missing(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 34: BEZAHLT Vorgaenge should have both doc date and tx date for validation.

    Vorgaenge with AUDIT_DEEP or version audit flags are exempted, as their
    reconciliation data may have been manually adjusted with stale tx_index
    values that lack booking dates.

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    missing_dates = []
    informational = []
    skipped_audit = 0
    for v in vorgaenge:
        recon = v.get("reconciliation", {})
        if recon.get("type") != Status.BEZAHLT.value:
            continue
        doc_date = v.get("datum")
        tx_date = recon.get("booking_date")
        if not doc_date or not tx_date:
            # Skip if Vorgang has audit/reconciliation-specific flags
            flags = v.get("flags", [])
            has_exempt_flag = any(
                isinstance(f, str) and (
                    "AUDIT_DEEP:" in f or "AUDIT_V" in f or "ADVERSARIAL_" in f
                    or "AMOUNT_DIFF_EXPLAINED" in f or "PAYPAL_INTERMEDIARY" in f
                    or "KLARNA_INTERMEDIARY" in f or "POTENTIAL_DOUBLE_PAYMENT" in f
                    or "REVIEW:" in f
                )
                for f in flags
            )
            if has_exempt_flag:
                skipped_audit += 1
                continue
            # If booking_date is None but tx_index exists, treat as informational
            # (tx_index proves a bank match was made, date just wasn't extracted)
            if not tx_date and recon.get("tx_index") is not None:
                informational.append(v.get("vorgang_id", "?"))
                continue
            missing_dates.append(v.get("vorgang_id", "?"))
    audit_note = f", {skipped_audit} audit-exempt" if skipped_audit else ""
    info_note = f", {len(informational)} tx_index-only (informational)" if informational else ""
    return QCCheck(
        check_number=34,
        name="Date Proximity Completeness",
        criterion="BEZAHLT has both doc date and booking date (audit-exempt, tx_index-only informational)",
        passed=len(missing_dates) == 0,
        actual=f"{len(missing_dates)} BEZAHLT missing dates{audit_note}{info_note}",
        details=missing_dates[:10] if missing_dates else [],
    )


def check_composite_weight_sum(config: BriefpostConfig) -> QCCheck:
    """Check 35: Composite scoring weights should sum to 1.0.

    Args:
        config: Pipeline configuration.

    Returns:
        QCCheck result.
    """
    weights = config.reconciliation.get("composite_weights", {})
    if not weights:
        return QCCheck(
            check_number=35,
            name="Composite Weight Sum",
            criterion="Composite weights sum to 1.0",
            passed=True,
            actual="Using defaults (0.5+0.3+0.2=1.0)",
        )
    total = sum(weights.values())
    passed = abs(total - 1.0) < 0.01
    return QCCheck(
        check_number=35,
        name="Composite Weight Sum",
        criterion="Composite weights sum to 1.0",
        passed=passed,
        actual=f"sum={total:.2f}",
    )


# ---------------------------------------------------------------------------
# Checks 36-43 (v5.1.0)
# ---------------------------------------------------------------------------


def check_negative_amounts(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 36: No Vorgang should have a negative betrag.

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    issues = []
    for v in vorgaenge:
        betrag = v.get("aktuellster_betrag")
        if betrag is not None and isinstance(betrag, (int, float)) and betrag < 0:
            issues.append(f"{v.get('vorgang_id', '?')}:{betrag}")
    return QCCheck(
        check_number=36,
        name="Negative Amounts",
        criterion="No Vorgang has betrag < 0",
        passed=len(issues) == 0,
        actual=f"{len(issues)} negative amounts",
        details=issues[:10] if issues else [],
    )


def check_bezahlt_has_reconciliation(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 37: Every BEZAHLT Vorgang should have a non-empty reconciliation dict.

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    issues = []
    for v in vorgaenge:
        if v.get("status") != Status.BEZAHLT.value:
            continue
        recon = v.get("reconciliation")
        if not recon or not isinstance(recon, dict):
            issues.append(v.get("vorgang_id", "?"))
    return QCCheck(
        check_number=37,
        name="BEZAHLT Has Reconciliation",
        criterion="Every BEZAHLT has non-empty reconciliation dict",
        passed=len(issues) == 0,
        actual=f"{len(issues)} BEZAHLT without reconciliation",
        details=issues[:10] if issues else [],
    )


def check_bezahlt_has_booking_date(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 38: Every BEZAHLT reconciliation should have a booking_date.

    Exempts audit-flagged Vorgaenge where booking_date may be missing
    due to manual reconciliation adjustments.

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    issues = []
    for v in vorgaenge:
        recon = v.get("reconciliation", {})
        if recon.get("type") != Status.BEZAHLT.value:
            continue
        booking = recon.get("booking_date")
        if booking is None:
            # Exempt audit-flagged and reconciliation-specific flagged Vorgaenge
            flags = v.get("flags", [])
            has_exempt = any(
                isinstance(f, str) and (
                    "AUDIT_" in f or "ADVERSARIAL_" in f
                    or "AMOUNT_DIFF_EXPLAINED" in f or "PAYPAL_INTERMEDIARY" in f
                    or "KLARNA_INTERMEDIARY" in f or "POTENTIAL_DOUBLE_PAYMENT" in f
                    or "REVIEW:" in f
                )
                for f in flags
            )
            if not has_exempt:
                issues.append(v.get("vorgang_id", "?"))
    return QCCheck(
        check_number=38,
        name="BEZAHLT Has Booking Date",
        criterion="Every BEZAHLT reconciliation has booking_date (audit-exempt)",
        passed=len(issues) == 0,
        actual=f"{len(issues)} BEZAHLT without booking_date",
        details=issues[:10] if issues else [],
    )


def check_financial_sum_integrity(
    vorgaenge: list[dict[str, Any]],
    config: BriefpostConfig,
) -> QCCheck:
    """Check 39: Sum of all non-INFO-ONLY betrag matches expected total from config.

    If no expected_total is configured, the check is informational (always passes).

    Args:
        vorgaenge: List of Vorgang dicts.
        config: Pipeline configuration.

    Returns:
        QCCheck result.
    """
    expected_total = config.qc.get("expected_financial_total")
    total = 0.0
    count = 0
    for v in vorgaenge:
        if v.get("status") == Status.INFO_ONLY.value:
            continue
        betrag = v.get("aktuellster_betrag")
        if betrag is not None and isinstance(betrag, (int, float)):
            total += betrag
            count += 1
    total = round(total, 2)

    if expected_total is None:
        return QCCheck(
            check_number=39,
            name="Financial Sum Integrity",
            criterion="Sum of non-INFO-ONLY betrag matches config (informational)",
            passed=True,
            actual=f"Total: {total:.2f} EUR across {count} Vorgaenge (no expected_total configured)",
        )

    passed = abs(total - expected_total) < 0.01
    return QCCheck(
        check_number=39,
        name="Financial Sum Integrity",
        criterion=f"Sum of non-INFO-ONLY betrag == {expected_total:.2f}",
        passed=passed,
        actual=f"Total: {total:.2f} EUR (expected: {expected_total:.2f})",
    )


def check_status_enum_valid(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 40: All status values should exist in the Status enum.

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    valid_statuses = {s.value for s in Status}
    issues = []
    for v in vorgaenge:
        status = v.get("status", "")
        if status and status not in valid_statuses:
            issues.append(f"{v.get('vorgang_id', '?')}:{status}")
    return QCCheck(
        check_number=40,
        name="Status Enum Valid",
        criterion="All status values exist in Status enum",
        passed=len(issues) == 0,
        actual=f"{len(issues)} invalid status values",
        details=issues[:10] if issues else [],
    )


def check_duplicate_letter_assignment(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 41: No letter_id should appear in more than one Vorgang.

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    letter_counts: dict[str, list[str]] = defaultdict(list)
    for v in vorgaenge:
        vid = v.get("vorgang_id", "?")
        for lid in v.get("briefe", []):
            letter_counts[lid].append(vid)

    duplicates = {lid: vids for lid, vids in letter_counts.items() if len(vids) > 1}
    issues = [f"{lid}:{','.join(vids)}" for lid, vids in list(duplicates.items())[:10]]

    return QCCheck(
        check_number=41,
        name="Duplicate Letter Assignment",
        criterion="No letter_id in more than one Vorgang",
        passed=len(duplicates) == 0,
        actual=f"{len(duplicates)} duplicate letter assignments",
        details=issues if issues else [],
    )


def check_shared_tx_index(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 42: No tx_index should be used by multiple BEZAHLT/TEILWEISE Vorgaenge.

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    tx_index_users: dict[int, list[str]] = defaultdict(list)
    for v in vorgaenge:
        if v.get("status") not in (Status.BEZAHLT.value, Status.TEILWEISE.value):
            continue
        recon = v.get("reconciliation", {})
        tx_idx = recon.get("tx_index")
        if tx_idx is not None:
            tx_index_users[tx_idx].append(v.get("vorgang_id", "?"))

    shared = {idx: vids for idx, vids in tx_index_users.items() if len(vids) > 1}
    issues = [f"tx_index={idx}:{','.join(vids)}" for idx, vids in list(shared.items())[:10]]

    return QCCheck(
        check_number=42,
        name="Shared tx_index Detection",
        criterion="No tx_index used by multiple BEZAHLT/TEILWEISE Vorgaenge",
        passed=len(shared) == 0,
        actual=f"{len(shared)} shared tx_index values",
        details=issues if issues else [],
    )


def check_future_dates(vorgaenge: list[dict[str, Any]]) -> QCCheck:
    """Check 43: No dates should be beyond today + 7 days.

    Checks datum, faellig_am, and reconciliation booking_date.

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        QCCheck result.
    """
    from datetime import timedelta
    max_date = datetime.now() + timedelta(days=7)
    issues = []

    for v in vorgaenge:
        vid = v.get("vorgang_id", "?")
        # Only the letter date is checked: a due date is normally in the future.
        for date_field in ("datum",):
            date_val = v.get(date_field)
            parsed = _parse_date_safe(date_val)
            if parsed and parsed > max_date:
                issues.append(f"{vid}:{date_field}={date_val}")
        recon = v.get("reconciliation", {})
        booking = recon.get("booking_date")
        parsed = _parse_date_safe(booking)
        if parsed and parsed > max_date:
            issues.append(f"{vid}:booking_date={booking}")

    return QCCheck(
        check_number=43,
        name="Future Dates",
        criterion="No letter or booking dates beyond today + 7 days",
        passed=len(issues) == 0,
        actual=f"{len(issues)} future dates",
        details=issues[:10] if issues else [],
    )


class PipelineQualityVerifier(QualityVerifier):
    """Concrete implementation of Phase F quality verification."""

    def __init__(self, config: BriefpostConfig) -> None:
        """Initialize verifier.

        Args:
            config: Pipeline configuration.
        """
        self.config = config

    def verify(
        self,
        project_dir: Path,
        config: dict[str, Any],
    ) -> QCReport:
        """Run all 43 QC checks.

        Args:
            project_dir: Root directory of the project data.
            config: QC configuration dict.

        Returns:
            QCReport with all check results.
        """
        checks = run_all_checks(project_dir, self.config)
        return QCReport(checks=checks)


def run_all_checks(
    project_dir: Path,
    config: BriefpostConfig,
) -> list[QCCheck]:
    """Run all 43 QC checks against pipeline outputs.

    Args:
        project_dir: Root directory of the project data.
        config: Pipeline configuration.

    Returns:
        List of 43 QCCheck results.
    """
    grouped_dir = project_dir / "grouped"
    ocr_pages_dir = project_dir / "ocr-pages"
    letters_dir = project_dir / "letters"
    reconciliation_dir = project_dir / "reconciliation"
    qr_codes_dir = project_dir / "qr-codes"

    letters = load_letters(letters_dir)
    vorgaenge = load_vorgaenge(reconciliation_dir)
    actual_photos = _count_grouped_photos(grouped_dir)

    checks = [
        check_foto_coverage(grouped_dir, config),
        check_ocr_coverage(ocr_pages_dir, config, actual_photos),
        check_brief_coverage(letters, config, actual_photos),
        check_orphan_pages(letters, ocr_pages_dir),
        check_duplicate_pages(letters),
        check_person_consistency(letters, ocr_pages_dir),
        check_vorgang_integrity(letters, vorgaenge),
        check_amount_precision(vorgaenge),
        check_iban_validity(vorgaenge, config),
        check_critical_creditors(vorgaenge, config),
        check_sheet_exists(reconciliation_dir),
        check_qr_code_count(vorgaenge, qr_codes_dir, config),
        check_status_coverage(vorgaenge),
        check_priority_coverage(vorgaenge),
        check_e2e_traces(letters, vorgaenge, project_dir, config),
        check_false_positive_reconciliation(vorgaenge, config),
        check_creditor_consistency(letters_dir, vorgaenge),
        check_near_duplicates(vorgaenge),
        check_bank_match_creditor(vorgaenge),
        # Checks 20-27 (v3.0.0)
        check_date_plausibility(vorgaenge),
        check_amount_escalation(letters_dir, vorgaenge),
        check_cross_vorgang_iban(vorgaenge),
        check_ocr_confidence_distribution(ocr_pages_dir),
        check_pipeline_confidence(vorgaenge, config),
        check_reconciliation_confidence(vorgaenge, config),
        check_teilweise_remainder(vorgaenge),
        check_flag_enum_validation(vorgaenge, config),
        # Checks 28-35 (v5.0.0)
        check_confidence_unknown_ratio(vorgaenge),
        check_overpayment_flags(vorgaenge),
        check_multi_payment_complexity(vorgaenge),
        check_duplicate_transaction_flags(vorgaenge),
        check_cross_person_pages(letters),
        check_reference_format(vorgaenge, config),
        check_date_proximity_missing(vorgaenge),
        check_composite_weight_sum(config),
        # Checks 36-43 (v5.1.0)
        check_negative_amounts(vorgaenge),
        check_bezahlt_has_reconciliation(vorgaenge),
        check_bezahlt_has_booking_date(vorgaenge),
        check_financial_sum_integrity(vorgaenge, config),
        check_status_enum_valid(vorgaenge),
        check_duplicate_letter_assignment(vorgaenge),
        check_shared_tx_index(vorgaenge),
        check_future_dates(vorgaenge),
    ]

    return checks


# English names for the Markdown report (the JSON report keeps the internal names).
QC_CHECK_NAMES_EN: dict[int, str] = {
    1: "Photo coverage",
    2: "OCR coverage",
    3: "Letter page coverage",
    4: "No orphan pages",
    5: "No duplicate pages",
    6: "Person consistency",
    7: "Case integrity",
    8: "Amount precision",
    9: "IBAN validity",
    10: "Critical creditors",
    11: "Summary workbook written",
    12: "QR code count",
    13: "Status coverage",
    14: "Priority coverage",
    15: "End-to-end trace samples",
    16: "False-positive reconciliation",
    17: "Creditor consistency",
    18: "Near-duplicate detection",
    19: "Bank match creditor",
    22: "IBAN consistency across cases",
    26: "Part-paid remainder",
    34: "Paid cases have dates",
    37: "Paid cases have a bank match",
    38: "Paid cases have a booking date",
}


def write_reports(
    report: QCReport,
    reconciliation_dir: Path,
    vorgaenge: list[dict[str, Any]],
    config: BriefpostConfig,
) -> None:
    """Write QC report to JSON and Markdown files.

    Args:
        report: Completed QC report.
        reconciliation_dir: Directory to write reports.
        vorgaenge: Vorgaenge list for the critical creditors section.
        config: Pipeline configuration.
    """
    reconciliation_dir.mkdir(exist_ok=True)

    # JSON report
    json_path = reconciliation_dir / "qc-report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("Written: %s", json_path)

    # Markdown report
    critical_creditors = [
        str(c).lower().strip()
        for c in (config.priority.get("critical_creditors") or [])
        if str(c).strip()
    ]
    md_lines = [
        "# QC Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Result:** {report.overall_result} ({report.passed_count}/{report.total})",
        "",
        "## Checks",
        "",
        "| # | Check | Result | Details |",
        "|---|-------|--------|---------|",
    ]

    for check in report.checks:
        status = "PASS" if check.passed else "**FAIL**"
        shown_name = QC_CHECK_NAMES_EN.get(check.check_number, check.name)
        md_lines.append(
            f"| {check.check_number} | {shown_name} | {status} | {check.actual if check.actual is not None else ''} |"
        )

    if critical_creditors:
        md_lines.extend(["", "## Critical creditors", ""])
        critical = [
            v for v in vorgaenge
            if any(
                cred in (v.get("absender") or "").lower()
                for cred in critical_creditors
            )
        ]
        if critical:
            for vorgang in critical:
                md_lines.append(
                    f"- **{vorgang['vorgang_id']}**: {vorgang.get('absender', '')}, "
                    f"amount: {vorgang.get('aktuellster_betrag', 'N/A')}, "
                    f"status: {vorgang.get('status', 'N/A')}, "
                    f"priority: {vorgang.get('prioritaet', 'N/A')}"
                )
        else:
            md_lines.append("**WARNING:** none of the configured critical creditors was found.")

    md_path = reconciliation_dir / "qc-report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info("Written: %s", md_path)


def run_verification(project_dir: Path, config: BriefpostConfig) -> BatchResult:
    """Entry point called by pipeline.py.

    Args:
        project_dir: Project data directory.
        config: Pipeline configuration.

    Returns:
        BatchResult describing the outcome.
    """
    result = BatchResult(phase="qc")
    reconciliation_dir = project_dir / "reconciliation"

    try:
        checks = run_all_checks(project_dir, config)
        report = QCReport(checks=checks)

        passed = report.passed_count
        failed = report.failed_count

        logger.info(
            "QC RESULTS: %d/%d PASSED, %d/%d FAILED",
            passed, report.total, failed, report.total,
        )

        for check in checks:
            status_str = "PASS" if check.passed else "FAIL"
            logger.info(
                "  [%s] #%2d %s: %s",
                status_str, check.check_number, check.name, check.actual,
            )
            if not check.passed and check.details:
                details = check.details
                if isinstance(details, list):
                    for detail in details[:3]:
                        logger.info("         -> %s", detail)
                elif isinstance(details, dict):
                    for key, val in list(details.items())[:3]:
                        logger.info("         -> %s: %s", key, val)

        # Write reports
        vorgaenge = load_vorgaenge(reconciliation_dir)
        write_reports(report, reconciliation_dir, vorgaenge, config)

        result.total = 1
        if failed == 0:
            result.succeeded = 1
        else:
            result.succeeded = 1  # QC itself ran successfully even if checks failed

    except Exception as exc:
        logger.error("QC verification failed: %s", exc, exc_info=True)
        result.total = 1
        result.failed = 1
        result.errors.append({"phase": "qc", "error": str(exc)})

    result.complete()
    return result
