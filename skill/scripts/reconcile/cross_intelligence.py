"""Automated cross-intelligence gap-filling.

Automates the forensic fixes from apply_v11_fixes.py and apply_v11b_fixes.py:
1. Infer person from creditor patterns
2. Look up IBAN from bank transactions
3. Fill betrag from related Vorgaenge (same Aktenzeichen)
4. Fill Aktenzeichen from bank CSV references
5. Categorize unfillable gaps with proper flags
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core.confidence import get_cross_intel_confidence as _get_phase_confidence
from core.config import BriefpostConfig
from core.constants import DocumentType, Flag, Status
from core.io import atomic_write_json
from core.models import BankTransaction, BatchResult
from core.utils import normalize_creditor as _normalize_creditor

logger = logging.getLogger(__name__)

def infer_person_from_creditor(
    vorgaenge: list[dict[str, Any]],
    config: BriefpostConfig,
) -> list[dict[str, Any]]:
    """Infer person for null-person Vorgaenge from creditor patterns.

    If creditor X has 5 Vorgaenge with person=A and 1 with person=null,
    infer the null as A (majority vote with confidence).

    Args:
        vorgaenge: List of Vorgang dicts.
        config: Pipeline configuration.

    Returns:
        List of change dicts with {vorgang_id, field, old_value, new_value, confidence, evidence}.
    """
    # Build creditor -> person mapping
    creditor_persons: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for vorgang in vorgaenge:
        creditor = _normalize_creditor(vorgang.get("absender", ""))
        person = vorgang.get("person")
        if creditor and person:
            creditor_persons[creditor][person] += 1

    changes: list[dict[str, Any]] = []
    for vorgang in vorgaenge:
        if vorgang.get("person") is not None:
            continue

        creditor = _normalize_creditor(vorgang.get("absender", ""))
        if not creditor or creditor not in creditor_persons:
            continue

        person_counts = creditor_persons[creditor]
        if not person_counts:
            continue

        # Majority vote
        total = sum(person_counts.values())
        best_person = max(person_counts, key=person_counts.get)
        confidence = person_counts[best_person] / total

        # F-CI-01: Require at least 2 data points for person inference
        # A singleton vote (1/1 = 100%) trivially passes any threshold
        if total < 2:
            continue
        # Only apply if confidence >= 0.6 (at least 60% of cases agree)
        if confidence < 0.6:
            continue

        vorgang["person"] = best_person
        # Remove NEEDS_VISION:PERSON or MISSING_PERSON flags
        vorgang["flags"] = [
            f for f in vorgang.get("flags", [])
            if "NEEDS_VISION:PERSON" not in f and f != Flag.MISSING_PERSON.value
        ]
        vorgang.setdefault("flags", []).append(
            f"{Flag.PERSON_INFERRED.value}:{creditor}_{confidence:.2f}"
        )

        changes.append({
            "vorgang_id": vorgang["vorgang_id"],
            "field": "person",
            "old_value": None,
            "new_value": best_person,
            "confidence": round(confidence, 2),
            "evidence": f"creditor '{vorgang.get('absender')}' has {person_counts[best_person]}/{total} {best_person}",
        })

    logger.info("Person inference: %d changes", len(changes))
    return changes


def lookup_iban_from_bank(
    vorgaenge: list[dict[str, Any]],
    transactions: list[dict[str, Any]],
    config: BriefpostConfig,
) -> list[dict[str, Any]]:
    """Reverse-match bank transactions to fill missing IBANs.

    Checks if a bank tx Verwendungszweck contains a Vorgang's Aktenzeichen.
    If found, fills IBAN from the bank tx's beneficiary IBAN.

    Args:
        vorgaenge: List of Vorgang dicts.
        transactions: List of bank transaction dicts.
        config: Pipeline configuration.

    Returns:
        List of change dicts.
    """
    ref_min_length = config.reconciliation.get("reference_min_length", 8)
    changes: list[dict[str, Any]] = []

    # Build reference -> transaction index
    tx_by_reference: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for tx in transactions:
        details = (tx.get("payment_details") or "").lower()
        ref = (tx.get("customer_reference") or "").lower()
        combined = f"{details} {ref}"
        tx_by_reference[combined].append(tx)

    for vorgang in vorgaenge:
        if vorgang.get("iban"):
            continue  # Already has IBAN

        # Check all references for this Vorgang
        refs_to_check = vorgang.get("alle_referenzen", [])
        aktenzeichen = vorgang.get("aktenzeichen")
        if aktenzeichen and aktenzeichen not in refs_to_check:
            refs_to_check = [aktenzeichen] + refs_to_check

        for ref in refs_to_check:
            if not ref or len(ref.strip()) < ref_min_length:
                continue

            ref_lower = ref.lower().strip()

            # Search through all transactions
            for combined_text, tx_list in tx_by_reference.items():
                if ref_lower in combined_text:
                    # Found a match - use the first tx's IBAN
                    tx = tx_list[0]
                    tx_iban = tx.get("iban", "")
                    if tx_iban:
                        vorgang["iban"] = tx_iban
                        vorgang["flags"] = [
                            f for f in vorgang.get("flags", [])
                            if "NEEDS_VISION:IBAN" not in f
                        ]
                        vorgang.setdefault("flags", []).append(
                            f"{Flag.BANK_PAYMENT_FOUND.value}:{tx.get('booking_date', '')}_{tx_iban}"
                        )

                        # C10: Dynamic confidence: use ref length relative to details text
                        details_text = tx.get("payment_details", "")
                        match_ratio = len(ref_lower) / max(len(details_text), 1)
                        base = _get_phase_confidence("iban_lookup")
                        # Scale linearly from floor (shortest match) to base (longest match)
                        # Cap ratio at 1.0 to avoid boosting above base
                        floor = 0.60
                        capped_ratio = min(match_ratio, 1.0)
                        phase_confidence = round(floor + capped_ratio * (base - floor), 4)

                        changes.append({
                            "vorgang_id": vorgang["vorgang_id"],
                            "field": "iban",
                            "old_value": None,
                            "new_value": tx_iban,
                            "confidence": phase_confidence,
                            "evidence": f"bank tx ref '{ref}' matched in payment details (ratio={match_ratio:.2f})",
                        })
                        break
            else:
                continue
            break  # IBAN was filled, move to next Vorgang

    logger.info("IBAN lookup from bank: %d changes", len(changes))
    return changes


def fill_betrag_from_related(
    vorgaenge: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fill zero-betrag Vorgaenge from related Vorgaenge with same Aktenzeichen.

    If V-082 (vollstreckung, 598.08) and V-086 (pfaendung, 0) share the
    same Aktenzeichen, V-086 gets betrag from V-082.

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        List of change dicts.
    """
    # Build Aktenzeichen -> Vorgaenge mapping
    az_vorgaenge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vorgang in vorgaenge:
        az = vorgang.get("aktenzeichen")
        if az:
            az_vorgaenge[az].append(vorgang)
        for ref in vorgang.get("alle_referenzen", []):
            if ref and ref != az:
                az_vorgaenge[ref].append(vorgang)

    changes: list[dict[str, Any]] = []

    for vorgang in vorgaenge:
        betrag = vorgang.get("aktuellster_betrag") or 0
        if betrag > 0:
            continue  # Already has amount

        az = vorgang.get("aktenzeichen")
        if not az:
            continue

        # Find related Vorgaenge with amounts
        related = az_vorgaenge.get(az, [])
        best_betrag = 0.0
        best_source = ""

        for related_vorgang in related:
            if related_vorgang["vorgang_id"] == vorgang["vorgang_id"]:
                continue
            related_betrag = related_vorgang.get("aktuellster_betrag") or 0
            if related_betrag > best_betrag:
                best_betrag = related_betrag
                best_source = related_vorgang["vorgang_id"]

        if best_betrag > 0:
            vorgang["aktuellster_betrag"] = best_betrag
            vorgang["flags"] = [
                f for f in vorgang.get("flags", [])
                if f not in (Flag.MISSING_AMOUNT.value, Flag.MISSING_BETRAG.value)
                and "NEEDS_VISION:BETRAG" not in f
            ]
            vorgang.setdefault("flags", []).append(
                f"{Flag.BETRAG_FROM_RELATED.value}:{best_source}"
            )

            # Dynamic confidence: more related sources + less variance → higher
            related_amounts = [
                rv.get("aktuellster_betrag", 0)
                for rv in related
                if rv["vorgang_id"] != vorgang["vorgang_id"]
                and (rv.get("aktuellster_betrag") or 0) > 0
            ]
            base = _get_phase_confidence("betrag_from_related")
            if len(related_amounts) >= 2:
                mean_amt = sum(related_amounts) / len(related_amounts)
                variance = sum((a - mean_amt) ** 2 for a in related_amounts) / len(related_amounts)
                cv = (variance ** 0.5) / mean_amt if mean_amt > 0 else 1.0
                # Low variance (cv < 0.1) → full confidence; high variance → penalize
                variance_penalty = min(cv * 0.3, 0.20)
                phase_confidence = round(max(0.50, base - variance_penalty), 4)
            else:
                # Single source: use base confidence
                phase_confidence = base

            changes.append({
                "vorgang_id": vorgang["vorgang_id"],
                "field": "aktuellster_betrag",
                "old_value": 0,
                "new_value": best_betrag,
                "confidence": phase_confidence,
                "evidence": f"same AZ '{az}' as {best_source} (betrag={best_betrag}, sources={len(related_amounts)})",
            })

    logger.info("Betrag from related: %d changes", len(changes))
    return changes


def fill_aktenzeichen_from_bank(
    vorgaenge: list[dict[str, Any]],
    transactions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fill missing Aktenzeichen from bank transaction references.

    If a bank tx reference field matches creditor name, extract the AZ
    from the tx reference.

    Args:
        vorgaenge: List of Vorgang dicts.
        transactions: List of bank transaction dicts.

    Returns:
        List of change dicts.
    """
    changes: list[dict[str, Any]] = []

    for vorgang in vorgaenge:
        if vorgang.get("aktenzeichen"):
            continue  # Already has AZ

        creditor = _normalize_creditor(vorgang.get("absender", ""))
        if not creditor or len(creditor) < 3:
            continue

        # Search transactions for matching creditor
        for tx in transactions:
            beneficiary = (tx.get("beneficiary") or "").lower()
            if creditor not in beneficiary and beneficiary not in creditor:
                continue

            # Extract reference from tx
            ref = tx.get("customer_reference", "").strip()
            details = tx.get("payment_details", "").strip()

            # Try customer_reference first
            if ref and len(ref) >= 5:
                # Clean reference (remove common suffixes)
                if "-" in ref:
                    parts = ref.rsplit("-", 1)
                    suffix = parts[1].strip()
                    # Only strip if suffix is short and non-numeric (e.g., "-FinA")
                    if len(suffix) < 5 and not suffix.isdigit():
                        clean_ref = parts[0].strip()
                    else:
                        clean_ref = ref
                else:
                    clean_ref = ref
                if len(clean_ref) >= 5:
                    vorgang["aktenzeichen"] = clean_ref
                    if clean_ref not in vorgang.get("alle_referenzen", []):
                        vorgang.setdefault("alle_referenzen", []).append(clean_ref)
                    vorgang["flags"] = [
                        f for f in vorgang.get("flags", [])
                        if "NEEDS_VISION:AKTENZEICHEN" not in f
                        and f != Flag.MISSING_AKTENZEICHEN.value
                    ]

                    # Dynamic confidence: longer AZ relative to ref → higher
                    ref_len = len(ref.strip())
                    az_len = len(clean_ref)
                    length_ratio = az_len / max(ref_len, 1)
                    base = _get_phase_confidence("aktenzeichen_from_bank")
                    # High ratio (az is most of ref) → boost; low ratio → penalize
                    phase_confidence = round(max(0.50, base * min(length_ratio, 1.0)), 4)

                    changes.append({
                        "vorgang_id": vorgang["vorgang_id"],
                        "field": "aktenzeichen",
                        "old_value": None,
                        "new_value": clean_ref,
                        "confidence": phase_confidence,
                        "evidence": f"bank tx ref '{ref}' for creditor '{vorgang.get('absender')}' (ratio={length_ratio:.2f})",
                    })
                    break

    logger.info("Aktenzeichen from bank: %d changes", len(changes))
    return changes


def categorize_unfillable(
    vorgaenge: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Categorize remaining null fields as unfillable with proper flags.

    Analyzes remaining gaps and classifies them:
    - DATA_NOT_APPLICABLE: DSGVO fragments, info letters, no-payment docs
    - IBAN_NOT_ON_AMOUNT_PAGE: cover page not photographed
    - IBAN_MANUAL_LOOKUP_REQUIRED: internal bank accounts
    - BETRAG_NOT_IN_DOCUMENT: warning letters without stated amounts

    Args:
        vorgaenge: List of Vorgang dicts.

    Returns:
        List of categorization change dicts.
    """
    changes: list[dict[str, Any]] = []

    info_doc_types = {
        DocumentType.INFO.value, DocumentType.KONTOAUSZUG.value,
        DocumentType.VERSICHERUNG.value, DocumentType.VERTRAG.value,
    }

    for vorgang in vorgaenge:
        flags = vorgang.get("flags", [])
        doc_type = vorgang.get("dokument_typ", "")

        # Skip already-categorized
        has_dna = any("DATA_NOT_APPLICABLE" in f for f in flags)
        if has_dna:
            continue

        categorized = False

        # Info letters: no payment expected
        if doc_type in info_doc_types:
            new_flags = []
            if not vorgang.get("aktuellster_betrag"):
                new_flags.append(f"{Flag.DATA_NOT_APPLICABLE.value}:BETRAG")
            if not vorgang.get("iban"):
                new_flags.append(f"{Flag.DATA_NOT_APPLICABLE.value}:IBAN")

            if new_flags:
                for nf in new_flags:
                    if nf not in flags:
                        vorgang.setdefault("flags", []).append(nf)
                        categorized = True

        # DSGVO fragments: privacy notices only
        if any("DSGVO" in f for f in flags):
            for field in ("BETRAG", "IBAN", "PERSON", "AKTENZEICHEN"):
                dna_flag = f"{Flag.DATA_NOT_APPLICABLE.value}:{field}"
                if dna_flag not in flags:
                    vorgang.setdefault("flags", []).append(dna_flag)
                    categorized = True

        # Betrag=0 for warning letters (mahnung without stated amount)
        if (
            not vorgang.get("aktuellster_betrag")
            and doc_type in (DocumentType.MAHNUNG_1.value, DocumentType.VERSICHERUNG.value)
            and not any(Flag.BETRAG_NOT_IN.value in f for f in flags)
        ):
            flag = f"{Flag.BETRAG_NOT_IN.value}:premium_amount_not_stated"
            if flag not in flags:
                vorgang.setdefault("flags", []).append(flag)
                categorized = True

        if categorized:
            changes.append({
                "vorgang_id": vorgang["vorgang_id"],
                "field": "categorization",
                "old_value": None,
                "new_value": "DATA_NOT_APPLICABLE",
                "confidence": _get_phase_confidence("unfillable_categorization"),
                "evidence": f"doc_type={doc_type}, flags added",
            })

    logger.info("Unfillable categorization: %d changes", len(changes))
    return changes


def run_all_intelligence(
    vorgaenge: list[dict[str, Any]],
    transactions: list[dict[str, Any]],
    config: BriefpostConfig,
) -> dict[str, Any]:
    """Orchestrate all 5 intelligence functions in sequence.

    Applies changes to vorgaenge in-place and returns a full audit trail.

    Args:
        vorgaenge: List of Vorgang dicts (modified in-place).
        transactions: List of bank transaction dicts.
        config: Pipeline configuration.

    Returns:
        Intelligence report dict with all changes and statistics.
    """
    all_changes: list[dict[str, Any]] = []

    # Phase 1: Person inference
    person_changes = infer_person_from_creditor(vorgaenge, config)
    all_changes.extend(person_changes)

    # Phase 2: IBAN lookup from bank
    iban_changes = lookup_iban_from_bank(vorgaenge, transactions, config)
    all_changes.extend(iban_changes)

    # Phase 3: Betrag from related Vorgaenge
    betrag_changes = fill_betrag_from_related(vorgaenge)
    all_changes.extend(betrag_changes)

    # Phase 4: Aktenzeichen from bank
    az_changes = fill_aktenzeichen_from_bank(vorgaenge, transactions)
    all_changes.extend(az_changes)

    # Phase 5: Categorize unfillable
    unfillable_changes = categorize_unfillable(vorgaenge)
    all_changes.extend(unfillable_changes)

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_changes": len(all_changes),
        "by_phase": {
            "person_inference": len(person_changes),
            "iban_lookup": len(iban_changes),
            "betrag_from_related": len(betrag_changes),
            "aktenzeichen_from_bank": len(az_changes),
            "unfillable_categorization": len(unfillable_changes),
        },
        "changes": all_changes,
    }

    logger.info(
        "Cross-intelligence complete: %d total changes "
        "(person=%d, iban=%d, betrag=%d, az=%d, unfillable=%d)",
        len(all_changes),
        len(person_changes), len(iban_changes),
        len(betrag_changes), len(az_changes),
        len(unfillable_changes),
    )

    return report


def _load_bank_transactions(
    project_dir: Path,
    config: BriefpostConfig,
) -> list[dict[str, Any]]:
    """Load bank transactions from CSV files.

    Args:
        project_dir: Project data directory.
        config: Pipeline configuration.

    Returns:
        List of transaction dicts.
    """
    bank_dir = project_dir / "bank-statements"
    if not bank_dir.exists():
        logger.warning("bank-statements/ not found: %s", bank_dir)
        return []

    # Reuse the reconciler's CSV parser so both phases read the same rows.
    from reconcile.bank_reconciler import load_bank_transactions

    transactions: list[dict[str, Any]] = []
    for tx in load_bank_transactions(bank_dir, config):
        transactions.append({
            "booking_date": tx.get("booking_date"),
            "beneficiary": tx.get("beneficiary", ""),
            "payment_details": tx.get("payment_details", ""),
            "iban": tx.get("iban", ""),
            "customer_reference": tx.get("customer_reference", ""),
            "debit": tx.get("debit"),
            "credit": tx.get("credit"),
            "source_account_iban": tx.get("source_account_iban", ""),
        })

    logger.info("Total bank transactions loaded: %d", len(transactions))
    return transactions


def run_cross_intelligence(project_dir: Path, config: BriefpostConfig) -> BatchResult:
    """Entry point called by pipeline.py.

    Args:
        project_dir: Project data directory.
        config: Pipeline configuration.

    Returns:
        BatchResult describing the outcome.
    """
    result = BatchResult(phase="cross_intelligence")
    reconciliation_dir = project_dir / "reconciliation"

    # Load Vorgaenge
    vorgaenge_path = reconciliation_dir / "vorgaenge-reconciled.json"
    if not vorgaenge_path.exists():
        vorgaenge_path = reconciliation_dir / "vorgaenge.json"
    if not vorgaenge_path.exists():
        logger.error("No vorgaenge file found in %s", reconciliation_dir)
        result.total = 1
        result.failed = 1
        result.errors.append({"error": "No vorgaenge file found"})
        result.complete()
        return result

    with open(vorgaenge_path) as f:
        data = json.load(f)
    vorgaenge = data.get("vorgaenge", [])
    logger.info("Loaded %d Vorgaenge from %s", len(vorgaenge), vorgaenge_path.name)

    # Load bank transactions
    transactions = _load_bank_transactions(project_dir, config)

    # RS-012: Checkpoint support for resume after crash
    checkpoint_path = reconciliation_dir / ".cross-intel-checkpoint.json"
    # RS-036: Clean up stale temp files from previous runs
    for stale_file in reconciliation_dir.glob(".cross-intel-*.tmp"):
        try:
            stale_age = datetime.now().timestamp() - stale_file.stat().st_mtime
            if stale_age > 3600:  # Older than 1 hour
                stale_file.unlink()
                logger.info("RS-036: Cleaned up stale temp file %s (age: %.0fs)", stale_file.name, stale_age)
        except OSError:
            pass

    try:
        # Run all intelligence
        report = run_all_intelligence(vorgaenge, transactions, config)

        # RS-012: Save checkpoint after intelligence completes
        try:
            atomic_write_json(checkpoint_path, {
                "phase": "cross_intel_complete",
                "total_changes": report.get("total_changes", 0),
                "timestamp": datetime.now().isoformat(),
            })
        except OSError:
            logger.warning("RS-012: Could not write cross-intelligence checkpoint")

        # Save updated Vorgaenge
        data["vorgaenge"] = vorgaenge
        data["cross_intelligence"] = report
        data["modified"] = datetime.now().isoformat()

        output_path = reconciliation_dir / "vorgaenge-cross-intel.json"
        atomic_write_json(output_path, data)
        logger.info("Saved updated Vorgaenge to %s", output_path)

        # Save intelligence report separately
        report_path = reconciliation_dir / "cross-intelligence-report.json"
        atomic_write_json(report_path, report)
        logger.info("Intelligence report saved to %s", report_path)

        result.total = 1
        result.succeeded = 1

        # RS-012: Clean up checkpoint on success
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    except Exception as exc:
        logger.error("Cross-intelligence failed: %s", exc, exc_info=True)
        result.total = 1
        result.failed = 1
        result.errors.append({"phase": "cross_intelligence", "error": str(exc)})

    result.complete()
    return result
