"""Phase D: Bank reconciliation - matches Vorgaenge against bank transactions.

Three-phase matching:
  Phase 1: Reference match (confidence 0.95) - Aktenzeichen/reference in payment details
  Phase 2: Amount match (confidence 0.85) - debit amount within tolerance + creditor check
  Phase 3: Multi-payment accumulation - sum of multiple bank tx for same creditor

Uses the built-in CSV parser (most bank export layouts) for bank CSV parsing with
manual fallback for edge cases.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from core.confidence import compute_recon_confidence
from core.config import BriefpostConfig
from core.constants import Flag, Priority, ReconciliationMethod, Status
from core.interfaces import BankReconciler
from core.io import atomic_write_json
from core.llm_helpers import call_gemini_flash
from core.models import BankTransaction, BatchResult, ReconciliationResult, Vorgang
from core.utils import normalize_creditor

logger = logging.getLogger(__name__)

# Finance-manager skill path


def _creditor_matches(
    beneficiary: str,
    absender: str,
    config: Optional[BriefpostConfig] = None,
) -> bool:
    """Check if bank beneficiary matches Vorgang creditor.

    Uses substring containment first. Falls back to fuzzy Jaro-Winkler
    matching when enable_fuzzy_matching is true in reconciliation config.

    Args:
        beneficiary: Bank transaction beneficiary name.
        absender: Vorgang creditor name.
        config: Optional pipeline config for fuzzy matching settings.

    Returns:
        True if names match via substring containment or fuzzy matching.
        False if either name is empty or too short for reliable matching.
    """
    norm_ben = normalize_creditor(beneficiary)
    norm_abs = normalize_creditor(absender)
    if not norm_ben or not norm_abs:
        return False  # Cannot verify = reject match
    if len(norm_ben) < 3 or len(norm_abs) < 3:
        return False  # Too short for reliable matching
    # RS-037: Cap normalized names at 100 chars to prevent slow fuzzy matching
    norm_ben = norm_ben[:100]
    norm_abs = norm_abs[:100]
    if norm_ben in norm_abs or norm_abs in norm_ben:
        return True
    # Fuzzy fallback via Jaro-Winkler
    if config is not None:
        recon_config = config.reconciliation
        if recon_config.get("enable_fuzzy_matching", False):
            from core.utils import fuzzy_creditor_match
            threshold = recon_config.get("fuzzy_threshold", 0.90)
            if fuzzy_creditor_match(beneficiary[:200], absender[:200], threshold):
                return True
    return False


def _detect_account_from_config(filename: str, config: BriefpostConfig) -> dict:
    """Detect bank account from filename using config.

    Args:
        filename: Bank CSV filename.
        config: Pipeline configuration with person accounts.

    Returns:
        Dict with iban, person, label keys or empty dict.
    """
    accounts = config.persons.get("accounts", {})
    for person, accts in accounts.items():
        if isinstance(accts, list):
            for acct in accts:
                if isinstance(acct, dict):
                    label = acct.get("label", "")
                    if label and label in filename:
                        return {"iban": acct["iban"], "person": person, "label": label}
    return {}


def detect_account_iban(filename: str, config: Optional[BriefpostConfig] = None) -> str:
    """Detect account IBAN from filename pattern.

    Uses the account labels in config (persons.accounts[].label).

    Args:
        filename: Bank CSV filename.
        config: Optional pipeline configuration.

    Returns:
        IBAN string or empty string if not detected.
    """
    if config:
        detected = _detect_account_from_config(filename, config)
        if detected:
            return detected["iban"]

    return ""


def detect_account_owner(iban: str, config: BriefpostConfig) -> str:
    """Detect account owner from IBAN using config.

    Args:
        iban: Account IBAN.
        config: Pipeline configuration with person accounts.

    Returns:
        Person name or "Unknown".
    """
    accounts = config.persons.get("accounts", {})
    for person, person_accounts in accounts.items():
        if isinstance(person_accounts, list):
            for acct in person_accounts:
                acct_iban = acct.get("iban", "") if isinstance(acct, dict) else str(acct)
                if acct_iban == iban:
                    return person
    return "Unknown"


def load_bank_transactions(
    bank_dir: Path,
    config: BriefpostConfig,
) -> list[dict[str, Any]]:
    """Load all bank transactions from CSVs.

    Uses the built-in CSV parser (see _manual_parse_csvs).

    Args:
        bank_dir: Directory containing bank CSV files.
        config: Pipeline configuration.

    Returns:
        List of transaction dicts.
    """
    all_transactions: list[dict[str, Any]] = []
    csv_files = sorted(bank_dir.glob("*.csv"))

    if not csv_files:
        logger.warning("No bank CSV files found in %s; every case stays open", bank_dir)
        return []

    all_transactions = _manual_parse_csvs(csv_files, config)

    logger.info("Total transactions: %d", len(all_transactions))
    return all_transactions


def _manual_parse_csvs(
    csv_files: list[Path],
    config: BriefpostConfig,
) -> list[dict[str, Any]]:
    """Parse bank CSV exports (header row, delimiter and columns are detected).

    Args:
        csv_files: List of CSV file paths.
        config: Pipeline configuration.

    Returns:
        List of transaction dicts.
    """
    bank_config = config.bank
    encoding_fallback = bank_config.get("encoding_fallback", ["utf-8-sig", "utf-8", "latin-1", "cp1252"])
    delimiter = bank_config.get("delimiter", ";")

    transactions: list[dict[str, Any]] = []

    for csv_path in csv_files:
        account_iban = detect_account_iban(csv_path.name, config)
        account_owner = detect_account_owner(account_iban, config)

        raw = None
        # RS-030: Sniff BOM to select correct encoding before fallback chain
        try:
            with open(csv_path, "rb") as bf:
                bom_bytes = bf.read(4)
            if bom_bytes[:3] == b'\xef\xbb\xbf':
                encoding_fallback = ["utf-8-sig"] + [e for e in encoding_fallback if e != "utf-8-sig"]
            elif bom_bytes[:2] in (b'\xff\xfe', b'\xfe\xff'):
                encoding_fallback = ["utf-16"] + list(encoding_fallback)
        except OSError:
            pass
        for encoding in encoding_fallback:
            try:
                with open(csv_path, encoding=encoding) as f:
                    raw = f.read()
                # RS-030: Verify that decoded text contains expected characters
                # latin-1 never raises UnicodeDecodeError but may garble umlauts
                if encoding in ("latin-1", "cp1252") and raw:
                    # If file has UTF-8 multi-byte sequences decoded as latin-1,
                    # common German umlauts appear as Ã¤, Ã¶, Ã¼ instead of ae, oe, ue
                    if "\xc3\xa4" in raw or "\xc3\xb6" in raw or "\xc3\xbc" in raw:
                        logger.warning("RS-030: File %s likely UTF-8 but decoded as %s, retrying",
                                       csv_path.name, encoding)
                        continue
                break
            except UnicodeDecodeError:
                continue

        if not raw:
            logger.error("Could not read %s", csv_path.name)
            continue

        lines = raw.splitlines()

        layout = _detect_csv_layout(lines, bank_config)
        if layout is None:
            logger.error(
                "CSV header detection failed for %s: no row with a date column and an "
                "amount column was found. Set bank.columns in config.yaml.",
                csv_path.name,
            )
            continue

        header_row, file_delimiter, columns = layout
        date_format = bank_config.get("date_format")
        reader = csv.reader(lines[header_row:], delimiter=file_delimiter)
        headers = [h.strip() for h in next(reader)]
        file_count = 0

        for row in reader:
            if len(row) < len(headers):
                continue
            row_dict = dict(zip(headers, (cell.strip() for cell in row)))

            def cell(role: str) -> str:
                name = columns.get(role)
                return row_dict.get(name, "") if name else ""

            debit = _parse_manual_amount(cell("debit")) if cell("debit") else None
            credit = _parse_manual_amount(cell("credit")) if cell("credit") else None
            if debit is None and credit is None and cell("amount"):
                signed = _parse_manual_amount(cell("amount"))
                if signed is not None:
                    if signed < 0:
                        debit = abs(signed)
                    elif signed > 0:
                        credit = signed

            booking_date = cell("date")
            if not booking_date or (debit is None and credit is None):
                continue  # Not a transaction row (totals, blank lines, footers)
            if date_format:
                try:
                    booking_date = datetime.strptime(booking_date, date_format).strftime("%Y-%m-%d")
                except ValueError:
                    logger.warning("Date %r in %s does not match bank.date_format %r",
                                   booking_date, csv_path.name, date_format)

            payment_details = cell("details")
            beneficiary = cell("payee") or payment_details

            transactions.append({
                "beneficiary": beneficiary,
                "payment_details": payment_details,
                "iban": cell("iban"),
                "debit": debit,
                "credit": credit,
                "amount": (-abs(debit) if debit else abs(credit) if credit else 0),
                "booking_date": booking_date,
                "source_account_iban": account_iban,
                "source_account_owner": account_owner,
            })
            file_count += 1

        logger.info("  -> %d transactions (%s)", file_count, csv_path.name)

    return transactions


# Column-name aliases for common bank exports (compared lower-case).
_CSV_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "date": (
        "booking date", "buchungstag", "buchungsdatum", "transaction date",
        "posting date", "posted date", "post date", "date", "value date", "valuta",
    ),
    "details": (
        "payment details", "verwendungszweck", "description", "transaction description",
        "details", "reference", "memo", "narrative", "notes",
    ),
    "payee": (
        "beneficiary / originator", "auftraggeber / beg\u00fcnstigter", "payee", "merchant",
        "counterparty", "counter party", "name", "beneficiary",
    ),
    "iban": ("iban / account number", "iban", "account number"),
    "debit": (
        "debit", "soll", "paid out", "money out", "withdrawal", "withdrawals",
        "debit amount", "amount debit",
    ),
    "credit": (
        "credit", "haben", "paid in", "money in", "deposit", "deposits",
        "credit amount", "amount credit",
    ),
    "amount": ("amount", "betrag", "value", "transaction amount", "amount (gbp)", "amount (eur)", "amount (usd)"),
}


def _detect_csv_layout(
    lines: list[str],
    bank_config: dict[str, Any],
) -> Optional[tuple[int, str, dict[str, str]]]:
    """Find the header row, delimiter and column mapping of a bank CSV.

    Works for the Deutsche Bank export (German or English headers) and for
    most other bank exports that have a date column plus either separate
    debit and credit columns or one signed amount column (negative = money out).

    Explicit names in config win: bank.columns = {date, details, payee, iban,
    debit, credit, amount}. bank.delimiter is tried first, then ";" "," and tab.

    Args:
        lines: All lines of the CSV file.
        bank_config: The "bank" section of the configuration.

    Returns:
        (header_row_index, delimiter, {role: column_name}) or None if no
        usable header row is found in the first 40 lines.
    """
    configured = {
        role: str(name).strip()
        for role, name in (bank_config.get("columns") or {}).items()
        if name
    }
    delimiters: list[str] = []
    for candidate in (bank_config.get("delimiter"), ";", ",", "\t"):
        if candidate and candidate not in delimiters:
            delimiters.append(candidate)

    for row_index, line in enumerate(lines[:40]):
        for delimiter in delimiters:
            if delimiter not in line:
                continue
            try:
                cells = [c.strip() for c in next(csv.reader([line], delimiter=delimiter))]
            except (csv.Error, StopIteration):
                continue
            by_lower = {c.lower(): c for c in cells if c}
            columns: dict[str, str] = {}
            for role, aliases in _CSV_COLUMN_ALIASES.items():
                if role in configured:
                    if configured[role].lower() in by_lower:
                        columns[role] = by_lower[configured[role].lower()]
                    continue
                for alias in aliases:
                    if alias in by_lower and by_lower[alias] not in columns.values():
                        columns[role] = by_lower[alias]
                        break
            has_amount = "amount" in columns or "debit" in columns or "credit" in columns
            if "date" in columns and has_amount:
                return row_index, delimiter, columns
    return None


def _parse_manual_amount(amount_str: str) -> Optional[float]:
    """Parse amount string handling both German and US formats.

    German format: "1.234,56" (dots as thousands, comma as decimal).
    US format: "1,234.56" or "-3,515" (commas as thousands, dot as decimal).

    Disambiguation when only comma present (no dot):
    - If exactly 3 digits follow the last comma, treat as US thousands separator.
    - Otherwise treat as German decimal separator.

    Args:
        amount_str: Raw amount string.

    Returns:
        Parsed float or None.
    """
    if not amount_str:
        return None
    amount_str = amount_str.strip()
    if ',' in amount_str and '.' in amount_str:
        # Both present: determine order to detect format
        last_comma = amount_str.rfind(',')
        last_dot = amount_str.rfind('.')
        if last_comma > last_dot:
            # German: "1.234,56" → strip dots, comma→dot
            amount_str = amount_str.replace('.', '').replace(',', '.')
        else:
            # US: "1,234.56" → strip commas
            amount_str = amount_str.replace(',', '')
    elif ',' in amount_str:
        # Only comma, no dot: disambiguate
        # Extract digits after the last comma
        last_comma = amount_str.rfind(',')
        after_comma = amount_str[last_comma + 1:].rstrip('-').lstrip('+')
        if len(after_comma) == 3 and after_comma.isdigit():
            # US thousands separator: "-3,515" → "-3515"
            amount_str = amount_str.replace(',', '')
        else:
            # German decimal: "13,99" → "13.99"
            amount_str = amount_str.replace(',', '.')
    else:
        # US format or plain number: "1234.56" or "1234"
        pass
    try:
        return float(amount_str)
    except ValueError:
        return None


def _get_tx_amount(txn: Any) -> float:
    """Get transaction amount (works with both dict and Transaction objects)."""
    if isinstance(txn, dict):
        return txn.get("amount", 0)
    return txn.amount if hasattr(txn, "amount") else 0


def _get_tx_details(txn: Any) -> str:
    """Get transaction payment details."""
    if isinstance(txn, dict):
        return txn.get("payment_details", "")
    return txn.payment_details if hasattr(txn, "payment_details") else ""


def _get_tx_beneficiary(txn: Any) -> str:
    """Get transaction beneficiary/originator name."""
    if isinstance(txn, dict):
        return txn.get("beneficiary", "")
    return txn.beneficiary if hasattr(txn, "beneficiary") else ""


def _get_tx_account(txn: Any) -> str:
    """Get transaction source account IBAN."""
    if isinstance(txn, dict):
        return txn.get("source_account_iban", "")
    return txn.source_account_iban if hasattr(txn, "source_account_iban") else ""


def _is_debit(txn: Any) -> bool:
    """Check if transaction is a debit (outgoing payment)."""
    if isinstance(txn, dict):
        return (txn.get("debit") or 0) != 0
    return txn.is_debit if hasattr(txn, "is_debit") else False


def _get_tx_booking_date(txn: Any) -> Optional[str]:
    """Get transaction booking date string."""
    if isinstance(txn, dict):
        return txn.get("booking_date")
    return txn.booking_date if hasattr(txn, "booking_date") else None


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse a date string in common formats to datetime.

    Supports: YYYY-MM-DD, DD.MM.YYYY, DD/MM/YYYY.

    Args:
        date_str: Date string or None.

    Returns:
        Parsed datetime or None if unparseable.
    """
    if not date_str:
        return None
    date_str = date_str.strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _compute_date_proximity(doc_date_str: Optional[str], tx_date_str: Optional[str]) -> float:
    """Compute date proximity score between document date and transaction date.

    D1: Temporal validation scoring.
    - Within 90 days: 1.0 (full confidence)
    - 90-180 days: 0.90
    - 180-365 days: 0.80
    - >365 days: 0.70

    Args:
        doc_date_str: Vorgang document date.
        tx_date_str: Transaction booking date.

    Returns:
        Proximity score (0.70-1.0), or 1.0 if dates unavailable.
    """
    doc_date = _parse_date(doc_date_str)
    tx_date = _parse_date(tx_date_str)
    if not doc_date or not tx_date:
        return 0.50  # C5: Cannot validate, missing data significantly reduces confidence
    delta_days = abs((tx_date - doc_date).days)
    if delta_days <= 90:
        return 1.0
    elif delta_days <= 180:
        return 0.90
    elif delta_days <= 365:
        return 0.80
    else:
        return 0.70


def _creditor_similarity(
    beneficiary: str,
    absender: str,
    config: Optional[BriefpostConfig] = None,
) -> float:
    """Compute creditor name similarity score for composite scoring.

    Returns 1.0 for match, 0.0 for no match. Uses the same substring
    containment logic as _creditor_matches but returns a float.

    Args:
        beneficiary: Bank transaction beneficiary.
        absender: Vorgang creditor name.
        config: Optional pipeline config for fuzzy matching settings.

    Returns:
        Similarity score 0.0 or 1.0.
    """
    return 1.0 if _creditor_matches(beneficiary, absender, config) else 0.0


def _compute_composite_score(
    amount: float,
    tx_amount: float,
    beneficiary: str,
    absender: str,
    doc_date_str: Optional[str],
    tx_date_str: Optional[str],
    config: Optional[BriefpostConfig] = None,
) -> float:
    """Compute composite match score for Phase 2 best-match selection.

    D2: Score = (1 - amount_diff/amount)*0.5 + creditor_similarity*0.3 + date_proximity*0.2

    Args:
        amount: Vorgang amount.
        tx_amount: Transaction amount (absolute).
        beneficiary: Transaction beneficiary name.
        absender: Vorgang creditor name.
        doc_date_str: Vorgang document date.
        tx_date_str: Transaction booking date.
        config: Optional pipeline config for fuzzy matching settings.

    Returns:
        Composite score (0.0-1.0).
    """
    # C8: Read composite weights from config (defaults match hardcoded)
    weights = {"amount": 0.5, "creditor": 0.3, "date": 0.2}
    if config is not None:
        cfg_weights = config.reconciliation.get("composite_weights", {})
        if cfg_weights:
            weights = {
                "amount": cfg_weights.get("amount", 0.5),
                "creditor": cfg_weights.get("creditor", 0.3),
                "date": cfg_weights.get("date", 0.2),
            }
    if amount <= 0:
        amount_score = 0.0
    else:
        amount_diff_ratio = abs(tx_amount - amount) / amount
        amount_score = max(0.0, 1.0 - amount_diff_ratio)
    cred_score = _creditor_similarity(beneficiary, absender, config)
    date_score = _compute_date_proximity(doc_date_str, tx_date_str)
    return amount_score * weights["amount"] + cred_score * weights["creditor"] + date_score * weights["date"]


def compute_tx_hash(txn: Any) -> str:
    """Compute content-addressing hash for a transaction.

    D5: hash(booking_date + beneficiary + amount + reference).

    Args:
        txn: Transaction dict or object.

    Returns:
        SHA-256 hex digest (first 16 chars).
    """
    booking_date = _get_tx_booking_date(txn) or ""
    beneficiary = _get_tx_beneficiary(txn)
    amount = str(_get_tx_amount(txn))
    reference = _get_tx_details(txn)
    content = f"{booking_date}|{beneficiary}|{amount}|{reference}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _apply_review_flags(vorgang: dict[str, Any], config: BriefpostConfig) -> None:
    """Apply confidence-gated human review flags post-reconciliation.

    D4: Flags BEZAHLT matches that have low confidence or are high-value.

    Args:
        vorgang: Reconciled Vorgang dict (modified in place).
        config: Pipeline configuration.
    """
    recon = vorgang.get("reconciliation", {})
    if not recon or vorgang.get("status") != Status.BEZAHLT.value:
        return

    confidence = recon.get("confidence", 0)
    amount = vorgang.get("aktuellster_betrag", 0) or 0
    flags = vorgang.get("flags", [])

    recon_config = config.reconciliation
    review_threshold = recon_config.get("review_confidence_threshold", 0.90)
    high_value_eur = recon_config.get("high_value_threshold_eur", 1000)
    high_value_conf = recon_config.get("high_value_confidence_threshold", 0.95)

    if confidence < review_threshold:
        flag = f"REVIEW:LOW_CONFIDENCE_BEZAHLT:{confidence:.2f}"
        if flag not in flags:
            flags.append(flag)

    if amount > high_value_eur and confidence < high_value_conf:
        flag = f"REVIEW:HIGH_VALUE_LOW_CONFIDENCE:{amount:.2f}_{confidence:.2f}"
        if flag not in flags:
            flags.append(flag)

    vorgang["flags"] = flags


def _apply_date_mismatch_flag(
    vorgang: dict[str, Any],
    txn: Any,
    config: BriefpostConfig,
) -> None:
    """Apply DATE_MISMATCH_WARNING flag if transaction date is far from document date.

    D1: Flag transactions that are >365 days from document date.

    Args:
        vorgang: Vorgang dict (modified in place).
        txn: Matched transaction.
        config: Pipeline configuration.
    """
    doc_date = _parse_date(vorgang.get("datum"))
    tx_date = _parse_date(_get_tx_booking_date(txn))
    if not doc_date or not tx_date:
        return

    date_tolerance = config.reconciliation.get("date_tolerance_days", 365)
    delta_days = abs((tx_date - doc_date).days)
    if delta_days > date_tolerance:
        flags = vorgang.get("flags", [])
        flag = Flag.DATE_MISMATCH_WARNING.value
        if flag not in flags:
            flags.append(flag)
            vorgang["flags"] = flags


def compute_priority_with_status(vorgang: dict[str, Any], config: BriefpostConfig) -> str:
    """Compute priority considering payment status.

    Args:
        vorgang: Vorgang data dict.
        config: Pipeline configuration.

    Returns:
        Priority string value.
    """
    if vorgang.get("status") == Status.BEZAHLT.value:
        return Priority.ERLEDIGT.value

    priority_config = config.priority
    esc = vorgang.get("hoechste_eskalation", 0)
    creditor = (vorgang.get("absender") or "").lower()
    critical_creditors = [c.lower() for c in priority_config.get("critical_creditors", [])]
    is_critical_creditor = any(cc in creditor for cc in critical_creditors) if critical_creditors else False
    dringend_amount = priority_config.get("dringend_amount", 500)

    if esc >= priority_config.get("gerichtlich_threshold", 4):
        return Priority.GERICHTLICH.value
    elif esc >= priority_config.get("kritisch_threshold", 3) or is_critical_creditor:
        return Priority.KRITISCH.value
    elif esc == 2 or (vorgang.get("aktuellster_betrag", 0) or 0) > dringend_amount:
        return Priority.DRINGEND.value
    else:
        return Priority.NORMAL.value


def _llm_reconcile_vorgang(
    vorgang: dict[str, Any],
    eligible_txns: list[tuple[int, Any]],
    tx_hashes: dict[int, str],
    config: BriefpostConfig,
) -> Optional[dict[str, Any]]:
    """Use Gemini Flash to match a Vorgang against candidate transactions.

    Called after algorithmic phases fail. Filters candidate transactions
    by same person's accounts, within 1 year, debit only, then sends
    top candidates to LLM for semantic matching.

    Args:
        vorgang: Vorgang dict to reconcile.
        eligible_txns: Pre-filtered (idx, txn) pairs from person's accounts.
        tx_hashes: Pre-computed transaction content hashes.
        config: Pipeline configuration.

    Returns:
        Reconciliation match dict, or None if no match found.
    """
    amount = vorgang.get("aktuellster_betrag", 0) or 0
    absender = vorgang.get("absender", "")
    refs = vorgang.get("alle_referenzen", [])
    doc_date_str = vorgang.get("datum")

    # Filter to debit-only candidates within 1 year of doc date
    doc_date = _parse_date(doc_date_str)
    candidates: list[tuple[int, Any]] = []
    for idx, txn in eligible_txns:
        if not _is_debit(txn):
            continue
        tx_date = _parse_date(_get_tx_booking_date(txn))
        if doc_date and tx_date:
            delta = abs((tx_date - doc_date).days)
            if delta > 365:
                continue
        candidates.append((idx, txn))

    if not candidates:
        return None

    # Take top 10 by amount proximity
    candidates.sort(key=lambda c: abs(abs(_get_tx_amount(c[1])) - amount))
    top_candidates = candidates[:10]

    # Build prompt table
    tx_rows: list[str] = []
    for idx, txn in top_candidates:
        tx_date = _get_tx_booking_date(txn) or ""
        tx_amount = abs(_get_tx_amount(txn))
        beneficiary = _get_tx_beneficiary(txn)
        details = _get_tx_details(txn)[:80]
        tx_rows.append(f"| {tx_date} | EUR {tx_amount:.2f} | {beneficiary} | {details} |")

    tx_table = "\n".join(tx_rows)
    refs_str = ", ".join(refs[:5]) if refs else "keine"

    prompt = f"""OFFENER VORGANG:
- Absender: {absender}
- Betrag: EUR {amount:.2f}
- Aktenzeichen: {refs_str}
- Datum: {doc_date_str or "unbekannt"}
- Typ: {vorgang.get("dokument_typ", "unknown")}

BANK-TRANSAKTIONEN (moegliche Zahlungen):
| Datum | Betrag | Empfaenger | Verwendungszweck |
{tx_table}

Welche Transaktion gehoert zu diesem Vorgang?
Begruende deine Entscheidung. Wenn KEINE passt, sage "KEINE ZUORDNUNG".

Antworte mit JSON:
{{
  "match": {{"tx_index": 0, "confidence": 0.85, "reasoning": "..."}}
}}

Oder wenn keine passt:
{{
  "match": null
}}

WICHTIG: tx_index bezieht sich auf die Position in der obigen Tabelle (0-indexed).
Nur zuordnen, wenn du dir sicher bist (confidence >= 0.70)."""

    result = call_gemini_flash(prompt, temperature=0.0)
    if not result:
        return None

    llm_match = result.get("match")
    if not llm_match or llm_match.get("tx_index") is None:
        return None

    candidate_idx = llm_match["tx_index"]
    confidence = llm_match.get("confidence", 0.75)

    # Validate candidate_idx range
    if candidate_idx < 0 or candidate_idx >= len(top_candidates):
        logger.warning("LLM returned invalid tx_index %d (max %d)", candidate_idx, len(top_candidates) - 1)
        return None

    if confidence < 0.70:
        logger.info("LLM match confidence too low (%.2f) for %s", confidence, vorgang.get("vorgang_id"))
        return None

    real_idx, matched_txn = top_candidates[candidate_idx]
    tx_amount = abs(_get_tx_amount(matched_txn))

    return {
        "type": Status.BEZAHLT.value,
        "method": ReconciliationMethod.LLM_MATCH.value,
        "confidence": round(confidence, 4),
        "tx_index": real_idx,
        "tx_amount": tx_amount,
        "amount_diff": round(tx_amount - amount, 2),
        "tx_hash": tx_hashes.get(real_idx, ""),
        "booking_date": _get_tx_booking_date(matched_txn),
        "bank_beneficiary": _get_tx_beneficiary(matched_txn),
        "bank_account": _get_tx_account(matched_txn),
        "llm_reasoning": llm_match.get("reasoning", ""),
    }


def reconcile(
    vorgaenge: list[dict[str, Any]],
    transactions: list[Any],
    config: BriefpostConfig,
) -> tuple[list[dict[str, Any]], set[int]]:
    """Match Vorgaenge against bank transactions.

    Args:
        vorgaenge: List of Vorgang dicts.
        transactions: List of transaction objects/dicts.
        config: Pipeline configuration.

    Returns:
        Tuple of (reconciled vorgaenge, set of matched transaction indices).
    """
    recon_config = config.reconciliation
    amount_tolerance = recon_config.get("amount_tolerance_eur", 0.50)
    reference_confidence = recon_config.get("reference_confidence", 0.95)
    amount_date_confidence = recon_config.get("amount_date_confidence", 0.85)
    reference_min_length = recon_config.get("reference_min_length", 8)
    bezahlt_threshold_pct = recon_config.get("bezahlt_threshold_pct", 99) / 100.0
    teilweise_lower_pct = recon_config.get("teilweise_lower_pct", 90) / 100.0

    # D5: Pre-compute tx_hash for all transactions and detect duplicates
    tx_hashes: dict[int, str] = {}
    hash_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, txn in enumerate(transactions):
        h = compute_tx_hash(txn)
        tx_hashes[i] = h
        hash_to_indices[h].append(i)

    duplicate_tx_hashes: set[str] = {
        h for h, indices in hash_to_indices.items() if len(indices) > 1
    }
    if duplicate_tx_hashes:
        logger.warning(
            "D5: %d duplicate transaction hashes detected across %d transactions",
            len(duplicate_tx_hashes),
            sum(len(hash_to_indices[h]) for h in duplicate_tx_hashes),
        )

    # C9: Date-aware duplicate detection: flag transactions with same hash AND same date
    duplicate_tx_indices: set[int] = set()
    for h in duplicate_tx_hashes:
        indices = hash_to_indices[h]
        date_groups: dict[str, list[int]] = defaultdict(list)
        for idx in indices:
            booking = _get_tx_booking_date(transactions[idx]) or "unknown"
            date_groups[booking].append(idx)
        for date_key, group_indices in date_groups.items():
            if len(group_indices) > 1:
                # Same hash AND same date = true duplicate
                duplicate_tx_indices.update(group_indices)
    if duplicate_tx_indices:
        logger.warning("C9: %d true duplicate transactions (same hash + same date)", len(duplicate_tx_indices))

    # Build person-accounts mapping
    person_accounts: dict[str, list[str]] = {}
    accounts_config = config.persons.get("accounts", {})
    for person, accts in accounts_config.items():
        if isinstance(accts, list):
            person_accounts[person] = [
                a.get("iban", "") if isinstance(a, dict) else str(a) for a in accts
            ]
    # RS-033: Pre-index transactions by account IBAN for O(V + T) lookup instead of O(V*T)
    tx_by_account: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    for i, t in enumerate(transactions):
        account = _get_tx_account(t)
        tx_by_account[account].append((i, t))

    results: list[dict[str, Any]] = []
    matched_tx_indices: set[int] = set()

    for vorgang in vorgaenge:
        raw_amount = vorgang.get("aktuellster_betrag", 0) or 0
        # RS-028: Use absolute value so negative amounts still participate in reconciliation
        amount = abs(raw_amount)
        if raw_amount < 0:
            logger.info("RS-028: Vorgang %s has negative amount %.2f, using abs for matching",
                        vorgang.get("vorgang_id"), raw_amount)
        refs = vorgang.get("alle_referenzen", [])
        person = vorgang.get("person")
        doc_date_str = vorgang.get("datum")

        # Filter transactions to person's accounts
        # RS-033: Use pre-indexed lookup instead of scanning all transactions
        eligible_accounts = [a for a in person_accounts.get(person, []) if a]
        if eligible_accounts:
            # Statements that could not be tied to a configured account ("")
            # stay available to everyone.
            eligible_txns = [
                (i, t) for acct in eligible_accounts + [""]
                for i, t in tx_by_account.get(acct, [])
                if i not in matched_tx_indices
            ]
        else:
            # No accounts configured for this person: check every statement.
            eligible_txns = [
                (i, t) for i, t in enumerate(transactions)
                if i not in matched_tx_indices
            ]

        match: Optional[dict[str, Any]] = None
        matched_txn: Any = None  # Track matched txn for D1 date flag

        # Phase 1: Reference match: collect ALL ref-matching txns, pick best
        ref_candidates: list[tuple[int, Any, float]] = []  # (idx, txn, amount_diff)
        for ref in refs:
            if not ref:
                continue
            norm_ref = ref.lower().strip()
            if len(norm_ref) < reference_min_length:
                continue
            # C1: Reference format validation: must contain both digit AND letter
            has_digit = any(c.isdigit() for c in norm_ref)
            has_letter = any(c.isalpha() for c in norm_ref)
            if not (has_digit and has_letter):
                continue
            for idx, txn in eligible_txns:
                details = _get_tx_details(txn).lower()
                if re.search(r'(?:^|\b)' + re.escape(norm_ref) + r'(?:\b|$)', details):
                    # Skip credits/refunds: only debit (outgoing) transactions confirm payment
                    raw_amount = _get_tx_amount(txn)
                    if raw_amount >= 0 and not _is_debit(txn):
                        continue
                    tx_amount = abs(_get_tx_amount(txn))
                    if tx_amount > 0:
                        ref_candidates.append((idx, txn, abs(tx_amount - amount) if amount else tx_amount))

        if ref_candidates:
            # Pick the best match by smallest absolute amount difference
            ref_candidates.sort(key=lambda c: c[2])
            best_idx, best_txn, _ = ref_candidates[0]
            best_tx_amount = abs(_get_tx_amount(best_txn))
            # D1: Apply date proximity to confidence
            date_prox = _compute_date_proximity(doc_date_str, _get_tx_booking_date(best_txn))
            if amount and abs(best_tx_amount - amount) <= amount_tolerance:
                match = {
                    "type": Status.BEZAHLT.value,
                    "method": "reference_exact",
                    "confidence": compute_recon_confidence(reference_confidence, date_prox),
                    "tx_index": best_idx,
                    "tx_amount": best_tx_amount,
                    "amount_diff": round(best_tx_amount - amount, 2),
                    "tx_hash": tx_hashes.get(best_idx, ""),
                    "booking_date": _get_tx_booking_date(best_txn),
                    "bank_beneficiary": _get_tx_beneficiary(best_txn),
                    "bank_account": _get_tx_account(best_txn),
                }
                matched_txn = best_txn
            elif amount and best_tx_amount > 0:
                # D3: Use bezahlt_threshold_pct (99%) for BEZAHLT determination
                paid_ratio = best_tx_amount / amount if amount > 0 else 0
                if paid_ratio >= bezahlt_threshold_pct:
                    match_type = Status.BEZAHLT.value
                    remaining = None
                elif paid_ratio >= teilweise_lower_pct:
                    match_type = Status.TEILWEISE.value
                    remaining = round(amount - best_tx_amount, 2)
                elif best_tx_amount < amount:
                    match_type = Status.TEILWEISE.value
                    remaining = round(amount - best_tx_amount, 2)
                else:
                    match_type = Status.BEZAHLT.value
                    remaining = None
                ref_diff_base = recon_config.get("reference_amount_diff_confidence", 0.80)
                match = {
                    "type": match_type,
                    "method": "reference_amount_diff",
                    "confidence": compute_recon_confidence(ref_diff_base, date_prox),
                    "tx_index": best_idx,
                    "tx_amount": best_tx_amount,
                    "amount_diff": round(best_tx_amount - amount, 2),
                    "tx_hash": tx_hashes.get(best_idx, ""),
                    "booking_date": _get_tx_booking_date(best_txn),
                    "bank_beneficiary": _get_tx_beneficiary(best_txn),
                    "bank_account": _get_tx_account(best_txn),
                }
                if remaining is not None:
                    match["remaining_amount"] = remaining
                matched_txn = best_txn

        # Phase 2: Amount match (debit transactions only) + creditor verification
        # D2: Collect ALL candidates, select best by composite score
        require_creditor = recon_config.get("require_creditor_match", True)
        absender = vorgang.get("absender", "")
        # C3: Absolute tolerance floor: at least EUR 2.00 even for small amounts
        abs_floor = recon_config.get("amount_tolerance_abs_floor", 2.00)
        effective_tolerance = max(amount_tolerance, abs_floor)
        if not match and amount and amount > 0:
            phase2_candidates: list[tuple[int, Any, float]] = []  # (idx, txn, composite_score)
            for idx, txn in eligible_txns:
                # Exclude credits/refunds: raw amount must be negative (outgoing)
                raw_amount = _get_tx_amount(txn)
                if raw_amount >= 0 and not _is_debit(txn):
                    continue
                tx_amount = abs(raw_amount)
                if tx_amount > 0 and abs(tx_amount - amount) <= effective_tolerance:
                    # Creditor verification: skip if beneficiary doesn't match absender
                    if require_creditor:
                        beneficiary = _get_tx_beneficiary(txn)
                        if not _creditor_matches(beneficiary, absender, config):
                            continue
                    # D2: Composite scoring
                    score = _compute_composite_score(
                        amount, tx_amount,
                        _get_tx_beneficiary(txn), absender,
                        doc_date_str, _get_tx_booking_date(txn),
                        config,
                    )
                    phase2_candidates.append((idx, txn, score))

            if phase2_candidates:
                # D2: Pick best by highest composite score
                phase2_candidates.sort(key=lambda c: c[2], reverse=True)
                best_idx, best_txn, best_score = phase2_candidates[0]
                best_tx_amount = abs(_get_tx_amount(best_txn))
                # D1: Apply date proximity to confidence
                date_prox = _compute_date_proximity(doc_date_str, _get_tx_booking_date(best_txn))
                match = {
                    "type": Status.BEZAHLT.value,
                    "method": "amount_match",
                    "confidence": compute_recon_confidence(amount_date_confidence, date_prox),
                    "tx_index": best_idx,
                    "tx_amount": best_tx_amount,
                    "amount_diff": round(best_tx_amount - amount, 2),
                    "tx_hash": tx_hashes.get(best_idx, ""),
                    "booking_date": _get_tx_booking_date(best_txn),
                    "bank_beneficiary": _get_tx_beneficiary(best_txn),
                    "bank_account": _get_tx_account(best_txn),
                }
                matched_txn = best_txn

        # Phase 3: Multi-payment accumulation (sum of multiple bank tx for same creditor)
        # Only accumulate txns whose payment details contain a Vorgang reference
        if not match and amount and amount > 0 and absender:
            creditor_txns = []
            for idx, txn in eligible_txns:
                if not _is_debit(txn):
                    continue
                beneficiary = _get_tx_beneficiary(txn)
                if not _creditor_matches(beneficiary, absender, config):
                    continue
                # Reference overlap guard: require at least one ref in payment details
                details_lower = _get_tx_details(txn).lower()
                has_ref_overlap = any(
                    r.lower().strip() in details_lower
                    for r in refs
                    if r and len(r.strip()) >= reference_min_length
                )
                if refs and not has_ref_overlap:
                    continue
                # C-06: When refs is empty, require strong fuzzy creditor match
                # to prevent accumulating ALL creditor payments as false BEZAHLT
                if not refs:
                    from core.utils import fuzzy_creditor_match as _fuzzy_match
                    if not _fuzzy_match(beneficiary, absender, threshold=0.85):
                        continue
                creditor_txns.append((idx, txn))

            # C6: Multi-payment cap: limit accumulated transactions
            max_multi_payments = 5
            if len(creditor_txns) > max_multi_payments:
                # Sort by amount descending and take top N
                creditor_txns.sort(key=lambda c: abs(_get_tx_amount(c[1])), reverse=True)
                creditor_txns = creditor_txns[:max_multi_payments]
                flags = vorgang.get("flags", [])
                if Flag.MULTI_PAYMENT_COMPLEX.value not in flags:
                    flags.append(Flag.MULTI_PAYMENT_COMPLEX.value)
                    vorgang["flags"] = flags

            if creditor_txns:
                total_paid = sum(abs(_get_tx_amount(t)) for _, t in creditor_txns)
                beneficiaries = list({_get_tx_beneficiary(t) for _, t in creditor_txns})
                accounts = sorted({_get_tx_account(t) for _, t in creditor_txns} - {""})
                # Latest booking date among accumulated transactions
                latest_booking = max(
                    (_get_tx_booking_date(t) or "" for _, t in creditor_txns),
                    default=None,
                )
                # Date proximity factor for multi-payment confidence
                mp_date_prox = _compute_date_proximity(doc_date_str, latest_booking)
                compound_conf = recon_config.get("compound_match_confidence", 0.75)
                mp_confidence = compute_recon_confidence(compound_conf, mp_date_prox)
                # D3: Use bezahlt_threshold_pct (99%) instead of hardcoded 0.95
                paid_ratio = total_paid / amount if amount > 0 else 0
                if paid_ratio >= bezahlt_threshold_pct:
                    match = {
                        "type": Status.BEZAHLT.value,
                        "method": "multi_payment",
                        "confidence": mp_confidence,
                        "tx_indices": [idx for idx, _ in creditor_txns],
                        "tx_amount": round(total_paid, 2),
                        "amount_diff": round(total_paid - amount, 2),
                        "payment_count": len(creditor_txns),
                        "booking_date": latest_booking,
                        "bank_beneficiary": beneficiaries[0] if len(beneficiaries) == 1 else "; ".join(beneficiaries),
                        "bank_account": "; ".join(accounts),
                    }
                elif total_paid > 0:
                    remaining = round(amount - total_paid, 2)
                    match = {
                        "type": Status.TEILWEISE.value,
                        "method": "multi_payment",
                        "confidence": mp_confidence,
                        "tx_indices": [idx for idx, _ in creditor_txns],
                        "tx_amount": round(total_paid, 2),
                        "amount_diff": round(total_paid - amount, 2),
                        "remaining_amount": remaining,
                        "payment_count": len(creditor_txns),
                        "booking_date": latest_booking,
                        "bank_beneficiary": beneficiaries[0] if len(beneficiaries) == 1 else "; ".join(beneficiaries),
                        "bank_account": "; ".join(accounts),
                    }

        # C-06: Flag zero-ref multi-payment matches for manual review
        if match and match.get("method") == "multi_payment" and not refs:
            mp_flags = vorgang.get("flags", [])
            zero_ref_flag = "ZERO_REF_MULTI_PAYMENT"
            if zero_ref_flag not in mp_flags:
                mp_flags.append(zero_ref_flag)
                vorgang["flags"] = mp_flags

        # Phase 4: LLM-assisted reconciliation for unmatched Vorgaenge
        enable_llm = recon_config.get("enable_llm_reconciliation", False)
        if not match and enable_llm and amount and amount > 0:
            llm_match = _llm_reconcile_vorgang(
                vorgang, eligible_txns, tx_hashes, config,
            )
            if llm_match:
                match = llm_match
                matched_txn = None  # LLM match doesn't track individual txn object

        # C4: Overpayment detection: flag when tx paid significantly more than owed
        if match and amount and amount > 0:
            tx_paid = match.get("tx_amount", 0)
            if tx_paid > amount * 1.10:
                flags = vorgang.get("flags", [])
                overpay_flag = Flag.POTENTIAL_OVERPAYMENT.value
                if overpay_flag not in flags:
                    flags.append(overpay_flag)
                    vorgang["flags"] = flags

        # Apply match result
        if match:
            vorgang["status"] = match["type"]
            vorgang["reconciliation"] = match
            # Mark matched transaction(s) as used
            if "tx_index" in match:
                matched_tx_indices.add(match["tx_index"])
            elif "tx_indices" in match:
                matched_tx_indices.update(match["tx_indices"])
            # C9: Flag if matched transaction is a known duplicate
            matched_idx = match.get("tx_index")
            if matched_idx is not None and matched_idx in duplicate_tx_indices:
                dup_flags = vorgang.get("flags", [])
                if Flag.DUPLICATE_TRANSACTION.value not in dup_flags:
                    dup_flags.append(Flag.DUPLICATE_TRANSACTION.value)
                    vorgang["flags"] = dup_flags
            # D1: Apply date mismatch flag
            if matched_txn is not None:
                _apply_date_mismatch_flag(vorgang, matched_txn, config)
            # D4: Apply confidence-gated review flags
            _apply_review_flags(vorgang, config)
        else:
            vorgang["status"] = Status.OFFEN.value
            vorgang["reconciliation"] = {"type": Status.OFFEN.value, "method": "no_match", "confidence": 0}

        vorgang["prioritaet"] = compute_priority_with_status(vorgang, config)
        results.append(vorgang)

    return results, matched_tx_indices


class SkillBankReconciler(BankReconciler):
    """Concrete implementation of Phase D bank reconciliation."""

    def __init__(self, config: BriefpostConfig) -> None:
        """Initialize reconciler.

        Args:
            config: Pipeline configuration.
        """
        self.config = config

    def reconcile(
        self,
        vorgaenge: list[Vorgang],
        transactions: list[BankTransaction],
        config: dict[str, Any],
    ) -> tuple[list[Vorgang], set[int]]:
        """Match Vorgaenge against bank transactions.

        Note: This ABC interface uses model objects. The actual implementation
        works with dicts. Use run_reconciliation() for pipeline execution.
        """
        raise NotImplementedError("Use run_reconciliation() directly")


def run_reconciliation(project_dir: Path, config: BriefpostConfig) -> BatchResult:
    """Entry point called by pipeline.py.

    Args:
        project_dir: Project data directory.
        config: Pipeline configuration.

    Returns:
        BatchResult describing the outcome.
    """
    result = BatchResult(phase="reconcile")
    reconciliation_dir = project_dir / "reconciliation"
    bank_dir = project_dir / "bank-statements"

    # Load Vorgaenge
    vorgaenge_path = reconciliation_dir / "vorgaenge.json"
    if not vorgaenge_path.exists():
        logger.error("vorgaenge.json not found: %s", vorgaenge_path)
        result.total = 1
        result.failed = 1
        result.errors.append({"error": "vorgaenge.json not found"})
        result.complete()
        return result

    with open(vorgaenge_path) as f:
        vorgaenge_data = json.load(f)
    vorgaenge = vorgaenge_data.get("vorgaenge", [])
    logger.info("Vorgaenge loaded: %d", len(vorgaenge))

    # Load transactions
    transactions = load_bank_transactions(bank_dir, config)
    logger.info("Transactions loaded: %d", len(transactions))

    if not vorgaenge:
        logger.error("No Vorgaenge to reconcile")
        result.total = 0
        result.complete()
        return result

    # RS-011/RS-034: Checkpoint support for LLM reconciliation crash recovery
    checkpoint_path = reconciliation_dir / ".reconciliation-checkpoint.json"

    try:
        # Reconcile
        reconciled, matched_tx_indices = reconcile(vorgaenge, transactions, config)

        # RS-011: Save checkpoint after reconciliation completes
        try:
            atomic_write_json(checkpoint_path, {
                "phase": "reconcile_complete",
                "vorgaenge_count": len(reconciled),
                "matched_tx_count": len(matched_tx_indices),
                "timestamp": datetime.now().isoformat(),
            })
        except OSError:
            logger.warning("RS-011: Could not write reconciliation checkpoint")

        # Summary
        status_counts: dict[str, int] = defaultdict(int)
        priority_counts: dict[str, int] = defaultdict(int)
        for vorgang in reconciled:
            status_counts[vorgang["status"]] += 1
            priority_counts[vorgang["prioritaet"]] += 1

        logger.info("Reconciliation results:")
        for status, count in sorted(status_counts.items()):
            logger.info("  Status %s: %d", status, count)
        for prio, count in sorted(priority_counts.items()):
            logger.info("  Priority %s: %d", prio, count)
        logger.info("Bank transactions matched: %d/%d", len(matched_tx_indices), len(transactions))

        # Write output
        reconciliation_dir.mkdir(exist_ok=True)

        out_path = reconciliation_dir / "vorgaenge-reconciled.json"
        atomic_write_json(out_path, {
            "vorgaenge": reconciled,
            "created": datetime.now().isoformat(),
        })
        logger.info("Written: %s", out_path)

        report_path = reconciliation_dir / "reconciliation-report.json"
        atomic_write_json(report_path, {
            "created": datetime.now().isoformat(),
            "total_vorgaenge": len(reconciled),
            "total_transactions": len(transactions),
            "matched_transactions": len(matched_tx_indices),
            "status_distribution": dict(status_counts),
            "priority_distribution": dict(priority_counts),
        })
        logger.info("Written: %s", report_path)

        result.total = len(reconciled)
        result.succeeded = len(reconciled)

        # RS-011: Clean up checkpoint on success
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    except Exception as exc:
        logger.error("Reconciliation failed: %s", exc, exc_info=True)
        result.total = 1
        result.failed = 1
        result.errors.append({"phase": "reconcile", "error": str(exc)})

    result.complete()
    return result
