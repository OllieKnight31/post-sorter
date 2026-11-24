"""Phase E2: payment QR codes for open cases (EPC QR, the SEPA "scan to pay" code).

One PNG per open case that has a valid IBAN and a positive amount:
    <project>/qr-codes/V-001-payment.png

EPC QR codes only exist for euro SEPA transfers. They are generated when
output.currency is EUR (the default when the key is missing). For any other
currency the step is skipped and the folder is left empty.

Needs the small "segno" package. Without it the step is skipped with a
warning; the rest of the pipeline is not affected.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from core.config import BriefpostConfig
from core.constants import Status
from core.models import BatchResult

logger = logging.getLogger(__name__)

# Source files in order of preference (same order as the sheet builder).
VORGAENGE_FILES = (
    "vorgaenge-cross-intel.json",
    "vorgaenge-reconciled.json",
    "vorgaenge.json",
)

EPC_MAX_PAYLOAD_BYTES = 331
EPC_MAX_AMOUNT = 999_999_999.99

# IBAN length per country (ISO 13616 registry).
IBAN_LENGTHS: dict[str, int] = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16, "BG": 22,
    "BH": 22, "BR": 29, "BY": 28, "CH": 21, "CR": 22, "CY": 28, "CZ": 24, "DE": 22,
    "DK": 18, "DO": 28, "EE": 20, "EG": 29, "ES": 24, "FI": 18, "FO": 18, "FR": 27,
    "GB": 22, "GE": 22, "GI": 23, "GL": 18, "GR": 27, "GT": 28, "HR": 21, "HU": 28,
    "IE": 22, "IL": 23, "IQ": 23, "IS": 26, "IT": 27, "JO": 30, "KW": 30, "KZ": 20,
    "LB": 28, "LC": 32, "LI": 21, "LT": 20, "LU": 20, "LV": 21, "MC": 27, "MD": 24,
    "ME": 22, "MK": 19, "MR": 27, "MT": 31, "MU": 30, "NL": 18, "NO": 15, "PK": 24,
    "PL": 28, "PS": 29, "PT": 25, "QA": 29, "RO": 24, "RS": 22, "SA": 24, "SC": 31,
    "SE": 24, "SI": 19, "SK": 24, "SM": 27, "ST": 25, "SV": 28, "TL": 23, "TN": 24,
    "TR": 26, "UA": 29, "VA": 22, "VG": 24, "XK": 20,
}


def validate_iban(iban: Optional[str]) -> tuple[bool, str]:
    """Check an IBAN: characters, country length and the mod-97 checksum.

    Args:
        iban: IBAN with or without spaces.

    Returns:
        (True, "") if valid, otherwise (False, reason).
    """
    if not iban or not isinstance(iban, str):
        return False, "empty"
    normalized = re.sub(r"\s+", "", iban).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", normalized):
        return False, "bad format"
    expected_length = IBAN_LENGTHS.get(normalized[:2])
    if expected_length is None:
        if not 15 <= len(normalized) <= 34:
            return False, "bad length"
    elif len(normalized) != expected_length:
        return False, f"length {len(normalized)}, expected {expected_length} for {normalized[:2]}"
    rearranged = normalized[4:] + normalized[:4]
    digits = "".join(str(int(ch, 36)) for ch in rearranged)
    if int(digits) % 97 != 1:
        return False, "checksum"
    return True, ""


def _clean_line(value: Any, max_chars: int) -> str:
    """One payload line: no line breaks, trimmed, cut to max_chars."""
    text = "" if value is None else str(value)
    text = re.sub(r"[\r\n\t]+", " ", text).strip()
    return text[:max_chars]


def build_epc_payload(
    iban: str,
    recipient: str,
    amount: float,
    bic: Optional[str] = None,
    reference: Optional[str] = None,
) -> str:
    """Build the EPC069-12 payload (version 002, UTF-8, SEPA credit transfer).

    The recipient is cut to 70 characters and the reference to 140. If the
    payload is still over the 331 byte limit (multi-byte characters), the
    reference is shortened until it fits.

    Args:
        iban: Payee IBAN.
        recipient: Payee name.
        amount: Amount in euro.
        bic: Payee BIC (optional in version 002).
        reference: Unstructured payment reference.

    Returns:
        Payload text for the QR code.
    """
    iban_line = re.sub(r"\s+", "", iban or "").upper()
    name_line = _clean_line(recipient, 70)
    bic_line = _clean_line(bic, 11).upper()
    reference_line = _clean_line(reference, 140)

    def assemble(ref: str) -> str:
        return "\n".join([
            "BCD", "002", "1", "SCT",
            bic_line, name_line, iban_line,
            f"EUR{amount:.2f}",
            "", "", ref,
        ])

    payload = assemble(reference_line)
    while len(payload.encode("utf-8")) > EPC_MAX_PAYLOAD_BYTES and reference_line:
        reference_line = reference_line[:-1]
        payload = assemble(reference_line)
    return payload


def generate_single_qr(payload: str, output_path: Path, scale: int = 10, border: int = 4) -> bool:
    """Write one QR code PNG. The file appears only when it is complete.

    Args:
        payload: Text to encode.
        output_path: Target .png path.
        scale: Pixels per module.
        border: Quiet zone in modules.

    Returns:
        True if the file was written, False on any failure (nothing is left behind).
    """
    try:
        import segno
    except ImportError:
        logger.warning("QR codes skipped: the 'segno' package is not installed (python3 -m pip install segno)")
        return False

    output_path = Path(output_path)
    tmp_name: Optional[str] = None
    try:
        qr = segno.make(payload, error="m", micro=False)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(output_path.parent), prefix=f".{output_path.stem}-", suffix=".part",
        )
        with os.fdopen(fd, "wb") as handle:
            qr.save(handle, kind="png", scale=scale, border=border)
        os.replace(tmp_name, str(output_path))
        tmp_name = None
        return True
    except Exception as exc:
        logger.error("QR code for %s failed: %s", output_path.name, exc)
        return False
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def qr_codes_enabled(config: BriefpostConfig) -> bool:
    """EPC QR codes only make sense for euro payments."""
    currency = str(config.output.get("currency", "EUR") or "EUR").strip().upper()
    return currency == "EUR"


def is_qr_candidate(vorgang: dict[str, Any]) -> bool:
    """True if a case should get a payment QR code."""
    status = vorgang.get("status") or Status.OFFEN.value
    if status not in (Status.OFFEN.value, Status.TEILWEISE.value):
        return False
    amount = vorgang.get("aktuellster_betrag")
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        return False
    if amount <= 0 or amount > EPC_MAX_AMOUNT:
        return False
    valid, _reason = validate_iban(vorgang.get("iban"))
    return valid


def generate_qr_codes(vorgaenge: list[dict[str, Any]], config: BriefpostConfig) -> int:
    """Generate payment QR codes for all eligible open cases.

    Args:
        vorgaenge: Cases from the reconciliation JSON.
        config: Pipeline configuration (project_dir, output.qr_scale, output.qr_border).

    Returns:
        Number of QR code files written.
    """
    project_dir = config.project_dir
    if project_dir is None:
        raise ValueError("pipeline.project_dir must be set to write QR codes")
    qr_dir = project_dir / "qr-codes"
    qr_dir.mkdir(parents=True, exist_ok=True)

    if not qr_codes_enabled(config):
        logger.info("QR codes skipped: EPC QR codes are euro only (output.currency is not EUR)")
        return 0

    scale = int(config.output.get("qr_scale", 10) or 10)
    border = int(config.output.get("qr_border", 4) or 4)
    written = 0

    for vorgang in vorgaenge:
        vorgang_id = str(vorgang.get("vorgang_id") or "").strip()
        if not vorgang_id or not is_qr_candidate(vorgang):
            if vorgang.get("iban") and (vorgang.get("status") or Status.OFFEN.value) in (
                Status.OFFEN.value, Status.TEILWEISE.value,
            ):
                valid, reason = validate_iban(vorgang.get("iban"))
                if not valid:
                    logger.warning("No QR code for %s: IBAN not valid (%s)", vorgang_id or "?", reason)
            continue

        amount = float(vorgang["aktuellster_betrag"])
        if vorgang.get("status") == Status.TEILWEISE.value:
            remaining = (vorgang.get("reconciliation") or {}).get("remaining_amount")
            if isinstance(remaining, (int, float)) and not isinstance(remaining, bool) and remaining > 0:
                amount = float(remaining)

        payload = build_epc_payload(
            vorgang["iban"],
            vorgang.get("empfaenger_name") or vorgang.get("absender") or "",
            amount,
            vorgang.get("bic"),
            vorgang.get("verwendungszweck") or vorgang.get("aktenzeichen") or "",
        )
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", vorgang_id)
        if generate_single_qr(payload, qr_dir / f"{safe_id}-payment.png", scale, border):
            written += 1

    logger.info("QR codes written: %d", written)
    return written


def run_qr_generator(project_dir: Path, config: BriefpostConfig) -> BatchResult:
    """Entry point called by pipeline.py.

    Args:
        project_dir: Project data directory.
        config: Pipeline configuration.

    Returns:
        BatchResult with phase "output_qr".
    """
    result = BatchResult(phase="output_qr")
    result.total = 1
    project_dir = Path(project_dir)
    reconciliation_dir = project_dir / "reconciliation"

    try:
        vorgaenge = None
        for name in VORGAENGE_FILES:
            path = reconciliation_dir / name
            if path.is_file():
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
                vorgaenge = data.get("vorgaenge", []) if isinstance(data, dict) else data
                break

        if vorgaenge is None:
            result.failed = 1
            result.errors.append({"phase": "output", "error": "No vorgaenge JSON found"})
            result.complete()
            return result

        (project_dir / "qr-codes").mkdir(parents=True, exist_ok=True)
        generate_qr_codes(list(vorgaenge or []), config)
        result.succeeded = 1

    except Exception as exc:
        logger.error("QR generation failed: %s", exc, exc_info=True)
        result.failed = 1
        result.errors.append({"phase": "output", "error": str(exc)})

    result.complete()
    return result
