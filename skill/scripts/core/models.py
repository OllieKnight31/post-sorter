"""Domain data models for briefpost-management skill.

All dataclasses correspond to entities flowing through the pipeline phases:
  Phase A (Group) -> Phase B (OCR) -> Phase B3 (Page Match) ->
  Phase C (Vorgang Match) -> Phase D (Reconcile) -> Phase E (Output) -> Phase F (QC)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase A: Grouping
# ---------------------------------------------------------------------------

@dataclass
class DocumentGroup:
    """A group of photos determined to be from the same physical document."""

    doc_id: str
    pages: list[str]
    timestamps: list[str]
    bursts: int
    gap_before_seconds: Optional[float]
    gap_after_seconds: Optional[float]
    confidence: float
    flags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id must not be empty")
        if self.confidence < 0 or self.confidence > 1.0:
            raise ValueError(f"confidence must be 0.0-1.0, got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "pages": self.pages,
            "timestamps": self.timestamps,
            "bursts": self.bursts,
            "gap_before_seconds": self.gap_before_seconds,
            "gap_after_seconds": self.gap_after_seconds,
            "confidence": self.confidence,
            "flags": self.flags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentGroup:
        # CQ-024: Validate required field
        doc_id = data.get("doc_id")
        if not doc_id:
            raise ValueError("DocumentGroup.from_dict: missing required 'doc_id'")
        return cls(
            doc_id=doc_id,
            pages=data.get("pages", []),
            timestamps=data.get("timestamps", []),
            bursts=data.get("bursts", 1),
            gap_before_seconds=data.get("gap_before_seconds"),
            gap_after_seconds=data.get("gap_after_seconds"),
            confidence=data.get("confidence", 0.0),
            flags=data.get("flags", []),
        )


# ---------------------------------------------------------------------------
# Phase B: OCR
# ---------------------------------------------------------------------------

@dataclass
class Page:
    """A single OCR-processed page from a document group."""

    page_id: str
    doc_group: str
    page_number: int
    ocr_text: str
    ocr_confidence: float
    ocr_provider: str
    signals: dict[str, Any] = field(default_factory=dict)
    gemini_extraction: dict[str, Any] = field(default_factory=dict)
    source_image: str = ""
    postprocessed_at: Optional[str] = None
    flags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.page_id:
            raise ValueError("page_id must not be empty")
        if self.ocr_confidence < 0 or self.ocr_confidence > 1.0:
            logger.warning(
                "Page %s has confidence %.2f outside 0-1 range, clamping",
                self.page_id, self.ocr_confidence,
            )
            self.ocr_confidence = max(0.0, min(1.0, self.ocr_confidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "doc_group": self.doc_group,
            "page_number": self.page_number,
            "ocr_text": self.ocr_text,
            "ocr_confidence": self.ocr_confidence,
            "ocr_provider": self.ocr_provider,
            "signals": self.signals,
            "gemini_extraction": self.gemini_extraction,
            "source_image": self.source_image,
            "postprocessed_at": self.postprocessed_at,
            "flags": self.flags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Page:
        return cls(
            page_id=data["page_id"],
            doc_group=data.get("doc_group", ""),
            page_number=data.get("page_number", 0),
            ocr_text=data.get("ocr_text", ""),
            ocr_confidence=data.get("ocr_confidence", 0.0),
            ocr_provider=data.get("ocr_provider", "unknown"),
            signals=data.get("signals", {}),
            gemini_extraction=data.get("gemini_extraction", {}),
            source_image=data.get("source_image", ""),
            postprocessed_at=data.get("postprocessed_at"),
            flags=data.get("flags", []),
        )


# ---------------------------------------------------------------------------
# Phase B3: Letter (Brief)
# ---------------------------------------------------------------------------

@dataclass
class Creditor:
    """Creditor/sender information extracted from a letter."""

    name: str = ""
    adresse: str = ""
    iban: str = ""
    bic: str = ""

    def to_dict(self) -> dict[str, Any]:
        # CQ-048: Always include name; omit empty optional fields
        result: dict[str, Any] = {"name": self.name}
        if self.adresse:
            result["adresse"] = self.adresse
        if self.iban:
            result["iban"] = self.iban
        if self.bic:
            result["bic"] = self.bic
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Creditor:
        if isinstance(data, str):
            logger.warning("Creditor.from_dict received str instead of dict: %r", data)
            return cls(name=data)
        return cls(
            name=data.get("name", ""),
            adresse=data.get("adresse", ""),
            iban=data.get("iban", ""),
            bic=data.get("bic", ""),
        )


@dataclass
class Brief:
    """A single letter composed of one or more OCR pages.

    Corresponds to letters/LTR-XXX.json in the project directory.
    """

    letter_id: str
    pages: list[dict[str, Any]] = field(default_factory=list)
    person: Optional[str] = None
    creditor: Optional[Creditor] = None
    document_type: str = "sonstig"
    eskalation: int = 0
    kategorie: str = "SONSTIGES"
    dates: dict[str, Any] = field(default_factory=dict)
    betreff: str = ""
    identifiers: list[str] = field(default_factory=list)
    amounts: dict[str, Any] = field(default_factory=dict)
    payment_info: dict[str, Any] = field(default_factory=dict)
    urgency_signals: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    notizen: str = ""
    ocr_confidence: float = 0.0
    ocr_provider: str = "gemini"
    source_images: list[str] = field(default_factory=list)
    raw_ocr_files: list[str] = field(default_factory=list)
    # CQ-047: match_confidence/match_method used during page-to-letter matching
    # (Phase B3), then propagated to Vorgang level in Phase C.
    match_confidence: float = 0.0
    match_method: str = ""

    def __post_init__(self) -> None:
        if not self.letter_id:
            raise ValueError("letter_id must not be empty")
        # CQ-025: Validate pages structure
        if self.pages:
            for i, page in enumerate(self.pages):
                if not isinstance(page, dict):
                    logger.warning(
                        "Brief %s: pages[%d] is %s, expected dict",
                        self.letter_id, i, type(page).__name__,
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "letter_id": self.letter_id,
            "pages": self.pages,
            "person": self.person,
            "creditor": self.creditor.to_dict() if self.creditor else None,
            "document_type": self.document_type,
            "eskalation": self.eskalation,
            "kategorie": self.kategorie,
            "dates": self.dates,
            "betreff": self.betreff,
            "identifiers": self.identifiers,
            "amounts": self.amounts,
            "payment_info": self.payment_info,
            "urgency_signals": self.urgency_signals,
            "flags": self.flags,
            "notizen": self.notizen,
            "ocr_confidence": self.ocr_confidence,
            "ocr_provider": self.ocr_provider,
            "source_images": self.source_images,
            "raw_ocr_files": self.raw_ocr_files,
            "match_confidence": self.match_confidence,
            "match_method": self.match_method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Brief:
        # RS-025: Guard against missing letter_id
        letter_id = data.get("letter_id")
        if not letter_id:
            raise ValueError("Brief.from_dict: missing required 'letter_id' in data")
        creditor_data = data.get("creditor")
        creditor = Creditor.from_dict(creditor_data) if creditor_data else None
        return cls(
            letter_id=letter_id,
            pages=data.get("pages", []),
            person=data.get("person"),
            creditor=creditor,
            document_type=data.get("document_type", "sonstig"),
            eskalation=data.get("eskalation", 0),
            kategorie=data.get("kategorie", "SONSTIGES"),
            dates=data.get("dates", {}),
            betreff=data.get("betreff", ""),
            identifiers=data.get("identifiers", []),
            amounts=data.get("amounts", {}),
            payment_info=data.get("payment_info", {}),
            urgency_signals=data.get("urgency_signals", []),
            flags=data.get("flags", []),
            notizen=data.get("notizen", ""),
            ocr_confidence=data.get("ocr_confidence", 0.0),
            ocr_provider=data.get("ocr_provider", "gemini"),
            source_images=data.get("source_images", []),
            raw_ocr_files=data.get("raw_ocr_files", []),
            match_confidence=data.get("match_confidence", 0.0),
            match_method=data.get("match_method", ""),
        )


# ---------------------------------------------------------------------------
# Phase C-D: Vorgang (Case/Proceeding)
# ---------------------------------------------------------------------------

@dataclass
class ReconciliationResult:
    """Result of matching a Vorgang against bank transactions."""

    match_type: str = "OFFEN"
    method: str = "no_match"
    confidence: float = 0.0
    tx_index: Optional[int] = None
    tx_amount: Optional[float] = None
    amount_diff: Optional[float] = None
    remaining_amount: Optional[float] = None
    tx_hash: Optional[str] = None
    bank_date: Optional[str] = None
    bank_account: Optional[str] = None
    bank_beneficiary: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        # CQ-006: Serialize as "match_type" (primary) and "type" (backward compat)
        result: dict[str, Any] = {
            "match_type": self.match_type,
            "type": self.match_type,  # backward compat alias
            "method": self.method,
            "confidence": self.confidence,
        }
        if self.tx_index is not None:
            result["tx_index"] = self.tx_index
        if self.tx_amount is not None:
            result["tx_amount"] = self.tx_amount
        if self.amount_diff is not None:
            result["amount_diff"] = self.amount_diff
        if self.remaining_amount is not None:
            result["remaining_amount"] = self.remaining_amount
        if self.tx_hash is not None:
            result["tx_hash"] = self.tx_hash
        if self.bank_date:
            result["bank_date"] = self.bank_date
        if self.bank_account:
            result["bank_account"] = self.bank_account
        if self.bank_beneficiary:
            result["bank_beneficiary"] = self.bank_beneficiary
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReconciliationResult:
        if not data:
            return cls()
        # CQ-006: Accept both "match_type" (new) and "type" (legacy) keys
        return cls(
            match_type=data.get("match_type", data.get("type", "OFFEN")),
            method=data.get("method", "no_match"),
            confidence=data.get("confidence", 0.0),
            tx_index=data.get("tx_index"),
            tx_amount=data.get("tx_amount"),
            amount_diff=data.get("amount_diff"),
            remaining_amount=data.get("remaining_amount"),
            tx_hash=data.get("tx_hash"),
            bank_date=data.get("bank_date"),
            bank_account=data.get("bank_account"),
            bank_beneficiary=data.get("bank_beneficiary"),
        )


@dataclass
class Vorgang:
    """A case/proceeding grouping one or more related letters.

    Corresponds to entries in reconciliation/vorgaenge.json and
    reconciliation/vorgaenge-reconciled.json.
    """

    vorgang_id: str
    absender: str = ""
    person: Optional[str] = None
    briefe: list[str] = field(default_factory=list)
    hoechste_eskalation: int = 0
    dokument_typ: str = "unknown"
    aktuellster_betrag: Optional[float] = None
    aktenzeichen: Optional[str] = None
    alle_referenzen: list[str] = field(default_factory=list)
    iban: Optional[str] = None
    bic: Optional[str] = None
    empfaenger_name: Optional[str] = None
    verwendungszweck: Optional[str] = None
    kategorie: str = "SONSTIGES"
    datum: Optional[str] = None
    faellig_am: Optional[str] = None
    urgency_signals: list[str] = field(default_factory=list)
    status: str = "OFFEN"
    prioritaet: str = "NORMAL"
    match_method: str = ""
    match_confidence: float = 0.0
    pipeline_confidence: Optional[float] = None
    reconciliation: Optional[ReconciliationResult] = None
    flags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.vorgang_id:
            raise ValueError("vorgang_id must not be empty")

    # CQ-011: brief_count as computed property instead of stored field
    @property
    def brief_count(self) -> int:
        return len(self.briefe)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vorgang_id": self.vorgang_id,
            "absender": self.absender,
            "person": self.person,
            "briefe": self.briefe,
            "hoechste_eskalation": self.hoechste_eskalation,
            "dokument_typ": self.dokument_typ,
            "aktuellster_betrag": self.aktuellster_betrag,
            "aktenzeichen": self.aktenzeichen,
            "alle_referenzen": self.alle_referenzen,
            "iban": self.iban,
            "bic": self.bic,
            "empfaenger_name": self.empfaenger_name,
            "verwendungszweck": self.verwendungszweck,
            "kategorie": self.kategorie,
            "datum": self.datum,
            "faellig_am": self.faellig_am,
            "urgency_signals": self.urgency_signals,
            "status": self.status,
            "prioritaet": self.prioritaet,
            "match_method": self.match_method,
            "match_confidence": self.match_confidence,
            "pipeline_confidence": self.pipeline_confidence,
            "reconciliation": self.reconciliation.to_dict() if self.reconciliation else None,
            "flags": self.flags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Vorgang:
        recon_data = data.get("reconciliation")
        reconciliation = ReconciliationResult.from_dict(recon_data) if recon_data else None
        return cls(
            vorgang_id=data["vorgang_id"],
            absender=data.get("absender", ""),
            person=data.get("person"),
            briefe=data.get("briefe", []),
            hoechste_eskalation=data.get("hoechste_eskalation", 0),
            dokument_typ=data.get("dokument_typ", "unknown"),
            aktuellster_betrag=data.get("aktuellster_betrag"),
            aktenzeichen=data.get("aktenzeichen"),
            alle_referenzen=data.get("alle_referenzen", []),
            iban=data.get("iban"),
            bic=data.get("bic"),
            empfaenger_name=data.get("empfaenger_name"),
            verwendungszweck=data.get("verwendungszweck"),
            kategorie=data.get("kategorie", "SONSTIGES"),
            datum=data.get("datum"),
            faellig_am=data.get("faellig_am"),
            urgency_signals=data.get("urgency_signals", []),
            status=data.get("status", "OFFEN"),
            prioritaet=data.get("prioritaet", "NORMAL"),
            match_method=data.get("match_method", ""),
            match_confidence=data.get("match_confidence", 0.0),
            pipeline_confidence=data.get("pipeline_confidence"),
            reconciliation=reconciliation,
            flags=data.get("flags", []),
        )


# ---------------------------------------------------------------------------
# Phase D: Bank Transaction
# ---------------------------------------------------------------------------

@dataclass
class BankTransaction:
    """A single bank transaction parsed from a CSV export."""

    booking_date: Optional[str] = None
    value_date: Optional[str] = None
    beneficiary: str = ""
    payment_details: str = ""
    iban: str = ""
    bic: str = ""
    customer_reference: str = ""
    debit: Optional[float] = None
    credit: Optional[float] = None
    currency: str = "EUR"
    source_account_iban: str = ""
    source_account_owner: str = ""

    @property
    def amount(self) -> float:
        """Signed amount: negative for debits, positive for credits.

        Returns 0.0 when both debit and credit are None or zero.
        Use has_amount to distinguish 0.0 from missing data.
        """
        if self.debit is not None and self.debit != 0:
            return -abs(self.debit)
        if self.credit is not None and self.credit != 0:
            return abs(self.credit)
        return 0.0

    @property
    def has_amount(self) -> bool:
        """CQ-036: True if debit or credit has a non-None value."""
        return self.debit is not None or self.credit is not None

    @property
    def is_debit(self) -> bool:
        return self.amount < 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "booking_date": self.booking_date,
            "value_date": self.value_date,
            "beneficiary": self.beneficiary,
            "payment_details": self.payment_details,
            "iban": self.iban,
            "bic": self.bic,
            "customer_reference": self.customer_reference,
            "debit": self.debit,
            "credit": self.credit,
            "currency": self.currency,
            "source_account_iban": self.source_account_iban,
            "source_account_owner": self.source_account_owner,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BankTransaction:
        return cls(
            booking_date=data.get("booking_date"),
            value_date=data.get("value_date"),
            beneficiary=data.get("beneficiary", ""),
            payment_details=data.get("payment_details", ""),
            iban=data.get("iban", ""),
            bic=data.get("bic", ""),
            customer_reference=data.get("customer_reference", ""),
            debit=data.get("debit"),
            credit=data.get("credit"),
            currency=data.get("currency", "EUR"),
            source_account_iban=data.get("source_account_iban", ""),
            source_account_owner=data.get("source_account_owner", ""),
        )


# ---------------------------------------------------------------------------
# Phase F: QC
# ---------------------------------------------------------------------------

@dataclass
class QCCheck:
    """A single quality control check result."""

    check_number: int
    name: str
    criterion: str
    passed: bool
    actual: Any = None
    expected: Any = None
    details: Any = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "check": self.check_number,
            "check_number": self.check_number,  # DF-015: forward compat alias
            "name": self.name,
            "criterion": self.criterion,
            "passed": self.passed,
            "actual": self.actual,
        }
        if self.expected is not None:
            result["expected"] = self.expected
        if self.details is not None:
            result["details"] = self.details
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QCCheck:
        # DF-015: Accept both "check_number" (new) and "check" (legacy)
        return cls(
            check_number=data.get("check_number", data.get("check", 0)),
            name=data.get("name", ""),
            criterion=data.get("criterion", ""),
            passed=data.get("passed", False),
            actual=data.get("actual"),
            expected=data.get("expected"),
            details=data.get("details"),
        )


@dataclass
class QCReport:
    """Complete QC verification report."""

    created: str = ""
    passed_count: int = 0
    failed_count: int = 0
    total: int = 0
    overall_result: str = "FAIL"
    checks: list[QCCheck] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.created:
            self.created = datetime.now().isoformat()
        self.total = len(self.checks)
        self.passed_count = sum(1 for c in self.checks if c.passed)
        self.failed_count = self.total - self.passed_count
        self.overall_result = "PASS" if self.failed_count == 0 else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "total": self.total,
            "overall_result": self.overall_result,
            "checks": [c.to_dict() for c in self.checks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QCReport:
        """Deserialize QCReport. DF-016: Preserve all fields from serialized data."""
        checks = [QCCheck.from_dict(c) for c in data.get("checks", [])]
        report = cls(
            created=data.get("created", ""),
            checks=checks,
        )
        # DF-016: Override computed fields with serialized values if present
        # (post_init computes from checks, but serialized may differ)
        if "overall_result" in data:
            report.overall_result = data["overall_result"]
        return report


# ---------------------------------------------------------------------------
# Pipeline Orchestration
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    """Result of running a pipeline phase."""

    phase: str
    started_at: str = ""
    completed_at: str = ""
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = datetime.now().isoformat()

    def complete(self) -> None:
        """Mark the batch as completed with current timestamp."""
        self.completed_at = datetime.now().isoformat()

    @property
    def success_rate(self) -> float:
        """CQ-050/DF-017: Returns 0.0 for empty batches (total=0)."""
        if self.total == 0:
            return 0.0
        return self.succeeded / self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "success_rate": round(self.success_rate, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchResult:
        return cls(
            phase=data.get("phase", ""),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            total=data.get("total", 0),
            succeeded=data.get("succeeded", 0),
            failed=data.get("failed", 0),
            skipped=data.get("skipped", 0),
            errors=data.get("errors", []),
        )


@dataclass
class PipelineResult:
    """Aggregate result of the full pipeline run."""

    project_dir: str
    started_at: str = ""
    completed_at: str = ""
    phase_results: list[BatchResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = datetime.now().isoformat()

    def complete(self) -> None:
        """Mark the pipeline as completed."""
        self.completed_at = datetime.now().isoformat()

    @property
    def all_succeeded(self) -> bool:
        return all(r.failed == 0 for r in self.phase_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_dir": self.project_dir,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "all_succeeded": self.all_succeeded,
            "phases": [r.to_dict() for r in self.phase_results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineResult:
        """Deserialize from dict. CQ-049: Implement missing from_dict."""
        phases = [BatchResult.from_dict(p) for p in data.get("phases", [])]
        result = cls(
            project_dir=data.get("project_dir", ""),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            phase_results=phases,
        )
        return result
