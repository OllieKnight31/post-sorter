#!/usr/bin/env python3
"""Type definitions for Gemini extraction results.

Provides dataclasses and type definitions for structured output
from German financial document extraction.

v4.0: Added TypedIdentifier, CourtInfo, DebtorInfo, FeesBreakdown,
related_references, category, person. Updated GermanDocumentTypeEnum
to 25 types. Updated FieldConfidence with person field.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .category_detector import DebtCategory


# ============================================================================
# Enums
# ============================================================================

class GermanDocumentTypeEnum(Enum):
    """Document types for German financial documents (v4: 25 types)."""
    # Financial Documents
    RECHNUNG = "rechnung"
    MAHNUNG_1 = "mahnung_1"
    MAHNUNG_2 = "mahnung_2"
    MAHNUNG_3 = "mahnung_3"
    ZAHLUNGSERINNERUNG = "zahlungserinnerung"
    KONTOAUSZUG = "kontoauszug"
    GEHALTSABRECHNUNG = "gehaltsabrechnung"
    # Legal Documents - Debt Collection
    INKASSO = "inkasso"
    MAHNBESCHEID = "mahnbescheid"
    VOLLSTRECKUNGSBESCHEID = "vollstreckungsbescheid"
    PFAENDUNG = "pfaendung"
    # Legal Documents - Advanced
    RAEUMUNGSKLAGE = "raeumungsklage"
    GERICHTSVOLLZIEHER = "gerichtsvollzieher"
    INSOLVENZVERFAHREN = "insolvenzverfahren"
    KOSTENFESTSETZUNGSANTRAG = "kostenfestsetzungsantrag"
    SCHULDTITEL = "schuldtitel"
    VERJAEHRUNGSBESCHEID = "verjaehrungsbescheid"
    RESTSCHULDBEFREIUNG = "restschuldbefreiung"
    # Contracts and Agreements
    VERTRAG = "vertrag"
    KUENDIGUNG = "kuendigung"
    # Official Documents
    BESCHEID = "bescheid"
    STEUERBESCHEID = "steuerbescheid"
    BUSSGELDBESCHEID = "bussgeldbescheid"
    # Correspondence
    ALLGEMEINE_KORRESPONDENZ = "allgemeine_korrespondenz"
    # Unknown
    UNKNOWN = "unknown"

    # === Backwards compatibility aliases ===
    # Old code using MAHNUNG or VOLLSTRECKUNG still works
    @classmethod
    def _missing_(cls, value: str) -> Optional["GermanDocumentTypeEnum"]:
        """Handle backwards-compatible lookups for old enum values."""
        compat_map = {
            "mahnung": cls.MAHNUNG_1,
            "vollstreckung": cls.VOLLSTRECKUNGSBESCHEID,
        }
        return compat_map.get(value)


class DebtStatus(Enum):
    """Status of a debt/claim."""
    OPEN = "open"
    PAID = "paid"
    DISPUTED = "disputed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class ThinkingLevel(Enum):
    """Thinking level configuration for Gemini.

    Higher thinking levels produce more accurate results
    but take longer and cost more.
    """
    NONE = "none"         # No extended thinking
    LOW = "low"           # ~10 seconds
    MEDIUM = "medium"     # ~30 seconds
    HIGH = "high"         # ~4 minutes


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GeminiExtractionConfig:
    """Configuration for Gemini extraction.

    Attributes:
        thinking_level: Level of extended thinking (none, low, medium, high)
        temperature: Temperature for generation (0.0-2.0, default 1.0)
        use_structured_output: Whether to use JSON schema enforcement
        include_raw_text: Whether to include raw text in output
        extract_tables: Whether to extract tables
        table_format: Format for tables (markdown, html)
        german_mode: Whether to use German financial prompts
        max_retries: Maximum retries for extraction
        validate_result: Whether to validate extraction result
    """
    thinking_level: ThinkingLevel = ThinkingLevel.MEDIUM
    temperature: float = 1.0
    use_structured_output: bool = True
    include_raw_text: bool = True
    extract_tables: bool = True
    table_format: str = "markdown"
    german_mode: bool = True
    max_retries: int = 2
    validate_result: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")

        if isinstance(self.thinking_level, str):
            self.thinking_level = ThinkingLevel(self.thinking_level)

    @property
    def thinking_budget_tokens(self) -> int:
        """Get thinking budget in tokens based on level.

        Returns:
            Token budget for thinking (0 for none)
        """
        budgets = {
            ThinkingLevel.NONE: 0,
            ThinkingLevel.LOW: 1024,
            ThinkingLevel.MEDIUM: 8192,
            ThinkingLevel.HIGH: 32768,
        }
        return budgets.get(self.thinking_level, 0)


# ============================================================================
# Field Confidence (v4: added person field)
# ============================================================================

@dataclass
class FieldConfidence:
    """Confidence scores per extraction field.

    Each field score represents the model's confidence (0.0-1.0) in
    the accuracy of that specific extraction. Used for review decisions.

    Attributes:
        creditor: Confidence in creditor name extraction
        identifiers: Confidence in reference number extraction
        amounts: Confidence in amount extraction accuracy
        dates: Confidence in date extraction accuracy
        payment_info: Confidence in IBAN/BIC extraction
        document_type: Confidence in document classification
        person: Confidence in person assignment (v4)
    """
    creditor: float = 0.0
    identifiers: float = 0.0
    amounts: float = 0.0
    dates: float = 0.0
    payment_info: float = 0.0
    document_type: float = 0.0
    person: float = 0.0

    @property
    def overall(self) -> float:
        """Calculate weighted average confidence.

        Weights reflect field importance for financial documents:
        - creditor/amounts: 22% each (critical for payment)
        - identifiers: 18% (important for tracking)
        - dates: 13% (important for deadlines)
        - payment_info: 10% (can be looked up)
        - document_type: 5% (classification)
        - person: 10% (critical for estate cases)

        Returns:
            Weighted average confidence (0.0-1.0)
        """
        weights = {
            'creditor': 0.22,
            'identifiers': 0.18,
            'amounts': 0.22,
            'dates': 0.13,
            'payment_info': 0.10,
            'document_type': 0.05,
            'person': 0.10,
        }
        return sum(getattr(self, k) * v for k, v in weights.items())

    @property
    def low_confidence_fields(self) -> List[str]:
        """Get fields with confidence below 0.7.

        Returns:
            List of field names with low confidence
        """
        threshold = 0.7
        fields = ['creditor', 'identifiers', 'amounts', 'dates', 'payment_info', 'document_type', 'person']
        return [f for f in fields if getattr(self, f) < threshold]

    @property
    def needs_review(self) -> bool:
        """Check if extraction needs manual review.

        Returns True if overall confidence is below 0.7 or
        any critical field (creditor, amounts) is below 0.6.

        Returns:
            True if manual review recommended
        """
        if self.overall < 0.7:
            return True
        if self.creditor < 0.6 or self.amounts < 0.6:
            return True
        return False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldConfidence":
        """Create from dictionary."""
        return cls(
            creditor=data.get("creditor", 0.0),
            identifiers=data.get("identifiers", 0.0),
            amounts=data.get("amounts", 0.0),
            dates=data.get("dates", 0.0),
            payment_info=data.get("payment_info", 0.0),
            document_type=data.get("document_type", 0.0),
            person=data.get("person", 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "creditor": self.creditor,
            "identifiers": self.identifiers,
            "amounts": self.amounts,
            "dates": self.dates,
            "payment_info": self.payment_info,
            "document_type": self.document_type,
            "person": self.person,
            "overall": round(self.overall, 3),
            "needs_review": self.needs_review,
            "low_confidence_fields": self.low_confidence_fields,
        }


# ============================================================================
# Result Types
# ============================================================================

@dataclass
class CreditorInfo:
    """Information about the creditor.

    Attributes:
        name: Name of the creditor
        is_inkasso: Whether this is a collection agency
        original_creditor: Original creditor if this is inkasso
    """
    name: str
    is_inkasso: bool = False
    original_creditor: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CreditorInfo":
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            is_inkasso=data.get("is_inkasso", False),
            original_creditor=data.get("original_creditor"),
        )


@dataclass
class TypedIdentifier:
    """A typed reference number with confidence (v4).

    Attributes:
        value: The identifier string (exact copy from document)
        type: Type of identifier (aktenzeichen, kundennummer, etc.)
        confidence: Confidence in this specific identifier (0.0-1.0)
    """
    value: str
    type: str
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TypedIdentifier":
        """Create from dictionary."""
        return cls(
            value=data.get("value", ""),
            type=data.get("type", "aktenzeichen"),
            confidence=data.get("confidence", 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "type": self.type,
            "confidence": self.confidence,
        }


@dataclass
class FeesBreakdown:
    """Breakdown of fees by type (v4).

    Attributes:
        mahngebuehren: Late payment fees
        zinsen: Interest
        gerichtskosten: Court costs
        anwaltskosten: Attorney fees
        zustellungskosten: Delivery/service costs
        sonstige: Other fees
    """
    mahngebuehren: float = 0.0
    zinsen: float = 0.0
    gerichtskosten: float = 0.0
    anwaltskosten: float = 0.0
    zustellungskosten: float = 0.0
    sonstige: float = 0.0

    @property
    def total(self) -> float:
        """Sum of all fee components."""
        return (self.mahngebuehren + self.zinsen + self.gerichtskosten +
                self.anwaltskosten + self.zustellungskosten + self.sonstige)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeesBreakdown":
        """Create from dictionary."""
        return cls(
            mahngebuehren=data.get("mahngebuehren", 0.0),
            zinsen=data.get("zinsen", 0.0),
            gerichtskosten=data.get("gerichtskosten", 0.0),
            anwaltskosten=data.get("anwaltskosten", 0.0),
            zustellungskosten=data.get("zustellungskosten", 0.0),
            sonstige=data.get("sonstige", 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mahngebuehren": self.mahngebuehren,
            "zinsen": self.zinsen,
            "gerichtskosten": self.gerichtskosten,
            "anwaltskosten": self.anwaltskosten,
            "zustellungskosten": self.zustellungskosten,
            "sonstige": self.sonstige,
        }


@dataclass
class AmountInfo:
    """Structured amount information.

    Distinguishes between original amount, current amount,
    and fees/interest.

    Attributes:
        original: Original invoice amount (before fees)
        current: Current total amount (including fees)
        fees: Fees, interest, and costs separately (backwards compat)
        fees_breakdown: Detailed fee breakdown by type (v4)
    """
    original: Optional[float] = None
    current: Optional[float] = None
    fees: Optional[float] = None
    fees_breakdown: Optional[FeesBreakdown] = None

    @property
    def total(self) -> float:
        """Calculate total amount."""
        if self.current is not None:
            return self.current
        if self.original is not None:
            return self.original + (self.fees or 0)
        return 0.0

    @property
    def has_fees(self) -> bool:
        """Check if fees are present."""
        if self.fees_breakdown and self.fees_breakdown.total > 0:
            return True
        return self.fees is not None and self.fees > 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AmountInfo":
        """Create from dictionary."""
        fees_breakdown = None
        if "fees_breakdown" in data and data["fees_breakdown"]:
            fees_breakdown = FeesBreakdown.from_dict(data["fees_breakdown"])

        return cls(
            original=data.get("original"),
            current=data.get("current"),
            fees=data.get("fees"),
            fees_breakdown=fees_breakdown,
        )


@dataclass
class DateInfo:
    """Structured date information.

    Attributes:
        document_date: Date of the document
        due_date: Payment due date
        original_invoice_date: Original invoice date (for reminders)
    """
    document_date: Optional[str] = None
    due_date: Optional[str] = None
    original_invoice_date: Optional[str] = None

    def get_document_date_parsed(self) -> Optional[date]:
        """Parse document date to date object."""
        return self._parse_german_date(self.document_date)

    def get_due_date_parsed(self) -> Optional[date]:
        """Parse due date to date object."""
        return self._parse_german_date(self.due_date)

    def _parse_german_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parse German date format (DD.MM.YYYY) to date."""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%d.%m.%Y").date()
        except ValueError:
            try:
                return datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                return None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DateInfo":
        """Create from dictionary."""
        return cls(
            document_date=data.get("document_date"),
            due_date=data.get("due_date"),
            original_invoice_date=data.get("original_invoice_date"),
        )


@dataclass
class PaymentInfo:
    """Payment information from document.

    Attributes:
        iban: IBAN for payment
        bic: BIC/SWIFT code
        reference: Payment reference/Verwendungszweck
        recipient_name: Name of payment recipient
    """
    iban: Optional[str] = None
    bic: Optional[str] = None
    reference: Optional[str] = None
    recipient_name: Optional[str] = None

    @property
    def has_iban(self) -> bool:
        """Check if IBAN is present."""
        return self.iban is not None and len(self.iban) > 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaymentInfo":
        """Create from dictionary."""
        return cls(
            iban=data.get("iban"),
            bic=data.get("bic"),
            reference=data.get("reference"),
            recipient_name=data.get("recipient_name"),
        )


@dataclass
class CourtInfo:
    """Court information for legal documents (v4).

    Attributes:
        gericht: Name of the court
        zustellungsdatum: Date of service/delivery
        widerspruchsfrist: Objection deadline
        verfahrenskosten: Court proceedings costs
    """
    gericht: Optional[str] = None
    zustellungsdatum: Optional[str] = None
    widerspruchsfrist: Optional[str] = None
    verfahrenskosten: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CourtInfo":
        """Create from dictionary."""
        return cls(
            gericht=data.get("gericht"),
            zustellungsdatum=data.get("zustellungsdatum"),
            widerspruchsfrist=data.get("widerspruchsfrist"),
            verfahrenskosten=data.get("verfahrenskosten"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gericht": self.gericht,
            "zustellungsdatum": self.zustellungsdatum,
            "widerspruchsfrist": self.widerspruchsfrist,
            "verfahrenskosten": self.verfahrenskosten,
        }


@dataclass
class DebtorInfo:
    """Debtor/addressee information (v4).

    Attributes:
        name: Name of the debtor/addressee
        address_line: Address line
    """
    name: Optional[str] = None
    address_line: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DebtorInfo":
        """Create from dictionary."""
        return cls(
            name=data.get("name"),
            address_line=data.get("address_line"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "address_line": self.address_line,
        }


@dataclass
class PageInfo:
    """Page information for multi-page documents.

    Attributes:
        current_page: Current page number
        total_pages: Total number of pages
        page_hint_text: Original page text (e.g., "Seite 1 von 3")
    """
    current_page: Optional[int] = None
    total_pages: Optional[int] = None
    page_hint_text: Optional[str] = None

    @property
    def is_multi_page(self) -> bool:
        """Check if document is multi-page."""
        return self.total_pages is not None and self.total_pages > 1

    @property
    def is_last_page(self) -> bool:
        """Check if this is the last page."""
        if self.current_page is None or self.total_pages is None:
            return True
        return self.current_page >= self.total_pages

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageInfo":
        """Create from dictionary."""
        return cls(
            current_page=data.get("current_page"),
            total_pages=data.get("total_pages"),
            page_hint_text=data.get("page_hint_text"),
        )


@dataclass
class DisputeInfo:
    """Information about dispute status.

    Attributes:
        status: Current status of the debt
        dispute_reason: Reason for dispute if disputed
        dispute_date: Date dispute was filed
        evidence_urls: URLs to evidence documents
    """
    status: DebtStatus = DebtStatus.OPEN
    dispute_reason: Optional[str] = None
    dispute_date: Optional[str] = None
    evidence_urls: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisputeInfo":
        """Create from dictionary."""
        status_str = data.get("status", "open")
        try:
            status = DebtStatus(status_str)
        except ValueError:
            status = DebtStatus.OPEN

        return cls(
            status=status,
            dispute_reason=data.get("dispute_reason"),
            dispute_date=data.get("dispute_date"),
            evidence_urls=data.get("evidence_urls", []),
        )


# ============================================================================
# Main Result Type
# ============================================================================

@dataclass
class GermanExtractionResult:
    """Complete extraction result from German financial document.

    This is the main result type returned by Gemini extraction
    for German financial documents.

    v4.0: Added typed_identifiers, fees_breakdown, court_info, debtor,
    related_references, category, person fields.

    Attributes:
        extraction_successful: Whether extraction succeeded
        error_reason: Error reason if extraction failed
        document_type: Type of document (25 types in v4)
        is_court_case: Whether this is a court document
        creditor: Creditor information
        debtor: Debtor/addressee information (v4)
        person: Assigned person (one of the configured names, or None)
        identifiers: All reference numbers found (flat list, backwards compat)
        typed_identifiers: Typed reference numbers with confidence (v4)
        amounts: Amount information (v4: includes fees_breakdown)
        dates: Date information
        payment_info: Payment details
        court_info: Court information for legal documents (v4)
        page_info: Page information
        related_references: Cross-reference texts (v4)
        urgency_signals: Urgency keywords found
        raw_text: Full extracted text
        confidence: Extraction confidence (0-100)
        category: Detected document category (v4)
        category_confidence: Category detection confidence (0.0-1.0)
        field_confidence: Per-field confidence scores
    """
    extraction_successful: bool
    error_reason: Optional[str] = None
    document_type: GermanDocumentTypeEnum = GermanDocumentTypeEnum.UNKNOWN
    is_court_case: bool = False
    creditor: Optional[CreditorInfo] = None
    debtor: Optional[DebtorInfo] = None
    person: Optional[str] = None
    identifiers: List[str] = field(default_factory=list)
    typed_identifiers: List[TypedIdentifier] = field(default_factory=list)
    amounts: Optional[AmountInfo] = None
    dates: Optional[DateInfo] = None
    payment_info: Optional[PaymentInfo] = None
    court_info: Optional[CourtInfo] = None
    page_info: Optional[PageInfo] = None
    related_references: List[str] = field(default_factory=list)
    urgency_signals: List[str] = field(default_factory=list)
    raw_text: Optional[str] = None
    confidence: float = 0.0
    category: Optional[str] = None
    category_confidence: float = 0.0
    field_confidence: Optional[FieldConfidence] = None

    @property
    def is_urgent(self) -> bool:
        """Check if document signals urgency."""
        return len(self.urgency_signals) > 0

    @property
    def is_legal_document(self) -> bool:
        """Check if this is a legal/court document."""
        legal_types = [
            GermanDocumentTypeEnum.MAHNBESCHEID,
            GermanDocumentTypeEnum.VOLLSTRECKUNGSBESCHEID,
            GermanDocumentTypeEnum.PFAENDUNG,
            GermanDocumentTypeEnum.RAEUMUNGSKLAGE,
            GermanDocumentTypeEnum.GERICHTSVOLLZIEHER,
            GermanDocumentTypeEnum.INSOLVENZVERFAHREN,
            GermanDocumentTypeEnum.KOSTENFESTSETZUNGSANTRAG,
            GermanDocumentTypeEnum.SCHULDTITEL,
        ]
        return self.is_court_case or self.document_type in legal_types

    @property
    def primary_amount(self) -> Optional[float]:
        """Get the primary (current) amount."""
        if self.amounts:
            return self.amounts.total
        return None

    @property
    def primary_identifier(self) -> Optional[str]:
        """Get the first/primary identifier."""
        if self.identifiers:
            return self.identifiers[0]
        return None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GermanExtractionResult":
        """Create from dictionary (e.g., from JSON response)."""
        # Handle extraction failure
        if not data.get("extraction_successful", True):
            return cls(
                extraction_successful=False,
                error_reason=data.get("error_reason", "Unknown error"),
            )

        # Parse document type
        doc_type_str = data.get("document_type", "unknown")
        try:
            doc_type = GermanDocumentTypeEnum(doc_type_str)
        except ValueError:
            doc_type = GermanDocumentTypeEnum.UNKNOWN

        # Parse nested objects
        creditor = None
        if "creditor" in data and data["creditor"]:
            creditor = CreditorInfo.from_dict(data["creditor"])

        debtor = None
        if "debtor" in data and data["debtor"]:
            debtor = DebtorInfo.from_dict(data["debtor"])

        amounts = None
        if "amounts" in data and data["amounts"]:
            amounts = AmountInfo.from_dict(data["amounts"])

        dates = None
        if "dates" in data and data["dates"]:
            dates = DateInfo.from_dict(data["dates"])

        payment_info = None
        if "payment_info" in data and data["payment_info"]:
            payment_info = PaymentInfo.from_dict(data["payment_info"])

        court_info = None
        if "court_info" in data and data["court_info"]:
            court_info = CourtInfo.from_dict(data["court_info"])

        page_info = None
        if "page_info" in data and data["page_info"]:
            page_info = PageInfo.from_dict(data["page_info"])

        field_confidence = None
        if "field_confidence" in data and data["field_confidence"]:
            field_confidence = FieldConfidence.from_dict(data["field_confidence"])

        # Parse typed identifiers (v4)
        typed_identifiers = []
        if "typed_identifiers" in data and data["typed_identifiers"]:
            typed_identifiers = [
                TypedIdentifier.from_dict(ti) for ti in data["typed_identifiers"]
            ]

        return cls(
            extraction_successful=True,
            document_type=doc_type,
            is_court_case=data.get("is_court_case", False),
            creditor=creditor,
            debtor=debtor,
            person=data.get("person"),
            identifiers=data.get("identifiers", []),
            typed_identifiers=typed_identifiers,
            amounts=amounts,
            dates=dates,
            payment_info=payment_info,
            court_info=court_info,
            page_info=page_info,
            related_references=data.get("related_references", []),
            urgency_signals=data.get("urgency_signals", []),
            raw_text=data.get("raw_text"),
            confidence=data.get("confidence", 0.0),
            category=data.get("category"),
            category_confidence=data.get("category_confidence", 0.0),
            field_confidence=field_confidence,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "extraction_successful": self.extraction_successful,
            "document_type": self.document_type.value,
            "is_court_case": self.is_court_case,
            "confidence": self.confidence,
        }

        if self.error_reason:
            result["error_reason"] = self.error_reason

        if self.creditor:
            result["creditor"] = {
                "name": self.creditor.name,
                "is_inkasso": self.creditor.is_inkasso,
                "original_creditor": self.creditor.original_creditor,
            }

        if self.debtor:
            result["debtor"] = self.debtor.to_dict()

        if self.person:
            result["person"] = self.person

        if self.identifiers:
            result["identifiers"] = self.identifiers

        if self.typed_identifiers:
            result["typed_identifiers"] = [ti.to_dict() for ti in self.typed_identifiers]

        if self.amounts:
            result["amounts"] = {
                "original": self.amounts.original,
                "current": self.amounts.current,
                "fees": self.amounts.fees,
            }
            if self.amounts.fees_breakdown:
                result["amounts"]["fees_breakdown"] = self.amounts.fees_breakdown.to_dict()

        if self.dates:
            result["dates"] = {
                "document_date": self.dates.document_date,
                "due_date": self.dates.due_date,
                "original_invoice_date": self.dates.original_invoice_date,
            }

        if self.payment_info:
            result["payment_info"] = {
                "iban": self.payment_info.iban,
                "bic": self.payment_info.bic,
                "reference": self.payment_info.reference,
                "recipient_name": self.payment_info.recipient_name,
            }

        if self.court_info:
            result["court_info"] = self.court_info.to_dict()

        if self.page_info:
            result["page_info"] = {
                "current_page": self.page_info.current_page,
                "total_pages": self.page_info.total_pages,
                "page_hint_text": self.page_info.page_hint_text,
            }

        if self.related_references:
            result["related_references"] = self.related_references

        if self.urgency_signals:
            result["urgency_signals"] = self.urgency_signals

        if self.raw_text:
            result["raw_text"] = self.raw_text

        if self.category:
            result["category"] = self.category
            result["category_confidence"] = self.category_confidence

        if self.field_confidence:
            result["field_confidence"] = self.field_confidence.to_dict()

        return result


# ============================================================================
# Validation
# ============================================================================

def validate_nested_objects(data: Dict[str, Any]) -> List[str]:
    """Validate nested objects in extraction result.

    Recursively validates creditor, amounts, dates, and payment_info
    objects for required fields and value types.

    Args:
        data: Extraction result dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Validate creditor object
    if creditor := data.get("creditor"):
        if not isinstance(creditor, dict):
            errors.append("creditor must be an object")
        else:
            if not creditor.get("name"):
                errors.append("creditor.name is required when creditor is present")
            if "is_inkasso" in creditor and not isinstance(creditor["is_inkasso"], bool):
                errors.append("creditor.is_inkasso must be a boolean")

    # Validate amounts object
    if amounts := data.get("amounts"):
        if not isinstance(amounts, dict):
            errors.append("amounts must be an object")
        else:
            for field_name in ["original", "current", "fees"]:
                if field_name in amounts and amounts[field_name] is not None:
                    if not isinstance(amounts[field_name], (int, float)):
                        errors.append(f"amounts.{field_name} must be numeric")
                    elif amounts[field_name] < 0:
                        errors.append(f"amounts.{field_name} cannot be negative")

            # Validate amount relationships
            original = amounts.get("original")
            current = amounts.get("current")
            if original is not None and current is not None:
                # current should generally be >= original (fees added)
                # but allow some tolerance for rounding
                if current < original * 0.99:
                    errors.append(
                        f"amounts.current ({current}) is less than amounts.original ({original})"
                    )

            # Validate fees_breakdown (v4)
            if fees_breakdown := amounts.get("fees_breakdown"):
                if not isinstance(fees_breakdown, dict):
                    errors.append("amounts.fees_breakdown must be an object")
                else:
                    for fee_field in ["mahngebuehren", "zinsen", "gerichtskosten",
                                      "anwaltskosten", "zustellungskosten", "sonstige"]:
                        if fee_field in fees_breakdown and fees_breakdown[fee_field] is not None:
                            if not isinstance(fees_breakdown[fee_field], (int, float)):
                                errors.append(f"amounts.fees_breakdown.{fee_field} must be numeric")
                            elif fees_breakdown[fee_field] < 0:
                                errors.append(f"amounts.fees_breakdown.{fee_field} cannot be negative")

    # Validate dates object
    if dates := data.get("dates"):
        if not isinstance(dates, dict):
            errors.append("dates must be an object")
        else:
            import re
            date_pattern = re.compile(r"^\d{2}\.\d{2}\.\d{4}$|^\d{4}-\d{2}-\d{2}$")

            for field_name in ["document_date", "due_date", "original_invoice_date"]:
                if field_name in dates and dates[field_name] is not None:
                    if not isinstance(dates[field_name], str):
                        errors.append(f"dates.{field_name} must be a string")
                    elif not date_pattern.match(dates[field_name]):
                        errors.append(
                            f"dates.{field_name} must be in DD.MM.YYYY or YYYY-MM-DD format, "
                            f"got: {dates[field_name]}"
                        )

    # Validate payment_info object
    if payment_info := data.get("payment_info"):
        if not isinstance(payment_info, dict):
            errors.append("payment_info must be an object")
        else:
            import re

            # Validate IBAN format (basic check: DE + 20 digits, or other country codes)
            if iban := payment_info.get("iban"):
                if not isinstance(iban, str):
                    errors.append("payment_info.iban must be a string")
                else:
                    # Remove spaces for validation
                    iban_clean = iban.replace(" ", "")
                    if not re.match(r"^[A-Z]{2}\d{2}[A-Z0-9]{10,30}$", iban_clean):
                        errors.append(f"payment_info.iban has invalid format: {iban}")

            # Validate BIC format (8 or 11 characters)
            if bic := payment_info.get("bic"):
                if not isinstance(bic, str):
                    errors.append("payment_info.bic must be a string")
                elif len(bic) not in [8, 11]:
                    errors.append(f"payment_info.bic must be 8 or 11 characters, got: {len(bic)}")

    # Validate court_info object (v4)
    if court_info := data.get("court_info"):
        if not isinstance(court_info, dict):
            errors.append("court_info must be an object")
        else:
            if "verfahrenskosten" in court_info and court_info["verfahrenskosten"] is not None:
                if not isinstance(court_info["verfahrenskosten"], (int, float)):
                    errors.append("court_info.verfahrenskosten must be numeric")
                elif court_info["verfahrenskosten"] < 0:
                    errors.append("court_info.verfahrenskosten cannot be negative")

    # Validate page_info object
    if page_info := data.get("page_info"):
        if not isinstance(page_info, dict):
            errors.append("page_info must be an object")
        else:
            for field_name in ["current_page", "total_pages"]:
                if field_name in page_info and page_info[field_name] is not None:
                    if not isinstance(page_info[field_name], int):
                        errors.append(f"page_info.{field_name} must be an integer")
                    elif page_info[field_name] < 1:
                        errors.append(f"page_info.{field_name} must be >= 1")

            # current_page should not exceed total_pages
            current = page_info.get("current_page")
            total = page_info.get("total_pages")
            if current is not None and total is not None and current > total:
                errors.append(
                    f"page_info.current_page ({current}) exceeds total_pages ({total})"
                )

    # Validate typed_identifiers (v4)
    if typed_ids := data.get("typed_identifiers"):
        if not isinstance(typed_ids, list):
            errors.append("typed_identifiers must be a list")
        else:
            from .extraction_schemas import IDENTIFIER_TYPE_ENUM
            for i, tid in enumerate(typed_ids):
                if not isinstance(tid, dict):
                    errors.append(f"typed_identifiers[{i}] must be an object")
                elif not tid.get("value"):
                    errors.append(f"typed_identifiers[{i}].value is required")
                elif tid.get("type") and tid["type"] not in IDENTIFIER_TYPE_ENUM:
                    errors.append(f"typed_identifiers[{i}].type '{tid['type']}' is not valid")

    return errors


def validate_extraction_result(data: Any) -> bool:
    """Validate that data matches GermanExtractionResult structure.

    Args:
        data: Data to validate

    Returns:
        True if data is valid, False otherwise
    """
    result = validate_extraction_result_detailed(data)
    return result["valid"]


def validate_extraction_result_detailed(data: Any) -> Dict[str, Any]:
    """Validate extraction result with detailed error reporting.

    Performs comprehensive validation including:
    - Top-level structure validation
    - Document type validation
    - Confidence range validation
    - Nested object validation (creditor, amounts, dates, payment_info)

    Args:
        data: Data to validate

    Returns:
        Dictionary with:
        - valid: bool - whether validation passed
        - errors: List[str] - list of validation errors
        - warnings: List[str] - non-fatal issues
    """
    errors = []
    warnings = []

    if not isinstance(data, dict):
        return {"valid": False, "errors": ["Data must be a dictionary"], "warnings": []}

    # Must have extraction_successful boolean
    if "extraction_successful" not in data:
        errors.append("Missing required field: extraction_successful")
        return {"valid": False, "errors": errors, "warnings": warnings}

    if not isinstance(data["extraction_successful"], bool):
        errors.append("extraction_successful must be a boolean")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # If extraction failed, should have error_reason
    if data["extraction_successful"] is False:
        if "error_reason" not in data:
            warnings.append("Failed extraction should have error_reason")
        return {"valid": True, "errors": errors, "warnings": warnings}

    # If extraction succeeded, validate structure
    if data["extraction_successful"] is True:
        # Document type validation (v4: uses DOCUMENT_TYPE_ENUM)
        if "document_type" in data:
            from .extraction_schemas import DOCUMENT_TYPE_ENUM
            if data["document_type"] not in DOCUMENT_TYPE_ENUM:
                errors.append(
                    f"Invalid document_type: {data['document_type']}. "
                    f"Must be one of the 25 valid types."
                )

        # Confidence validation
        if "confidence" in data:
            if not isinstance(data["confidence"], (int, float)):
                errors.append("confidence must be numeric")
            elif not 0 <= data["confidence"] <= 100:
                errors.append(f"confidence must be 0-100, got: {data['confidence']}")

        # Identifiers validation
        if "identifiers" in data:
            if not isinstance(data["identifiers"], list):
                errors.append("identifiers must be a list")

        # Category validation (v4)
        if "category" in data and data["category"] is not None:
            from .extraction_schemas import CATEGORY_ENUM
            if data["category"] not in CATEGORY_ENUM:
                errors.append(
                    f"Invalid category: {data['category']}. "
                    f"Must be one of: {CATEGORY_ENUM}"
                )

        # Nested object validation (creditor, amounts, dates, payment_info, page_info, court_info)
        nested_errors = validate_nested_objects(data)
        errors.extend(nested_errors)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ============================================================================
# German Number Format Verification
# ============================================================================

def verify_german_number_format(
    raw_text: str,
    extracted_amounts: Dict[str, Optional[float]],
) -> Dict[str, Any]:
    """Verify extracted amounts match German number format in source text.

    Detects common extraction errors:
    - 100x error: Amount incorrectly extracted as cents (103,79 EUR -> 10379)
    - 1000x error: Thousands separator misinterpreted (1.234,56 EUR -> 123456)

    Args:
        raw_text: Original OCR text containing German-formatted numbers
        extracted_amounts: Dictionary with amounts (original, current, fees)

    Returns:
        Dictionary with:
        - valid: bool - whether amounts appear correct
        - issues: List of detected format issues
        - suggested_corrections: Dict mapping field to corrected value
    """
    import re

    issues = []
    corrections = {}

    # Pattern to find German currency amounts in text
    # Matches: 103,79 EUR, 1.234,56EUR, EUR 99,00, EUR12,50
    german_amount_pattern = re.compile(
        r'(?:EUR\s*|€\s*)?'                    # Optional EUR or EUR prefix
        r'(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)'    # German number format
        r'(?:\s*(?:EUR|€))?'                    # Optional EUR or EUR suffix
        r'|'
        r'(\d+,\d{2})'                          # Simple decimal (99,00)
        r'(?:\s*(?:EUR|€))?',
        re.IGNORECASE
    )

    # Extract all amounts from raw text and convert to floats
    text_amounts = []
    for match in german_amount_pattern.finditer(raw_text):
        amount_str = match.group(1) or match.group(2)
        if amount_str:
            # Convert German format to float: 1.234,56 -> 1234.56
            normalized = amount_str.replace(".", "").replace(",", ".")
            try:
                amount = float(normalized)
                if amount > 0:  # Ignore zero amounts
                    text_amounts.append(amount)
            except ValueError:
                continue

    if not text_amounts:
        return {"valid": True, "issues": [], "suggested_corrections": {}}

    # Check each extracted amount against text amounts
    for field_name, extracted in extracted_amounts.items():
        if extracted is None or extracted <= 0:
            continue

        # Check for 100x error (cents vs euros)
        expected_euros = extracted / 100.0
        for text_amount in text_amounts:
            # If dividing by 100 gives us a number that matches text, it's a 100x error
            if abs(expected_euros - text_amount) < 0.01:
                issues.append(
                    f"Potential 100x error in {field_name}: extracted {extracted}, "
                    f"but text shows {text_amount:.2f} (appears to be in cents)"
                )
                corrections[field_name] = text_amount
                break

        # Check for 1000x error (missing decimal)
        expected_with_decimal = extracted / 1000.0
        for text_amount in text_amounts:
            if abs(expected_with_decimal - text_amount) < 0.01:
                issues.append(
                    f"Potential 1000x error in {field_name}: extracted {extracted}, "
                    f"but text shows {text_amount:.2f}"
                )
                corrections[field_name] = text_amount
                break

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "suggested_corrections": corrections,
    }


def detect_and_fix_amount_errors(
    extraction_result: "GermanExtractionResult",
    raw_text: str,
) -> "GermanExtractionResult":
    """Detect and automatically fix German number format errors.

    This is a post-processing step that checks extracted amounts against
    the source text and corrects common conversion errors.

    Args:
        extraction_result: The extraction result to check
        raw_text: Original OCR text

    Returns:
        Updated extraction result with corrected amounts (if needed)
    """
    if not extraction_result.extraction_successful:
        return extraction_result

    if not extraction_result.amounts:
        return extraction_result

    amounts_dict = {
        "original": extraction_result.amounts.original,
        "current": extraction_result.amounts.current,
        "fees": extraction_result.amounts.fees,
    }

    verification = verify_german_number_format(raw_text, amounts_dict)

    if verification["valid"]:
        return extraction_result

    # Apply corrections
    for field_name, corrected_value in verification["suggested_corrections"].items():
        if field_name == "original":
            extraction_result.amounts.original = corrected_value
        elif field_name == "current":
            extraction_result.amounts.current = corrected_value
        elif field_name == "fees":
            extraction_result.amounts.fees = corrected_value

    # Log the corrections (if logging available)
    import logging
    logger = logging.getLogger("briefpost-ocr.validation")
    for issue in verification["issues"]:
        logger.warning(f"Amount format correction: {issue}")

    return extraction_result


# ============================================================================
# Confidence Adjustment
# ============================================================================

# Penalty weights for different validation rules
VALIDATION_PENALTIES: Dict[str, float] = {
    # High severity - significant penalties
    "current_gte_original": -0.20,      # Amount relationship violation
    "court_needs_aktenzeichen": -0.15,  # Missing required field for court docs
    "iban_format": -0.15,               # Invalid IBAN format
    "date_in_future": -0.25,            # Clearly impossible date
    # Medium severity - moderate penalties
    "due_after_document": -0.10,        # Date relationship violation
    "amount_plausibility": -0.08,       # Suspicious amount
    # Low severity - minor penalties
    "iban_country_code": -0.03,         # Non-German IBAN (might be valid)
    "date_too_old": -0.05,              # Old document (might be valid)
}


def adjust_confidence_for_validation(
    base_confidence: float,
    validation_issues: List[Dict[str, Any]],
) -> float:
    """Adjust extraction confidence based on validation failures.

    Applies penalties to the base confidence score when validation
    rules identify issues. Higher severity issues result in larger
    penalties.

    Args:
        base_confidence: Original confidence score (0.0-1.0 or 0-100)
        validation_issues: List of issues from SemanticValidator

    Returns:
        Adjusted confidence score, clamped to [0.0, 1.0] or [0, 100]

    Example:
        >>> issues = [{"rule": "current_gte_original", "severity": "high"}]
        >>> adjust_confidence_for_validation(0.9, issues)
        0.7  # -0.20 penalty applied
    """
    # Normalize to 0-1 scale if needed
    is_percentage = base_confidence > 1.0
    if is_percentage:
        confidence = base_confidence / 100.0
    else:
        confidence = base_confidence

    # Apply penalties for each validation issue
    for issue in validation_issues:
        rule = issue.get("rule", "")
        severity = issue.get("severity", "low")

        # Get specific penalty for rule, or use severity-based default
        penalty = VALIDATION_PENALTIES.get(rule)

        if penalty is None:
            # Default penalties by severity
            severity_defaults = {
                "high": -0.15,
                "medium": -0.08,
                "low": -0.03,
            }
            penalty = severity_defaults.get(severity, -0.05)

        confidence += penalty

    # Clamp to valid range
    confidence = max(0.0, min(1.0, confidence))

    # Convert back to percentage scale if needed
    if is_percentage:
        return round(confidence * 100.0, 1)
    return round(confidence, 3)


def calculate_overall_confidence(
    field_confidence: "FieldConfidence",
    validation_issues: Optional[List[Dict[str, Any]]] = None,
) -> float:
    """Calculate overall confidence from field-level scores and validation.

    Combines weighted field confidence with validation penalties to
    produce a single overall confidence score.

    Args:
        field_confidence: Per-field confidence scores
        validation_issues: Optional list of validation issues

    Returns:
        Overall confidence score (0.0-1.0)
    """
    # Start with weighted average from field confidence
    overall = field_confidence.overall

    # Apply validation penalties if provided
    if validation_issues:
        overall = adjust_confidence_for_validation(overall, validation_issues)

    return overall
