"""Constants and enumerations for briefpost-management skill.

Every hardcoded string value from the original pipeline scripts
is mapped to an enum member or constant here.
"""

from __future__ import annotations

from enum import Enum


# ---------------------------------------------------------------------------
# Document Types (Section 9 of CLAUDE.md)
# ---------------------------------------------------------------------------

class DocumentType(str, Enum):
    """15 document types with their German names."""

    RECHNUNG = "rechnung"
    MAHNUNG_1 = "mahnung_1"
    MAHNUNG_2 = "mahnung_2"
    MAHNUNG_3 = "mahnung_3"
    MAHNBESCHEID = "mahnbescheid"
    VOLLSTRECKUNG = "vollstreckung"
    PFAENDUNG = "pfaendung"
    BESCHEID = "bescheid"
    VERSICHERUNG = "versicherung"
    VERTRAG = "vertrag"
    KUENDIGUNG = "kuendigung"
    INFO = "info"
    KONTOAUSZUG = "kontoauszug"
    ERBSCHAFT = "erbschaft"
    SONSTIG = "sonstig"


# ---------------------------------------------------------------------------
# Categories (Section 10)
# ---------------------------------------------------------------------------

class Category(str, Enum):
    """8 document categories."""

    GESUNDHEIT = "GESUNDHEIT"
    VERSICHERUNG = "VERSICHERUNG"
    WOHNEN = "WOHNEN"
    KOMMUNIKATION = "KOMMUNIKATION"
    STAAT_RECHT = "STAAT_RECHT"
    ERBSCHAFT = "ERBSCHAFT"
    FINANZEN = "FINANZEN"
    SONSTIGES = "SONSTIGES"


# ---------------------------------------------------------------------------
# Priority (Section 11)
# ---------------------------------------------------------------------------

class Priority(str, Enum):
    """5 priority levels, ordered from most to least urgent."""

    GERICHTLICH = "GERICHTLICH"
    KRITISCH = "KRITISCH"
    DRINGEND = "DRINGEND"
    NORMAL = "NORMAL"
    ERLEDIGT = "ERLEDIGT"

    @property
    def sort_order(self) -> int:
        """Numeric sort order (lower = more urgent)."""
        return _PRIORITY_ORDER.get(self, 99)


_PRIORITY_ORDER: dict[Priority, int] = {
    Priority.GERICHTLICH: 0,
    Priority.KRITISCH: 1,
    Priority.DRINGEND: 2,
    Priority.NORMAL: 3,
    Priority.ERLEDIGT: 4,
}


# ---------------------------------------------------------------------------
# Reconciliation Status (Section 13)
# ---------------------------------------------------------------------------

class Status(str, Enum):
    """5 reconciliation status values."""

    OFFEN = "OFFEN"
    BEZAHLT = "BEZAHLT"
    TEILWEISE = "TEILWEISE"
    UNKLAR = "UNKLAR"
    INFO_ONLY = "INFO-ONLY"


# ---------------------------------------------------------------------------
# Escalation Levels
# ---------------------------------------------------------------------------

class EscalationLevel(int, Enum):
    """Escalation levels mapped from document types (0-6)."""

    RECHNUNG = 0
    MAHNUNG_1 = 1
    MAHNUNG_2 = 2
    MAHNUNG_3 = 3
    MAHNBESCHEID = 4
    VOLLSTRECKUNG = 5
    PFAENDUNG = 6


# Mapping from DocumentType to escalation level
ESCALATION_MAP: dict[DocumentType, int] = {
    DocumentType.RECHNUNG: EscalationLevel.RECHNUNG,
    DocumentType.MAHNUNG_1: EscalationLevel.MAHNUNG_1,
    DocumentType.MAHNUNG_2: EscalationLevel.MAHNUNG_2,
    DocumentType.MAHNUNG_3: EscalationLevel.MAHNUNG_3,
    DocumentType.MAHNBESCHEID: EscalationLevel.MAHNBESCHEID,
    DocumentType.VOLLSTRECKUNG: EscalationLevel.VOLLSTRECKUNG,
    DocumentType.PFAENDUNG: EscalationLevel.PFAENDUNG,
}


def get_escalation(doc_type: str) -> int:
    """Get escalation level for a document type string.

    Returns 0 for unknown document types (non-escalating).
    """
    try:
        dt = DocumentType(doc_type)
        return ESCALATION_MAP.get(dt, 0)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Match Methods
# ---------------------------------------------------------------------------

class MatchMethod(str, Enum):
    """6 methods used to match letters into Vorgaenge or reconcile with bank."""

    AKTENZEICHEN_EXACT = "aktenzeichen_exact"
    CREDITOR_AMOUNT = "creditor_amount"
    SINGLE_CREDITOR = "single_creditor"
    SINGLE_UNKNOWN = "single_unknown"
    ORPHAN = "orphan"
    LLM = "llm_matched"


class ReconciliationMethod(str, Enum):
    """Methods for bank reconciliation matching."""

    REFERENCE_EXACT = "reference_exact"
    REFERENCE_AMOUNT_DIFF = "reference_amount_diff"
    AMOUNT_MATCH = "amount_match"
    FUZZY_REFERENCE = "fuzzy_reference"
    COMPOUND = "compound"
    LLM_MATCH = "llm_match"
    NO_MATCH = "no_match"


# ---------------------------------------------------------------------------
# OCR Providers
# ---------------------------------------------------------------------------

class OCRProvider(str, Enum):
    """OCR processing providers."""

    GEMINI = "gemini"
    GEMINI_PRO = "gemini_pro"
    CONSENSUS = "consensus"


# ---------------------------------------------------------------------------
# QC/Pipeline Flags
# ---------------------------------------------------------------------------

class Flag(str, Enum):
    """All flags used throughout the pipeline.

    CQ-056: Grouped by origin phase and purpose for maintainability.
    When adding new flags, place them under the correct phase section.
    """

    # Phase A: Grouping flags
    OVERSIZED_GROUP = "OVERSIZED_GROUP"
    SINGLETON = "SINGLETON"
    AMBIGUOUS_BOUNDARY = "AMBIGUOUS_BOUNDARY"
    MODERATE_CONFIDENCE_GROUP = "MODERATE_CONFIDENCE_GROUP"

    # Phase B: OCR flags
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    VERY_LOW_CONFIDENCE = "VERY_LOW_CONFIDENCE"
    MISSING_AMOUNT = "MISSING_AMOUNT"
    MISSING_IBAN = "MISSING_IBAN"
    INVALID_IBAN = "INVALID_IBAN"
    INVALID_BIC_LENGTH = "INVALID_BIC_LENGTH"
    HAS_UNREADABLE_PAGE = "HAS_UNREADABLE_PAGE"
    NEEDS_VISION = "NEEDS_VISION"
    LOW_IMAGE_QUALITY = "LOW_IMAGE_QUALITY"
    CROSS_PAGE_INCONSISTENCY = "CROSS_PAGE_INCONSISTENCY"

    # Phase B3: Letter matching flags
    AMBIGUOUS_TYPE = "AMBIGUOUS_TYPE"
    MULTI_INVOICE = "MULTI_INVOICE"

    # Phase C: Vorgang matching flags
    UNKNOWN_CREDITOR = "UNKNOWN_CREDITOR"
    ORPHAN_LETTER = "ORPHAN_LETTER"
    REVIEW = "REVIEW"

    # Phase D: Reconciliation flags
    DUPLICATE_SUSPECT = "DUPLICATE_SUSPECT"
    DATE_MISMATCH_WARNING = "DATE_MISMATCH_WARNING"
    REVIEW_LOW_CONFIDENCE_BEZAHLT = "REVIEW:LOW_CONFIDENCE_BEZAHLT"
    REVIEW_HIGH_VALUE_LOW_CONFIDENCE = "REVIEW:HIGH_VALUE_LOW_CONFIDENCE"
    TX_DUPLICATE_HASH = "TX_DUPLICATE_HASH"

    # Phase E: Output routing flags
    ERBSCHAFT_ROUTE = "ERBSCHAFT_ROUTE"
    URGENT_LEGAL = "URGENT_LEGAL"

    # Cross-intelligence / gap-filling flag prefixes
    DATA_NOT_APPLICABLE = "DATA_NOT_APPLICABLE"
    BETRAG_NOT_IN_DOCUMENT = "BETRAG_NOT_IN_DOCUMENT"
    BETRAG_NOT_IN = "BETRAG_NOT_IN"
    BETRAG_UNREADABLE_PAGE = "BETRAG_UNREADABLE_PAGE"
    BETRAG_FROM_RELATED = "BETRAG_FROM_RELATED"
    IBAN_NOT_ON_AMOUNT_PAGE = "IBAN_NOT_ON_AMOUNT_PAGE"
    IBAN_MANUAL_LOOKUP_REQUIRED = "IBAN_MANUAL_LOOKUP_REQUIRED"
    IBAN_OCR_CORRECTED = "IBAN_OCR_CORRECTED"
    IBAN_NOT_REAL_IBAN = "IBAN_NOT_REAL_IBAN"
    PERSON_INFERRED = "PERSON_INFERRED"
    BANK_PAYMENT_FOUND = "BANK_PAYMENT_FOUND"
    INFO_LETTER = "INFO_LETTER"
    DSGVO_FRAGMENT = "DSGVO_FRAGMENT"
    LIKELY_CRITICAL_CREDITOR = "LIKELY_CRITICAL_CREDITOR"

    # Amount format correction
    AMOUNT_FORMAT_CORRECTED = "AMOUNT_FORMAT_CORRECTED"

    # V11 fix tracking
    V11_BANK_DISCOVERY = "V11_BANK_DISCOVERY"
    FALSE_POSITIVE_REMOVED = "FALSE_POSITIVE_REMOVED"

    # Phase C: Cross-person and over-merge guards
    CROSS_PERSON_BLOCKED = "CROSS_PERSON_BLOCKED"
    OVERSIZED_COMPONENT = "OVERSIZED_COMPONENT"
    TRANSITIVE_MERGE_WARNING = "TRANSITIVE_MERGE_WARNING"
    LOW_PIPELINE_CONFIDENCE = "LOW_PIPELINE_CONFIDENCE"
    LLM_MATCHED = "LLM_MATCHED"
    LLM_SPLIT = "LLM_SPLIT"

    # Phase F: Dual-provider consensus flags
    CONSENSUS_DISAGREEMENT = "CONSENSUS_DISAGREEMENT"
    NEEDS_MANUAL_REVIEW = "NEEDS_MANUAL_REVIEW"

    # Gap inventory
    MISSING_BETRAG = "MISSING_BETRAG"
    MISSING_PERSON = "MISSING_PERSON"
    MISSING_AKTENZEICHEN = "MISSING_AKTENZEICHEN"
    MISSING_ABSENDER = "MISSING_ABSENDER"

    # v5.0.0: Confidence calibration flags
    CONFIDENCE_UNKNOWN = "CONFIDENCE_UNKNOWN"

    # v5.0.0: Reconciliation hardening flags
    POTENTIAL_OVERPAYMENT = "POTENTIAL_OVERPAYMENT"
    MULTI_PAYMENT_COMPLEX = "MULTI_PAYMENT_COMPLEX"
    DUPLICATE_TRANSACTION = "DUPLICATE_TRANSACTION"
    CROSS_PERSON_PAGE = "CROSS_PERSON_PAGE"

    # v5.2.0: Extraction parse failure
    EXTRACTION_PARSE_FAILURE = "EXTRACTION_PARSE_FAILURE"


# ---------------------------------------------------------------------------
# Company Suffixes (for creditor normalization)
# ---------------------------------------------------------------------------

COMPANY_SUFFIXES: list[str] = [
    "gmbh", "ag", "kg", "e.v.", "aör", "mbh", "se", "& co.",
]


# ---------------------------------------------------------------------------
# Boundary Scoring Weights (Phase B3: Page-to-Letter matching)
# ---------------------------------------------------------------------------

# Split signals (positive = more likely new letter)
SPLIT_SIGNAL_CREDITOR_CHANGE = 4
SPLIT_SIGNAL_ADDRESS_BLOCK = 4
SPLIT_SIGNAL_PAGE_RESET = 4
SPLIT_SIGNAL_IBAN_CHANGE = 3
SPLIT_SIGNAL_IDENTIFIER_CHANGE = 3
SPLIT_SIGNAL_PERSON_CHANGE = 3
SPLIT_SIGNAL_DOCTYPE_CHANGE = 1
SPLIT_SIGNAL_INTERBURST_GAP = 1

# Continuity signals (negative = more likely same letter)
CONTINUITY_SEQUENTIAL_PAGES = -5
CONTINUITY_SAME_IDENTIFIER = -3
CONTINUITY_SAME_CREDITOR = -2
CONTINUITY_SAME_IBAN = -2


# ---------------------------------------------------------------------------
# C6: Automatic Categorization Rules
# ---------------------------------------------------------------------------

# Maps keyword patterns (lowercase) to Category enum values.
# First match wins; order matters for specificity.
CATEGORIZATION_RULES: list[tuple[list[str], str]] = [
    (["versicherung", "insurance", "insurer", "huk", "allianz", "ergo", "axa", "generali", "aviva"],
     Category.VERSICHERUNG.value),
    (["gericht", "vollstreckung", "pfaendung", "amtsgericht", "mahnbescheid",
      "gerichtsvollzieher", "justiz", "court", "tribunal", "bailiff", "tax office",
      "hmrc", "revenue", "council"],
     Category.STAAT_RECHT.value),
    (["krankenkasse", "arzt", "aok", "barmer", "dak", "tkk", "apotheke",
      "klinik", "krankenhaus", "hospital", "clinic", "dental", "pharmacy", "medical"],
     Category.GESUNDHEIT.value),
    (["stadtwerke", "miete", "hausverwaltung", "wohnungsbau", "nebenkost",
      "strom", "gas", "wasser", "heizung", "energy", "electric", "water", "landlord",
      "letting", "housing", "utilities"],
     Category.WOHNEN.value),
    (["telekom", "vodafone", "o2", "telefonica", "internet", "mobilfunk",
      "congstar", "freenet", "phone", "mobile", "broadband", "telecom"],
     Category.KOMMUNIKATION.value),
    (["finanzamt", "erbschaft", "erbschein", "nachlassgericht", "nachlass",
      "erben", "steuerbescheid"],
     Category.ERBSCHAFT.value),
    (["bank", "kredit", "sparkasse", "darlehen", "zins", "konto", "girokonto",
      "credit", "loan", "finance", "card services"],
     Category.FINANZEN.value),
]
