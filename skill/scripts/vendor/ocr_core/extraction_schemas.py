#!/usr/bin/env python3
"""JSON Schema Definitions for Gemini Structured Output.

Defines the exact schema for Gemini AI to return extraction results
for German financial documents.

v4.0: Expanded from 5 to 25 document types, added field_confidence,
typed identifiers, fee breakdown, court_info, debtor, related_references,
and category. Full backwards compatibility with v3 consumers.

Based on PRD Section 14: Gemini Integration
"""

from typing import Any, Dict, List, Optional, TypedDict


# ============================================================================
# Document Type Enum Values (25 types from document_classifier.py)
# ============================================================================

DOCUMENT_TYPE_ENUM: List[str] = [
    # Financial Documents
    "rechnung",
    "mahnung_1",
    "mahnung_2",
    "mahnung_3",
    "zahlungserinnerung",
    "kontoauszug",
    "gehaltsabrechnung",
    # Legal Documents - Debt Collection
    "inkasso",
    "mahnbescheid",
    "vollstreckungsbescheid",
    "pfaendung",
    # Legal Documents - Advanced
    "raeumungsklage",
    "gerichtsvollzieher",
    "insolvenzverfahren",
    "kostenfestsetzungsantrag",
    "schuldtitel",
    "verjaehrungsbescheid",
    "restschuldbefreiung",
    # Contracts and Agreements
    "vertrag",
    "kuendigung",
    # Official Documents
    "bescheid",
    "steuerbescheid",
    "bussgeldbescheid",
    # Correspondence
    "allgemeine_korrespondenz",
    # Unknown
    "unknown",
]

# Category enum values (from briefpost-management constants.py)
CATEGORY_ENUM: List[str] = [
    "GESUNDHEIT",
    "VERSICHERUNG",
    "WOHNEN",
    "KOMMUNIKATION",
    "STAAT_RECHT",
    "ERBSCHAFT",
    "FINANZEN",
    "SONSTIGES",
]

# Identifier type enum
IDENTIFIER_TYPE_ENUM: List[str] = [
    "aktenzeichen",
    "kundennummer",
    "rechnungsnummer",
    "vorgangsnummer",
    "geschaeftszeichen",
    "vertragsnummer",
    "schadennummer",
    "policennummer",
    "steuernummer",
]


# ============================================================================
# Type Definitions for Schema
# ============================================================================

class CreditorSchema(TypedDict, total=False):
    """Schema for creditor information."""
    name: str
    is_inkasso: bool
    original_creditor: Optional[str]


class AmountsSchema(TypedDict, total=False):
    """Schema for amount information (v4: expanded with fee breakdown)."""
    original: float
    current: float
    fees: float  # Backwards compat: sum of all fees
    fees_breakdown: Optional[Dict[str, float]]
    total: float


class DatesSchema(TypedDict, total=False):
    """Schema for date information."""
    document_date: Optional[str]
    due_date: Optional[str]
    original_invoice_date: Optional[str]


class PaymentInfoSchema(TypedDict, total=False):
    """Schema for payment information."""
    iban: Optional[str]
    bic: Optional[str]
    reference: Optional[str]
    recipient_name: Optional[str]


class PageInfoSchema(TypedDict, total=False):
    """Schema for page information."""
    current_page: Optional[int]
    total_pages: Optional[int]
    page_hint_text: Optional[str]


class IdentifierSchema(TypedDict, total=False):
    """Schema for a typed identifier (v4)."""
    value: str
    type: str
    confidence: float


class CourtInfoSchema(TypedDict, total=False):
    """Schema for court information (v4)."""
    gericht: Optional[str]
    zustellungsdatum: Optional[str]
    widerspruchsfrist: Optional[str]
    verfahrenskosten: Optional[float]


class DebtorSchema(TypedDict, total=False):
    """Schema for debtor/addressee information (v4)."""
    name: Optional[str]
    address_line: Optional[str]


class FieldConfidenceSchema(TypedDict, total=False):
    """Schema for per-field confidence scores (v4)."""
    creditor: float
    identifiers: float
    amounts: float
    dates: float
    payment_info: float
    document_type: float
    person: float


# ============================================================================
# Main Extraction Schema (v4.0 - Expanded)
# ============================================================================

GERMAN_EXTRACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["extraction_successful"],
    "properties": {
        "extraction_successful": {
            "type": "boolean",
            "description": "false wenn Dokument nicht lesbar/kein Finanzdokument",
        },
        "error_reason": {
            "type": "string",
            "description": "Grund wenn extraction_successful=false",
        },
        # === Document Classification ===
        "document_type": {
            "type": "string",
            "enum": DOCUMENT_TYPE_ENUM,
            "description": "Typ des Dokuments (25 Typen)",
        },
        "is_court_case": {
            "type": "boolean",
            "description": "true wenn gerichtliches Dokument (VI, RR, AG, Amtsgericht)",
        },
        "category": {
            "type": "string",
            "enum": CATEGORY_ENUM,
            "description": "Kategorie des Dokuments (GESUNDHEIT, VERSICHERUNG, WOHNEN, etc.)",
        },
        # === Parties ===
        "creditor": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name des Gläubigers/Absenders",
                },
                "is_inkasso": {
                    "type": "boolean",
                    "description": "true wenn Inkassobüro/Forderungseinzug",
                },
                "original_creditor": {
                    "type": "string",
                    "description": "Falls Inkasso: ursprünglicher Gläubiger",
                },
            },
            "required": ["name"],
        },
        "debtor": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name des Empfängers/Schuldners (vollständig, wie im Adressfeld gedruckt)",
                },
                "address_line": {
                    "type": "string",
                    "description": "Adresszeile des Empfängers",
                },
            },
        },
        "person": {
            "type": "string",
            "description": "Zugeordnete Person gemäß Personenregeln im Prompt, sonst leer lassen",
        },
        # === Identifiers (v4: typed array + backwards-compat flat list) ===
        "identifiers": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Alle Referenznummern als flache Liste (Aktenzeichen, Kundennummer, etc.) - ABWÄRTSKOMPATIBEL",
        },
        "typed_identifiers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "Der Wert der Referenznummer (exakt kopiert)",
                    },
                    "type": {
                        "type": "string",
                        "enum": IDENTIFIER_TYPE_ENUM,
                        "description": "Typ der Referenznummer",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Konfidenz für diese Referenznummer (0.0-1.0)",
                    },
                },
                "required": ["value", "type"],
            },
            "description": "Typisierte Referenznummern mit Typ und Konfidenz (v4 NEU)",
        },
        # === Amounts (v4: expanded with fee breakdown) ===
        "amounts": {
            "type": "object",
            "properties": {
                "original": {
                    "type": "number",
                    "description": "Ursprungsbetrag/Hauptforderung vor Gebühren",
                },
                "current": {
                    "type": "number",
                    "description": "Aktuell zu zahlender Gesamtbetrag inkl. aller Gebühren",
                },
                "fees": {
                    "type": "number",
                    "description": "Summe aller Gebühren/Zinsen (ABWÄRTSKOMPATIBEL)",
                },
                "fees_breakdown": {
                    "type": "object",
                    "properties": {
                        "mahngebuehren": {
                            "type": "number",
                            "description": "Mahngebühren",
                        },
                        "zinsen": {
                            "type": "number",
                            "description": "Verzugszinsen",
                        },
                        "gerichtskosten": {
                            "type": "number",
                            "description": "Gerichtskosten",
                        },
                        "anwaltskosten": {
                            "type": "number",
                            "description": "Rechtsanwaltskosten",
                        },
                        "zustellungskosten": {
                            "type": "number",
                            "description": "Zustellungskosten/Auslagen",
                        },
                        "sonstige": {
                            "type": "number",
                            "description": "Sonstige Kosten/Gebühren",
                        },
                    },
                    "description": "Aufschlüsselung der Gebühren nach Typ (v4 NEU)",
                },
                "total": {
                    "type": "number",
                    "description": "Berechneter Gesamtbetrag (original + alle fees_breakdown)",
                },
            },
        },
        # === Dates ===
        "dates": {
            "type": "object",
            "properties": {
                "document_date": {
                    "type": "string",
                    "description": "Dokumentdatum (DD.MM.YYYY)",
                },
                "due_date": {
                    "type": "string",
                    "description": "Fälligkeitsdatum (DD.MM.YYYY)",
                },
                "original_invoice_date": {
                    "type": "string",
                    "description": "Ursprüngliches Rechnungsdatum (DD.MM.YYYY)",
                },
            },
        },
        # === Payment Info ===
        "payment_info": {
            "type": "object",
            "properties": {
                "iban": {
                    "type": "string",
                    "description": "IBAN (DE + 20 Ziffern, ohne Leerzeichen)",
                },
                "bic": {
                    "type": "string",
                    "description": "BIC/SWIFT (8 oder 11 Zeichen)",
                },
                "reference": {
                    "type": "string",
                    "description": "Verwendungszweck für Überweisung",
                },
                "recipient_name": {
                    "type": "string",
                    "description": "Kontoinhaber/Zahlungsempfänger",
                },
            },
        },
        # === Court Info (v4 NEU) ===
        "court_info": {
            "type": "object",
            "properties": {
                "gericht": {
                    "type": "string",
                    "description": "Name des Gerichts (z.B. 'Amtsgericht Düsseldorf')",
                },
                "zustellungsdatum": {
                    "type": "string",
                    "description": "Datum der Zustellung (DD.MM.YYYY)",
                },
                "widerspruchsfrist": {
                    "type": "string",
                    "description": "Widerspruchs-/Einspruchsfrist (z.B. '2 Wochen ab Zustellung', oder konkretes Datum DD.MM.YYYY)",
                },
                "verfahrenskosten": {
                    "type": "number",
                    "description": "Gesamte Verfahrenskosten als Dezimalzahl in EUR",
                },
            },
            "description": "Gerichtsinformationen (nur bei gerichtlichen Dokumenten, v4 NEU)",
        },
        # === Page Info ===
        "page_info": {
            "type": "object",
            "properties": {
                "current_page": {
                    "type": "integer",
                    "description": "Aktuelle Seitennummer",
                },
                "total_pages": {
                    "type": "integer",
                    "description": "Gesamtzahl der Seiten",
                },
                "page_hint_text": {
                    "type": "string",
                    "description": "Text der Seitenzahl-Angabe (z.B. 'Seite 1 von 3')",
                },
            },
        },
        # === Related References (v4 NEU) ===
        "related_references": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Querverweis-Texte ('bezugnehmend auf...', 'zu unserem Schreiben vom...', etc., v4 NEU)",
        },
        # === Urgency and Signals ===
        "urgency_signals": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Gefundene Dringlichkeits-Formulierungen",
        },
        # === Raw Text ===
        "raw_text": {
            "type": "string",
            "description": "Vollständiger extrahierter Text des Dokuments",
        },
        # === Confidence (overall, 0-100) ===
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": "Gesamtkonfidenz der Extraktion (0-100)",
        },
        # === Field Confidence (v4: NOW IN SCHEMA) ===
        "field_confidence": {
            "type": "object",
            "properties": {
                "creditor": {
                    "type": "number",
                    "description": "Konfidenz Gläubigername (0.0-1.0): 1.0=klar lesbar, 0.5=teilweise unlesbar",
                },
                "identifiers": {
                    "type": "number",
                    "description": "Konfidenz Referenznummern (0.0-1.0): 1.0=alle deutlich, 0.7=einige undeutlich",
                },
                "amounts": {
                    "type": "number",
                    "description": "Konfidenz Beträge (0.0-1.0): 1.0=alle Zahlen klar, 0.6=Dezimalstellen unsicher",
                },
                "dates": {
                    "type": "number",
                    "description": "Konfidenz Datumsangaben (0.0-1.0): 1.0=Datum vollständig, 0.7=Jahr/Tag unsicher",
                },
                "payment_info": {
                    "type": "number",
                    "description": "Konfidenz IBAN/BIC (0.0-1.0): 1.0=IBAN vollständig lesbar, 0.8=lesbar aber unsicher",
                },
                "document_type": {
                    "type": "number",
                    "description": "Konfidenz Dokumenttyp (0.0-1.0): 1.0=eindeutige Signale, 0.5=mehrdeutig",
                },
                "person": {
                    "type": "number",
                    "description": "Konfidenz Personenzuordnung (0.0-1.0): 1.0=Name eindeutig, 0.5=Empfänger unklar",
                },
            },
            "description": "Feldweise Konfidenzwerte (0.0-1.0) für jedes extrahierte Feld (v4: jetzt im Schema)",
        },
    },
}


# ============================================================================
# Simplified Schemas
# ============================================================================

DOCUMENT_TYPE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["document_type", "is_court_case"],
    "properties": {
        "document_type": {
            "type": "string",
            "enum": DOCUMENT_TYPE_ENUM,
        },
        "is_court_case": {
            "type": "boolean",
        },
        "creditor_name": {
            "type": "string",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
        },
    },
}


VALIDATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["is_valid", "confidence"],
    "properties": {
        "is_valid": {
            "type": "boolean",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
        },
        "issues": {
            "type": "array",
            "items": {"type": "string"},
        },
        "suggestions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


# ============================================================================
# Schema Utilities
# ============================================================================

def get_focused_schema(focus_fields: List[str]) -> Dict[str, Any]:
    """Get a schema with additional required fields.

    Creates a modified schema that emphasizes specific fields
    for re-extraction attempts.

    Args:
        focus_fields: Fields to mark as required

    Returns:
        Modified schema with additional required fields
    """
    import copy
    focused = copy.deepcopy(GERMAN_EXTRACTION_SCHEMA)

    # Add focus fields to required list if they exist in properties
    current_required = focused.get("required", [])
    for field in focus_fields:
        if field in focused["properties"] and field not in current_required:
            current_required.append(field)

    focused["required"] = current_required
    return focused


def validate_document_type(doc_type: str) -> bool:
    """Validate that document type is in allowed values.

    Args:
        doc_type: Document type string to validate

    Returns:
        True if valid, False otherwise
    """
    return doc_type in DOCUMENT_TYPE_ENUM


def get_schema_for_mode(mode: str) -> Dict[str, Any]:
    """Get the appropriate schema based on extraction mode.

    Args:
        mode: Extraction mode ("full", "type_only", "validation")

    Returns:
        Schema dictionary for the specified mode
    """
    schemas = {
        "full": GERMAN_EXTRACTION_SCHEMA,
        "type_only": DOCUMENT_TYPE_SCHEMA,
        "validation": VALIDATION_SCHEMA,
    }
    return schemas.get(mode, GERMAN_EXTRACTION_SCHEMA)
