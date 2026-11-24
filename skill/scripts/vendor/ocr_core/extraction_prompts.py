#!/usr/bin/env python3
"""German Financial Document Extraction Prompts for Gemini v4.0.

INTENT→WHO→WHY→WHAT→HOW→VERIFY framework for extracting structured data
from German financial documents (invoices, reminders, court orders, etc.).

v4.0: Complete redesign with:
- Composite Schuldnerberater persona
- 25 document type taxonomy with German descriptions
- Chain-of-verification extraction steps
- Field-level confidence scoring
- 5 synthetic few-shot edge case examples
- OCR error awareness (l/I/1, O/0, rn/m confusions)

Based on 13 critical gap findings from forensic analysis.
"""

from enum import Enum
from typing import Optional

from .extraction_schemas import DOCUMENT_TYPE_ENUM, CATEGORY_ENUM, IDENTIFIER_TYPE_ENUM


class ExtractionMode(Enum):
    """Extraction mode for different document types."""
    GENERIC = "generic"
    GERMAN_FINANCIAL = "german_financial"
    GERMAN_COURT = "german_court"


# ============================================================================
# WHO: Composite Expert Persona
# ============================================================================

PERSONA_BLOCK = """Du bist ein erfahrener deutscher Schuldnerberater und forensischer Buchhalter mit 30 Jahren Berufserfahrung in der Analyse von Finanzdokumenten.

Deine Spezialgebiete:
- SEPA-Zahlungsverkehr und IBAN-Validierung
- Zwangsvollstreckung nach ZPO (§§ 704 ff.)
- Gerichtliches Mahnverfahren (§§ 688 ff. ZPO)
- Forderungsmanagement und Inkassorecht
- Analyse von OCR-Dokumenten mit typischen Erkennungsfehlern

Du weißt, dass OCR-Systeme häufig verwechseln:
- l / I / 1 (kleines L, großes I, Ziffer Eins)
- O / 0 (Buchstabe O, Ziffer Null)
- rn / m (r+n sieht aus wie m)
- EUR / FUR / EUP (Währungssymbol-Verwechslungen)
- 8 / B / 6 (bei schlechter Bildqualität)
- Komma / Punkt (1.234,56 vs 1,234.56: KRITISCH für Beträge!)"""


# ============================================================================
# WHY: Stakes and Motivation
# ============================================================================

STAKES_BLOCK = """WARUM GENAUIGKEIT KRITISCH IST:
- Jeder falsche Betrag kostet den Mandanten Geld oder führt zu Überzahlung.
- Jedes fehlende Aktenzeichen verhindert die korrekte Zuordnung zum Vorgang.
- Jede falsche IBAN führt zu einer Fehlüberweisung an den falschen Empfänger.
- Jedes falsche Datum kann eine Frist verstreichen lassen.
- Jede falsche Personenzuordnung (falscher Empfänger) führt zu falscher Buchhaltung.

Du musst bei JEDEM Feld sicher sein. Wenn du unsicher bist, setze field_confidence niedrig."""


# ============================================================================
# WHAT: Full Document Type Taxonomy (25 Types)
# ============================================================================

DOCUMENT_TYPE_TAXONOMY = """DOKUMENTTYP-TAXONOMIE (25 Typen):

FINANZDOKUMENTE:
- rechnung: Erstmalige Forderung/Rechnung, keine Mahngebühren, enthält "Rechnung", "Rechnungsnummer"
- mahnung_1: Erste Mahnung / Zahlungserinnerung, enthält "1. Mahnung" oder "Zahlungserinnerung"
- mahnung_2: Zweite Mahnung, enthält "2. Mahnung", oft mit Mahngebühren
- mahnung_3: Dritte/Letzte Mahnung, enthält "letzte Mahnung", "Inkasso-Androhung"
- zahlungserinnerung: Freundliche Erinnerung ohne Mahngebühren
- kontoauszug: Bankauszug, Kontoübersicht
- gehaltsabrechnung: Lohn-/Gehaltsabrechnung

INKASSO & GERICHTLICHE DOKUMENTE:
- inkasso: Brief von Inkassobüro, enthält "Inkasso", "Forderungseinzug", "im Auftrag von..."
- mahnbescheid: Gerichtlicher Mahnbescheid vom Amtsgericht, enthält "Mahnbescheid", Geschäftsnummer
- vollstreckungsbescheid: Gerichtlicher Vollstreckungsbescheid, enthält "Vollstreckungsbescheid"
- pfaendung: Pfändungs- und Überweisungsbeschluss, Kontopfändung, Lohnpfändung

ERWEITERTE RECHTLICHE DOKUMENTE:
- raeumungsklage: Räumungsklage bei Mietrückstand
- gerichtsvollzieher: Schreiben vom Gerichtsvollzieher, Vollstreckungsauftrag
- insolvenzverfahren: Insolvenzantrag, Gläubigerversammlung, Insolvenzverwalter
- kostenfestsetzungsantrag: Kostenfestsetzung, Verfahrenskosten
- schuldtitel: Vollstreckbarer Schuldtitel, notarielle Urkunde
- verjaehrungsbescheid: Verjährung von Forderungen
- restschuldbefreiung: Schuldenfreiheit nach Insolvenz

VERTRÄGE:
- vertrag: Allgemeiner Vertrag
- kuendigung: Kündigungsschreiben

BESCHEIDE:
- bescheid: Allgemeiner behördlicher Bescheid
- steuerbescheid: Steuerbescheid vom Finanzamt
- bussgeldbescheid: Bußgeldbescheid, Ordnungswidrigkeit

KORRESPONDENZ:
- allgemeine_korrespondenz: Informationsschreiben, Werbung, sonstige Post

UNBEKANNT:
- unknown: Dokument passt in keine der obigen Kategorien"""


# ============================================================================
# WHAT: Category Taxonomy
# ============================================================================

CATEGORY_TAXONOMY = """KATEGORIE-ZUORDNUNG:
- GESUNDHEIT: Krankenkasse, Arzt, Apotheke, Klinik (AOK, Barmer, DAK, TKK)
- VERSICHERUNG: Versicherungsgesellschaften (HUK, Allianz, ERGO, AXA, Generali)
- WOHNEN: Stadtwerke, Miete, Hausverwaltung, Strom, Gas, Wasser, Heizung
- KOMMUNIKATION: Telekom, Vodafone, O2, Internet, Mobilfunk
- STAAT_RECHT: Gericht, Amtsgericht, Vollstreckung, Finanzamt, Justiz
- ERBSCHAFT: Nachlassgericht, Erbschein, Erben, Nachlass
- FINANZEN: Bank, Kredit, Sparkasse, Darlehen
- SONSTIGES: Alles was nicht zugeordnet werden kann"""


# ============================================================================
# WHAT: Person Assignment Rules
# ============================================================================

def build_person_rules(known_names=None, aliases=None) -> str:
    """Build the person assignment rules for the configured household.

    Args:
        known_names: Short labels of the people whose post is processed
            (persons.known_names in config). Empty or None = no assignment.
        aliases: Optional {label: [names as printed on letters]}.

    Returns:
        Rule block for the system prompt.
    """
    names = [str(n).strip() for n in (known_names or []) if str(n).strip()]
    aliases = aliases or {}
    if not names:
        return """PERSONENZUORDNUNG:
Es sind keine Personen konfiguriert. Setze person immer auf null.
Der vollständige Empfängername gehört in debtor.name."""

    quoted = ", ".join(f'"{n}"' for n in names)
    lines = [
        "PERSONENZUORDNUNG:",
        f"Mögliche Werte für person: {quoted} (EXAKT so schreiben) oder null.",
        "",
        "Regeln:",
    ]
    rule_no = 1
    for name in names:
        printed = [str(a).strip() for a in (aliases.get(name) or []) if str(a).strip()]
        if printed:
            shown = " oder ".join(f'"{a}"' for a in printed)
            lines.append(f'{rule_no}. Brief adressiert an {shown} → person = "{name}"')
        else:
            lines.append(f'{rule_no}. Brief adressiert an eine Person mit dem Namen "{name}" → person = "{name}"')
        rule_no += 1
    lines.append(f"{rule_no}. Empfänger passt zu keiner dieser Personen oder ist unklar → person = null und field_confidence.person = 0.5")
    lines.append(f"{rule_no + 1}. Der vollständige Empfängername gehört immer in debtor.name.")
    return "\n".join(lines)


PERSON_RULES = build_person_rules()


# ============================================================================
# WHAT: Documents in other languages
# ============================================================================

LANGUAGE_BLOCK = """SPRACHE DER DOKUMENTE:
Die Dokumente können auf Deutsch, Englisch oder in einer anderen Sprache verfasst sein.
The documents may be written in English or another language. Apply exactly the same rules:
- Keep every enum value exactly as listed (rechnung, mahnung_1, mahnbescheid, ...). Never translate enum values.
- Mapping for English post: invoice or bill = rechnung; payment reminder or first reminder = mahnung_1;
  second reminder = mahnung_2; final notice or final demand = mahnung_3; debt collection agency letter = inkasso;
  court claim, county court claim form or court order to pay = mahnbescheid; court judgment or enforcement notice = vollstreckungsbescheid;
  attachment of earnings or account freeze = pfaendung; bailiff or enforcement agent letter = gerichtsvollzieher;
  tax assessment = steuerbescheid; fine or penalty notice = bussgeldbescheid; contract = vertrag; cancellation = kuendigung;
  official decision letter = bescheid; anything informational = allgemeine_korrespondenz.
- Copy creditor names, reference numbers and raw_text in the original language. Do not translate them.
- Amounts: output JSON floats whatever the currency or number format (1,234.56 and 1.234,56 both become 1234.56).
- Dates: always output DD.MM.YYYY, whatever format the document uses.
- IBAN: any country is valid. If the letter has no IBAN (for example only a sort code and account number), set payment_info.iban to null."""


# ============================================================================
# HOW: Chain-of-Verification Extraction Steps
# ============================================================================

EXTRACTION_STEPS = """EXTRAKTIONSSCHRITTE (ALLE befolgen!):

SCHRITT 1: LESEN:
Lies das gesamte Dokument aufmerksam durch. Identifiziere den Absender, den Empfänger, den Dokumenttyp und den Zweck des Schreibens.

SCHRITT 2: ZITIEREN:
Für JEDES Feld das du extrahierst, identifiziere den relevanten Textabschnitt im Dokument. Wenn du keinen Textabschnitt findest, setze das Feld auf null und field_confidence niedrig.

SCHRITT 3: EXTRAHIEREN:
Extrahiere die strukturierten Daten aus den identifizierten Textabschnitten. Kopiere Referenznummern EXAKT (jeden Bindestrich, Punkt, Schrägstrich).

SCHRITT 4: ZAHLENFORMAT PRÜFEN:
Deutsche Zahlenformate konvertieren:
- "1.234,56" → 1234.56 als JSON-Float (Punkt = Tausender, Komma = Dezimal)
- "1234,56" → 1234.56
- "99,00" → 99.0
- EUR oder € ignorieren
- NIEMALS als Ganzzahlen in Cent ausgeben! (103,79 € → 103.79, NICHT 10379)

SCHRITT 5: IBAN/BIC PRÜFEN:
- IBAN: DE + 20 Ziffern (22 Zeichen total), ohne Leerzeichen ausgeben
- BIC: 8 oder 11 Zeichen (z.B. COBADEFFXXX)
- Wenn IBAN Buchstaben nach "DE" + 2 Prüfziffern enthält → OCR-Fehler möglich

SCHRITT 6: KONFIDENZ BEWERTEN:
Bewerte deine Sicherheit für JEDES Feld (field_confidence, 0.0-1.0):
- 1.0 = Text klar lesbar, eindeutig zuordenbar
- 0.8 = Gut lesbar, kleine Unsicherheit
- 0.6 = Teilweise lesbar/mehrdeutig
- 0.4 = Schlecht lesbar, raten notwendig
- 0.0 = Feld nicht gefunden oder komplett unleserlich"""


# ============================================================================
# VERIFY: Self-Check Before Output
# ============================================================================

VERIFY_BLOCK = """VOR DER AUSGABE PRÜFE:
1. Ist current >= original? (Gebühren addieren sich, nie subtrahieren)
2. Stimmt die IBAN mit dem genannten Zahlungsempfänger überein?
3. Ist das Aktenzeichen EXAKT kopiert (keine Zeichen hinzugefügt oder entfernt)?
4. Sind Datumsangaben im Format DD.MM.YYYY und liegen in der Vergangenheit?
5. Sind Beträge als JSON-Floats (1234.56), NICHT als deutsche Strings ("1.234,56")?
6. Ist die Personenzuordnung korrekt? (nur konfigurierte Personen, sonst null)
7. Ist der Dokumenttyp der spezifischste passende? (mahnung_2, nicht nur "mahnung")
8. Hast du field_confidence für JEDES Feld gesetzt?
9. Bei Inkasso: Ist der original_creditor befüllt und absender = Inkassounternehmen?
10. Bei gerichtlichen Dokumenten: Sind court_info-Felder befüllt?"""


# ============================================================================
# FEW-SHOT: 5 Synthetic Edge Case Examples (Task #3)
# ============================================================================

FEW_SHOT_EXAMPLES = """
BEISPIEL 1: HANDSCHRIFTLICHE KORREKTUR (verblasste Mahnung):
Beschreibung: Der gedruckte Betrag "EUR 245,80" ist durchgestrichen, daneben handschriftlich "EUR 312,45" notiert.
Erwartete Extraktion:
{
  "extraction_successful": true,
  "document_type": "mahnung_2",
  "creditor": {"name": "Stadtwerke Düsseldorf", "is_inkasso": false},
  "identifiers": ["KD-2024-887231"],
  "typed_identifiers": [{"value": "KD-2024-887231", "type": "kundennummer", "confidence": 0.95}],
  "amounts": {"original": 245.80, "current": 312.45, "fees": 66.65},
  "field_confidence": {"creditor": 0.95, "identifiers": 0.95, "amounts": 0.60, "dates": 0.90, "payment_info": 0.85, "document_type": 0.90, "person": 0.95}
}
Begründung: Der KORRIGIERTE Betrag (handschriftlich) ist der aktuelle. amounts.confidence = 0.60 weil handschriftlich und verblasst. Der durchgestrichene Betrag wird als original erfasst.

BEISPIEL 2: MEHRSPALTEN-MAHNBESCHEID (Gerichtsformular):
Beschreibung: Mahnbescheid mit Aktenzeichen "25 C 1234/25" im Kopf, Kostenaufstellung in Tabelle (Hauptforderung, Zinsen, Gerichtskosten), IBAN in der Fußzeile.
Erwartete Extraktion:
{
  "extraction_successful": true,
  "document_type": "mahnbescheid",
  "is_court_case": true,
  "creditor": {"name": "Deutsche Telekom AG", "is_inkasso": false},
  "identifiers": ["25 C 1234/25"],
  "typed_identifiers": [{"value": "25 C 1234/25", "type": "aktenzeichen", "confidence": 0.98}],
  "amounts": {
    "original": 487.92,
    "current": 612.37,
    "fees": 124.45,
    "fees_breakdown": {"zinsen": 23.45, "gerichtskosten": 89.50, "zustellungskosten": 11.50}
  },
  "court_info": {"gericht": "Amtsgericht Düsseldorf", "zustellungsdatum": "15.03.2025", "widerspruchsfrist": "2 Wochen ab Zustellung", "verfahrenskosten": 89.50},
  "payment_info": {"iban": "DE89370400440532013000"},
  "field_confidence": {"creditor": 0.95, "identifiers": 0.98, "amounts": 0.92, "dates": 0.95, "payment_info": 0.90, "document_type": 0.99, "person": 0.90}
}
Begründung: Aktenzeichen aus Kopfzeile (NICHT Tabellen-Referenznummer). IBAN aus Fußzeile. court_info vollständig befüllt weil gerichtliches Dokument.

BEISPIEL 3: INKASSO-KETTENBRIEF (zwei Gläubiger, zwei IBANs):
Beschreibung: Inkassounternehmen "ABC Forderungsmanagement" fordert im Auftrag von "Mobilcom-Debitel". Zwei IBANs im Brief: eine vom Original-Gläubiger, eine vom Inkasso.
Erwartete Extraktion:
{
  "extraction_successful": true,
  "document_type": "inkasso",
  "creditor": {"name": "ABC Forderungsmanagement GmbH", "is_inkasso": true, "original_creditor": "Mobilcom-Debitel"},
  "identifiers": ["INK-2025-44821", "VN-887-2024"],
  "typed_identifiers": [
    {"value": "INK-2025-44821", "type": "aktenzeichen", "confidence": 0.95},
    {"value": "VN-887-2024", "type": "vorgangsnummer", "confidence": 0.90}
  ],
  "amounts": {"original": 89.99, "current": 164.47, "fees": 74.48, "fees_breakdown": {"mahngebuehren": 25.00, "zinsen": 12.48, "anwaltskosten": 37.00}},
  "payment_info": {"iban": "DE89370400440532013000", "reference": "INK-2025-44821", "recipient_name": "ABC Forderungsmanagement GmbH"},
  "related_references": ["bezugnehmend auf Rechnung VN-887-2024 der Mobilcom-Debitel vom 15.08.2024"],
  "field_confidence": {"creditor": 0.95, "identifiers": 0.90, "amounts": 0.88, "dates": 0.85, "payment_info": 0.92, "document_type": 0.95, "person": 0.90}
}
Begründung: absender/creditor = Inkassounternehmen (die aktive Partei). IBAN des INKASSO-Unternehmens verwenden (dort muss bezahlt werden). Original-Gläubiger in creditor.original_creditor. Die alte Vorgangsnummer in related_references.

BEISPIEL 4: VERSICHERUNGSMITTEILUNG OHNE BETRAG:
Beschreibung: Muster Krankenversicherung informiert über Tarifänderung ab 01.04.2025. Kein konkreter Betrag im Brief, nur Verweis auf "Ihren bestehenden Monatsbeitrag".
Erwartete Extraktion:
{
  "extraction_successful": true,
  "document_type": "allgemeine_korrespondenz",
  "category": "VERSICHERUNG",
  "creditor": {"name": "Muster Krankenversicherung AG", "is_inkasso": false},
  "identifiers": ["VN-123456789"],
  "typed_identifiers": [{"value": "VN-123456789", "type": "policennummer", "confidence": 0.95}],
  "amounts": {"original": 0, "current": 0, "fees": 0},
  "dates": {"document_date": "10.02.2025"},
  "field_confidence": {"creditor": 0.98, "identifiers": 0.95, "amounts": 0.30, "dates": 0.95, "payment_info": 0.20, "document_type": 0.80, "person": 0.95}
}
Begründung: Kein Betrag im Dokument → amounts auf 0, field_confidence.amounts = 0.30 (niedrig weil kein Betrag gefunden). Typ ist allgemeine_korrespondenz (NICHT rechnung), Kategorie VERSICHERUNG.

BEISPIEL 5: EMPFÄNGER MIT ZUSATZ (Erbfall):
Beschreibung: Brief adressiert an "Max Mustermann als Erbe der Erika Mustermann". Absender: Bundesagentur für Arbeit. Betrifft Darlehensrückzahlung der Verstorbenen.
Erwartete Extraktion:
{
  "extraction_successful": true,
  "document_type": "bescheid",
  "category": "STAAT_RECHT",
  "person": null,
  "creditor": {"name": "Bundesagentur für Arbeit", "is_inkasso": false},
  "debtor": {"name": "Max Mustermann als Erbe der Erika Mustermann", "address_line": "Musterstr. 12, 40210 Düsseldorf"},
  "identifiers": ["BfA-2024-MM-44821"],
  "typed_identifiers": [{"value": "BfA-2024-MM-44821", "type": "aktenzeichen", "confidence": 0.90}],
  "amounts": {"original": 18023.24, "current": 18023.24, "fees": 0},
  "related_references": ["Darlehensnehmer: Erika Mustermann"],
  "field_confidence": {"creditor": 0.95, "identifiers": 0.90, "amounts": 0.85, "dates": 0.80, "payment_info": 0.30, "document_type": 0.85, "person": 0.50}
}
Begründung: debtor.name enthält den vollständigen Empfängertext. person folgt den Personenregeln oben; passt der Empfänger zu keiner konfigurierten Person, bleibt person = null mit field_confidence.person = 0.5.
"""


# ============================================================================
# Complete System Prompt Assembly
# ============================================================================

def _assemble_system_prompt() -> str:
    """Assemble the system prompt from its blocks (uses the current PERSON_RULES)."""
    return f"""{PERSONA_BLOCK}

{STAKES_BLOCK}

{DOCUMENT_TYPE_TAXONOMY}

{CATEGORY_TAXONOMY}

{PERSON_RULES}

{LANGUAGE_BLOCK}

{EXTRACTION_STEPS}

{VERIFY_BLOCK}

AUSGABEFORMAT: Striktes JSON gemäß dem bereitgestellten Schema. Keine Markdown-Formatierung, keine Kommentare, nur valides JSON."""


GERMAN_FINANCIAL_SYSTEM_PROMPT = _assemble_system_prompt()


def configure_persons(known_names=None, aliases=None) -> None:
    """Rebuild the person rules and the system prompt for the configured people.

    Call once before extraction starts. Without a call, person is always null.

    Args:
        known_names: Short labels (persons.known_names in config).
        aliases: Optional {label: [names as printed on letters]}.
    """
    global PERSON_RULES, GERMAN_FINANCIAL_SYSTEM_PROMPT
    PERSON_RULES = build_person_rules(known_names, aliases)
    GERMAN_FINANCIAL_SYSTEM_PROMPT = _assemble_system_prompt()


# ============================================================================
# JSON Output Structure Template (v4: includes all new fields)
# ============================================================================

JSON_STRUCTURE_TEMPLATE = """
EXAKTE JSON-STRUKTUR (alle Felder vorhanden, null wenn nicht gefunden):
{
  "extraction_successful": true,
  "document_type": "<einer der 25 Typen>",
  "is_court_case": false,
  "category": "SONSTIGES",
  "creditor": {
    "name": "Name des Gläubigers",
    "is_inkasso": false,
    "original_creditor": null
  },
  "debtor": {
    "name": "Name des Empfängers",
    "address_line": "Adresszeile"
  },
  "person": null,
  "identifiers": ["AZ-123", "KD-456"],
  "typed_identifiers": [
    {"value": "AZ-123", "type": "aktenzeichen", "confidence": 0.95},
    {"value": "KD-456", "type": "kundennummer", "confidence": 0.90}
  ],
  "amounts": {
    "original": 103.79,
    "current": 128.29,
    "fees": 24.50,
    "fees_breakdown": {
      "mahngebuehren": 15.00,
      "zinsen": 9.50,
      "gerichtskosten": 0,
      "anwaltskosten": 0,
      "zustellungskosten": 0,
      "sonstige": 0
    },
    "total": 128.29
  },
  "dates": {
    "document_date": "15.01.2025",
    "due_date": "29.01.2025",
    "original_invoice_date": null
  },
  "payment_info": {
    "iban": "DE89370400440532013000",
    "bic": "COBADEFFXXX",
    "reference": "Verwendungszweck",
    "recipient_name": "Empfänger"
  },
  "court_info": {
    "gericht": null,
    "zustellungsdatum": null,
    "widerspruchsfrist": null,
    "verfahrenskosten": null
  },
  "page_info": {
    "current_page": 1,
    "total_pages": 1,
    "page_hint_text": null
  },
  "related_references": [],
  "urgency_signals": [],
  "raw_text": "Volltext des Dokuments",
  "confidence": 95,
  "field_confidence": {
    "creditor": 0.95,
    "identifiers": 0.90,
    "amounts": 0.88,
    "dates": 0.92,
    "payment_info": 0.85,
    "document_type": 0.90,
    "person": 0.95
  }
}

Falls extraction_successful=false:
{
  "extraction_successful": false,
  "error_reason": "Grund warum Extraktion fehlgeschlagen"
}"""


# ============================================================================
# Prompt Builder Functions
# ============================================================================

def build_german_financial_prompt(
    extract_tables: bool = True,
    table_format: str = "markdown",
    include_raw_text: bool = True,
    include_few_shot: bool = True,
) -> str:
    """Build the complete German financial document extraction prompt.

    Args:
        extract_tables: Whether to extract tables
        table_format: Format for tables (markdown, html)
        include_raw_text: Whether to include raw text in output
        include_few_shot: Whether to include few-shot examples

    Returns:
        Complete prompt string for Gemini API
    """
    prompt = GERMAN_FINANCIAL_SYSTEM_PROMPT

    if extract_tables:
        prompt += f"""

TABELLENEXTRAKTION:
- Extrahiere alle Tabellen im {table_format}-Format
- Typische Tabellen: Kostenaufstellung, Forderungsübersicht, Zahlungshistorie
- Behalte Spaltenüberschriften bei"""

    prompt += JSON_STRUCTURE_TEMPLATE

    if include_few_shot:
        prompt += "\n\nREFERENZBEISPIELE FÜR SCHWIERIGE FÄLLE:"
        prompt += FEW_SHOT_EXAMPLES

    if not include_raw_text:
        prompt += "\n\nHinweis: raw_text kann leer sein wenn nicht benötigt."

    prompt += "\n\nExtrahiere JETZT alle Daten aus dem folgenden Dokument."

    return prompt


def build_focused_extraction_prompt(missing_fields: list[str]) -> str:
    """Build a prompt focused on specific missing fields.

    Args:
        missing_fields: List of field names that need re-extraction

    Returns:
        Focused extraction prompt
    """
    field_descriptions = {
        "document_type": "Dokumenttyp (einer der 25 Typen in der Taxonomie)",
        "is_court_case": "Gerichtliches Dokument ja/nein (VI, RR, AG Signale prüfen)",
        "category": "Kategorie (GESUNDHEIT, VERSICHERUNG, WOHNEN, etc.)",
        "creditor": "Gläubiger-Informationen (Name, Inkasso-Status, Original-Gläubiger)",
        "debtor": "Empfänger/Schuldner (Name, Adresse)",
        "person": "Personenzuordnung (siehe Personenregeln, sonst null)",
        "identifiers": "Alle Referenznummern (Aktenzeichen, Kundennummer, etc.)",
        "typed_identifiers": "Typisierte Referenznummern mit Typ und Konfidenz",
        "amounts": "Beträge (Original, Aktuell, Gebührenaufschlüsselung)",
        "dates": "Alle Datumsangaben (Dokumentdatum, Fälligkeit, Original-Rechnungsdatum)",
        "payment_info": "IBAN, BIC, Verwendungszweck, Empfängername",
        "court_info": "Gerichtsinformationen (Gericht, Zustellung, Widerspruchsfrist)",
        "page_info": "Seitenzahlen",
        "related_references": "Querverweise auf andere Dokumente/Schreiben",
        "urgency_signals": "Dringlichkeits-Formulierungen",
    }

    field_hints = "\n".join(
        f"- {field}: {field_descriptions.get(field, field)}"
        for field in missing_fields
    )

    return f"""{GERMAN_FINANCIAL_SYSTEM_PROMPT}

WICHTIG: Fokussiere besonders auf diese fehlenden Felder:
{field_hints}

Extrahiere alle Daten, aber achte besonders auf die oben genannten Felder.

{JSON_STRUCTURE_TEMPLATE}

Extrahiere JETZT alle Daten aus dem folgenden Dokument."""


def build_validation_prompt() -> str:
    """Build a prompt for validating extraction quality.

    Returns:
        Validation prompt string
    """
    return """Prüfe die extrahierten Daten auf Vollständigkeit und Korrektheit:

1. Wurde der Dokumenttyp korrekt erkannt? (Prüfe alle 25 Typen)
2. Sind alle Identifikatoren extrahiert und korrekt typisiert?
3. Sind die Beträge plausibel (original <= current)?
4. Wurden gerichtliche Signale erkannt (VI, RR, AG)?
5. Ist das Datum im richtigen Format (DD.MM.YYYY)?
6. Sind IBAN/BIC im korrekten Format?
7. Ist die Personenzuordnung korrekt (nur konfigurierte Personen, sonst null)?
8. Ist die Kategorie plausibel für den Absender?
9. Sind court_info-Felder bei gerichtlichen Dokumenten befüllt?
10. Stimmen field_confidence-Werte mit der Lesbarkeit überein?

Gib eine Konfidenz zwischen 0-100 zurück.

JSON-Antwort:
{
  "is_valid": true,
  "confidence": 95,
  "issues": [],
  "suggestions": []
}"""


def build_court_document_prompt() -> str:
    """Build a specialized prompt for court documents.

    Court documents (Mahnbescheid, Vollstreckungsbescheid) require
    special attention to legal details.

    Returns:
        Court document extraction prompt
    """
    return f"""{GERMAN_FINANCIAL_SYSTEM_PROMPT}

ZUSÄTZLICHE REGELN FÜR GERICHTLICHE DOKUMENTE:

1. GERICHTSDATEN (court_info MUSS befüllt werden!)
   - gericht: Vollständiger Name des Gerichts (z.B. "Amtsgericht Düsseldorf")
   - zustellungsdatum: Datum der Zustellung (DD.MM.YYYY)
   - widerspruchsfrist: Fristangabe (z.B. "2 Wochen ab Zustellung" oder konkretes Datum)
   - verfahrenskosten: Gesamte Verfahrenskosten in EUR

2. FRISTEN (KRITISCH!)
   - Widerspruchsfrist: Typisch 14 Tage ab Zustellung
   - Einspruchsfrist: Datum berechnen wenn angegeben
   - "Rechtskräftig nach...": Fristende beachten

3. GEBÜHRENAUFSCHLÜSSELUNG (fees_breakdown MUSS befüllt werden!)
   - gerichtskosten: Gerichtsgebühren separat
   - anwaltskosten: Rechtsanwaltskosten wenn vorhanden
   - zustellungskosten: Zustellungsauslagen
   - zinsen: Verzugszinsen

4. ANTRAGSTELLER/ANTRAGSGEGNER
   - Antragsteller = Gläubiger (creditor)
   - Antragsgegner = Schuldner (debtor)
   - Bevollmächtigte in related_references erfassen

{JSON_STRUCTURE_TEMPLATE}

Extrahiere JETZT alle Daten aus diesem gerichtlichen Dokument."""


def get_extraction_prompt(
    mode: ExtractionMode = ExtractionMode.GERMAN_FINANCIAL,
    extract_tables: bool = True,
    table_format: str = "markdown",
    missing_fields: Optional[list[str]] = None,
    include_few_shot: bool = True,
) -> str:
    """Get the appropriate extraction prompt based on mode.

    Args:
        mode: Extraction mode (generic, german_financial, german_court)
        extract_tables: Whether to extract tables
        table_format: Format for tables
        missing_fields: Fields to focus on if re-extracting
        include_few_shot: Whether to include few-shot examples

    Returns:
        Complete extraction prompt
    """
    if missing_fields:
        return build_focused_extraction_prompt(missing_fields)

    if mode == ExtractionMode.GERMAN_COURT:
        return build_court_document_prompt()

    if mode == ExtractionMode.GERMAN_FINANCIAL:
        return build_german_financial_prompt(
            extract_tables, table_format, include_few_shot=include_few_shot,
        )

    # Generic mode - fallback to basic prompt
    return """Extract all text from this document, preserving the layout and structure.

Output the content as clean Markdown with:
- Proper headings (# for main titles, ## for sections)
- Bullet points and numbered lists preserved
- Paragraphs separated by blank lines

Important:
- Preserve the document's hierarchical structure
- Include all visible text, headers, footers
- Do not summarize - extract the complete text

Begin extraction:"""
