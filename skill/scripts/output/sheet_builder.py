"""Phase E1: build the summary workbook from the matched and reconciled cases.

Four tabs:
  1. All letters    one row per letter
  2. Open items     one row per open or part-paid case, most urgent first
  3. Summary        counts, totals, top creditors
  4. Bank matching  one row per case with the bank match that was found

Where the result goes is set by output.destination in the config:
  local   XLSX workbook plus one CSV per tab in <project>/output/ (no account needed)
  google  Google Sheet (needs Google sign-in, see output/gsheets.py)
  both    local files and a Google Sheet
The local files are written in every mode, so there is always a result on disk.

Labels are set by output.language: "en" (English) or "de" (the original
German labels and raw field names). Internal values in the JSON files are
never changed, only what is shown in the workbook.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from xml.sax.saxutils import escape as _xml_escape

from core.config import BriefpostConfig
from core.constants import Priority, Status
from core.models import BatchResult

logger = logging.getLogger(__name__)

GSHEETS_HELPER = Path(__file__).resolve().parent / "gsheets.py"

# Source files in order of preference. vorgaenge-rereconciled.json is a
# name from old documentation that no phase writes, so it is never read.
VORGAENGE_FILES = (
    "vorgaenge-cross-intel.json",
    "vorgaenge-reconciled.json",
    "vorgaenge.json",
)

# Column M (index 12) holds the priority in tabs 1 and 2.
PRIORITY_COLUMN_INDEX = 12

# Tab 1 and 2 columns. The keys are also the German-mode header names.
COLUMN_KEYS = [
    "letter_id", "vorgang_id", "person", "absender", "typ", "eskalation",
    "kategorie", "datum", "aktenzeichen", "betrag", "faellig_am",
    "status", "prioritaet", "bezahlt_am", "bezahlt_von",
    "empfaenger_iban", "verwendungszweck", "notizen", "ocr_confidence",
]

_LABELS: dict[str, dict[str, Any]] = {
    "de": {
        "tabs": ["Alle Dokumente", "Offene Posten", "Zusammenfassung", "Bank-Abgleich"],
        "columns": {key: key for key in COLUMN_KEYS},
        "summary_title": "Zusammenfassung",
        "created": "Erstellt",
        "case_count": "Anzahl Vorgaenge",
        "letter_count": "Anzahl Briefe",
        "open_total": "Gesamt-Schulden (offen)",
        "paid_total": "Gesamt bezahlt",
        "by_status": "Nach Status",
        "by_priority": "Nach Prioritaet",
        "top_creditors": "Top {n} Glaeubiger",
        "unknown_creditor": "Unbekannt",
        "bank_headers": [
            "Vorgang", "Absender", "Betrag", "Typ", "Methode", "Confidence",
            "Buchungsdatum", "Bank-Empfaenger", "Konto",
        ],
        "status": {},
        "priority": {},
        "doc_type": {},
        "category": {},
        "method": {},
    },
    "en": {
        "tabs": ["All letters", "Open items", "Summary", "Bank matching"],
        "columns": {
            "letter_id": "Letter", "vorgang_id": "Case", "person": "Person",
            "absender": "Sender", "typ": "Type", "eskalation": "Escalation level",
            "kategorie": "Category", "datum": "Letter date", "aktenzeichen": "Reference",
            "betrag": "Amount", "faellig_am": "Due date", "status": "Status",
            "prioritaet": "Priority", "bezahlt_am": "Paid on", "bezahlt_von": "Paid from account",
            "empfaenger_iban": "Payee IBAN", "verwendungszweck": "Payment reference",
            "notizen": "Notes", "ocr_confidence": "Confidence",
        },
        "summary_title": "Summary",
        "created": "Created",
        "case_count": "Cases",
        "letter_count": "Letters",
        "open_total": "Total still to pay",
        "paid_total": "Total already paid",
        "by_status": "By status",
        "by_priority": "By priority",
        "top_creditors": "Top {n} creditors (open amounts)",
        "unknown_creditor": "Unknown",
        "bank_headers": [
            "Case", "Sender", "Amount", "Result", "Match method", "Confidence",
            "Booking date", "Bank payee", "Account",
        ],
        "status": {
            "OFFEN": "Open", "BEZAHLT": "Paid", "TEILWEISE": "Part paid",
            "UNKLAR": "Unclear", "INFO-ONLY": "Info only",
        },
        "priority": {
            "GERICHTLICH": "Court (act now)", "KRITISCH": "Critical", "DRINGEND": "Soon",
            "NORMAL": "Can wait", "ERLEDIGT": "Done",
        },
        "doc_type": {
            "rechnung": "Invoice", "mahnung_1": "Reminder 1", "mahnung_2": "Reminder 2",
            "mahnung_3": "Final reminder", "zahlungserinnerung": "Payment reminder",
            "inkasso": "Debt collection", "mahnbescheid": "Court order to pay",
            "vollstreckung": "Enforcement", "vollstreckungsbescheid": "Enforcement order",
            "pfaendung": "Seizure", "gerichtsvollzieher": "Bailiff", "bescheid": "Official notice",
            "steuerbescheid": "Tax notice", "bussgeldbescheid": "Fine",
            "versicherung": "Insurance", "vertrag": "Contract", "kuendigung": "Cancellation",
            "info": "Information", "allgemeine_korrespondenz": "General letter",
            "kontoauszug": "Bank statement", "gehaltsabrechnung": "Payslip",
            "erbschaft": "Estate", "sonstig": "Other", "unknown": "Unknown",
        },
        "category": {
            "GESUNDHEIT": "Health", "VERSICHERUNG": "Insurance", "WOHNEN": "Housing and utilities",
            "KOMMUNIKATION": "Phone and internet", "STAAT_RECHT": "Government and legal",
            "ERBSCHAFT": "Estate", "FINANZEN": "Finance", "SONSTIGES": "Other",
        },
        "method": {
            "no_match": "No payment found", "reference_exact": "Reference found in bank statement",
            "reference_amount_diff": "Reference found, amount differs",
            "amount_match": "Amount and date match", "amount_date": "Amount and date match",
            "compound_match": "Several payments add up", "llm_match": "AI assisted match",
        },
    },
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def safe_str(value: Any) -> str:
    """Return str(value), with None as an empty string."""
    if value is None:
        return ""
    return str(value)


def _language(config: BriefpostConfig) -> str:
    lang = str(config.output.get("language", "de") or "de").lower()
    return lang if lang in _LABELS else "de"


def _labels(config: BriefpostConfig) -> dict[str, Any]:
    return _LABELS[_language(config)]


def _shown(labels: dict[str, Any], group: str, value: Any) -> str:
    """Display text for an internal enum value (falls back to the raw value)."""
    raw = safe_str(value)
    return labels[group].get(raw, raw)


def _status_of(vorgang: dict[str, Any]) -> str:
    """Status of a case. Cases that were never reconciled count as open."""
    return vorgang.get("status") or Status.OFFEN.value


def _amount_of(vorgang: dict[str, Any]) -> float:
    amount = vorgang.get("aktuellster_betrag")
    return float(amount) if isinstance(amount, (int, float)) and not isinstance(amount, bool) else 0.0


def _open_amount_of(vorgang: dict[str, Any]) -> float:
    """Amount still to pay: the remaining amount for part-paid cases, else the full amount."""
    if _status_of(vorgang) == Status.TEILWEISE.value:
        remaining = (vorgang.get("reconciliation") or {}).get("remaining_amount")
        if isinstance(remaining, (int, float)) and not isinstance(remaining, bool):
            return float(remaining)
    return _amount_of(vorgang)


def _amount_cell(value: Any, labels: dict[str, Any]) -> str:
    """Amount as shown in a table cell: two decimals in English mode, raw value in German mode."""
    if value is None:
        return ""
    if labels is _LABELS["en"] and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.2f}"
    return safe_str(value)


def _money(amount: float, config: BriefpostConfig) -> str:
    currency = safe_str(config.output.get("currency", "EUR")).strip()
    return f"{amount:.2f} {currency}".strip()


def _priority_rank(vorgang: dict[str, Any]) -> int:
    try:
        return Priority(vorgang.get("prioritaet")).sort_order
    except ValueError:
        return 99


def make_conditional_format(sheet_id: int, value: str, color: dict[str, float]) -> dict[str, Any]:
    """Build a Google Sheets request that colours a row's priority cell.

    Args:
        sheet_id: Numeric sheetId of the tab.
        value: Cell text to match exactly (for example "GERICHTLICH").
        color: Background colour as {"red": .., "green": .., "blue": ..}.

    Returns:
        One addConditionalFormatRule request for spreadsheets.batchUpdate.
    """
    return {
        "addConditionalFormatRule": {
            "rule": {
                "ranges": [{
                    "sheetId": sheet_id,
                    "startRowIndex": 1,
                    "startColumnIndex": PRIORITY_COLUMN_INDEX,
                    "endColumnIndex": PRIORITY_COLUMN_INDEX + 1,
                }],
                "booleanRule": {
                    "condition": {
                        "type": "TEXT_EQ",
                        "values": [{"userEnteredValue": value}],
                    },
                    "format": {"backgroundColor": color},
                },
            },
            "index": 0,
        }
    }


# ---------------------------------------------------------------------------
# Tab content
# ---------------------------------------------------------------------------

def _case_row(vorgang: dict[str, Any], letter_cell: str, labels: dict[str, Any]) -> list[str]:
    recon = vorgang.get("reconciliation") or {}
    return [
        letter_cell,
        safe_str(vorgang.get("vorgang_id")),
        safe_str(vorgang.get("person")),
        safe_str(vorgang.get("absender")),
        _shown(labels, "doc_type", vorgang.get("dokument_typ")),
        safe_str(vorgang.get("hoechste_eskalation")),
        _shown(labels, "category", vorgang.get("kategorie")),
        safe_str(vorgang.get("datum")),
        safe_str(vorgang.get("aktenzeichen")),
        _amount_cell(vorgang.get("aktuellster_betrag"), labels),
        safe_str(vorgang.get("faellig_am")),
        _shown(labels, "status", _status_of(vorgang)),
        _shown(labels, "priority", vorgang.get("prioritaet")),
        safe_str(recon.get("booking_date")),
        safe_str(recon.get("bank_account")),
        safe_str(vorgang.get("iban")),
        safe_str(vorgang.get("verwendungszweck")),
        "; ".join(safe_str(flag) for flag in (vorgang.get("flags") or [])),
        safe_str(vorgang.get("pipeline_confidence")),
    ]


def build_tabs(vorgaenge: list[dict[str, Any]], config: BriefpostConfig) -> list[tuple[str, list[list[str]]]]:
    """Build the four tabs as (tab name, rows). Every cell is a string.

    Args:
        vorgaenge: Cases from the reconciliation JSON.
        config: Pipeline configuration (language, currency, top_creditors).

    Returns:
        List of four (name, rows) tuples in display order.
    """
    labels = _labels(config)
    tab_names = labels["tabs"]
    headers = [labels["columns"][key] for key in COLUMN_KEYS]

    # Tab 1: one row per letter
    all_rows: list[list[str]] = [list(headers)]
    for vorgang in vorgaenge:
        briefe = vorgang.get("briefe")
        if briefe is None:
            briefe = [vorgang.get("vorgang_id")]
        for letter_id in briefe:
            all_rows.append(_case_row(vorgang, safe_str(letter_id), labels))

    # Tab 2: open and part-paid cases, most urgent first
    open_cases = [
        v for v in vorgaenge
        if _status_of(v) in (Status.OFFEN.value, Status.TEILWEISE.value)
    ]
    open_cases.sort(key=lambda v: (_priority_rank(v), safe_str(v.get("faellig_am")) or "9999", safe_str(v.get("vorgang_id"))))
    open_rows: list[list[str]] = [list(headers)]
    for vorgang in open_cases:
        letters = vorgang.get("briefe")
        letter_cell = ", ".join(safe_str(b) for b in letters) if letters else safe_str(vorgang.get("vorgang_id"))
        open_rows.append(_case_row(vorgang, letter_cell, labels))

    # Tab 3: summary
    open_total = sum(_open_amount_of(v) for v in open_cases)
    paid_total = sum(_amount_of(v) for v in vorgaenge if _status_of(v) == Status.BEZAHLT.value)
    letter_count = sum(len(v.get("briefe") or []) for v in vorgaenge)

    status_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    for vorgang in vorgaenge:
        status_counts[_status_of(vorgang)] = status_counts.get(_status_of(vorgang), 0) + 1
        prio = safe_str(vorgang.get("prioritaet"))
        if prio:
            priority_counts[prio] = priority_counts.get(prio, 0) + 1

    summary_rows: list[list[str]] = [
        [labels["summary_title"], ""],
        [labels["created"], datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["", ""],
        [labels["case_count"], str(len(vorgaenge))],
        [labels["letter_count"], str(letter_count)],
        [labels["open_total"], _money(open_total, config)],
        [labels["paid_total"], _money(paid_total, config)],
        ["", ""],
        [labels["by_status"], ""],
    ]
    for status in sorted(status_counts):
        summary_rows.append([_shown(labels, "status", status), str(status_counts[status])])
    summary_rows.append(["", ""])
    summary_rows.append([labels["by_priority"], ""])
    priority_order = [p.value for p in Priority]
    for prio in sorted(priority_counts, key=lambda p: priority_order.index(p) if p in priority_order else 99):
        summary_rows.append([_shown(labels, "priority", prio), str(priority_counts[prio])])
    summary_rows.append(["", ""])

    # Top creditors go last: everything below this heading is "name, amount".
    top_n = int(config.output.get("top_creditors", 10) or 10)
    creditor_totals: dict[str, float] = {}
    for vorgang in open_cases:
        name = safe_str(vorgang.get("absender")).strip() or labels["unknown_creditor"]
        creditor_totals[name] = creditor_totals.get(name, 0.0) + _open_amount_of(vorgang)
    summary_rows.append([labels["top_creditors"].format(n=top_n), ""])
    for name, total in sorted(creditor_totals.items(), key=lambda item: (-item[1], item[0]))[:top_n]:
        summary_rows.append([name, f"{total:.2f}"])

    # Tab 4: bank matching
    bank_rows: list[list[str]] = [list(labels["bank_headers"])]
    for vorgang in vorgaenge:
        recon = vorgang.get("reconciliation") or {}
        bank_rows.append([
            safe_str(vorgang.get("vorgang_id")),
            safe_str(vorgang.get("absender")),
            _amount_cell(vorgang.get("aktuellster_betrag"), labels),
            _shown(labels, "status", recon.get("type") or _status_of(vorgang)),
            _shown(labels, "method", recon.get("method")),
            safe_str(recon.get("confidence")),
            safe_str(recon.get("booking_date")),
            safe_str(recon.get("bank_beneficiary")),
            safe_str(recon.get("bank_account")),
        ])

    return [
        (tab_names[0], all_rows),
        (tab_names[1], open_rows),
        (tab_names[2], summary_rows),
        (tab_names[3], bank_rows),
    ]


# ---------------------------------------------------------------------------
# Local output: CSV per tab and one XLSX workbook (standard library only)
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?$")
_ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return cleaned or "tab"


def _column_letter(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _sheet_xml(rows: list[list[str]]) -> str:
    """Worksheet XML for one tab. Plain numbers become numeric cells, the rest inline text."""
    widths: dict[int, int] = {}
    xml_rows: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells: list[str] = []
        for col_index, value in enumerate(row):
            text = _ILLEGAL_XML_RE.sub("", safe_str(value))
            if text == "":
                continue
            widths[col_index] = max(widths.get(col_index, 0), min(len(text), 60))
            ref = f"{_column_letter(col_index)}{row_index}"
            style = ' s="1"' if row_index == 1 else ""
            if row_index > 1 and len(text) <= 15 and _NUMBER_RE.match(text):
                cells.append(f'<c r="{ref}"{style}><v>{text}</v></c>')
            else:
                cells.append(
                    f'<c r="{ref}"{style} t="inlineStr"><is><t xml:space="preserve">'
                    f"{_xml_escape(text)}</t></is></c>"
                )
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    cols = "".join(
        f'<col min="{i + 1}" max="{i + 1}" width="{width + 3}" customWidth="1"/>'
        for i, width in sorted(widths.items())
    )
    cols_xml = f"<cols>{cols}</cols>" if cols else ""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f'{cols_xml}<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    )


def write_xlsx(tabs: list[tuple[str, list[list[str]]]], path: Path) -> None:
    """Write the tabs to one .xlsx workbook using only the standard library.

    Opens in Excel, Numbers, LibreOffice and Google Sheets (File > Import).

    Args:
        tabs: (tab name, rows) pairs.
        path: Target .xlsx path. Written atomically.
    """
    sheet_names: list[str] = []
    for name, _rows in tabs:
        clean = re.sub(r"[\[\]\*\?/\\:]", " ", name).strip()[:31] or "Sheet"
        base, counter = clean, 2
        while clean in sheet_names:
            suffix = f" {counter}"
            clean = base[: 31 - len(suffix)] + suffix
            counter += 1
        sheet_names.append(clean)

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        + "".join(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for i in range(1, len(tabs) + 1)
        )
        + "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
        + "".join(
            f'<sheet name="{_xml_escape(name, {chr(34): "&quot;"})}" sheetId="{i}" r:id="rId{i}"/>'
            for i, name in enumerate(sheet_names, start=1)
        )
        + "</sheets></workbook>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, len(tabs) + 1)
        )
        + f'<Relationship Id="rId{len(tabs) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        "</Relationships>"
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
        "</styleSheet>"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", content_types)
            archive.writestr("_rels/.rels", root_rels)
            archive.writestr("xl/workbook.xml", workbook)
            archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
            archive.writestr("xl/styles.xml", styles)
            for i, (_name, rows) in enumerate(tabs, start=1):
                archive.writestr(f"xl/worksheets/sheet{i}.xml", _sheet_xml(rows))
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_local_outputs(
    tabs: list[tuple[str, list[list[str]]]],
    project_dir: Path,
    config: BriefpostConfig,
) -> dict[str, Any]:
    """Write the workbook and one CSV per tab into <project>/output/.

    Args:
        tabs: (tab name, rows) pairs from build_tabs.
        project_dir: Project data directory.
        config: Pipeline configuration (output.local_dir, output.sheet_prefix).

    Returns:
        {"xlsx": Path, "csv": [Path, ...]}.
    """
    out_dir = project_dir / safe_str(config.output.get("local_dir", "output") or "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = _safe_filename(safe_str(config.output.get("sheet_prefix", "post-summary")))

    xlsx_path = out_dir / f"{prefix}.xlsx"
    write_xlsx(tabs, xlsx_path)

    csv_paths: list[Path] = []
    for index, (name, rows) in enumerate(tabs, start=1):
        csv_path = out_dir / f"{prefix}-{index}-{_safe_filename(name)}.csv"
        # utf-8-sig so Excel shows accented characters correctly
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
            csv.writer(handle).writerows(rows)
        csv_paths.append(csv_path)

    logger.info("Local workbook written: %s", xlsx_path)
    return {"xlsx": xlsx_path, "csv": csv_paths}


# ---------------------------------------------------------------------------
# Google Sheet output (through the gsheets.py helper, run as a subprocess)
# ---------------------------------------------------------------------------

def _run_helper(args: list[str]) -> Optional[dict[str, Any]]:
    """Run gsheets.py with the given arguments and parse its JSON output.

    Returns:
        Parsed JSON dict, or None if the helper failed.
    """
    try:
        proc = subprocess.run(
            [sys.executable, str(GSHEETS_HELPER), *args],
            capture_output=True, text=True, timeout=180,
        )
    except Exception as exc:
        logger.error("Google Sheets helper could not be started: %s", exc)
        return None
    if proc.returncode != 0:
        message = (getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip()
        logger.error("Google Sheets helper failed: %s", str(message)[-400:])
        return None
    try:
        data = json.loads(proc.stdout or "{}")
    except (json.JSONDecodeError, TypeError):
        logger.error("Google Sheets helper returned unreadable output")
        return None
    return data if isinstance(data, dict) else None


def write_tab(
    tab_name: str,
    rows: list[list[str]],
    spreadsheet_id: Optional[str] = None,
    work_dir: Optional[Path] = None,
) -> bool:
    """Write all rows of one tab into the Google Sheet.

    Args:
        tab_name: Name of an existing tab.
        rows: Rows of strings (first row is the header).
        spreadsheet_id: Target spreadsheet.
        work_dir: Folder for the temporary rows file (default: system temp).

    Returns:
        True on success.
    """
    if not spreadsheet_id:
        return False
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix=".sheet-rows-", delete=False, encoding="utf-8",
        dir=str(work_dir) if work_dir and work_dir.is_dir() else None,
    )
    rows_path = Path(handle.name)
    try:
        with handle:
            json.dump(rows, handle, ensure_ascii=False)
        result = _run_helper([
            "write", "--spreadsheet-id", spreadsheet_id,
            "--tab", tab_name, "--rows-file", str(rows_path),
        ])
        return result is not None
    finally:
        if rows_path.exists():
            rows_path.unlink()


def create_google_sheet(vorgaenge: list[dict[str, Any]], config: BriefpostConfig) -> Optional[str]:
    """Create the Google Sheet with all four tabs.

    Args:
        vorgaenge: Cases from the reconciliation JSON.
        config: Pipeline configuration.

    Returns:
        Spreadsheet ID, or None if the sheet could not be created or filled.
    """
    labels = _labels(config)
    tabs = build_tabs(vorgaenge, config)
    prefix = safe_str(config.output.get("sheet_prefix", "Post summary")) or "Post summary"
    title = f"{prefix} {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    created = _run_helper([
        "create", "--title", title,
        "--tabs-json", json.dumps([name for name, _rows in tabs], ensure_ascii=False),
    ])
    if not created or not created.get("spreadsheet_id"):
        logger.error("Google Sheet was not created")
        return None
    spreadsheet_id = str(created["spreadsheet_id"])
    if created.get("url"):
        logger.info("Google Sheet: %s", created["url"])

    project_dir = config.project_dir
    work_dir = (project_dir / "reconciliation") if project_dir else None

    all_written = True
    for tab_name, rows in tabs:
        if not write_tab(tab_name, rows, spreadsheet_id, work_dir):
            logger.error("Could not write tab '%s'", tab_name)
            all_written = False

    # Colour the priority column of the open-items tab
    colors = config.output.get("conditional_format_colors", {}) or {}
    open_tab_name = tabs[1][0]
    sheet_id = None
    for sheet in created.get("sheets") or []:
        if isinstance(sheet, dict) and sheet.get("title") == open_tab_name:
            sheet_id = sheet.get("sheetId")
    if sheet_id is not None and colors:
        requests = []
        for key, priority in (
            ("gerichtlich", Priority.GERICHTLICH),
            ("kritisch", Priority.KRITISCH),
            ("dringend", Priority.DRINGEND),
        ):
            if key in colors:
                requests.append(make_conditional_format(
                    sheet_id, _shown(labels, "priority", priority.value), colors[key],
                ))
        if requests:
            handle = tempfile.NamedTemporaryFile(
                "w", suffix=".json", prefix=".sheet-format-", delete=False, encoding="utf-8",
                dir=str(work_dir) if work_dir and work_dir.is_dir() else None,
            )
            requests_path = Path(handle.name)
            try:
                with handle:
                    json.dump(requests, handle)
                if _run_helper([
                    "batch-update", "--spreadsheet-id", spreadsheet_id,
                    "--requests-file", str(requests_path),
                ]) is None:
                    logger.warning("Colour formatting was not applied (the data is complete)")
            finally:
                if requests_path.exists():
                    requests_path.unlink()

    return spreadsheet_id if all_written else None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_vorgaenge(reconciliation_dir: Path) -> Optional[list[dict[str, Any]]]:
    """Load cases from the best available reconciliation file.

    Order: cross-intel, reconciled, raw. Returns None if none exists.
    """
    for name in VORGAENGE_FILES:
        path = reconciliation_dir / name
        if path.is_file():
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            vorgaenge = data.get("vorgaenge", []) if isinstance(data, dict) else data
            logger.info("Loaded %d cases from %s", len(vorgaenge or []), name)
            return list(vorgaenge or [])
    return None


def run_sheet_builder(project_dir: Path, config: BriefpostConfig) -> BatchResult:
    """Entry point called by pipeline.py.

    Args:
        project_dir: Project data directory.
        config: Pipeline configuration.

    Returns:
        BatchResult with phase "output_sheet".
    """
    result = BatchResult(phase="output_sheet")
    result.total = 1
    project_dir = Path(project_dir)
    reconciliation_dir = project_dir / "reconciliation"

    try:
        vorgaenge = load_vorgaenge(reconciliation_dir) if reconciliation_dir.is_dir() else None
        if vorgaenge is None:
            result.failed = 1
            result.errors.append({"phase": "output", "error": "No vorgaenge JSON found"})
            result.complete()
            return result

        destination = safe_str(config.output.get("destination", "google") or "google").lower()
        if destination not in ("local", "google", "both"):
            logger.warning("Unknown output.destination '%s', using 'local'", destination)
            destination = "local"

        # The local files are always written so there is a result without any account.
        local = write_local_outputs(build_tabs(vorgaenge, config), project_dir, config)
        sheet_reference = f"local:{local['xlsx']}"

        if destination in ("google", "both"):
            sheet_id = create_google_sheet(vorgaenge, config)
            if sheet_id is None:
                result.failed = 1
                result.errors.append({
                    "phase": "output",
                    "error": (
                        "Google Sheet could not be created. The local workbook was written to "
                        f"{local['xlsx']}. Run 'python3 scripts/output/gsheets.py status' to check "
                        "the Google sign-in, or set output.destination to 'local'."
                    ),
                })
            else:
                sheet_reference = sheet_id

        # sheet-id.txt marks the output phase as done (read by the pipeline gate and QC)
        (reconciliation_dir / "sheet-id.txt").write_text(sheet_reference + "\n", encoding="utf-8")

        if result.failed == 0:
            result.succeeded = 1

    except Exception as exc:
        logger.error("Sheet builder failed: %s", exc, exc_info=True)
        result.failed = 1
        result.succeeded = 0
        result.errors.append({"phase": "output", "error": str(exc)})

    result.complete()
    return result
