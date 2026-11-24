#!/usr/bin/env python3
"""Offline self-test. Needs no API key, no Google account and no internet.

Copies the fictional demo letters and bank export into a temporary folder,
runs the real matching, bank reconciliation, gap filling, workbook and
quality-check phases on them, and checks the results.

Usage:
    python3 selftest.py            # run and clean up
    python3 selftest.py --keep     # keep the output folder and print its path

Exit code 0 = everything works. The OCR phases (reading photos with Gemini)
are not covered: they need a live API key.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.dont_write_bytecode = True

DEMO = SCRIPTS.parent / "examples" / "demo"

results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")


def main() -> int:
    keep = "--keep" in sys.argv
    logging.basicConfig(level=logging.ERROR)

    try:
        from core.config import BriefpostConfig
        from pipeline import run_pipeline
    except ImportError as exc:
        print(f"A required package is missing: {exc}")
        print(f"Run: python3 -m pip install -r {SCRIPTS.parent / 'requirements.txt'}")
        return 1

    if not DEMO.is_dir():
        print(f"Demo data not found at {DEMO}")
        return 1

    work = Path(tempfile.mkdtemp(prefix="briefpost-selftest-"))
    project = work / "project"
    shutil.copytree(DEMO, project)
    print(f"Self-test project: {project}")

    try:
        config = BriefpostConfig(
            project_dir=project,
            output={"destination": "local", "language": "en", "currency": "EUR"},
        )
        result = run_pipeline(
            project, config,
            phases=["match_vorgaenge", "reconcile", "cross_intelligence", "output", "qc"],
        )
        for phase in result.phase_results:
            check(f"phase {phase.phase} ran", phase.failed == 0,
                  "; ".join(str(e.get("error")) for e in phase.errors)[:200])
        check("all phases succeeded", result.all_succeeded)

        recon = project / "reconciliation"
        cases = json.loads((recon / "vorgaenge-cross-intel.json").read_text(encoding="utf-8"))["vorgaenge"]
        by_sender = {c["absender"]: c for c in cases}

        check("7 letters grouped into 6 cases", len(cases) == 6, f"{len(cases)} cases")
        energy = by_sender.get("Energy Supplier Ltd", {})
        check("reminder attached to its original bill",
              sorted(energy.get("briefe", [])) == ["LTR-001", "LTR-002"], str(energy.get("briefe")))
        check("paid bill found in the bank export",
              energy.get("status") == "BEZAHLT" and energy.get("prioritaet") == "ERLEDIGT",
              f"{energy.get('status')} via {(energy.get('reconciliation') or {}).get('method')}")
        check("court letter ranked most urgent",
              by_sender.get("County Court", {}).get("prioritaet") == "GERICHTLICH")
        check("final notice ranked critical",
              by_sender.get("Tax Office", {}).get("prioritaet") == "KRITISCH")
        check("second reminder ranked soon",
              by_sender.get("Water Company Ltd", {}).get("prioritaet") == "DRINGEND")
        check("ordinary bills can wait",
              by_sender.get("Phone Provider Ltd", {}).get("prioritaet") == "NORMAL"
              and by_sender.get("Home Insurer Ltd", {}).get("prioritaet") == "NORMAL")
        open_count = sum(1 for c in cases if c.get("status") == "OFFEN")
        check("5 cases still open", open_count == 5, f"{open_count} open")

        out = project / "output"
        xlsx = out / "post-summary.xlsx"
        check("workbook written", xlsx.is_file(), str(xlsx))
        if xlsx.is_file():
            with zipfile.ZipFile(xlsx) as archive:
                names = archive.namelist()
                check("workbook is a valid archive with 4 sheets",
                      archive.testzip() is None
                      and sum(1 for n in names if n.startswith("xl/worksheets/sheet")) == 4)
                workbook_xml = archive.read("xl/workbook.xml").decode("utf-8")
                check("workbook tabs are in English",
                      all(t in workbook_xml for t in ("All letters", "Open items", "Summary", "Bank matching")))

        open_csv = out / "post-summary-2-open-items.csv"
        check("open items CSV written", open_csv.is_file())
        if open_csv.is_file():
            with open(open_csv, encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
            header, data = rows[0], rows[1:]
            check("open items has 5 rows, court letter first",
                  len(data) == 5 and data[0][header.index("Sender")] == "County Court",
                  f"{len(data)} rows, first: {data[0][header.index('Sender')] if data else 'none'}")
            check("labels are English",
                  header[:4] == ["Letter", "Case", "Person", "Sender"]
                  and data[0][header.index("Priority")] == "Court (act now)"
                  and data[0][header.index("Status")] == "Open")

        summary_csv = out / "post-summary-3-summary.csv"
        if summary_csv.is_file():
            with open(summary_csv, encoding="utf-8-sig", newline="") as handle:
                summary = {r[0]: r[1] for r in csv.reader(handle) if len(r) > 1}
            check("total still to pay is 1899.65",
                  summary.get("Total still to pay") == "1899.65 EUR", summary.get("Total still to pay", ""))
            check("total already paid is 184.20",
                  summary.get("Total already paid") == "184.20 EUR", summary.get("Total already paid", ""))

        qr_dir = project / "qr-codes"
        qr_files = sorted(p.name for p in qr_dir.glob("*.png")) if qr_dir.is_dir() else []
        try:
            import segno  # noqa: F401
            check("payment QR code written for the one open case with an IBAN",
                  len(qr_files) == 1, ", ".join(qr_files) or "none")
        except ImportError:
            print("  [SKIP] payment QR code (optional 'segno' package not installed)")

        check("quality report written", (recon / "qc-report.md").is_file())

    except Exception as exc:  # A crash is a failed self-test, not a traceback dump
        logging.getLogger(__name__).error("self-test crashed", exc_info=True)
        check("self-test ran without crashing", False, f"{type(exc).__name__}: {exc}")

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")

    if keep:
        print(f"Output kept in: {project}")
    else:
        shutil.rmtree(work, ignore_errors=True)

    if failed:
        print("SELF-TEST FAILED")
        return 1
    print("SELF-TEST PASSED (OCR not covered: it needs a live Gemini API key)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
