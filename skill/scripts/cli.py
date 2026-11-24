#!/usr/bin/env python3
"""Command line entry point for the post pipeline.

Usage:
    python3 cli.py <project_dir> [--init] [--check] [--phases a,b] [--config path.yaml]
                   [--resume] [--dry-run] [--log-level DEBUG]

Examples:
    # Create an empty project folder with the right layout
    python3 cli.py ~/post-project --init

    # Check that everything needed is installed and set up
    python3 cli.py ~/post-project --check

    # Full pipeline
    python3 cli.py ~/post-project

    # Only matching, bank check and the workbook (no OCR, no API key needed)
    python3 cli.py ~/post-project --phases match_vorgaenge,reconcile,cross_intelligence,output,qc
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import shutil
import sys
from pathlib import Path

# Ensure the scripts directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

SKILL_DIR = Path(__file__).resolve().parent.parent
CONFIG_EXAMPLE = SKILL_DIR / "config" / "config.example.yaml"

# Phases that call the Gemini API
API_PHASES = {"classify", "ocr", "verify"}

# (import name, pip name, needed for)
REQUIRED_PACKAGES = [
    ("yaml", "pyyaml", "everything"),
    ("PIL", "pillow", "reading photos"),
    ("google.genai", "google-genai", "reading letters with Gemini"),
    ("httpx", "httpx", "reading letters with Gemini"),
]
OPTIONAL_PACKAGES = [
    ("segno", "segno", "SEPA payment QR codes (euro only)"),
    ("googleapiclient", "google-api-python-client", "writing a Google Sheet"),
    ("google_auth_oauthlib", "google-auth-oauthlib", "writing a Google Sheet"),
]


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def setup_logging(level: str) -> None:
    """Configure structured logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def init_project(project_dir: Path) -> int:
    """Create the project folder layout. Never overwrites an existing config.yaml."""
    for sub in ("raw-images/unsorted", "bank-statements"):
        (project_dir / sub).mkdir(parents=True, exist_ok=True)
    target = project_dir / "config.yaml"
    if target.exists():
        print(f"Kept existing settings file: {target}")
    elif CONFIG_EXAMPLE.exists():
        shutil.copyfile(CONFIG_EXAMPLE, target)
        print(f"Created settings file: {target}")
    print(f"Project folder ready: {project_dir}")
    print(f"  1. Put your letter photos in   {project_dir / 'raw-images' / 'unsorted'}")
    print(f"  2. Put bank CSV exports in     {project_dir / 'bank-statements'}  (optional)")
    print(f"  3. Run: {sys.executable} {Path(__file__).resolve()} {project_dir}")
    return 0


def run_checks(project_dir: Path, phases: list[str] | None) -> int:
    """Print a setup report. Returns 0 if the requested phases can run."""
    problems = 0
    print("Setup check")
    print(f"  Python {sys.version.split()[0]}: {'OK' if sys.version_info >= (3, 9) else 'TOO OLD (3.9 or newer needed)'}")
    if sys.version_info < (3, 9):
        problems += 1

    needs_api = phases is None or bool(API_PHASES & set(phases))
    for module, pip_name, purpose in REQUIRED_PACKAGES:
        found = _has_module(module)
        needed = purpose == "everything" or purpose == "reading photos" or needs_api
        state = "OK" if found else ("MISSING" if needed else "missing (not needed for these phases)")
        print(f"  Package {pip_name}: {state}  ({purpose})")
        if not found and needed:
            problems += 1
    for module, pip_name, purpose in OPTIONAL_PACKAGES:
        print(f"  Package {pip_name}: {'OK' if _has_module(module) else 'not installed'}  (optional, {purpose})")

    if needs_api:
        from core.extern import load_api_key
        has_key = load_api_key()
        print(f"  Gemini API key: {'found' if has_key else 'NOT FOUND (set GOOGLE_API_KEY, see README)'}")
        if not has_key:
            problems += 1
    else:
        print("  Gemini API key: not needed for these phases")

    try:
        from core.extern import ocr_core_dir
        ocr_core_dir()
        print("  OCR helper modules: OK")
    except ImportError as exc:
        print(f"  OCR helper modules: MISSING ({exc})")
        problems += 1

    photos_dir = project_dir / "raw-images" / "unsorted"
    photo_count = 0
    if photos_dir.is_dir():
        photo_count = sum(
            1 for f in photos_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".heif"}
        )
    print(f"  Photos in raw-images/unsorted: {photo_count}")
    bank_dir = project_dir / "bank-statements"
    csv_count = len(list(bank_dir.glob("*.csv"))) if bank_dir.is_dir() else 0
    print(f"  Bank CSV files: {csv_count}{'' if csv_count else ' (optional; without them every case stays open)'}")
    if (phases is None or "group" in phases) and photo_count == 0:
        print("  -> No photos found. Add photos, or run with --init to create the folders.")
        problems += 1

    print("Result: " + ("ready" if problems == 0 else f"{problems} problem(s) to fix"))
    return 0 if problems == 0 else 1


def print_result(result: dict, project_dir: Path | None = None) -> None:
    """Pretty-print pipeline results."""
    phases = result.get("phases", [])
    all_ok = result.get("all_succeeded", False)

    separator = "=" * 60
    print(f"\n{separator}")
    print("POST PIPELINE RESULTS")
    print(separator)
    print(f"  Project:    {result.get('project_dir', 'N/A')}")
    print(f"  Started:    {result.get('started_at', 'N/A')}")
    print(f"  Completed:  {result.get('completed_at', 'N/A')}")
    print(f"  Overall:    {'PASS' if all_ok else 'FAIL'}")
    print(f"{separator}")

    for phase in phases:
        status = "PASS" if phase.get("failed", 0) == 0 else "FAIL"
        name = phase.get("phase", "unknown")
        print(f"  [{status}] {name}")
        if phase.get("errors"):
            for error in phase["errors"][:3]:
                print(f"         Error: {error.get('error', 'unknown')}")

    print(separator)

    if project_dir is not None:
        sheet_ref = project_dir / "reconciliation" / "sheet-id.txt"
        if sheet_ref.exists():
            ref = sheet_ref.read_text(encoding="utf-8").strip()
            if ref.startswith("local:"):
                print(f"\n  Your summary workbook: {ref[len('local:'):]}")
            elif ref:
                print(f"\n  Your Google Sheet: https://docs.google.com/spreadsheets/d/{ref}")
        out_dir = project_dir / "output"
        if out_dir.is_dir():
            print(f"  Local files: {out_dir}")

    if not all_ok:
        failed_phases = [p["phase"] for p in phases if p.get("failed", 0) > 0]
        print(f"\n  FAILED PHASES: {', '.join(failed_phases)}")
        print("  Fix the errors above, then run again with --resume or --phases.")


def main() -> int:
    """Main CLI entry point."""
    try:
        from pipeline import PHASE_ORDER, run_pipeline
        from core.config import BriefpostConfig
    except ImportError as exc:
        print(f"ERROR: a required package is missing: {exc}", file=sys.stderr)
        print(f"Run: python3 -m pip install -r {SKILL_DIR / 'requirements.txt'}", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(
        description="Post pipeline: photographed letters to a prioritised summary workbook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available phases: {', '.join(PHASE_ORDER)}",
    )

    parser.add_argument("project_dir", type=str, help="Path to the project data directory")
    parser.add_argument("--init", action="store_true",
                        help="Create the project folder layout and a config.yaml, then exit")
    parser.add_argument("--check", action="store_true",
                        help="Check packages, API key and input files, then exit")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a custom YAML config file (merged with defaults)")
    parser.add_argument("--phases", type=str, default=None,
                        help="Comma-separated list of phases to run (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip phases that already finished and whose outputs are unchanged")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: INFO)")

    args = parser.parse_args()
    setup_logging(args.log_level)

    project_dir = Path(args.project_dir).expanduser().resolve()

    if args.init:
        return init_project(project_dir)

    if not project_dir.exists():
        print(f"ERROR: Project directory does not exist: {project_dir}", file=sys.stderr)
        print(f"Create it with: {sys.executable} {Path(__file__).resolve()} {project_dir} --init", file=sys.stderr)
        return 1

    # Parse phases
    phases = None
    if args.phases:
        phases = [p.strip().replace("-", "_") for p in args.phases.split(",") if p.strip()]
        invalid = [p for p in phases if p not in PHASE_ORDER]
        if invalid:
            print(f"ERROR: Unknown phases: {invalid}", file=sys.stderr)
            print(f"Valid phases: {', '.join(PHASE_ORDER)}", file=sys.stderr)
            return 1

    if args.check:
        return run_checks(project_dir, phases)

    # Load configuration
    try:
        if args.config:
            config = BriefpostConfig.from_yaml(args.config, project_dir=project_dir)
        else:
            config = BriefpostConfig.from_project_dir(project_dir)
    except Exception as exc:
        print(f"ERROR: could not read the settings file: {exc}", file=sys.stderr)
        return 1

    errors = config.validate()
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    # Stop early with a clear message when the API key is missing
    if not args.dry_run and (phases is None or API_PHASES & set(phases)):
        from core.extern import load_api_key
        if not load_api_key():
            print(
                "ERROR: no Gemini API key found. Set GOOGLE_API_KEY in your shell, or put\n"
                "  GOOGLE_API_KEY=your_key\n"
                "in ~/.config/briefpost-management/.env (see README, section Setup).",
                file=sys.stderr,
            )
            return 1

    # Run pipeline
    try:
        result = run_pipeline(
            project_dir=project_dir,
            config=config,
            phases=phases,
            dry_run=args.dry_run,
            resume=args.resume,
        )
        print_result(result.to_dict(), project_dir)
        return 0 if result.all_succeeded else 1

    except Exception as exc:
        logging.getLogger(__name__).error("Pipeline error: %s", exc, exc_info=True)
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
