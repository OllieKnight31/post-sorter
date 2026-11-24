"""Loader for the OCR helper modules (Gemini client, prompts, schemas, IBAN checks).

The helper files use relative imports ("from .base import ..."), so they
must be imported as members of a package. Loading them one file at a time
with importlib.util.spec_from_file_location() fails with "attempted relative
import with no known parent package". This module registers the helper
folder as a lightweight package and imports from it.

Lookup order for the helper folder:
  1. BRIEFPOST_OCR_CORE environment variable (a folder path)
  2. scripts/vendor/ocr_core/ inside this skill (bundled copy)
  3. ~/.claude/skills/ocr-processor/scripts/core (only if that separate skill is installed)
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any

_PACKAGE_NAME = "_briefpost_ocr_core"

_VENDORED_DIR = Path(__file__).resolve().parent.parent / "vendor" / "ocr_core"
_SIBLING_SKILL_DIR = Path("~/.claude/skills/ocr-processor/scripts/core").expanduser()


def ocr_core_dir() -> Path:
    """Return the folder that holds the OCR helper modules.

    Raises:
        ImportError: If no helper folder can be found.
    """
    override = os.environ.get("BRIEFPOST_OCR_CORE", "").strip()
    candidates = [Path(override).expanduser()] if override else []
    candidates += [_VENDORED_DIR, _SIBLING_SKILL_DIR]
    for candidate in candidates:
        if (candidate / "extraction_types.py").is_file():
            return candidate
    raise ImportError(
        "OCR helper modules not found. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def load_ocr_module(name: str) -> Any:
    """Import one OCR helper module by name, for example "gemini".

    Args:
        name: Module file name without ".py".

    Returns:
        The imported module.

    Raises:
        ImportError: If the helper folder or the module is missing.
    """
    core_dir = ocr_core_dir()
    package = sys.modules.get(_PACKAGE_NAME)
    if package is None or list(getattr(package, "__path__", [])) != [str(core_dir)]:
        # Drop modules cached from a different folder, then register the package.
        for cached in [m for m in sys.modules if m.startswith(_PACKAGE_NAME + ".")]:
            del sys.modules[cached]
        package = types.ModuleType(_PACKAGE_NAME)
        package.__path__ = [str(core_dir)]  # type: ignore[attr-defined]
        package.__package__ = _PACKAGE_NAME
        sys.modules[_PACKAGE_NAME] = package
    if not (core_dir / f"{name}.py").is_file():
        raise ImportError(f"Cannot find {name}.py in {core_dir}")
    return importlib.import_module(f"{_PACKAGE_NAME}.{name}")


def load_ocr_attr(module_name: str, attr_name: str) -> Any:
    """Import one attribute from an OCR helper module.

    Args:
        module_name: Module file name without ".py".
        attr_name: Attribute to return.

    Returns:
        The attribute.
    """
    module = load_ocr_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(f"{module_name}.py has no attribute {attr_name}") from exc


def load_api_key() -> bool:
    """Make sure GOOGLE_API_KEY is in the environment.

    Reads it from the environment (GOOGLE_API_KEY or GEMINI_API_KEY) or from
    one of these files, first match wins:
      ~/.config/briefpost-management/.env
      ~/.claude/.env

    Returns:
        True if a key is available, False otherwise. The key is never logged.
    """
    if os.environ.get("GOOGLE_API_KEY"):
        return True
    if os.environ.get("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
        return True
    for env_file in (
        Path("~/.config/briefpost-management/.env").expanduser(),
        Path("~/.claude/.env").expanduser(),
    ):
        if not env_file.is_file():
            continue
        try:
            lines = env_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("export "):
                stripped = stripped[len("export "):].lstrip()
            for key_name in ("GOOGLE_API_KEY=", "GEMINI_API_KEY="):
                if stripped.startswith(key_name):
                    value = stripped.split("=", 1)[1].strip().strip("'\"")
                    if value:
                        os.environ["GOOGLE_API_KEY"] = value
                        return True
    return False
