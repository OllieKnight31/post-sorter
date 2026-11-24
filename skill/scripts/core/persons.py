"""People whose post is being processed (the "persons" section of the config).

Everything that needs to know who lives in the household reads it from here,
so no name is ever written into the code:
  - regex signals on OCR text (address block, persons_detected)
  - the person rules inside the OCR prompts
  - the page classification prompt

Config shape (all optional):

    persons:
      known_names: ["Sam", "Jo"]          # short labels shown in the sheet
      address_surnames: ["Example"]         # surnames as printed in address blocks
      aliases:                              # names as printed on letters
        Sam: ["Samuel Example", "S. Example"]
      accounts:                             # bank accounts per person
        Sam:
          - {iban: "GB00XXXX00000000000000", label: "main"}

With nothing configured, person stays empty everywhere and all letters and
bank rows are treated as one household.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Salutations that open an address block (German and English post).
_SALUTATION = r"(?:Herrn?|Frau|Familie|Mr|Mrs|Ms|Miss|Mx|Dr)\.?"

_state: dict[str, Any] = {
    "known_names": [],
    "address_surnames": [],
    "aliases": {},
}


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def configure_persons(config: Any) -> None:
    """Load the persons section of the config and push it into the OCR prompts.

    Safe to call many times. Never raises: a missing OCR helper only means
    the prompts keep their neutral default (person stays empty).

    Args:
        config: BriefpostConfig, or a plain dict with the persons section.
    """
    persons = getattr(config, "persons", None)
    if persons is None and isinstance(config, dict):
        persons = config
    persons = persons or {}

    _state["known_names"] = _clean_list(persons.get("known_names"))
    _state["address_surnames"] = _clean_list(persons.get("address_surnames"))
    aliases = persons.get("aliases") or {}
    if not isinstance(aliases, dict):
        aliases = {}
    _state["aliases"] = {
        str(name).strip(): _clean_list(printed)
        for name, printed in aliases.items()
        if str(name).strip()
    }

    try:
        from core.extern import load_ocr_module
        prompts = load_ocr_module("extraction_prompts")
        if hasattr(prompts, "configure_persons"):
            prompts.configure_persons(_state["known_names"], _state["aliases"])
    except Exception as exc:  # ImportError or anything raised while importing
        logger.debug("OCR prompts not configured with persons: %s", exc)


def known_names() -> list[str]:
    """Short person labels from the config (may be empty)."""
    return list(_state["known_names"])


def person_prompt_hint() -> str:
    """One line for prompts that says which values the person field may take."""
    names = known_names()
    if not names:
        return "null (keine Personen konfiguriert)"
    return " oder ".join(f'"{n}"' for n in names) + " oder null"


def has_address_block(text: str) -> bool:
    """True if the text looks like it contains a postal address block.

    A salutation followed by a capitalised name counts, as does any configured
    surname from persons.address_surnames.
    """
    if re.search(_SALUTATION + r"\s+[A-ZÄÖÜ]", text):
        return True
    for surname in _state["address_surnames"]:
        if re.search(r"\b" + re.escape(surname) + r"\b", text, re.IGNORECASE):
            return True
    return False


def detect_persons(text: str) -> list[str]:
    """Return the configured person labels whose names appear in the text.

    A label matches through its aliases when aliases are configured for it,
    otherwise through the label itself (whole word, case-insensitive).
    """
    found: list[str] = []
    for name in _state["known_names"]:
        patterns = _state["aliases"].get(name) or [name]
        for printed in patterns:
            # Allow flexible whitespace between name parts ("Samuel  Example").
            parts = [re.escape(p) for p in printed.split()]
            if not parts:
                continue
            if re.search(r"\b" + r"\s+".join(parts) + r"\b", text, re.IGNORECASE):
                found.append(name)
                break
    return found


def reset_persons() -> None:
    """Clear the configured persons (used by tests)."""
    _state["known_names"] = []
    _state["address_surnames"] = []
    _state["aliases"] = {}


def get_state() -> Optional[dict[str, Any]]:
    """Return a copy of the current persons state (for diagnostics)."""
    return {k: (list(v) if isinstance(v, list) else dict(v)) for k, v in _state.items()}
