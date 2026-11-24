"""Phase C: Vorgang (case) matching - groups letters into proceedings.

Uses 3-tier matching:
1. Exact Aktenzeichen -> 100% confidence (connected components via DFS)
2. Same creditor + similar amount (+-5%) -> 90% confidence
3. Same creditor only -> 85% confidence (single creditor match)

Unmatched letters with unknown creditors get UNKNOWN_CREDITOR flag.
Orphan letters get ORPHAN_LETTER flag.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core.confidence import (
    compute_pipeline_confidence_from_letters,
    get_match_confidence,
)
from core.config import BriefpostConfig
from core.constants import (
    CATEGORIZATION_RULES,
    Flag,
    MatchMethod,
    Priority,
    get_escalation,
)
from core.interfaces import VorgangMatcher
from core.io import atomic_write_json
from core.llm_helpers import call_gemini_flash
from core.models import BatchResult, Brief, Vorgang
from core.utils import fuzzy_creditor_match, normalize_creditor

logger = logging.getLogger(__name__)


def normalize_reference(ref: str) -> str:
    """Normalize reference/Aktenzeichen for comparison.

    Args:
        ref: Raw reference string.

    Returns:
        Normalized reference with whitespace and punctuation removed.
    """
    if not ref:
        return ""
    return re.sub(r"[\s\-/.\(\)]", "", str(ref).lower().strip())


def get_letter_escalation(letter: dict[str, Any]) -> int:
    """Get escalation level from letter.

    Args:
        letter: Letter JSON data.

    Returns:
        Escalation level (0-6).
    """
    return letter.get("eskalation", 0)


def get_letter_amount(letter: dict[str, Any]) -> float:
    """Get current amount from letter.

    Args:
        letter: Letter JSON data.

    Returns:
        Current amount or 0.
    """
    amounts = letter.get("amounts", {})
    if isinstance(amounts, dict):
        return amounts.get("current") or amounts.get("original") or 0
    return 0


def compute_priority(vorgang: dict[str, Any], config: BriefpostConfig) -> str:
    """Compute priority based on escalation, amounts, and special rules.

    Args:
        vorgang: Vorgang data dict.
        config: Pipeline configuration for thresholds.

    Returns:
        Priority string value.
    """
    priority_config = config.priority
    esc = vorgang.get("hoechste_eskalation", 0)
    creditor = (vorgang.get("absender") or "").lower()

    gerichtlich_threshold = priority_config.get("gerichtlich_threshold", 4)
    kritisch_threshold = priority_config.get("kritisch_threshold", 3)
    dringend_amount = priority_config.get("dringend_amount", 500)
    critical_creditors = [c.lower() for c in priority_config.get("critical_creditors", [])]

    is_critical_creditor = any(cc in creditor for cc in critical_creditors) if critical_creditors else False

    if esc >= gerichtlich_threshold:
        return Priority.GERICHTLICH.value
    elif esc >= kritisch_threshold or is_critical_creditor:
        return Priority.KRITISCH.value
    elif esc == 2 or (vorgang.get("aktuellster_betrag", 0) or 0) > dringend_amount:
        return Priority.DRINGEND.value
    else:
        return Priority.NORMAL.value


def categorize_vorgang(absender: str, doc_type: str) -> str:
    """Determine category from creditor name and document type.

    Uses CATEGORIZATION_RULES: first keyword match wins.

    Args:
        absender: Creditor/sender name.
        doc_type: Document type string.

    Returns:
        Category string value (e.g. "VERSICHERUNG", "SONSTIGES").
    """
    search_text = f"{absender} {doc_type}".lower()
    for keywords, category in CATEGORIZATION_RULES:
        if any(kw in search_text for kw in keywords):
            return category
    return "SONSTIGES"


def compute_pipeline_confidence(letters: list[dict[str, Any]], match_confidence: float) -> float:
    """Compute pipeline_confidence = avg(letter_confidence) * match_confidence.

    Uses the centralized confidence module which reads letter_confidence,
    verified_confidence, confidence.overall, or ocr_confidence (in that order).

    Args:
        letters: List of letter JSON data dicts.
        match_confidence: Confidence of the matching step.

    Returns:
        Pipeline confidence score between 0.0 and 1.0.
    """
    return compute_pipeline_confidence_from_letters(letters, match_confidence)


def build_vorgang(
    counter: int,
    letters: list[dict[str, Any]],
    match_method: str,
    match_confidence: float,
    config: BriefpostConfig,
) -> dict[str, Any]:
    """Build a Vorgang record from a list of letters.

    Args:
        counter: Vorgang counter for ID generation.
        letters: List of letter JSON data dicts.
        match_method: Method used for matching.
        match_confidence: Confidence of the match.
        config: Pipeline configuration.

    Returns:
        Vorgang dict ready for JSON serialization.
    """
    vorgang_id = f"V-{counter:03d}"

    highest_esc = max((get_letter_escalation(letter) for letter in letters), default=0)
    best_letter = max(letters, key=lambda letter: get_letter_escalation(letter))

    creditor = best_letter.get("creditor", {})
    if isinstance(creditor, dict):
        absender = creditor.get("name", "")
    else:
        absender = str(creditor)

    # Person: majority vote
    person_votes: dict[str, int] = defaultdict(int)
    for letter in letters:
        person = letter.get("person")
        if person:
            person_votes[person] += 1
    person = max(person_votes, key=person_votes.get) if person_votes else None

    # All identifiers (deduplicated)
    all_refs: list[str] = []
    seen_refs: set[str] = set()
    for letter in letters:
        for ref in letter.get("identifiers", []):
            norm = normalize_reference(ref)
            if norm and norm not in seen_refs:
                all_refs.append(ref)
                seen_refs.add(norm)

    # Pick amount from highest-escalation letter that has a non-zero amount.
    # Vollstreckung/Pfaendung letters often have amount=0, so fall through.
    aktuellster_betrag = 0
    for letter_by_esc in sorted(letters, key=lambda l: get_letter_escalation(l), reverse=True):
        amt = get_letter_amount(letter_by_esc)
        if amt and amt > 0:
            aktuellster_betrag = amt
            break

    payment_info = best_letter.get("payment_info", {})

    # Dates
    all_doc_dates: list[str] = []
    all_due_dates: list[str] = []
    for letter in letters:
        dates = letter.get("dates", {})
        if isinstance(dates, dict):
            if dates.get("document_date"):
                all_doc_dates.append(dates["document_date"])
            if dates.get("due_date"):
                all_due_dates.append(dates["due_date"])

    # Urgency signals
    urgency: list[str] = []
    for letter in letters:
        for sig in letter.get("urgency_signals", []):
            if sig not in urgency:
                urgency.append(sig)

    # Flags
    flags: list[str] = []
    for letter in letters:
        for flag in letter.get("flags", []):
            if flag not in flags:
                flags.append(flag)

    vorgang = {
        "vorgang_id": vorgang_id,
        "absender": absender,
        "person": person,
        "briefe": [letter["letter_id"] for letter in letters],
        "brief_count": len(letters),
        "hoechste_eskalation": highest_esc,
        "dokument_typ": best_letter.get("document_type", "unknown"),
        "aktuellster_betrag": aktuellster_betrag,
        "aktenzeichen": all_refs[0] if all_refs else None,
        "alle_referenzen": all_refs,
        "iban": payment_info.get("iban") if isinstance(payment_info, dict) else None,
        "bic": payment_info.get("bic") if isinstance(payment_info, dict) else None,
        "empfaenger_name": payment_info.get("recipient_name") if isinstance(payment_info, dict) else None,
        "verwendungszweck": payment_info.get("reference") if isinstance(payment_info, dict) else None,
        "kategorie": categorize_vorgang(absender, best_letter.get("document_type", "")),
        "datum": sorted(all_doc_dates)[-1] if all_doc_dates else None,
        "faellig_am": sorted(all_due_dates)[0] if all_due_dates else None,
        "urgency_signals": urgency,
        "status": "OFFEN",
        "prioritaet": None,
        "match_method": match_method,
        "match_confidence": match_confidence,
        "flags": flags,
    }

    vorgang["prioritaet"] = compute_priority(vorgang, config)

    # C5: Pipeline confidence propagation
    vorgang["pipeline_confidence"] = compute_pipeline_confidence(letters, match_confidence)
    pipeline_threshold = config.qc.get("pipeline_confidence_threshold", 0.50)
    if vorgang["pipeline_confidence"] < pipeline_threshold:
        if Flag.LOW_PIPELINE_CONFIDENCE.value not in vorgang["flags"]:
            vorgang["flags"].append(Flag.LOW_PIPELINE_CONFIDENCE.value)

    return vorgang


def _sanitize_for_prompt(value: str, max_len: int = 200) -> str:
    """Sanitize a string for safe interpolation into LLM prompts.

    Strips control characters and newlines that could break prompt
    structure, and truncates to max_len.

    Args:
        value: Raw string from OCR extraction.
        max_len: Maximum character length.

    Returns:
        Sanitized string safe for prompt interpolation.
    """
    if not value:
        return value
    # Strip newlines and control chars that could break prompt structure
    cleaned = value.replace("\n", " ").replace("\r", " ")
    cleaned = cleaned.replace("{", "(").replace("}", ")")
    # Collapse multiple spaces
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned.strip()[:max_len]


def _build_letter_summary(letter: dict[str, Any]) -> str:
    """Build a text summary of a letter for LLM prompts.

    All interpolated values are sanitized to prevent prompt injection.

    Args:
        letter: Letter JSON data dict.

    Returns:
        Multiline summary string.
    """
    creditor = letter.get("creditor", {})
    cred_name = creditor.get("name", "") if isinstance(creditor, dict) else str(creditor)
    cred_name = _sanitize_for_prompt(cred_name, 100)
    amounts = letter.get("amounts", {})
    amt = amounts.get("current") or amounts.get("original") or 0 if isinstance(amounts, dict) else 0
    dates = letter.get("dates", {})
    doc_date = dates.get("document_date", "") if isinstance(dates, dict) else ""
    doc_type = letter.get("document_type", "unknown")
    person = _sanitize_for_prompt(letter.get("person", "unbekannt"), 50)
    identifiers = letter.get("identifiers", [])
    refs_str = ", ".join(_sanitize_for_prompt(str(i), 60) for i in identifiers[:5]) if identifiers else "keine"
    esc = letter.get("eskalation", 0)
    betreff = _sanitize_for_prompt(letter.get("betreff", ""), 80)

    return (
        f"  - {letter['letter_id']}: {cred_name}, Typ={doc_type}, "
        f"Eskalation={esc}, Betrag=EUR {amt}, Datum={doc_date}, "
        f"Person={person}, Referenzen=[{refs_str}]"
        + (f", Betreff: {betreff}" if betreff else "")
    )


def llm_group_same_creditor(
    creditor_name: str,
    letters: list[dict[str, Any]],
) -> Optional[list[list[dict[str, Any]]]]:
    """Use Gemini Flash to decide if same-creditor letters belong to one Vorgang.

    Only called when multiple letters from the same creditor have no shared
    Aktenzeichen/reference number. The LLM analyzes escalation sequences,
    amounts, and temporal patterns to determine grouping.

    Args:
        creditor_name: Normalized creditor name.
        letters: Letters from the same creditor without shared refs.

    Returns:
        List of letter groups, or None if LLM call fails.
    """
    if len(letters) < 2:
        return None

    summaries = "\n".join(_build_letter_summary(letter) for letter in letters)
    letter_ids = [letter["letter_id"] for letter in letters]

    prompt = f"""Hier sind {len(letters)} Briefe vom gleichen Absender ({creditor_name}).
Keiner teilt eine gemeinsame Aktenzeichen/Referenznummer.

BRIEFE:
{summaries}

Gehoeren diese Briefe zum GLEICHEN Vorgang (Fall/Angelegenheit)?

SIGNALE FUER GLEICHEN VORGANG:
- Eskalationsfolge: Rechnung -> Mahnung 1 -> Mahnung 2 -> Mahnbescheid
- Aehnlicher Betrag (mit akkumulierten Gebuehren)
- Zeitliche Abfolge (spaetere Briefe referenzieren fruehere)
- Brief erwaehnt fruehere Korrespondenz

SIGNALE FUER VERSCHIEDENE VORGAENGE:
- Komplett verschiedene Betraege ohne erklaerbare Gebuehren
- Verschiedene Rechnungsnummern/Kundennummern
- Verschiedene Personen (unterschiedliche Empfaenger)
- Brief erwaehnt explizit eine NEUE Forderung

Antworte mit JSON:
{{
  "groups": [
    {{"letter_ids": ["LTR-001", "LTR-005"], "reasoning": "Eskalationsfolge...", "confidence": 0.85}},
    {{"letter_ids": ["LTR-009"], "reasoning": "Separate Forderung...", "confidence": 0.90}}
  ]
}}

Verwende NUR die folgenden letter_ids: {letter_ids}
Jeder Brief muss in genau einer Gruppe erscheinen."""

    result = call_gemini_flash(prompt, temperature=0.0)
    if not result or "groups" not in result:
        return None

    # Validate response: every letter_id must appear exactly once
    letter_id_set = set(letter_ids)
    seen: set[str] = set()
    groups: list[list[dict[str, Any]]] = []
    letter_map = {letter["letter_id"]: letter for letter in letters}

    for group_data in result["groups"]:
        group_ids = group_data.get("letter_ids", [])
        group_letters: list[dict[str, Any]] = []
        for lid in group_ids:
            if lid in letter_id_set and lid not in seen:
                seen.add(lid)
                group_letters.append(letter_map[lid])
        if group_letters:
            groups.append(group_letters)

    # If LLM missed some letters, put them in their own group
    missing = letter_id_set - seen
    for lid in sorted(missing):
        groups.append([letter_map[lid]])

    return groups if groups else None


def llm_resolve_overmerge(
    letters: list[dict[str, Any]],
    vorgang_flags: list[str],
) -> Optional[list[list[dict[str, Any]]]]:
    """Use Gemini Flash to split an over-merged component.

    Called for Tier 1 components flagged OVERSIZED_COMPONENT or
    TRANSITIVE_MERGE_WARNING. Asks the LLM to determine the correct
    sub-grouping.

    Args:
        letters: Letters in the over-merged component.
        vorgang_flags: Existing flags on the component.

    Returns:
        List of letter groups, or None if LLM call fails or decides no split.
    """
    if len(letters) < 3:
        return None

    summaries = "\n".join(_build_letter_summary(letter) for letter in letters)
    letter_ids = [letter["letter_id"] for letter in letters]

    prompt = f"""Diese {len(letters)} Briefe wurden algorithmisch zu einem einzigen Vorgang zusammengefasst,
aber es gibt Warnsignale fuer eine moegliche Ueber-Zusammenfuehrung:
Flags: {vorgang_flags}

BRIEFE:
{summaries}

Sollen diese Briefe in MEHRERE Vorgaenge aufgeteilt werden?

KRITERIEN FUER AUFTEILUNG:
- Verschiedene Personen sollten separate Vorgaenge haben
- Verschiedene Forderungen (unterschiedliche Betraege ohne erklaerbare Eskalation)
- Referenznummern verbinden NUR bestimmte Briefe miteinander

KRITERIEN GEGEN AUFTEILUNG:
- Eskalationsfolge vom gleichen Absender
- Gleiche Referenznummer
- Gleicher Betrag oder nachvollziehbare Erhoehung

Antworte mit JSON:
{{
  "should_split": true,
  "groups": [
    {{"letter_ids": ["LTR-001", "LTR-005"], "reasoning": "Gleiche Forderung"}},
    {{"letter_ids": ["LTR-009"], "reasoning": "Andere Person/Forderung"}}
  ]
}}

Verwende NUR die folgenden letter_ids: {letter_ids}
Jeder Brief muss in genau einer Gruppe erscheinen.
Wenn KEINE Aufteilung noetig ist, antworte: {{"should_split": false, "groups": []}}"""

    result = call_gemini_flash(prompt, temperature=0.0)
    if not result:
        return None

    if not result.get("should_split", False):
        return None

    # Validate and build groups
    letter_id_set = set(letter_ids)
    seen: set[str] = set()
    groups: list[list[dict[str, Any]]] = []
    letter_map = {letter["letter_id"]: letter for letter in letters}

    for group_data in result.get("groups", []):
        group_ids = group_data.get("letter_ids", [])
        group_letters: list[dict[str, Any]] = []
        for lid in group_ids:
            if lid in letter_id_set and lid not in seen:
                seen.add(lid)
                group_letters.append(letter_map[lid])
        if group_letters:
            groups.append(group_letters)

    # Assign any missed letters to their own group
    missing = letter_id_set - seen
    for lid in sorted(missing):
        groups.append([letter_map[lid]])

    # Only return split if it produced >1 group
    return groups if len(groups) > 1 else None


def match_letters_to_vorgaenge(
    letters: list[dict[str, Any]],
    config: BriefpostConfig,
) -> list[dict[str, Any]]:
    """Group letters into Vorgaenge using 3-tier matching.

    Args:
        letters: All letter JSON data dicts.
        config: Pipeline configuration.

    Returns:
        List of Vorgang dicts.
    """
    matching_config = config.matching
    amount_tolerance_pct = matching_config.get("amount_tolerance_percent", 5) / 100.0
    amount_tolerance_abs = matching_config.get("amount_tolerance_absolute_eur", 50.0)
    max_letters_per_vorgang = matching_config.get("max_letters_per_vorgang", 8)

    vorgaenge: list[dict[str, Any]] = []
    vorgang_counter = 0
    assigned_letter_ids: set[str] = set()

    # Build index of references -> letter IDs
    # Skip short references (< 6 chars) to prevent unrelated Vorgaenge merging
    ref_min_length = matching_config.get("reference_min_length", 8)
    ref_index: dict[str, list[str]] = defaultdict(list)
    for letter in letters:
        for ref in letter.get("identifiers", []):
            norm_ref = normalize_reference(ref)
            if norm_ref and len(norm_ref) >= ref_min_length:
                ref_index[norm_ref].append(letter["letter_id"])

    # Tier 1: Exact Aktenzeichen match (connected components)
    ref_groups: dict[str, set[str]] = defaultdict(set)
    for norm_ref, letter_ids in ref_index.items():
        if len(letter_ids) > 1:
            for lid in letter_ids:
                for other_lid in letter_ids:
                    if lid != other_lid:
                        ref_groups[lid].add(other_lid)

    # RS-029: Detect duplicate letter_ids before building map
    letter_map: dict[str, dict[str, Any]] = {}
    for letter in letters:
        lid = letter["letter_id"]
        if lid in letter_map:
            logger.warning("RS-029: Duplicate letter_id '%s' detected, keeping first occurrence", lid)
        else:
            letter_map[lid] = letter
    visited: set[str] = set()
    components: list[set[str]] = []
    # B-03: Track cross-person blocked letters per component (not globally)
    component_cross_person_blocked: list[list[str]] = []
    cross_person_blocked: list[str] = []

    # C2: Page-level cross-person detection: flag letters with multiple persons on pages
    for letter in letters:
        page_persons: set[str] = set()
        for page in letter.get("pages", []):
            page_person = page.get("person")
            if page_person:
                page_persons.add(page_person)
        if len(page_persons) > 1:
            flags = letter.get("flags", [])
            if Flag.CROSS_PERSON_PAGE.value not in flags:
                flags.append(Flag.CROSS_PERSON_PAGE.value)
                letter["flags"] = flags
                logger.info("C2: Cross-person page detected in %s: %s", letter["letter_id"], page_persons)

    # C1: Cross-Person Guard During DFS
    def dfs(letter_id: str, component: set[str], component_persons: set[str]) -> None:
        if letter_id in visited:
            return
        letter_person = letter_map[letter_id].get("person")
        # Check person consistency: if adding this letter introduces a person conflict, block
        if letter_person and component_persons and letter_person not in component_persons:
            cross_person_blocked.append(letter_id)
            logger.warning("C1: Cross-person blocked %s (person=%s, component persons=%s)",
                           letter_id, letter_person, component_persons)
            return
        visited.add(letter_id)
        component.add(letter_id)
        if letter_person:
            component_persons.add(letter_person)
        for neighbor in ref_groups.get(letter_id, set()):
            dfs(neighbor, component, component_persons)

    for letter in letters:
        lid = letter["letter_id"]
        if lid not in visited and lid in ref_groups:
            comp: set[str] = set()
            persons: set[str] = set()
            # B-03: Clear per-component tracking before each DFS
            cross_person_blocked.clear()
            dfs(lid, comp, persons)
            components.append(comp)
            component_cross_person_blocked.append(list(cross_person_blocked))

    enable_llm = matching_config.get("enable_llm_matching", False)

    for comp_idx, comp in enumerate(components):
        # C2: Over-Merge Prevention - check max component size
        comp_flags: list[str] = []
        if len(comp) > max_letters_per_vorgang:
            comp_flags.append(Flag.OVERSIZED_COMPONENT.value)
            logger.warning("C2: Oversized component (%d letters > %d max): %s",
                           len(comp), max_letters_per_vorgang, sorted(comp))

        # C2: Intra-component validation - check all letters share at least one common ref directly
        comp_letter_refs: dict[str, set[str]] = {}
        for lid in comp:
            letter_refs: set[str] = set()
            for ref in letter_map[lid].get("identifiers", []):
                norm = normalize_reference(ref)
                if norm and len(norm) >= ref_min_length:
                    letter_refs.add(norm)
            comp_letter_refs[lid] = letter_refs
        # Find if there is any single reference shared by ALL letters in the component
        if len(comp) > 1:
            all_ref_sets = list(comp_letter_refs.values())
            common_refs = all_ref_sets[0].intersection(*all_ref_sets[1:]) if all_ref_sets else set()
            if not common_refs:
                comp_flags.append(Flag.TRANSITIVE_MERGE_WARNING.value)
                logger.info("C2: Transitive merge warning for component %s (no single common ref)", sorted(comp))

        # LLM over-merge resolution for flagged components
        vorgang_letters = [letter_map[lid] for lid in sorted(comp)]
        if enable_llm and comp_flags and len(comp) >= 3:
            split_groups = llm_resolve_overmerge(vorgang_letters, comp_flags)
            if split_groups:
                logger.info("LLM split component %s into %d groups", sorted(comp), len(split_groups))
                for group in split_groups:
                    vorgang_counter += 1
                    vorgang = build_vorgang(
                        vorgang_counter, group,
                        MatchMethod.LLM.value, 0.75, config,
                    )
                    if Flag.LLM_SPLIT.value not in vorgang["flags"]:
                        vorgang["flags"].append(Flag.LLM_SPLIT.value)
                    if Flag.LLM_MATCHED.value not in vorgang["flags"]:
                        vorgang["flags"].append(Flag.LLM_MATCHED.value)
                    vorgaenge.append(vorgang)
                    for letter in group:
                        assigned_letter_ids.add(letter["letter_id"])
                continue  # Skip default single-Vorgang creation

        vorgang_counter += 1
        vorgang = build_vorgang(
            vorgang_counter, vorgang_letters,
            MatchMethod.AKTENZEICHEN_EXACT.value, 1.0, config,
        )
        for flag in comp_flags:
            if flag not in vorgang["flags"]:
                vorgang["flags"].append(flag)
        # B-03: Add cross-person blocked flags only for THIS component's blocked letters
        for blocked_lid in component_cross_person_blocked[comp_idx]:
            flag_val = f"{Flag.CROSS_PERSON_BLOCKED.value}:{blocked_lid}"
            if flag_val not in vorgang["flags"]:
                vorgang["flags"].append(flag_val)
        vorgaenge.append(vorgang)
        assigned_letter_ids.update(comp)

    # Tier 2 & 3: Creditor-based matching for unassigned letters
    unassigned = [letter for letter in letters if letter["letter_id"] not in assigned_letter_ids]

    creditor_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for letter in unassigned:
        creditor = letter.get("creditor", {})
        if isinstance(creditor, dict):
            cred_name = normalize_creditor(creditor.get("name", ""))
        else:
            cred_name = normalize_creditor(str(creditor))
        creditor_groups[cred_name].append(letter)

    for cred_name, cred_letters in creditor_groups.items():
        if not cred_name:
            # Unknown creditor: each letter is its own Vorgang
            for letter in cred_letters:
                vorgang_counter += 1
                vorgang = build_vorgang(
                    vorgang_counter, [letter],
                    MatchMethod.SINGLE_UNKNOWN.value, 0.5, config,
                )
                vorgang["flags"] = vorgang.get("flags", []) + [Flag.UNKNOWN_CREDITOR.value]
                vorgaenge.append(vorgang)
                assigned_letter_ids.add(letter["letter_id"])
            continue

        if len(cred_letters) == 1:
            vorgang_counter += 1
            vorgang = build_vorgang(
                vorgang_counter, cred_letters,
                MatchMethod.SINGLE_CREDITOR.value, 0.85, config,
            )
            vorgaenge.append(vorgang)
            assigned_letter_ids.add(cred_letters[0]["letter_id"])
            continue

        # Multiple letters from same creditor: try amount-based sub-grouping
        # Sort by amount descending, letter_id ascending for deterministic grouping
        cred_letters.sort(key=lambda l: (-get_letter_amount(l), l.get("letter_id", "")))
        amount_groups: list[list[dict[str, Any]]] = []
        remaining = list(cred_letters)

        while remaining:
            current = remaining.pop(0)
            current_amount = get_letter_amount(current)
            group = [current]

            still_remaining: list[dict[str, Any]] = []
            for other in remaining:
                other_amount = get_letter_amount(other)
                if current_amount and other_amount:
                    abs_diff = abs(current_amount - other_amount)
                    # C4: Absolute tolerance cap - use max(pct-based, absolute cap)
                    tolerance = max(current_amount * amount_tolerance_pct, amount_tolerance_abs)
                    if abs_diff <= tolerance:
                        group.append(other)
                        continue
                still_remaining.append(other)
            remaining = still_remaining
            amount_groups.append(group)

        # LLM Tier 3: If algorithmic grouping produced only singletons from
        # multiple letters, ask LLM whether they belong together
        singleton_count = sum(1 for g in amount_groups if len(g) == 1)
        if (
            enable_llm
            and len(cred_letters) >= 2
            and singleton_count == len(amount_groups)
        ):
            creditor_display = cred_letters[0].get("creditor", {})
            if isinstance(creditor_display, dict):
                display_name = creditor_display.get("name", cred_name)
            else:
                display_name = cred_name
            llm_groups = llm_group_same_creditor(display_name, cred_letters)
            if llm_groups:
                logger.info(
                    "LLM grouped %d letters from '%s' into %d groups",
                    len(cred_letters), cred_name, len(llm_groups),
                )
                for group in llm_groups:
                    vorgang_counter += 1
                    vorgang = build_vorgang(
                        vorgang_counter, group,
                        MatchMethod.LLM.value, 0.75, config,
                    )
                    if Flag.LLM_MATCHED.value not in vorgang["flags"]:
                        vorgang["flags"].append(Flag.LLM_MATCHED.value)
                    vorgaenge.append(vorgang)
                    for letter in group:
                        assigned_letter_ids.add(letter["letter_id"])
                continue  # Skip default amount_groups processing

        for group in amount_groups:
            vorgang_counter += 1
            if len(group) > 1:
                vorgang = build_vorgang(
                    vorgang_counter, group,
                    MatchMethod.CREDITOR_AMOUNT.value, 0.90, config,
                )
            else:
                vorgang = build_vorgang(
                    vorgang_counter, group,
                    MatchMethod.SINGLE_CREDITOR.value, 0.85, config,
                )
            vorgaenge.append(vorgang)
            for letter in group:
                assigned_letter_ids.add(letter["letter_id"])

    # Catch orphan letters
    all_letter_ids = {letter["letter_id"] for letter in letters}
    orphan_letters = all_letter_ids - assigned_letter_ids
    if orphan_letters:
        logger.warning("%d orphan letters not assigned to any Vorgang", len(orphan_letters))
        for lid in sorted(orphan_letters):
            vorgang_counter += 1
            vorgang = build_vorgang(
                vorgang_counter, [letter_map[lid]],
                MatchMethod.ORPHAN.value, 0.3, config,
            )
            vorgang["flags"] = vorgang.get("flags", []) + [Flag.ORPHAN_LETTER.value]
            vorgaenge.append(vorgang)

    return vorgaenge


def validate_vorgaenge(
    vorgaenge: list[dict[str, Any]],
    letters: list[dict[str, Any]],
    config: Optional[BriefpostConfig] = None,
) -> list[dict[str, Any]]:
    """Run validation checks on Vorgaenge.

    Args:
        vorgaenge: List of Vorgang dicts.
        letters: List of letter JSON data dicts.

    Returns:
        List of issue dicts with check, severity, and details.
    """
    issues: list[dict[str, Any]] = []

    # Check 1: Every letter in exactly 1 Vorgang
    letter_assignments: dict[str, list[str]] = defaultdict(list)
    for vorgang in vorgaenge:
        for lid in vorgang["briefe"]:
            letter_assignments[lid].append(vorgang["vorgang_id"])

    all_letter_ids = {letter["letter_id"] for letter in letters}
    assigned_ids = set(letter_assignments.keys())

    orphans = all_letter_ids - assigned_ids
    if orphans:
        issues.append({
            "check": "orphan_letters",
            "severity": "ERROR",
            "count": len(orphans),
            "details": list(orphans)[:10],
        })

    duplicates = {lid: vids for lid, vids in letter_assignments.items() if len(vids) > 1}
    if duplicates:
        issues.append({
            "check": "duplicate_assignments",
            "severity": "ERROR",
            "count": len(duplicates),
            "details": {k: v for k, v in list(duplicates.items())[:10]},
        })

    # Check 2: No cross-person mixes
    letter_map = {letter["letter_id"]: letter for letter in letters}
    for vorgang in vorgaenge:
        persons: set[str] = set()
        for lid in vorgang["briefe"]:
            letter = letter_map.get(lid, {})
            person = letter.get("person")
            if person:
                persons.add(person)
        if len(persons) > 1:
            issues.append({
                "check": "cross_person",
                "severity": "ERROR",
                "vorgang_id": vorgang["vorgang_id"],
                "persons": list(persons),
            })

    # Check 3: Every configured critical creditor shows up at least once
    critical_creditors = (config.priority.get("critical_creditors") or []) if config else []
    for critical in critical_creditors:
        needle = str(critical).lower().strip()
        if needle and not any(needle in (v.get("absender") or "").lower() for v in vorgaenge):
            issues.append({
                "check": "critical_creditor_missing",
                "severity": "WARNING",
                "details": f"Critical creditor '{critical}' not found in any case",
            })

    # Check 4: Vorgang IDs sequential
    expected_ids = {f"V-{i:03d}" for i in range(1, len(vorgaenge) + 1)}
    actual_ids = {v["vorgang_id"] for v in vorgaenge}
    if expected_ids != actual_ids:
        issues.append({
            "check": "id_gaps",
            "severity": "WARNING",
            "expected": len(expected_ids),
            "actual": len(actual_ids),
        })

    # Check 5: Dates plausible (2024-2026)
    bad_dates = 0
    for vorgang in vorgaenge:
        for date_field in ["datum", "faellig_am"]:
            date_val = vorgang.get(date_field)
            if date_val and isinstance(date_val, str):
                try:
                    if "." in date_val:
                        parts = date_val.split(".")
                        year = int(parts[2])
                    elif "-" in date_val:
                        year = int(date_val.split("-")[0])
                    else:
                        continue
                    current_year = datetime.now().year
                    if year < current_year - 2 or year > current_year + 1:
                        bad_dates += 1
                except (ValueError, IndexError):
                    pass
    if bad_dates:
        issues.append({"check": "date_plausibility", "severity": "WARNING", "count": bad_dates})

    return issues


def run_vorgang_matching(project_dir: Path, config: BriefpostConfig) -> BatchResult:
    """Entry point called by pipeline.py.

    Args:
        project_dir: Project data directory.
        config: Pipeline configuration.

    Returns:
        BatchResult describing the outcome.
    """
    result = BatchResult(phase="match_vorgaenge")
    letters_dir = project_dir / "letters"
    reconciliation_dir = project_dir / "reconciliation"

    # Load letters
    letters: list[dict[str, Any]] = []
    for json_path in sorted(letters_dir.glob("LTR-*.json")):
        with open(json_path) as f:
            letters.append(json.load(f))

    logger.info("Loaded %d letters", len(letters))

    if not letters:
        logger.error("No letters found in %s", letters_dir)
        result.total = 0
        result.complete()
        return result

    # RS-010/RS-034: Checkpoint support for LLM mode crash recovery
    checkpoint_path = reconciliation_dir / ".vorgang-matching-checkpoint.json"

    try:
        # Match
        vorgaenge = match_letters_to_vorgaenge(letters, config)
        logger.info("Created %d Vorgaenge", len(vorgaenge))

        # RS-010: Save checkpoint after matching completes (before validation)
        try:
            atomic_write_json(checkpoint_path, {
                "phase": "match_complete",
                "vorgaenge_count": len(vorgaenge),
                "timestamp": datetime.now().isoformat(),
            })
        except OSError:
            logger.warning("RS-010: Could not write matching checkpoint")

        # Validate
        issues = validate_vorgaenge(vorgaenge, letters, config)
        errors = [i for i in issues if i["severity"] == "ERROR"]
        warnings = [i for i in issues if i["severity"] == "WARNING"]
        logger.info("Validation: %d errors, %d warnings", len(errors), len(warnings))

        for issue in issues:
            if issue["severity"] == "ERROR":
                logger.error("[%s] %s: %s", issue["severity"], issue["check"], issue.get("details", ""))
            else:
                logger.warning("[%s] %s: %s", issue["severity"], issue["check"], issue.get("details", issue.get("count", "")))

        # Write output
        reconciliation_dir.mkdir(exist_ok=True)

        vorgaenge_path = reconciliation_dir / "vorgaenge.json"
        atomic_write_json(vorgaenge_path, {"vorgaenge": vorgaenge, "created": datetime.now().isoformat()})
        logger.info("Written: %s", vorgaenge_path)

        # Priority distribution for logging
        priority_counts: dict[str, int] = defaultdict(int)
        for vorgang in vorgaenge:
            priority_counts[vorgang["prioritaet"]] += 1
        for prio, count in sorted(priority_counts.items()):
            logger.info("  %s: %d", prio, count)

        report_path = reconciliation_dir / "validation-report.json"
        atomic_write_json(report_path, {
            "created": datetime.now().isoformat(),
            "total_letters": len(letters),
            "total_vorgaenge": len(vorgaenge),
            "priority_distribution": dict(priority_counts),
            "issues": issues,
            "errors": len(errors),
            "warnings": len(warnings),
        })
        logger.info("Written: %s", report_path)

        result.total = len(vorgaenge)
        result.succeeded = len(vorgaenge)

        # RS-010: Clean up checkpoint on success
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    except Exception as exc:
        logger.error("Vorgang matching failed: %s", exc, exc_info=True)
        result.total = 1
        result.failed = 1
        result.errors.append({"phase": "match_vorgaenge", "error": str(exc)})

    result.complete()
    return result
