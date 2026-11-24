#!/usr/bin/env python3
"""IBAN and BIC validation and extraction.

Validates IBANs according to ISO 13616 standard.
Extracts IBANs and BICs from text content.
"""

import re
from dataclasses import dataclass
from typing import Optional, List


# ============================================================================
# IBAN Country Specifications
# ============================================================================

# IBAN lengths by country (ISO 3166-1 alpha-2)
IBAN_LENGTHS = {
    "AL": 28,  # Albania
    "AD": 24,  # Andorra
    "AT": 20,  # Austria
    "AZ": 28,  # Azerbaijan
    "BH": 22,  # Bahrain
    "BY": 28,  # Belarus
    "BE": 16,  # Belgium
    "BA": 20,  # Bosnia and Herzegovina
    "BR": 29,  # Brazil
    "BG": 22,  # Bulgaria
    "CR": 22,  # Costa Rica
    "HR": 21,  # Croatia
    "CY": 28,  # Cyprus
    "CZ": 24,  # Czech Republic
    "DK": 18,  # Denmark
    "DO": 28,  # Dominican Republic
    "TL": 23,  # Timor-Leste
    "EE": 20,  # Estonia
    "FO": 18,  # Faroe Islands
    "FI": 18,  # Finland
    "FR": 27,  # France
    "GE": 22,  # Georgia
    "DE": 22,  # Germany
    "GI": 23,  # Gibraltar
    "GR": 27,  # Greece
    "GL": 18,  # Greenland
    "GT": 28,  # Guatemala
    "HU": 28,  # Hungary
    "IS": 26,  # Iceland
    "IQ": 23,  # Iraq
    "IE": 22,  # Ireland
    "IL": 23,  # Israel
    "IT": 27,  # Italy
    "JO": 30,  # Jordan
    "KZ": 20,  # Kazakhstan
    "XK": 20,  # Kosovo
    "KW": 30,  # Kuwait
    "LV": 21,  # Latvia
    "LB": 28,  # Lebanon
    "LI": 21,  # Liechtenstein
    "LT": 20,  # Lithuania
    "LU": 20,  # Luxembourg
    "MK": 19,  # North Macedonia
    "MT": 31,  # Malta
    "MR": 27,  # Mauritania
    "MU": 30,  # Mauritius
    "MC": 27,  # Monaco
    "MD": 24,  # Moldova
    "ME": 22,  # Montenegro
    "NL": 18,  # Netherlands
    "NO": 15,  # Norway
    "PK": 24,  # Pakistan
    "PS": 29,  # Palestine
    "PL": 28,  # Poland
    "PT": 25,  # Portugal
    "QA": 29,  # Qatar
    "RO": 24,  # Romania
    "LC": 32,  # Saint Lucia
    "SM": 27,  # San Marino
    "ST": 25,  # Sao Tome and Principe
    "SA": 24,  # Saudi Arabia
    "RS": 22,  # Serbia
    "SC": 31,  # Seychelles
    "SK": 24,  # Slovakia
    "SI": 19,  # Slovenia
    "ES": 24,  # Spain
    "SE": 24,  # Sweden
    "CH": 21,  # Switzerland
    "TN": 24,  # Tunisia
    "TR": 26,  # Turkey
    "UA": 29,  # Ukraine
    "AE": 23,  # United Arab Emirates
    "GB": 22,  # United Kingdom
    "VA": 22,  # Vatican City
    "VG": 24,  # British Virgin Islands
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class IBANValidationResult:
    """Result of IBAN validation."""
    original: str              # Original input
    iban: str                  # Normalized IBAN (no spaces)
    valid: bool                # Validation result
    country: str               # Country code (e.g., "DE")
    country_name: str          # Country name (e.g., "Germany")
    formatted: str             # Formatted IBAN (with spaces)
    error: Optional[str] = None

    @property
    def bank_code(self) -> Optional[str]:
        """Extract bank code (country-specific)."""
        if not self.valid:
            return None
        # German IBAN: DExx BBBB BBBB AAAA AAAA AA (B=bank, A=account)
        if self.country == "DE" and len(self.iban) >= 12:
            return self.iban[4:12]  # 8-digit Bankleitzahl
        return None


@dataclass
class BICValidationResult:
    """Result of BIC/SWIFT code validation."""
    original: str
    bic: str                   # Normalized BIC
    valid: bool
    bank_code: str             # 4-char bank identifier
    country_code: str          # 2-char country
    location_code: str         # 2-char location
    branch_code: Optional[str] # 3-char branch (optional)
    error: Optional[str] = None


# Country names mapping
COUNTRY_NAMES = {
    "DE": "Germany",
    "AT": "Austria",
    "CH": "Switzerland",
    "FR": "France",
    "GB": "United Kingdom",
    "NL": "Netherlands",
    "BE": "Belgium",
    "IT": "Italy",
    "ES": "Spain",
    "PT": "Portugal",
    "PL": "Poland",
    "CZ": "Czech Republic",
    "HU": "Hungary",
    "SE": "Sweden",
    "DK": "Denmark",
    "NO": "Norway",
    "FI": "Finland",
    "IE": "Ireland",
    "LU": "Luxembourg",
    "LI": "Liechtenstein",
}


# ============================================================================
# IBAN Validation
# ============================================================================

def validate_iban(iban: str) -> IBANValidationResult:
    """Validate an IBAN according to ISO 13616.

    Checks:
    1. Format (alphanumeric only)
    2. Country code validity
    3. Length for country
    4. Check digits (mod 97)

    Args:
        iban: IBAN string (with or without spaces)

    Returns:
        IBANValidationResult with validation details
    """
    original = iban

    # Normalize: remove spaces, uppercase
    iban = iban.upper().replace(" ", "").replace("-", "")

    # Check minimum length
    if len(iban) < 5:
        return IBANValidationResult(
            original=original,
            iban=iban,
            valid=False,
            country="",
            country_name="",
            formatted="",
            error="IBAN too short",
        )

    # Extract country code
    country = iban[:2]

    # Check country code
    if not country.isalpha():
        return IBANValidationResult(
            original=original,
            iban=iban,
            valid=False,
            country=country,
            country_name="",
            formatted="",
            error=f"Invalid country code: {country}",
        )

    if country not in IBAN_LENGTHS:
        return IBANValidationResult(
            original=original,
            iban=iban,
            valid=False,
            country=country,
            country_name="Unknown",
            formatted="",
            error=f"Unknown country code: {country}",
        )

    # Check length
    expected_length = IBAN_LENGTHS[country]
    if len(iban) != expected_length:
        return IBANValidationResult(
            original=original,
            iban=iban,
            valid=False,
            country=country,
            country_name=COUNTRY_NAMES.get(country, "Unknown"),
            formatted="",
            error=f"Invalid length for {country}: expected {expected_length}, got {len(iban)}",
        )

    # Check alphanumeric
    if not iban.isalnum():
        return IBANValidationResult(
            original=original,
            iban=iban,
            valid=False,
            country=country,
            country_name=COUNTRY_NAMES.get(country, "Unknown"),
            formatted="",
            error="IBAN contains invalid characters",
        )

    # Check digits validation (mod 97)
    if not _validate_iban_checksum(iban):
        return IBANValidationResult(
            original=original,
            iban=iban,
            valid=False,
            country=country,
            country_name=COUNTRY_NAMES.get(country, "Unknown"),
            formatted="",
            error="Invalid check digits",
        )

    # Valid!
    formatted = format_iban(iban)

    return IBANValidationResult(
        original=original,
        iban=iban,
        valid=True,
        country=country,
        country_name=COUNTRY_NAMES.get(country, "Unknown"),
        formatted=formatted,
    )


def _validate_iban_checksum(iban: str) -> bool:
    """Validate IBAN checksum using mod 97.

    Move first 4 chars to end, convert letters to numbers,
    and check if result mod 97 == 1.
    """
    # Move first 4 chars to end
    rearranged = iban[4:] + iban[:4]

    # Convert letters to numbers (A=10, B=11, ..., Z=35)
    numeric = ""
    for char in rearranged:
        if char.isdigit():
            numeric += char
        else:
            numeric += str(ord(char) - ord("A") + 10)

    # Check mod 97
    return int(numeric) % 97 == 1


def format_iban(iban: str) -> str:
    """Format IBAN with standard spacing.

    Groups of 4 characters separated by spaces.

    Args:
        iban: Normalized IBAN (no spaces)

    Returns:
        Formatted IBAN (e.g., "DE89 3704 0044 0532 0130 00")
    """
    iban = iban.upper().replace(" ", "")
    return " ".join(iban[i:i+4] for i in range(0, len(iban), 4))


# ============================================================================
# BIC Validation
# ============================================================================

def validate_bic(bic: str) -> BICValidationResult:
    """Validate a BIC/SWIFT code.

    Format: BBBB CC LL [XXX]
    - BBBB: Bank code (4 letters)
    - CC: Country code (2 letters)
    - LL: Location code (2 alphanumeric)
    - XXX: Branch code (3 alphanumeric, optional)

    Args:
        bic: BIC/SWIFT code

    Returns:
        BICValidationResult with validation details
    """
    original = bic
    bic = bic.upper().replace(" ", "")

    # Check length (8 or 11)
    if len(bic) not in (8, 11):
        return BICValidationResult(
            original=original,
            bic=bic,
            valid=False,
            bank_code="",
            country_code="",
            location_code="",
            branch_code=None,
            error=f"Invalid BIC length: {len(bic)} (must be 8 or 11)",
        )

    # Extract components
    bank_code = bic[:4]
    country_code = bic[4:6]
    location_code = bic[6:8]
    branch_code = bic[8:] if len(bic) == 11 else None

    # Validate bank code (4 letters)
    if not bank_code.isalpha():
        return BICValidationResult(
            original=original,
            bic=bic,
            valid=False,
            bank_code=bank_code,
            country_code=country_code,
            location_code=location_code,
            branch_code=branch_code,
            error=f"Invalid bank code: {bank_code} (must be 4 letters)",
        )

    # Validate country code (must be valid ISO country code)
    if not country_code.isalpha():
        return BICValidationResult(
            original=original,
            bic=bic,
            valid=False,
            bank_code=bank_code,
            country_code=country_code,
            location_code=location_code,
            branch_code=branch_code,
            error=f"Invalid country code: {country_code}",
        )

    # Check against known country codes (use IBAN_LENGTHS as reference)
    if country_code not in IBAN_LENGTHS:
        return BICValidationResult(
            original=original,
            bic=bic,
            valid=False,
            bank_code=bank_code,
            country_code=country_code,
            location_code=location_code,
            branch_code=branch_code,
            error=f"Unknown country code: {country_code}",
        )

    # Validate location code (alphanumeric)
    if not location_code.isalnum():
        return BICValidationResult(
            original=original,
            bic=bic,
            valid=False,
            bank_code=bank_code,
            country_code=country_code,
            location_code=location_code,
            branch_code=branch_code,
            error=f"Invalid location code: {location_code}",
        )

    # Validate branch code if present
    if branch_code and not branch_code.isalnum():
        return BICValidationResult(
            original=original,
            bic=bic,
            valid=False,
            bank_code=bank_code,
            country_code=country_code,
            location_code=location_code,
            branch_code=branch_code,
            error=f"Invalid branch code: {branch_code}",
        )

    return BICValidationResult(
        original=original,
        bic=bic,
        valid=True,
        bank_code=bank_code,
        country_code=country_code,
        location_code=location_code,
        branch_code=branch_code,
    )


# ============================================================================
# Text Extraction
# ============================================================================

def extract_iban_from_text(text: str) -> Optional[str]:
    """Extract a single IBAN from text.

    Args:
        text: Text to search

    Returns:
        First valid IBAN found, or None
    """
    ibans = extract_all_ibans_from_text(text)
    return ibans[0] if ibans else None


def extract_all_ibans_from_text(text: str) -> List[str]:
    """Extract all valid IBANs from text.

    Args:
        text: Text to search

    Returns:
        List of valid IBAN strings (normalized, no spaces)
    """
    ibans = []
    seen = set()

    # Pattern: 2 letters + 2 digits + alphanumeric chars (with optional spaces)
    # Uses word boundary to avoid matching into adjacent text
    # Matches formats like: DE89370400440532013000 or DE89 3704 0044 0532 0130 00
    # NOTE: Use [ ] instead of [\s] to avoid matching newlines across lines
    pattern = r"\b([A-Z]{2})[ ]?(\d{2})[ ]?((?:[A-Z0-9][ ]?){10,30})\b"

    for match in re.finditer(pattern, text.upper()):
        candidate = (match.group(1) + match.group(2) + match.group(3)).replace(" ", "")
        # Skip if we've already seen this IBAN
        if candidate in seen:
            continue
        result = validate_iban(candidate)
        if result.valid:
            ibans.append(result.iban)
            seen.add(result.iban)

    return ibans


def extract_bic_from_text(text: str) -> Optional[str]:
    """Extract a single BIC from text.

    Args:
        text: Text to search

    Returns:
        First valid BIC found, or None
    """
    bics = extract_all_bics_from_text(text)
    return bics[0] if bics else None


def extract_all_bics_from_text(text: str) -> List[str]:
    """Extract all valid BICs from text.

    Args:
        text: Text to search

    Returns:
        List of valid BIC strings
    """
    bics = []

    # Pattern: 8 or 11 alphanumeric characters that look like a BIC
    # 4 letters + 2 letters + 2 alphanumeric + optional 3 alphanumeric
    pattern = r"\b([A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)\b"

    for match in re.finditer(pattern, text.upper()):
        candidate = match.group(1)
        result = validate_bic(candidate)
        if result.valid:
            bics.append(result.bic)

    return bics


def extract_bank_details_from_text(text: str) -> dict:
    """Extract all bank-related details from text.

    Args:
        text: Text to search

    Returns:
        Dictionary with ibans, bics, and validation results
    """
    ibans = []
    bics = []

    for iban in extract_all_ibans_from_text(text):
        result = validate_iban(iban)
        if result.valid:
            ibans.append({
                "iban": result.iban,
                "formatted": result.formatted,
                "country": result.country,
                "country_name": result.country_name,
                "bank_code": result.bank_code,
            })

    for bic in extract_all_bics_from_text(text):
        result = validate_bic(bic)
        if result.valid:
            bics.append({
                "bic": result.bic,
                "bank_code": result.bank_code,
                "country": result.country_code,
            })

    return {
        "ibans": ibans,
        "bics": bics,
        "has_bank_details": len(ibans) > 0 or len(bics) > 0,
    }
