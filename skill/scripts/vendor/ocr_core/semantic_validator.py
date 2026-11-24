#!/usr/bin/env python3
"""Semantic validation layer for extraction results.

Performs post-extraction plausibility checks to flag impossible states
or suspicious values. This is FLAG-ONLY mode - no auto-corrections.

Usage:
    from core.semantic_validator import SemanticValidator

    validator = SemanticValidator()
    issues = validator.validate(extraction_result)

    for issue in issues:
        print(f"{issue.severity}: {issue.message}")
"""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .extraction_types import GermanExtractionResult


@dataclass
class SemanticIssue:
    """A semantic validation issue.

    Attributes:
        field: Field name where issue was detected
        rule: Rule identifier that triggered this issue
        message: Human-readable description of the issue
        severity: Issue severity (high, medium, low)
    """
    field: str
    rule: str
    message: str
    severity: str  # "high", "medium", "low"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "rule": self.rule,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class SemanticValidationResult:
    """Result of semantic validation.

    Attributes:
        issues: List of detected semantic issues
        high_severity_count: Count of high-severity issues
        passed: True if no high-severity issues found
    """
    issues: List[SemanticIssue] = field(default_factory=list)

    @property
    def high_severity_count(self) -> int:
        """Count of high-severity issues."""
        return sum(1 for i in self.issues if i.severity == "high")

    @property
    def medium_severity_count(self) -> int:
        """Count of medium-severity issues."""
        return sum(1 for i in self.issues if i.severity == "medium")

    @property
    def passed(self) -> bool:
        """True if no high-severity issues."""
        return self.high_severity_count == 0

    @property
    def needs_review(self) -> bool:
        """True if any issues require manual review."""
        return len(self.issues) > 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "issues": [i.to_dict() for i in self.issues],
            "high_severity_count": self.high_severity_count,
            "medium_severity_count": self.medium_severity_count,
            "passed": self.passed,
            "needs_review": self.needs_review,
        }


class SemanticValidator:
    """Post-extraction semantic plausibility checks.

    This validator performs FLAG-ONLY mode validation - it identifies
    potential issues but does not auto-correct them. All suspicious
    values are flagged for manual review.

    Rules implemented:
    1. current_gte_original: Current amount >= original (fees add, not subtract)
    2. due_after_document: Due date >= document date
    3. court_needs_aktenzeichen: Court cases must have identifiers
    4. iban_country_code: Flag non-German IBANs
    5. amount_plausibility: Flag unusually large or small amounts
    6. date_in_future: Flag document dates in the future
    7. date_too_old: Flag document dates older than 10 years
    8. iban_format: Validate IBAN format (DE + 20 chars)
    """

    # Reasonable amount bounds for German debt documents
    MIN_REASONABLE_AMOUNT = 1.0  # Less than 1 EUR is suspicious
    MAX_REASONABLE_AMOUNT = 1_000_000.0  # More than 1M EUR is suspicious

    def validate(self, result: "GermanExtractionResult") -> SemanticValidationResult:
        """Validate extraction result for semantic plausibility.

        Args:
            result: Extraction result to validate

        Returns:
            SemanticValidationResult with all detected issues
        """
        validation_result = SemanticValidationResult()

        # Skip validation for failed extractions
        if not result.extraction_successful:
            return validation_result

        # Run all validation rules
        self._check_current_gte_original(result, validation_result)
        self._check_due_after_document(result, validation_result)
        self._check_court_needs_aktenzeichen(result, validation_result)
        self._check_iban_country_code(result, validation_result)
        self._check_iban_format(result, validation_result)
        self._check_amount_plausibility(result, validation_result)
        self._check_date_in_future(result, validation_result)
        self._check_date_too_old(result, validation_result)

        return validation_result

    def _check_current_gte_original(
        self,
        result: "GermanExtractionResult",
        validation_result: SemanticValidationResult,
    ) -> None:
        """Rule 1: Current amount should be >= original (fees add).

        In debt collection, the current amount is the original plus fees.
        If current < original, something is wrong.
        """
        if not result.amounts:
            return

        if result.amounts.current is not None and result.amounts.original is not None:
            if result.amounts.current < result.amounts.original:
                validation_result.issues.append(SemanticIssue(
                    field="amounts",
                    rule="current_gte_original",
                    message=(
                        f"Current amount ({result.amounts.current:.2f}) is less than "
                        f"original ({result.amounts.original:.2f}). "
                        "Fees should increase the total, not decrease it."
                    ),
                    severity="high",
                ))

    def _check_due_after_document(
        self,
        result: "GermanExtractionResult",
        validation_result: SemanticValidationResult,
    ) -> None:
        """Rule 2: Due date should be >= document date.

        A payment cannot be due before the document was created.
        """
        if not result.dates:
            return

        due_date = result.dates.get_due_date_parsed()
        doc_date = result.dates.get_document_date_parsed()

        if due_date and doc_date:
            if due_date < doc_date:
                validation_result.issues.append(SemanticIssue(
                    field="dates",
                    rule="due_after_document",
                    message=(
                        f"Due date ({due_date}) is before document date ({doc_date}). "
                        "Payment cannot be due before the document was created."
                    ),
                    severity="medium",
                ))

    def _check_court_needs_aktenzeichen(
        self,
        result: "GermanExtractionResult",
        validation_result: SemanticValidationResult,
    ) -> None:
        """Rule 3: Court cases must have identifiers (Aktenzeichen).

        All court documents (Mahnbescheid, Vollstreckung) should have
        a case reference number.
        """
        if result.is_court_case and not result.identifiers:
            validation_result.issues.append(SemanticIssue(
                field="identifiers",
                rule="court_needs_aktenzeichen",
                message=(
                    "Court case document is missing Aktenzeichen. "
                    "All court documents should have a case reference number."
                ),
                severity="high",
            ))

    def _check_iban_country_code(
        self,
        result: "GermanExtractionResult",
        validation_result: SemanticValidationResult,
    ) -> None:
        """Rule 4: Flag non-German IBANs.

        German debt documents typically have German IBANs (DE prefix).
        Non-German IBANs are unusual and should be reviewed.
        """
        if not result.payment_info or not result.payment_info.iban:
            return

        iban = result.payment_info.iban.upper().replace(" ", "")

        if not iban.startswith("DE"):
            country_code = iban[:2] if len(iban) >= 2 else "??"
            validation_result.issues.append(SemanticIssue(
                field="payment_info",
                rule="iban_country_code",
                message=(
                    f"Non-German IBAN detected (country: {country_code}). "
                    "Verify this is correct for a German debt document."
                ),
                severity="low",
            ))

    def _check_iban_format(
        self,
        result: "GermanExtractionResult",
        validation_result: SemanticValidationResult,
    ) -> None:
        """Rule 5: Validate German IBAN format.

        German IBANs should be exactly 22 characters: DE + 2 check digits + 18 chars.
        """
        if not result.payment_info or not result.payment_info.iban:
            return

        iban = result.payment_info.iban.upper().replace(" ", "")

        # Check if it looks like a German IBAN
        if iban.startswith("DE"):
            if len(iban) != 22:
                validation_result.issues.append(SemanticIssue(
                    field="payment_info",
                    rule="iban_format",
                    message=(
                        f"German IBAN has invalid length ({len(iban)} chars, expected 22). "
                        f"IBAN: {iban}"
                    ),
                    severity="high",
                ))
            elif not re.match(r"^DE\d{20}$", iban):
                validation_result.issues.append(SemanticIssue(
                    field="payment_info",
                    rule="iban_format",
                    message=(
                        f"German IBAN has invalid format. Expected DE + 20 digits. "
                        f"IBAN: {iban}"
                    ),
                    severity="high",
                ))

    def _check_amount_plausibility(
        self,
        result: "GermanExtractionResult",
        validation_result: SemanticValidationResult,
    ) -> None:
        """Rule 6: Flag unusually large or small amounts.

        Very small amounts (<1 EUR) or very large amounts (>1M EUR)
        in debt documents are unusual and should be verified.
        """
        if not result.amounts:
            return

        amounts_to_check = [
            ("original", result.amounts.original),
            ("current", result.amounts.current),
        ]

        for name, amount in amounts_to_check:
            if amount is None:
                continue

            if amount < self.MIN_REASONABLE_AMOUNT:
                validation_result.issues.append(SemanticIssue(
                    field="amounts",
                    rule="amount_plausibility",
                    message=(
                        f"Amount {name} ({amount:.2f} EUR) is unusually small. "
                        "Verify this is correct."
                    ),
                    severity="medium",
                ))
            elif amount > self.MAX_REASONABLE_AMOUNT:
                validation_result.issues.append(SemanticIssue(
                    field="amounts",
                    rule="amount_plausibility",
                    message=(
                        f"Amount {name} ({amount:,.2f} EUR) is unusually large. "
                        "Verify this is correct."
                    ),
                    severity="medium",
                ))

    def _check_date_in_future(
        self,
        result: "GermanExtractionResult",
        validation_result: SemanticValidationResult,
    ) -> None:
        """Rule 7: Flag document dates in the future.

        Document date should not be in the future.
        """
        if not result.dates:
            return

        doc_date = result.dates.get_document_date_parsed()
        if doc_date and doc_date > date.today():
            validation_result.issues.append(SemanticIssue(
                field="dates",
                rule="date_in_future",
                message=(
                    f"Document date ({doc_date}) is in the future. "
                    "This may indicate a parsing error."
                ),
                severity="high",
            ))

    def _check_date_too_old(
        self,
        result: "GermanExtractionResult",
        validation_result: SemanticValidationResult,
    ) -> None:
        """Rule 8: Flag document dates older than 10 years.

        Very old documents are unusual for active debt collection.
        """
        if not result.dates:
            return

        doc_date = result.dates.get_document_date_parsed()
        if doc_date:
            years_old = (date.today() - doc_date).days / 365

            if years_old > 10:
                validation_result.issues.append(SemanticIssue(
                    field="dates",
                    rule="date_too_old",
                    message=(
                        f"Document date ({doc_date}) is over 10 years old. "
                        "Verify this is the correct document."
                    ),
                    severity="low",
                ))


def validate_extraction(result: "GermanExtractionResult") -> SemanticValidationResult:
    """Convenience function to validate an extraction result.

    Args:
        result: Extraction result to validate

    Returns:
        SemanticValidationResult with all detected issues
    """
    validator = SemanticValidator()
    return validator.validate(result)
