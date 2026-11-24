"""Abstract base classes for the briefpost-management pipeline.

Each interface defines the contract for one pipeline phase.
Concrete implementations live in the corresponding phase modules
(group/, ocr/, match/, reconcile/, output/, qc/).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .models import (
    BankTransaction,
    BatchResult,
    Brief,
    DocumentGroup,
    Page,
    QCReport,
    Vorgang,
)


class DocumentGrouper(ABC):
    """Phase A: Groups photos into document clusters using timestamp analysis."""

    @abstractmethod
    def group_photos(
        self,
        image_dir: Path,
        output_dir: Path,
        config: dict[str, Any],
    ) -> list[DocumentGroup]:
        """Group photos from image_dir into document clusters.

        Args:
            image_dir: Directory containing unsorted JPG photos.
            output_dir: Directory to write grouped/doc-XXX/ subdirectories.
            config: Grouping configuration (burst_threshold_seconds, doc_threshold_seconds, etc.).

        Returns:
            List of DocumentGroup records describing each group.
        """
        ...


class OCRExtractor(ABC):
    """Phase B1: Extracts text and structured data from a single page image."""

    @abstractmethod
    def extract_page(
        self,
        image_path: Path,
        config: dict[str, Any],
    ) -> Page:
        """Run OCR on a single page image.

        Args:
            image_path: Path to the page image (JPG).
            config: OCR configuration (provider, model, rate limits, etc.).

        Returns:
            Page with OCR text, confidence, signals, and extraction data.
        """
        ...


class OCRPostprocessor(ABC):
    """Phase B2: Normalizes and fixes OCR outputs in-place."""

    @abstractmethod
    def postprocess(
        self,
        ocr_pages_dir: Path,
        config: dict[str, Any],
    ) -> BatchResult:
        """Apply post-processing fixes to all OCR JSON files.

        Fixes: confidence normalization, signal regeneration, IBAN validation,
        missing field defaults, timestamps.

        Args:
            ocr_pages_dir: Directory containing per-page .json files.
            config: OCR configuration.

        Returns:
            BatchResult with counts of processed/fixed files.
        """
        ...


class PageMatcher(ABC):
    """Phase B3: Matches OCR pages into logical letters using boundary detection."""

    @abstractmethod
    def match_pages_to_letters(
        self,
        pages: list[Page],
        groups: list[DocumentGroup],
        config: dict[str, Any],
    ) -> list[Brief]:
        """Group pages into letters using signal-weighted boundary scoring.

        Args:
            pages: All OCR-processed pages.
            groups: Document groups from Phase A.
            config: Matching configuration (split_threshold, company_suffixes, etc.).

        Returns:
            List of Brief (letter) records, each containing one or more pages.
        """
        ...


class VorgangMatcher(ABC):
    """Phase C: Groups letters into Vorgaenge (cases) using 3-tier matching."""

    @abstractmethod
    def match_letters_to_vorgaenge(
        self,
        letters: list[Brief],
        config: dict[str, Any],
    ) -> list[Vorgang]:
        """Group letters into Vorgaenge using reference, creditor, and amount matching.

        Matching tiers:
          1. Exact Aktenzeichen -> 100% confidence
          2. Same creditor + similar amount (+-5%) -> 90% confidence
          3. Same creditor + similar subject -> 70% confidence (REVIEW flag)

        Args:
            letters: All letters from Phase B3.
            config: Matching configuration.

        Returns:
            List of Vorgang records, each containing one or more letters.
        """
        ...


class BankReconciler(ABC):
    """Phase D: Reconciles Vorgaenge against bank transactions."""

    @abstractmethod
    def reconcile(
        self,
        vorgaenge: list[Vorgang],
        transactions: list[BankTransaction],
        config: dict[str, Any],
    ) -> tuple[list[Vorgang], set[int]]:
        """Match Vorgaenge against bank transactions to determine payment status.

        Two-phase matching:
          Phase 1: Reference match (confidence 0.95)
          Phase 2: Amount+date fallback (confidence 0.85)

        Args:
            vorgaenge: All Vorgaenge from Phase C.
            transactions: All parsed bank transactions.
            config: Reconciliation configuration (tolerances, thresholds).

        Returns:
            Tuple of (reconciled Vorgaenge with status/priority updated,
            set of matched transaction indices).
        """
        ...


class OutputBuilder(ABC):
    """Phase E: Generates Google Sheet and QR codes."""

    @abstractmethod
    def build_sheet(
        self,
        vorgaenge: list[Vorgang],
        config: dict[str, Any],
    ) -> str | None:
        """Create a Google Sheet with 4 tabs from reconciled Vorgaenge.

        Tabs: Alle Dokumente, Offene Posten, Zusammenfassung, Reconciliation.

        Args:
            vorgaenge: Reconciled Vorgaenge from Phase D.
            config: Output configuration (sheet_prefix, conditional_format_colors).

        Returns:
            Spreadsheet ID string, or None if creation failed.
        """
        ...

    @abstractmethod
    def generate_qr_codes(
        self,
        vorgaenge: list[Vorgang],
        output_dir: Path,
        config: dict[str, Any],
    ) -> list[Path]:
        """Generate EPC QR codes for open Vorgaenge with payment info.

        Args:
            vorgaenge: Reconciled Vorgaenge from Phase D.
            output_dir: Directory to write QR PNGs.
            config: Output configuration (qr_scale, qr_border).

        Returns:
            List of Paths to generated QR code PNG files.
        """
        ...


class QualityVerifier(ABC):
    """Phase F: Runs end-to-end quality control checks."""

    @abstractmethod
    def verify(
        self,
        project_dir: Path,
        config: dict[str, Any],
    ) -> QCReport:
        """Run all QC checks against pipeline outputs.

        Checks include: foto coverage, OCR coverage, brief coverage,
        orphan pages, duplicate pages, person consistency, Vorgang integrity,
        amount precision, IBAN validity, critical creditor identification,
        sheet existence, QR code count, status/priority coverage,
        E2E traces, false-positive reconciliation detection.

        Args:
            project_dir: Root directory of the project data.
            config: QC configuration (e2e_sample_size, date_range, etc.).

        Returns:
            QCReport with all check results.
        """
        ...
