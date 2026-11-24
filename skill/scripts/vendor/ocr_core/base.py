#!/usr/bin/env python3
"""Base classes for OCR providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import os
import re


# ============================================================================
# Security Exceptions
# ============================================================================

class PathValidationError(Exception):
    """Raised when file path fails security validation.

    This exception is raised for:
    - Path traversal attempts (e.g., ../../etc/passwd)
    - Symlink attacks
    - Files exceeding size limits
    - Non-file paths (directories, devices)
    """
    pass


# ============================================================================
# Path Validation
# ============================================================================

# Maximum safe file size for memory (20MB base64 encoded = ~53MB peak RAM)
MAX_SAFE_MEMORY_MB = 20


def validate_file_path(
    file_path: str,
    max_size_bytes: Optional[int] = None,
    allowed_extensions: Optional[List[str]] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    """Validate and resolve file path securely.

    Security checks performed:
    1. Resolve to absolute path (handles symlinks)
    2. Verify file exists and is a regular file
    3. Check for path traversal attempts
    4. Validate file extension if specified
    5. Check file size if limit specified
    6. Warn about large files that may cause memory pressure

    Args:
        file_path: Path to validate (can be relative or absolute)
        max_size_bytes: Optional maximum file size in bytes
        allowed_extensions: Optional list of allowed extensions (e.g., [".pdf", ".png"])
        base_dir: Optional base directory to restrict paths to

    Returns:
        Resolved absolute Path object

    Raises:
        FileNotFoundError: If file does not exist
        PathValidationError: If path fails security validation
    """
    # Detect suspicious path patterns FIRST (before file existence check)
    # These patterns are always suspicious regardless of resolution
    original_str = str(file_path)
    if ".." in original_str:
        # Check if the original path tried to traverse
        original_normalized = os.path.normpath(original_str)
        if original_normalized.startswith(".."):
            raise PathValidationError(
                f"Path traversal attempt detected: {file_path}"
            )

    # Convert to Path and resolve (follows symlinks, makes absolute)
    try:
        resolved = Path(file_path).resolve()
    except (OSError, ValueError) as e:
        raise PathValidationError(f"Cannot resolve path: {file_path} - {e}")

    # Check file exists
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Must be a regular file (not directory, device, etc.)
    if not resolved.is_file():
        raise PathValidationError(f"Not a regular file: {file_path}")

    # Path traversal check: resolved path should not escape base_dir
    if base_dir is not None:
        base_resolved = Path(base_dir).resolve()
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            raise PathValidationError(
                f"Path traversal detected: {file_path} escapes {base_dir}"
            )

    # Extension validation
    if allowed_extensions is not None:
        ext = resolved.suffix.lower()
        allowed_lower = [e.lower() for e in allowed_extensions]
        if ext not in allowed_lower:
            raise PathValidationError(
                f"Invalid file extension: {ext}. Allowed: {allowed_extensions}"
            )

    # File size validation
    file_size = resolved.stat().st_size

    if max_size_bytes is not None and file_size > max_size_bytes:
        raise PathValidationError(
            f"File too large: {file_size / 1024 / 1024:.1f}MB "
            f"(max: {max_size_bytes / 1024 / 1024:.1f}MB)"
        )

    # Memory warning (file + base64 encoding = ~2.5x memory usage)
    if file_size > MAX_SAFE_MEMORY_MB * 1024 * 1024:
        # Import logger lazily to avoid circular imports
        from .logger import get_logger
        get_logger().warning(
            f"Large file ({file_size / 1024 / 1024:.1f}MB) may cause memory pressure. "
            f"Peak RAM usage could reach ~{file_size * 2.5 / 1024 / 1024:.0f}MB"
        )

    return resolved


class ProviderType(Enum):
    """Available OCR providers."""
    MISTRAL = "mistral"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    AUTO = "auto"


class DocumentType(Enum):
    """Supported document types."""
    PDF = "pdf"
    IMAGE = "image"
    HANDWRITTEN = "handwritten"
    TABLE = "table"


class DebtStatus(Enum):
    """Status of a debt/claim for dispute tracking.

    Used to track the lifecycle of a debt from open to resolved.
    """
    OPEN = "open"           # Active debt, not yet addressed
    PAID = "paid"           # Debt has been paid
    DISPUTED = "disputed"   # Debt is being disputed
    PARTIAL = "partial"     # Partially paid
    CANCELLED = "cancelled" # Debt was cancelled/forgiven
    NEGOTIATING = "negotiating"  # In settlement negotiation
    LEGAL_ACTION = "legal_action"  # Legal proceedings initiated


@dataclass
class DisputeInfo:
    """Information about a debt dispute.

    Tracks dispute status, reasons, and evidence for debt management.

    Attributes:
        status: Current status of the debt
        dispute_reason: Reason for disputing the debt
        dispute_date: Date the dispute was filed
        evidence_urls: URLs/paths to evidence documents
        notes: Additional notes about the dispute
        response_deadline: Deadline for response (if any)
        legal_case_number: Associated legal case number
    """
    status: DebtStatus = DebtStatus.OPEN
    dispute_reason: Optional[str] = None
    dispute_date: Optional[str] = None
    evidence_urls: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    response_deadline: Optional[str] = None
    legal_case_number: Optional[str] = None

    def is_resolved(self) -> bool:
        """Check if the debt is resolved (paid or cancelled)."""
        return self.status in (DebtStatus.PAID, DebtStatus.CANCELLED)

    def is_active(self) -> bool:
        """Check if the debt requires attention."""
        return self.status in (
            DebtStatus.OPEN,
            DebtStatus.DISPUTED,
            DebtStatus.PARTIAL,
            DebtStatus.NEGOTIATING,
            DebtStatus.LEGAL_ACTION,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "dispute_reason": self.dispute_reason,
            "dispute_date": self.dispute_date,
            "evidence_urls": self.evidence_urls,
            "notes": self.notes,
            "response_deadline": self.response_deadline,
            "legal_case_number": self.legal_case_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisputeInfo":
        """Create from dictionary."""
        status_str = data.get("status", "open")
        try:
            status = DebtStatus(status_str)
        except ValueError:
            status = DebtStatus.OPEN

        return cls(
            status=status,
            dispute_reason=data.get("dispute_reason"),
            dispute_date=data.get("dispute_date"),
            evidence_urls=data.get("evidence_urls", []),
            notes=data.get("notes"),
            response_deadline=data.get("response_deadline"),
            legal_case_number=data.get("legal_case_number"),
        )


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""
    provider: ProviderType = ProviderType.AUTO
    confidence_threshold: float = 0.85
    auto_route: bool = True
    table_format: str = "markdown"  # markdown, html, csv, json
    parallel_workers: int = 4
    timeout: int = 120
    track_costs: bool = True

    # Vault paths
    vault_path: str = "~/Documents"
    input_dir: str = "0-Inbox/ocr-input"
    output_dir: str = "0-Inbox/ocr-output"
    review_dir: str = "0-Inbox/ocr-review"

    def __post_init__(self):
        """Validate configuration values after initialization."""
        # Timeout must be positive
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")

        # Workers must be positive
        if self.parallel_workers <= 0:
            raise ValueError(f"parallel_workers must be positive, got {self.parallel_workers}")

        # Confidence threshold must be 0.0-1.0
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.confidence_threshold}"
            )

        # Table format must be valid
        valid_formats = {"markdown", "html", "csv", "json"}
        if self.table_format not in valid_formats:
            raise ValueError(
                f"table_format must be one of {valid_formats}, "
                f"got {self.table_format}"
            )

        # Reasonable timeout limits (1 second to 10 minutes)
        if self.timeout > 600:
            raise ValueError(
                f"timeout exceeds maximum of 600 seconds, got {self.timeout}"
            )

    @classmethod
    def from_env(cls) -> "OCRConfig":
        """Load config from environment variables."""
        return cls(
            provider=ProviderType(os.getenv("OCR_DEFAULT_PROVIDER", "auto")),
            confidence_threshold=float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.85")),
            auto_route=os.getenv("OCR_AUTO_ROUTE", "true").lower() == "true",
            table_format=os.getenv("OCR_TABLE_FORMAT", "markdown"),
            parallel_workers=int(os.getenv("OCR_PARALLEL_WORKERS", "4")),
            timeout=int(os.getenv("OCR_TIMEOUT", "120")),
            track_costs=os.getenv("OCR_TRACK_COSTS", "true").lower() == "true",
            vault_path=os.getenv("VAULT_PATH", "~/Documents"),
            input_dir=os.getenv("OCR_INPUT_DIR", "0-Inbox/ocr-input"),
            output_dir=os.getenv("OCR_OUTPUT_DIR", "0-Inbox/ocr-output"),
            review_dir=os.getenv("OCR_REVIEW_DIR", "0-Inbox/ocr-review"),
        )


@dataclass
class PageResult:
    """Result for a single page."""
    index: int
    markdown: str
    images: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    header: Optional[str] = None
    footer: Optional[str] = None
    dimensions: Optional[Dict[str, int]] = None


@dataclass
class OCRResult:
    """Result from OCR processing."""
    success: bool
    provider: ProviderType
    original_file: str
    pages: List[PageResult]

    # Metadata
    confidence_score: float = 0.0
    page_count: int = 0
    word_count: int = 0
    language: str = "en"
    processing_time: float = 0.0

    # Cost tracking
    cost_estimate: float = 0.0
    tokens_used: int = 0

    # Error handling
    error: Optional[str] = None

    @property
    def full_text(self) -> str:
        """Get concatenated text from all pages."""
        return "\n\n".join(page.markdown for page in self.pages)

    def to_frontmatter(self, project: Optional[str] = None) -> str:
        """Generate YAML frontmatter for vault file."""
        today = datetime.now().strftime("%Y-%m-%d")
        fm = f"""---
created: {today}
tags:
  - type/extracted-text
  - source/ocr
  - provider/{self.provider.value}
original-file: "{Path(self.original_file).name}"
original-location: "{self.original_file}"
extraction-method: {self.provider.value}
confidence-score: {self.confidence_score:.2f}
page-count: {self.page_count}
word-count: {self.word_count}
language: {self.language}
extracted-date: {today}"""

        if project:
            fm += f'\nsource-project: "[[1-Projects/{project}]]"'

        fm += "\n---"
        return fm

    def to_markdown(self, project: Optional[str] = None) -> str:
        """Generate complete Markdown file content."""
        frontmatter = self.to_frontmatter(project)

        content = f"{frontmatter}\n\n"
        content += f"# OCR Extraction: {Path(self.original_file).stem}\n\n"

        # Source info table
        content += "## Source Information\n\n"
        content += "| Field | Value |\n"
        content += "|-------|-------|\n"
        content += f"| **Original File** | {Path(self.original_file).name} |\n"
        content += f"| **Extraction Date** | {datetime.now().strftime('%Y-%m-%d')} |\n"
        content += f"| **Provider** | {self.provider.value} |\n"
        content += f"| **Confidence** | {self.confidence_score:.0%} |\n"
        content += f"| **Pages** | {self.page_count} |\n"
        content += f"| **Words** | {self.word_count} |\n"
        content += "\n"

        # Extracted content
        content += "## Extracted Text\n\n"
        content += self.full_text
        content += "\n"

        # Tables if present
        all_tables = [t for p in self.pages for t in p.tables]
        if all_tables:
            content += "\n## Extracted Tables\n\n"
            for i, table in enumerate(all_tables, 1):
                content += f"### Table {i}\n\n"
                content += table.get("content", "")
                content += "\n\n"

        return content


# ============================================================================
# Extraction Result (Unified Pipeline Output)
# ============================================================================

@dataclass
class ExtractionResult:
    """Unified result from the full extraction pipeline.

    Combines OCR, German format parsing, classification, and anomaly detection
    into a single result object for easy handling.

    Example:
        result = await router.process_with_anomaly_detection(file_path)

        if result.is_blocked:
            print("CRITICAL: Processing blocked")
        elif result.requires_review:
            print("Flagged for review")
        else:
            print(f"Extracted {len(result.amounts)} amounts")
    """
    success: bool
    ocr_result: Optional["OCRResult"] = None
    dates: List[Any] = field(default_factory=list)      # List[ParsedGermanDate]
    amounts: List[Any] = field(default_factory=list)    # List[ParsedGermanAmount]
    ibans: List[Any] = field(default_factory=list)      # List[IBANValidationResult]
    classification: Optional[Any] = None                 # ClassificationResult
    anomaly_report: Optional[Any] = None                 # AnomalyReport
    error: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_blocked(self) -> bool:
        """Check if processing was blocked by a critical anomaly."""
        if self.anomaly_report is None:
            return False
        return getattr(self.anomaly_report, "processing_blocked", False)

    @property
    def requires_review(self) -> bool:
        """Check if the result requires manual review."""
        if self.anomaly_report is None:
            return False
        return getattr(self.anomaly_report, "requires_review", False)

    @property
    def full_text(self) -> str:
        """Get full extracted text from OCR result."""
        if self.ocr_result:
            return self.ocr_result.full_text
        return ""

    @property
    def confidence_score(self) -> float:
        """Get confidence score from OCR result."""
        if self.ocr_result:
            return self.ocr_result.confidence_score
        return 0.0

    def get_primary_amount(self) -> Optional[float]:
        """Get the primary (usually largest) amount found."""
        if not self.amounts:
            return None
        # Return the largest amount (usually the total)
        valid_amounts = [a for a in self.amounts if hasattr(a, "amount") and a.success]
        if not valid_amounts:
            return None
        return max(a.amount for a in valid_amounts)

    def get_primary_date(self) -> Optional[str]:
        """Get the primary (usually first) date found in ISO format."""
        if not self.dates:
            return None
        valid_dates = [d for d in self.dates if hasattr(d, "iso_date") and d.success]
        if not valid_dates:
            return None
        return valid_dates[0].iso_date

    def get_primary_iban(self) -> Optional[str]:
        """Get the first valid IBAN found."""
        if not self.ibans:
            return None
        valid_ibans = [i for i in self.ibans if hasattr(i, "valid") and i.valid]
        if not valid_ibans:
            return None
        return valid_ibans[0].iban

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "error": self.error,
            "processing_time": self.processing_time,
            "is_blocked": self.is_blocked,
            "requires_review": self.requires_review,
            "confidence_score": self.confidence_score,
            "dates_found": len(self.dates),
            "amounts_found": len(self.amounts),
            "ibans_found": len(self.ibans),
            "primary_amount": self.get_primary_amount(),
            "primary_date": self.get_primary_date(),
            "primary_iban": self.get_primary_iban(),
            "metadata": self.metadata,
        }


# ============================================================================
# Enhanced Result Types (for Entity Resolution & Clustering)
# ============================================================================

@dataclass
class EnhancedExtractionResult(ExtractionResult):
    """Extended extraction result with entity resolution and clustering info.

    Adds entity resolution, clustering, and urgency scoring to the base
    ExtractionResult for complete pipeline output.
    """
    # Entity resolution results
    entity_resolution: Optional[Dict[str, Any]] = None
    resolved_person_id: Optional[int] = None
    resolved_creditor_id: Optional[int] = None
    resolved_debt_id: Optional[int] = None

    # Clustering info
    cluster_id: Optional[int] = None
    is_cluster_representative: bool = False

    # Urgency scoring (150-point system)
    urgency_score: int = 0
    urgency_level: int = 1
    urgency_breakdown: Dict[str, int] = field(default_factory=dict)

    # Database tracking
    document_id: Optional[int] = None
    file_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        base = super().to_dict()
        base.update({
            "entity_resolution": self.entity_resolution,
            "resolved_person_id": self.resolved_person_id,
            "resolved_creditor_id": self.resolved_creditor_id,
            "resolved_debt_id": self.resolved_debt_id,
            "cluster_id": self.cluster_id,
            "is_cluster_representative": self.is_cluster_representative,
            "urgency_score": self.urgency_score,
            "urgency_level": self.urgency_level,
            "urgency_breakdown": self.urgency_breakdown,
            "document_id": self.document_id,
            "file_hash": self.file_hash,
        })
        return base


@dataclass
class BatchProcessingResult:
    """Result from batch processing multiple documents.

    Aggregates results from processing multiple documents, including
    clustering information and statistics.
    """
    total_documents: int = 0
    successful: int = 0
    failed: int = 0
    requires_review: int = 0
    blocked: int = 0

    results: List[EnhancedExtractionResult] = field(default_factory=list)
    clusters_formed: int = 0
    debts_identified: int = 0
    new_entities_created: int = 0

    processing_time: float = 0.0
    total_cost: float = 0.0

    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_documents == 0:
            return 0.0
        return self.successful / self.total_documents

    @property
    def consolidation_ratio(self) -> float:
        """Calculate document-to-debt consolidation ratio."""
        if self.debts_identified == 0:
            return 0.0
        return self.successful / self.debts_identified

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "successful": self.successful,
            "failed": self.failed,
            "requires_review": self.requires_review,
            "blocked": self.blocked,
            "success_rate": f"{self.success_rate:.1%}",
            "clusters_formed": self.clusters_formed,
            "debts_identified": self.debts_identified,
            "consolidation_ratio": f"{self.consolidation_ratio:.2f}",
            "new_entities_created": self.new_entities_created,
            "processing_time": self.processing_time,
            "total_cost": self.total_cost,
            "error_count": len(self.errors),
        }


@dataclass
class PipelineConfig:
    """Configuration for the full OCR processing pipeline.

    Controls all aspects of processing including:
    - Provider selection and fallback
    - Entity resolution thresholds
    - Clustering parameters
    - Anomaly detection settings
    """
    # Provider settings
    primary_provider: str = "auto"
    fallback_providers: List[str] = field(default_factory=lambda: ["mistral", "gemini", "deepseek"])
    enable_circuit_breaker: bool = True

    # Entity resolution
    enable_entity_resolution: bool = True
    auto_link_threshold: int = 95  # Score for automatic linking
    review_threshold: int = 60     # Score requiring manual review

    # Clustering
    enable_clustering: bool = True
    cluster_threshold: float = 70.0  # Similarity threshold

    # Anomaly detection
    enable_anomaly_detection: bool = True
    block_on_identity_mismatch: bool = True
    min_confidence_threshold: float = 0.85

    # Persistence
    enable_persistence: bool = True
    database_path: str = "~/.config/briefpost-management/ocr_metadata.db"

    # Processing
    parallel_workers: int = 4
    timeout_seconds: int = 120
    retry_count: int = 3
    track_costs: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.auto_link_threshold < self.review_threshold:
            raise ValueError("auto_link_threshold must be >= review_threshold")
        if not 0 <= self.cluster_threshold <= 100:
            raise ValueError("cluster_threshold must be between 0 and 100")
        if not 0 <= self.min_confidence_threshold <= 1:
            raise ValueError("min_confidence_threshold must be between 0 and 1")


class OCRProvider(ABC):
    """Abstract base class for OCR providers."""

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig.from_env()

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Provider type enum."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available (API key set, etc.)."""
        pass

    @abstractmethod
    async def process_file(
        self,
        file_path: str,
        pages: Optional[List[int]] = None,
        extract_tables: bool = True,
        table_format: str = "markdown",
    ) -> OCRResult:
        """Process a file and return OCR result."""
        pass

    @abstractmethod
    async def process_url(
        self,
        url: str,
        pages: Optional[List[int]] = None,
        extract_tables: bool = True,
        table_format: str = "markdown",
    ) -> OCRResult:
        """Process a document from URL."""
        pass

    def estimate_cost(self, page_count: int) -> float:
        """Estimate cost for processing given page count."""
        return 0.0  # Override in subclasses

    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())

    def _detect_language(self, text: str) -> str:
        """Detect language from text content.

        Enhanced detection for German and other European languages.
        Uses character patterns and keyword frequency.

        Args:
            text: Text to analyze

        Returns:
            ISO 639-1 language code (e.g., "de", "en", "fr")
        """
        if not text or len(text) < 20:
            return "en"

        text_lower = text.lower()

        # German-specific characters
        german_chars = set("äöüß")
        has_german_chars = bool(german_chars & set(text_lower))

        # German keywords (high frequency words)
        german_keywords = {
            "der", "die", "das", "und", "ist", "ein", "eine", "für", "mit",
            "auf", "nicht", "sich", "bei", "auch", "von", "nach", "über",
            "werden", "haben", "wird", "sind", "ihre", "oder", "aber",
            # Document-specific
            "rechnung", "mahnung", "betrag", "zahlung", "bitte", "gmbh",
            "straße", "sehr", "geehrte", "freundlichen", "grüßen",
        }

        # French keywords
        french_keywords = {
            "le", "la", "les", "de", "du", "des", "et", "est", "un", "une",
            "pour", "dans", "qui", "que", "sur", "avec", "sont", "nous",
            "vous", "être", "avoir", "faire", "plus", "tout", "cette",
        }

        # Spanish keywords
        spanish_keywords = {
            "el", "la", "los", "las", "de", "del", "que", "en", "es", "un",
            "una", "por", "con", "para", "como", "más", "pero", "sus",
            "están", "ser", "también", "todos", "cuando", "muy", "puede",
        }

        # Dutch keywords
        dutch_keywords = {
            "de", "het", "een", "van", "en", "in", "is", "dat", "op", "voor",
            "met", "zijn", "niet", "worden", "naar", "ook", "heeft", "worden",
        }

        # Italian keywords
        italian_keywords = {
            "il", "la", "di", "che", "è", "un", "per", "non", "una", "sono",
            "come", "da", "con", "nel", "della", "anche", "più", "essere",
        }

        # Count word matches
        words = set(text_lower.split())

        scores = {
            "de": len(german_keywords & words) * 2 + (10 if has_german_chars else 0),
            "fr": len(french_keywords & words),
            "es": len(spanish_keywords & words),
            "nl": len(dutch_keywords & words),
            "it": len(italian_keywords & words),
        }

        # Additional German patterns
        # Date format DD.MM.YYYY is common in German
        if re.search(r"\d{1,2}\.\d{1,2}\.\d{4}", text):
            scores["de"] += 3

        # Currency format with comma decimal
        if re.search(r"\d+,\d{2}\s*€", text):
            scores["de"] += 5

        # German document keywords
        german_doc_words = {
            "rechnung", "mahnung", "mahnbescheid", "gmbh", "bescheid",
            "gläubiger", "schuldner", "forderung", "betrag", "iban",
        }
        if german_doc_words & words:
            scores["de"] += 15

        # Find best match
        best_lang = max(scores, key=scores.get)
        best_score = scores[best_lang]

        # Require minimum confidence
        if best_score >= 5:
            return best_lang

        return "en"  # Default to English
