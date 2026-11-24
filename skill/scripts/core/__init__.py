"""Core abstractions: models, interfaces, configuration, and constants."""

from .config import BriefpostConfig
from .constants import (
    Category,
    DocumentType,
    EscalationLevel,
    Flag,
    MatchMethod,
    OCRProvider,
    Priority,
    ReconciliationMethod,
    Status,
)
from .models import (
    BankTransaction,
    BatchResult,
    Brief,
    Creditor,
    DocumentGroup,
    Page,
    PipelineResult,
    QCCheck,
    QCReport,
    ReconciliationResult,
    Vorgang,
)

__all__ = [
    "BriefpostConfig",
    "Category",
    "DocumentType",
    "EscalationLevel",
    "Flag",
    "MatchMethod",
    "OCRProvider",
    "Priority",
    "ReconciliationMethod",
    "Status",
    "BankTransaction",
    "BatchResult",
    "Brief",
    "Creditor",
    "DocumentGroup",
    "Page",
    "PipelineResult",
    "QCCheck",
    "QCReport",
    "ReconciliationResult",
    "Vorgang",
]
