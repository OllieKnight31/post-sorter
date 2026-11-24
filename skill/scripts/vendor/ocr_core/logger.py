#!/usr/bin/env python3
"""Structured logging for OCR processor.

Provides JSON-formatted logs with context for debugging, monitoring, and analytics.
Logs are written to ~/.config/briefpost-management/logs/ocr.log with rotation support.
"""

import json
import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum


class LogLevel(Enum):
    """Log levels with semantic meaning."""
    DEBUG = "debug"      # API request/response details
    INFO = "info"        # Processing start/complete
    WARNING = "warning"  # Retries, fallbacks
    ERROR = "error"      # Failures with stack traces


@dataclass
class LogContext:
    """Structured context for log entries."""
    operation: str                      # e.g., "process_file", "select_provider"
    provider: Optional[str] = None      # e.g., "deepseek", "gemini"
    file_path: Optional[str] = None     # Document being processed
    page_count: Optional[int] = None
    duration_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    tokens: Optional[int] = None
    confidence: Optional[float] = None
    retry_count: Optional[int] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON line."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add context if present
        if hasattr(record, "context") and record.context:
            if isinstance(record.context, LogContext):
                log_entry["context"] = record.context.to_dict()
            elif isinstance(record.context, dict):
                log_entry["context"] = record.context

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": "".join(traceback.format_exception(*record.exc_info)),
            }

        return json.dumps(log_entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "RESET": "\033[0m",
    }

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console."""
        level = record.levelname
        color = self.COLORS.get(level, "") if self.use_colors else ""
        reset = self.COLORS["RESET"] if self.use_colors else ""

        # Build message parts
        timestamp = datetime.now().strftime("%H:%M:%S")
        parts = [f"{color}[{timestamp}] [{level:7}]{reset} {record.getMessage()}"]

        # Add context summary
        if hasattr(record, "context") and record.context:
            ctx = record.context
            if isinstance(ctx, LogContext):
                ctx_parts = []
                if ctx.provider:
                    ctx_parts.append(f"provider={ctx.provider}")
                if ctx.duration_ms is not None:
                    ctx_parts.append(f"duration={ctx.duration_ms:.0f}ms")
                if ctx.confidence is not None:
                    ctx_parts.append(f"confidence={ctx.confidence:.0%}")
                if ctx.retry_count is not None:
                    ctx_parts.append(f"retry={ctx.retry_count}")
                if ctx_parts:
                    parts.append(f"  ({', '.join(ctx_parts)})")
            elif isinstance(ctx, dict):
                ctx_str = ", ".join(f"{k}={v}" for k, v in ctx.items())
                if ctx_str:
                    parts.append(f"  ({ctx_str})")

        return "".join(parts)


class OCRLogger:
    """Structured logger for OCR processor operations."""

    _instance: Optional["OCRLogger"] = None
    _initialized: bool = False

    def __new__(cls) -> "OCRLogger":
        """Singleton pattern for consistent logging across modules."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize logger (only once due to singleton)."""
        if OCRLogger._initialized:
            return

        self.logger = logging.getLogger("briefpost-ocr")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # File handler with rotation (JSON format)
        try:
            log_dir = Path.home() / ".config" / "briefpost-management" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "ocr.log"

            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(file_handler)
        except OSError:
            # A read-only home folder must not stop OCR; console logging still works.
            pass

        # Console handler (human-readable)
        console_handler = logging.StreamHandler(sys.stderr)
        console_level = os.getenv("OCR_LOG_LEVEL", "INFO").upper()
        console_handler.setLevel(getattr(logging, console_level, logging.INFO))
        console_handler.setFormatter(ConsoleFormatter())
        self.logger.addHandler(console_handler)

        OCRLogger._initialized = True

    def _log(
        self,
        level: int,
        message: str,
        context: Optional[Union[LogContext, Dict[str, Any]]] = None,
        exc_info: bool = False,
    ) -> None:
        """Internal logging method."""
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=sys.exc_info() if exc_info else None,
        )
        record.context = context
        self.logger.handle(record)

    def debug(
        self,
        message: str,
        context: Optional[Union[LogContext, Dict[str, Any]]] = None,
    ) -> None:
        """Log debug message (API request/response details)."""
        self._log(logging.DEBUG, message, context)

    def info(
        self,
        message: str,
        context: Optional[Union[LogContext, Dict[str, Any]]] = None,
    ) -> None:
        """Log info message (processing start/complete)."""
        self._log(logging.INFO, message, context)

    def warning(
        self,
        message: str,
        context: Optional[Union[LogContext, Dict[str, Any]]] = None,
    ) -> None:
        """Log warning message (retries, fallbacks)."""
        self._log(logging.WARNING, message, context)

    def error(
        self,
        message: str,
        context: Optional[Union[LogContext, Dict[str, Any]]] = None,
        exc_info: bool = True,
    ) -> None:
        """Log error message (failures with stack traces)."""
        self._log(logging.ERROR, message, context, exc_info=exc_info)

    # Convenience methods for common operations

    def log_processing_start(
        self,
        file_path: str,
        provider: str,
        page_count: Optional[int] = None,
    ) -> None:
        """Log start of file processing."""
        self.info(
            f"Starting OCR processing: {Path(file_path).name}",
            LogContext(
                operation="process_start",
                provider=provider,
                file_path=file_path,
                page_count=page_count,
            ),
        )

    def log_processing_complete(
        self,
        file_path: str,
        provider: str,
        duration_ms: float,
        page_count: int,
        word_count: int,
        confidence: float,
        cost_usd: float = 0.0,
        tokens: int = 0,
    ) -> None:
        """Log successful completion of processing."""
        self.info(
            f"Processing complete: {Path(file_path).name} ({page_count} pages, {word_count} words)",
            LogContext(
                operation="process_complete",
                provider=provider,
                file_path=file_path,
                page_count=page_count,
                duration_ms=duration_ms,
                confidence=confidence,
                cost_usd=cost_usd,
                tokens=tokens,
                extra={"word_count": word_count},
            ),
        )

    def log_processing_failed(
        self,
        file_path: str,
        provider: str,
        error_type: str,
        error_message: str,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Log failed processing attempt."""
        self.error(
            f"Processing failed: {Path(file_path).name} - {error_message}",
            LogContext(
                operation="process_failed",
                provider=provider,
                file_path=file_path,
                duration_ms=duration_ms,
                error_type=error_type,
                error_message=error_message,
            ),
            exc_info=False,
        )

    def log_retry(
        self,
        file_path: str,
        provider: str,
        retry_count: int,
        max_retries: int,
        error_type: str,
        delay_seconds: float,
    ) -> None:
        """Log retry attempt."""
        self.warning(
            f"Retrying ({retry_count}/{max_retries}) after {delay_seconds:.1f}s: {error_type}",
            LogContext(
                operation="retry",
                provider=provider,
                file_path=file_path,
                retry_count=retry_count,
                error_type=error_type,
                extra={"max_retries": max_retries, "delay_seconds": delay_seconds},
            ),
        )

    def log_fallback(
        self,
        file_path: str,
        from_provider: str,
        to_provider: str,
        reason: str,
    ) -> None:
        """Log fallback to alternative provider."""
        self.warning(
            f"Falling back from {from_provider} to {to_provider}: {reason}",
            LogContext(
                operation="fallback",
                provider=to_provider,
                file_path=file_path,
                extra={"from_provider": from_provider, "reason": reason},
            ),
        )

    def log_provider_selected(
        self,
        file_path: str,
        provider: str,
        reason: str,
    ) -> None:
        """Log provider selection decision."""
        self.info(
            f"Selected provider: {provider} ({reason})",
            LogContext(
                operation="provider_selected",
                provider=provider,
                file_path=file_path,
                extra={"reason": reason},
            ),
        )

    def log_api_request(
        self,
        provider: str,
        endpoint: str,
        request_size: int,
    ) -> None:
        """Log API request details (debug level)."""
        self.debug(
            f"API request to {provider}: {endpoint}",
            LogContext(
                operation="api_request",
                provider=provider,
                extra={"endpoint": endpoint, "request_size": request_size},
            ),
        )

    def log_api_response(
        self,
        provider: str,
        status_code: int,
        response_size: int,
        duration_ms: float,
    ) -> None:
        """Log API response details (debug level)."""
        self.debug(
            f"API response from {provider}: {status_code}",
            LogContext(
                operation="api_response",
                provider=provider,
                duration_ms=duration_ms,
                extra={"status_code": status_code, "response_size": response_size},
            ),
        )

    def log_cost(
        self,
        provider: str,
        file_path: str,
        page_count: int,
        cost_usd: float,
        tokens: int = 0,
    ) -> None:
        """Log cost tracking entry."""
        self.debug(
            f"Cost logged: ${cost_usd:.4f} for {page_count} pages via {provider}",
            LogContext(
                operation="cost_logged",
                provider=provider,
                file_path=file_path,
                page_count=page_count,
                cost_usd=cost_usd,
                tokens=tokens,
            ),
        )


# Global logger instance
logger = OCRLogger()


def get_logger() -> OCRLogger:
    """Get the global OCR logger instance."""
    return logger
