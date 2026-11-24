"""Shared LLM utility functions for briefpost pipeline.

Provides Gemini Flash and Pro helpers with structured JSON output,
used by matching (Phase C) and reconciliation (Phase D) for
ambiguous cases that algorithmic methods cannot resolve.

A2: Retry with exponential backoff on 429/500/503.
A3: 30-second timeout on all API calls.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# A2: Retry configuration
MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]  # seconds
RETRYABLE_STATUS_CODES = {429, 500, 503}

# A3: Timeout in seconds for API calls
API_TIMEOUT_SECONDS = 30


# Model IDs. Override with ocr.llm_flash_model / ocr.llm_pro_model in the
# config (see configure_models) or the BRIEFPOST_FLASH_MODEL /
# BRIEFPOST_PRO_MODEL environment variables.
FLASH_MODEL = os.environ.get("BRIEFPOST_FLASH_MODEL", "gemini-3.6-flash")
PRO_MODEL = os.environ.get("BRIEFPOST_PRO_MODEL", "gemini-3.8-flash")


def configure_models(config: Any) -> None:
    """Take the helper model IDs from the config, if set.

    Args:
        config: BriefpostConfig (reads ocr.llm_flash_model and ocr.llm_pro_model).
    """
    global FLASH_MODEL, PRO_MODEL
    ocr = getattr(config, "ocr", None) or {}
    if ocr.get("llm_flash_model"):
        FLASH_MODEL = str(ocr["llm_flash_model"])
    if ocr.get("llm_pro_model"):
        PRO_MODEL = str(ocr["llm_pro_model"])


def generate_content_safe(client: Any, model: str, contents: Any, config: dict[str, Any]) -> Any:
    """Call generate_content; if the model rejects the thinking settings, retry without them.

    Newer Gemini models changed how thinking is configured. A request that
    fails with an error mentioning "thinking" is sent once more with the
    thinking_config removed, so a model change does not stop the pipeline.

    Args:
        client: google.genai Client.
        model: Model ID.
        contents: Request parts.
        config: Generation config dict (not modified).

    Returns:
        API response object.
    """
    try:
        return client.models.generate_content(model=model, contents=contents, config=config)
    except Exception as exc:
        if "thinking_config" in config and "thinking" in str(exc).lower():
            logger.warning(
                "Model %s rejected the thinking settings (%s). Retrying without them.",
                model, str(exc)[:120],
            )
            reduced = {k: v for k, v in config.items() if k != "thinking_config"}
            return client.models.generate_content(model=model, contents=contents, config=reduced)
        raise


def _ensure_api_key() -> None:
    """Load GOOGLE_API_KEY from the environment or an .env file."""
    from core.extern import load_api_key
    load_api_key()


def validate_api_key() -> bool:
    """Validate that GOOGLE_API_KEY is available.

    Attempts to load the key from .env if not in environment, then
    checks if the key is set and non-empty.

    Returns:
        True if API key is available, False otherwise.
    """
    _ensure_api_key()
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        logger.warning("GOOGLE_API_KEY not set - all LLM features will be disabled")
        return False
    return True


def _get_genai_client() -> Any:
    """Get google.genai Client."""
    _ensure_api_key()
    try:
        from google import genai
        return genai.Client()
    except ImportError:
        raise ImportError("google-genai package required for LLM-assisted pipeline")


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is retryable (429, 500, 503).

    Uses isinstance() for google.api_core.exceptions when available,
    with fallback to string matching for other exception types.

    Args:
        exc: The exception to check.

    Returns:
        True if the request should be retried.
    """
    # CQ-007: Prefer isinstance() over string name matching
    try:
        from google.api_core import exceptions as api_exceptions
        if isinstance(exc, (
            api_exceptions.ResourceExhausted,
            api_exceptions.ServiceUnavailable,
            api_exceptions.InternalServerError,
            api_exceptions.DeadlineExceeded,
        )):
            return True
    except ImportError:
        # Fallback: check type name if google.api_core not available
        exc_type = type(exc).__name__
        if exc_type in (
            "ResourceExhausted", "ServiceUnavailable",
            "InternalServerError", "DeadlineExceeded",
        ):
            return True

    # Check for HTTP status codes in exception message
    exc_str = str(exc)
    for code in RETRYABLE_STATUS_CODES:
        if str(code) in exc_str:
            return True

    return False


def _call_with_retry(
    client: Any,
    model: str,
    contents: str,
    config: dict[str, Any],
) -> Any:
    """Call generate_content with retry and exponential backoff.

    Args:
        client: google.genai Client.
        model: Model name.
        contents: Prompt text.
        config: Generation config dict.

    Returns:
        API response object.

    Raises:
        Exception: If all retry attempts fail.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(MAX_RETRIES):
        try:
            return generate_content_safe(client, model, contents, config)
        except ImportError:
            raise  # Never retry import errors
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt == MAX_RETRIES - 1:
                # CQ-008: Log exhaustion before final raise
                logger.error(
                    "All %d retries exhausted for API call",
                    MAX_RETRIES, exc_info=True,
                )
                raise
            delay = RETRY_DELAYS[attempt] if attempt < len(RETRY_DELAYS) else RETRY_DELAYS[-1]
            logger.warning(
                "API call attempt %d/%d failed (%s), retrying in %ds...",
                attempt + 1, MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)

    raise last_exc  # Should not reach here, but satisfy type checker


def _parse_json_response(text: str) -> Optional[dict]:
    """Extract and parse JSON from LLM response text.

    Handles responses with markdown fences, preamble text, etc.
    CQ-038: Brace counting is aware of string literals.

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed dict or None if parsing fails.
    """
    if not text:
        return None
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code fence
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # CQ-038: Find first { ... } block, skipping braces inside string literals
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        in_string = False
        escape_next = False
        for i in range(brace_start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i + 1])
                    except json.JSONDecodeError:
                        break
    # RS-020: Log raw response on parse failure
    logger.debug("Failed to parse JSON from response (len=%d): %.200s", len(text), text)
    return None


def call_gemini_flash(
    prompt: str,
    temperature: float = 0.0,
    max_output_tokens: int = 4096,
) -> Optional[dict]:
    """Call Gemini Flash with text prompt, expecting JSON response.

    Includes retry with exponential backoff (A2) and timeout (A3).

    Args:
        prompt: Text prompt for the model.
        temperature: Sampling temperature (0.0 for deterministic).
        max_output_tokens: Maximum response length.

    Returns:
        Parsed JSON dict or None if call fails.
    """
    try:
        client = _get_genai_client()
        gen_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": "application/json",
            "http_options": {"timeout": API_TIMEOUT_SECONDS * 1000},
        }
        # CQ-023: ThreadPoolExecutor timeout as safety net
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _call_with_retry, client, FLASH_MODEL, prompt, gen_config,
            )
            response = future.result(timeout=API_TIMEOUT_SECONDS)
        response_text = response.text if hasattr(response, "text") else ""
        return _parse_json_response(response_text)
    except ImportError:
        logger.warning("google-genai not available, LLM call skipped")
        return None
    except concurrent.futures.TimeoutError:
        logger.error("Gemini Flash call timed out after %ds", API_TIMEOUT_SECONDS)
        return None
    except Exception as exc:
        # CQ-010: Preserve exception type and traceback
        logger.error("Gemini Flash call failed", exc_info=True)
        return None


def call_gemini_pro(
    prompt: str,
    temperature: float = 0.0,
    max_output_tokens: int = 8192,
    thinking_budget: int = 32768,
) -> Optional[dict]:
    """Call Gemini Pro with extended thinking, expecting JSON response.

    Includes retry with exponential backoff (A2) and timeout (A3).

    Args:
        prompt: Text prompt for the model.
        temperature: Sampling temperature.
        max_output_tokens: Maximum response length.
        thinking_budget: Unused. Newer models take a thinking level instead of a token budget.

    Returns:
        Parsed JSON dict or None if call fails.
    """
    try:
        client = _get_genai_client()
        gen_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            # Gemini 3.5 and newer take a thinking level; a token budget is rejected.
            # thinking_budget is kept in the signature for older callers only.
            "thinking_config": {"thinking_level": "high"},
            "response_mime_type": "application/json",
            "http_options": {"timeout": API_TIMEOUT_SECONDS * 1000},
        }
        # CQ-023: ThreadPoolExecutor timeout as safety net
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _call_with_retry, client, PRO_MODEL, prompt, gen_config,
            )
            response = future.result(timeout=API_TIMEOUT_SECONDS)
        response_text = response.text if hasattr(response, "text") else ""
        return _parse_json_response(response_text)
    except ImportError:
        logger.warning("google-genai not available, LLM call skipped")
        return None
    except concurrent.futures.TimeoutError:
        logger.error("Gemini Pro call timed out after %ds", API_TIMEOUT_SECONDS)
        return None
    except Exception as exc:
        # CQ-010: Preserve exception type and traceback
        logger.error("Gemini Pro call failed", exc_info=True)
        return None
