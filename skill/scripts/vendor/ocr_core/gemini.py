#!/usr/bin/env python3
"""Gemini Vision OCR provider implementation.

Enhanced with German financial document extraction capabilities,
thinking level configuration, and structured JSON output.
"""

import os
import asyncio
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import httpx

from .base import (
    OCRProvider, OCRResult, OCRConfig, ProviderType, PageResult,
    validate_file_path, PathValidationError,
)
from .logger import get_logger
from .extraction_prompts import (
    ExtractionMode,
    get_extraction_prompt,
    build_german_financial_prompt,
)
from .extraction_schemas import GERMAN_EXTRACTION_SCHEMA
from .extraction_types import (
    GeminiExtractionConfig,
    GermanExtractionResult,
    ThinkingLevel,
)


class GeminiOCR(OCRProvider):
    """Google Gemini 3 Vision provider.

    Features:
    - Native PDF understanding (not just text extraction)
    - Up to 1000 pages, 50MB per document
    - "Derendering" - converts visuals to HTML/LaTeX/Markdown
    - Superior for complex layouts, math equations
    - media_resolution control (low/medium/high)
    - Multi-document comparison
    """

    MODEL = "gemini-3.6-flash"
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_PAGES = 1000
    TOKENS_PER_PAGE = 258  # Approximate

    def __init__(self, config: Optional[OCRConfig] = None):
        super().__init__(config)
        self._client = None

    @property
    def name(self) -> str:
        return "Gemini 3 Pro Vision"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.GEMINI

    def is_available(self) -> bool:
        """Check if Google API key is available."""
        api_key = os.getenv("GOOGLE_API_KEY")
        return api_key is not None and len(api_key) > 0

    def _get_client(self):
        """Get or create Gemini client."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client()
            except ImportError:
                raise ImportError("google-genai package not installed. Run: pip install google-genai")
        return self._client

    def estimate_cost(self, page_count: int) -> float:
        """Estimate cost for processing.

        Gemini uses token-based pricing. Flash model is very cost-effective.
        Rough estimate: ~$0.00025 per page for flash model.
        """
        return page_count * 0.00025

    def _is_pdf_encrypted(self, file_path: str) -> bool:
        """Check if a PDF is encrypted/password-protected.

        Returns:
            True if the PDF is encrypted, False otherwise
        """
        try:
            import pypdf
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                return reader.is_encrypted
        except ImportError:
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    return reader.is_encrypted
            except ImportError:
                # If neither library is available, skip check
                get_logger().debug("Neither pypdf nor PyPDF2 available for encryption check")
                return False
        except Exception as e:
            get_logger().warning(f"Error checking PDF encryption: {e}")
            return False

    async def process_file(
        self,
        file_path: str,
        pages: Optional[List[int]] = None,
        extract_tables: bool = True,
        table_format: str = "markdown",
        resolution: str = "high",
    ) -> OCRResult:
        """Process a local file with Gemini Vision."""
        start_time = time.time()

        # Security: Validate file path before processing
        try:
            path = validate_file_path(
                file_path,
                max_size_bytes=self.MAX_FILE_SIZE,
                allowed_extensions=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"],
            )
        except FileNotFoundError as e:
            return OCRResult(
                success=False,
                provider=self.provider_type,
                original_file=file_path,
                pages=[],
                error=str(e),
            )
        except PathValidationError as e:
            get_logger().error(f"Path validation failed: {e}", exc_info=False)
            return OCRResult(
                success=False,
                provider=self.provider_type,
                original_file=file_path,
                pages=[],
                error=f"Security validation failed: {e}",
            )

        # Check for encrypted PDF before processing
        if path.suffix.lower() == ".pdf" and self._is_pdf_encrypted(str(path)):
            return OCRResult(
                success=False,
                provider=self.provider_type,
                original_file=file_path,
                pages=[],
                error="PDF is password-protected. Please remove protection before processing. "
                      "You can use tools like 'qpdf --decrypt input.pdf output.pdf' or "
                      "'pdftk input.pdf output output.pdf' to remove password protection.",
                processing_time=time.time() - start_time,
            )

        try:
            client = self._get_client()
            from google.genai import types

            # Read file
            file_data = path.read_bytes()

            # Determine MIME type
            mime_types = {
                ".pdf": "application/pdf",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".heic": "image/heic",
                ".heif": "image/heif",
            }
            mime_type = mime_types.get(path.suffix.lower(), "application/pdf")

            # Build prompt for OCR
            prompt = self._build_ocr_prompt(extract_tables, table_format)

            # Create content parts
            contents = [
                types.Part.from_bytes(data=file_data, mime_type=mime_type),
                prompt,
            ]

            # Generate content
            response = await asyncio.to_thread(
                lambda: client.models.generate_content(
                    model=self.MODEL,
                    contents=contents,
                )
            )

            # Parse response
            extracted_text = response.text if hasattr(response, "text") else ""

            # Get usage info
            tokens_used = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                um = response.usage_metadata
                tokens_used = getattr(um, "total_token_count", 0) or (
                    getattr(um, "prompt_token_count", 0) + getattr(um, "candidates_token_count", 0)
                )

            # Create page result (Gemini returns all text at once)
            page_results = [
                PageResult(
                    index=0,
                    markdown=extracted_text,
                    images=[],
                    tables=[],
                )
            ]

            # Estimate page count from tokens
            page_count = max(1, tokens_used // self.TOKENS_PER_PAGE)
            word_count = self._count_words(extracted_text)

            processing_time = time.time() - start_time

            return OCRResult(
                success=True,
                provider=self.provider_type,
                original_file=file_path,
                pages=page_results,
                confidence_score=0.95,  # Gemini doesn't return confidence
                page_count=page_count,
                word_count=word_count,
                language=self._detect_language(extracted_text),
                processing_time=processing_time,
                cost_estimate=self.estimate_cost(page_count),
                tokens_used=tokens_used,
            )

        except Exception as e:
            return OCRResult(
                success=False,
                provider=self.provider_type,
                original_file=file_path,
                pages=[],
                error=str(e),
            )

    async def process_url(
        self,
        url: str,
        pages: Optional[List[int]] = None,
        extract_tables: bool = True,
        table_format: str = "markdown",
        resolution: str = "high",
    ) -> OCRResult:
        """Process a document from URL using Files API."""
        start_time = time.time()

        try:
            import io
            import httpx
            from google.genai import types

            client = self._get_client()

            # Download file with timeout configuration
            timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout) as http_client:
                response = await http_client.get(url)
                response.raise_for_status()
                file_data = io.BytesIO(response.content)

            # Determine MIME type from URL
            is_pdf = url.lower().endswith(".pdf")
            mime_type = "application/pdf" if is_pdf else "image/png"

            # Upload to Files API for better handling
            uploaded_file = await asyncio.to_thread(
                lambda: client.files.upload(
                    file=file_data,
                    config=dict(mime_type=mime_type)
                )
            )

            # Build prompt
            prompt = self._build_ocr_prompt(extract_tables, table_format)

            # Generate content with uploaded file
            response = await asyncio.to_thread(
                lambda: client.models.generate_content(
                    model=self.MODEL,
                    contents=[uploaded_file, prompt],
                )
            )

            extracted_text = response.text if hasattr(response, "text") else ""

            tokens_used = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                um = response.usage_metadata
                tokens_used = getattr(um, "total_token_count", 0) or (
                    getattr(um, "prompt_token_count", 0) + getattr(um, "candidates_token_count", 0)
                )

            page_results = [
                PageResult(
                    index=0,
                    markdown=extracted_text,
                    images=[],
                    tables=[],
                )
            ]

            page_count = max(1, tokens_used // self.TOKENS_PER_PAGE)
            word_count = self._count_words(extracted_text)
            processing_time = time.time() - start_time

            return OCRResult(
                success=True,
                provider=self.provider_type,
                original_file=url,
                pages=page_results,
                confidence_score=0.95,
                page_count=page_count,
                word_count=word_count,
                language=self._detect_language(extracted_text),
                processing_time=processing_time,
                cost_estimate=self.estimate_cost(page_count),
                tokens_used=tokens_used,
            )

        except httpx.TimeoutException as e:
            get_logger().error(f"URL download timeout: {url}", exc_info=False)
            return OCRResult(
                success=False,
                provider=self.provider_type,
                original_file=url,
                pages=[],
                error=f"Download timeout: {e}",
            )
        except httpx.HTTPStatusError as e:
            get_logger().error(f"HTTP error downloading {url}: {e.response.status_code}", exc_info=False)
            return OCRResult(
                success=False,
                provider=self.provider_type,
                original_file=url,
                pages=[],
                error=f"HTTP {e.response.status_code}: {e}",
            )
        except Exception as e:
            get_logger().error(f"Failed to process URL: {url}", exc_info=True)
            return OCRResult(
                success=False,
                provider=self.provider_type,
                original_file=url,
                pages=[],
                error=str(e),
            )

    def _build_ocr_prompt(
        self,
        extract_tables: bool,
        table_format: str,
        german_mode: bool = False,
    ) -> str:
        """Build the OCR extraction prompt.

        Args:
            extract_tables: Whether to extract tables
            table_format: Format for tables (markdown, html)
            german_mode: Whether to use German financial mode

        Returns:
            Prompt string for the API
        """
        if german_mode:
            return build_german_financial_prompt(extract_tables, table_format)

        prompt = """Extract all text from this document, preserving the layout and structure.

Output the content as clean Markdown with:
- Proper headings (# for main titles, ## for sections)
- Bullet points and numbered lists preserved
- Paragraphs separated by blank lines
"""

        if extract_tables:
            if table_format == "html":
                prompt += """
- Extract all tables as HTML tables with proper <table>, <tr>, <th>, <td> tags
- Include colspan/rowspan attributes where applicable
"""
            else:
                prompt += """
- Extract all tables as Markdown tables with proper alignment
- Use | for column separators and --- for header rows
"""

        prompt += """
Important:
- Preserve the document's hierarchical structure
- Include all visible text, headers, footers
- Do not summarize - extract the complete text
- If there are images with captions, include the captions
- For math equations, use LaTeX notation ($ for inline, $$ for display)

Begin extraction:"""

        return prompt

    async def extract_german_financial(
        self,
        file_path: str,
        config: Optional[GeminiExtractionConfig] = None,
    ) -> GermanExtractionResult:
        """Extract structured data from German financial document.

        Uses enhanced prompts with 8 critical German rules for accurate
        extraction of invoices, reminders, and court documents.

        Args:
            file_path: Path to the document file
            config: Extraction configuration

        Returns:
            GermanExtractionResult with structured data
        """
        config = config or GeminiExtractionConfig()

        # Security: Validate file path
        try:
            path = validate_file_path(
                file_path,
                max_size_bytes=self.MAX_FILE_SIZE,
                allowed_extensions=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"],
            )
        except (FileNotFoundError, PathValidationError) as e:
            return GermanExtractionResult(
                extraction_successful=False,
                error_reason=str(e),
            )

        try:
            client = self._get_client()
            from google.genai import types

            # Read file
            file_data = path.read_bytes()

            # Determine MIME type
            mime_types = {
                ".pdf": "application/pdf",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".heic": "image/heic",
                ".heif": "image/heif",
            }
            mime_type = mime_types.get(path.suffix.lower(), "application/pdf")

            # Build German financial prompt
            prompt = get_extraction_prompt(
                mode=ExtractionMode.GERMAN_FINANCIAL,
                extract_tables=config.extract_tables,
                table_format=config.table_format,
            )

            # Create content parts
            contents = [
                types.Part.from_bytes(data=file_data, mime_type=mime_type),
                prompt,
            ]

            # Build generation config with thinking level
            gen_config = self._build_generation_config(config)

            # Generate content with generation config (always pass for thinking level)
            # Note: gen_config includes thinking_config for improved accuracy
            def _generate(active_config):
                return client.models.generate_content(
                    model=self.MODEL,
                    contents=contents,
                    config=active_config,
                )

            try:
                response = await asyncio.to_thread(lambda: _generate(gen_config))
            except Exception as thinking_exc:
                # Newer models changed how thinking is configured. If the
                # request is rejected because of it, retry once without it.
                if "thinking_config" in gen_config and "thinking" in str(thinking_exc).lower():
                    get_logger().warning(
                        f"Model {self.MODEL} rejected the thinking settings, retrying without them"
                    )
                    reduced = {k: v for k, v in gen_config.items() if k != "thinking_config"}
                    response = await asyncio.to_thread(lambda: _generate(reduced))
                else:
                    raise

            # Parse response
            response_text = response.text if hasattr(response, "text") else ""

            # Try to parse as JSON
            try:
                # Clean up response - remove markdown code blocks if present
                json_text = response_text
                if json_text.startswith("```json"):
                    json_text = json_text[7:]
                if json_text.startswith("```"):
                    json_text = json_text[3:]
                if json_text.endswith("```"):
                    json_text = json_text[:-3]
                json_text = json_text.strip()

                result_data = json.loads(json_text)
                # Handle Gemini returning a list instead of dict
                if isinstance(result_data, list):
                    result_data = result_data[0] if result_data else {}
                return GermanExtractionResult.from_dict(result_data)

            except json.JSONDecodeError as e:
                get_logger().warning(f"Failed to parse JSON response: {e}")
                # Return raw text as fallback
                return GermanExtractionResult(
                    extraction_successful=True,
                    raw_text=response_text,
                    confidence=50.0,
                )

        except Exception as e:
            get_logger().error(f"German financial extraction failed: {e}", exc_info=True)
            return GermanExtractionResult(
                extraction_successful=False,
                error_reason=str(e),
            )

    def _build_generation_config(
        self,
        config: GeminiExtractionConfig,
    ) -> Dict[str, Any]:
        """Build generation config for Gemini API.

        Args:
            config: Extraction configuration

        Returns:
            Dictionary with generation config parameters
        """
        gen_config: Dict[str, Any] = {
            "temperature": config.temperature,
        }

        # Add thinking configuration if not NONE
        if config.thinking_level != ThinkingLevel.NONE:
            # Gemini 3.5 and newer take a thinking level ("low", "medium",
            # "high"); a token budget is rejected by those models.
            gen_config["thinking_config"] = {
                "thinking_level": config.thinking_level.value,
            }

        # Add response schema for structured output
        if config.use_structured_output:
            gen_config["response_mime_type"] = "application/json"
            gen_config["response_schema"] = GERMAN_EXTRACTION_SCHEMA

        return gen_config
