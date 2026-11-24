"""LLM-enhanced OCR fallback strategies.

Provides multi-provider OCR fallback for low-confidence pages:
1. Cross-provider consensus for NEEDS_VISION flagged pages
2. Extended thinking for court documents (eskalation >= 4) OR confidence < 0.70
3. Gemini Pro fallback for confidence < low_confidence_threshold
4. Extraction merging (highest confidence fields win)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from core.config import BriefpostConfig
from core.constants import Flag

logger = logging.getLogger(__name__)

def _import_from_ocr_skill(module_path: str, attr_name: str) -> Any:
    """Import an attribute from the bundled OCR helpers.

    Args:
        module_path: Module file path such as "core/gemini.py" (only the file name is used).
        attr_name: Name of the attribute to import.

    Returns:
        The imported attribute.
    """
    from core.extern import load_ocr_attr
    return load_ocr_attr(Path(module_path).stem, attr_name)


def extract_with_extended_thinking(
    image_path: Path,
    config: BriefpostConfig,
) -> dict[str, Any]:
    """Extract using Gemini with extended thinking for court documents.

    Triggered for documents with eskalation >= 4 (Mahnbescheid and above).
    Uses ThinkingLevel.HIGH for maximum accuracy on complex legal text.

    Args:
        image_path: Path to the page image.
        config: Pipeline configuration.

    Returns:
        Dict with extraction result including confidence and structured data.
    """
    try:
        import asyncio
        GeminiOCR = _import_from_ocr_skill("core/gemini.py", "GeminiOCR")
        GeminiExtractionConfig = _import_from_ocr_skill("core/extraction_types.py", "GeminiExtractionConfig")
        ThinkingLevel = _import_from_ocr_skill("core/extraction_types.py", "ThinkingLevel")

        ocr_config = config.ocr
        budget = ocr_config.get("thinking_budget_tokens", 15000)

        # HIGH thinking for court docs, MEDIUM otherwise
        extraction_config = GeminiExtractionConfig(
            thinking_level=ThinkingLevel.HIGH,
            temperature=0.5,
            use_structured_output=True,
            german_mode=True,
        )

        from core.extern import load_api_key
        load_api_key()
        gemini = GeminiOCR()
        if not gemini.is_available():
            logger.error("Gemini API not available (GOOGLE_API_KEY missing)")
            return {"success": False, "error": "Gemini not available"}
        gemini.MODEL = config.ocr.get("model", gemini.MODEL)

        result = asyncio.run(
            gemini.extract_german_financial(str(image_path), config=extraction_config)
        )

        return {
            "success": result.extraction_successful,
            "confidence": result.confidence,
            "extraction": result.to_dict() if result.extraction_successful else None,
            "provider": "gemini_extended_thinking",
            "error": result.error_reason if not result.extraction_successful else None,
        }

    except Exception as exc:
        logger.error("Extended thinking extraction failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


def extract_with_gemini_pro(
    image_path: Path,
    gemini_text: str,
    config: BriefpostConfig,
) -> dict[str, Any]:
    """Extract using Gemini 3 Pro for independent second opinion.

    Triggered when Gemini Flash confidence < low_confidence_threshold.
    Sends the actual image to Gemini Pro for independent extraction.

    Args:
        image_path: Path to the page image.
        gemini_text: Gemini Flash's OCR text for context/comparison.
        config: Pipeline configuration.

    Returns:
        Dict with extraction result from Gemini Pro.
    """
    try:
        import asyncio
        GeminiOCR = _import_from_ocr_skill("core/gemini.py", "GeminiOCR")
        GeminiExtractionConfig = _import_from_ocr_skill("core/extraction_types.py", "GeminiExtractionConfig")
        ThinkingLevel = _import_from_ocr_skill("core/extraction_types.py", "ThinkingLevel")

        ocr_config = config.ocr
        temperature = ocr_config.get("temperature", 0.3)

        extraction_config = GeminiExtractionConfig(
            thinking_level=ThinkingLevel.NONE,
            temperature=temperature,
            use_structured_output=True,
            german_mode=True,
        )

        from core.extern import load_api_key
        load_api_key()
        gemini = GeminiOCR()
        if not gemini.is_available():
            logger.error("Gemini API not available (GOOGLE_API_KEY missing)")
            return {"success": False, "error": "Gemini not available"}

        # Second opinion from a different (stronger) model than the first pass
        gemini.MODEL = config.ocr.get("fallback_model", "gemini-3.8-flash")

        result = asyncio.run(
            gemini.extract_german_financial(str(image_path), config=extraction_config)
        )

        return {
            "success": result.extraction_successful,
            "confidence": result.confidence,
            "extraction": result.to_dict() if result.extraction_successful else None,
            "provider": "gemini_pro",
            "error": result.error_reason if not result.extraction_successful else None,
        }

    except Exception as exc:
        logger.error("Gemini Pro extraction failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


def extract_with_consensus(
    image_path: Path,
    providers: list[str],
    config: BriefpostConfig,
) -> dict[str, Any]:
    """Extract using multiple providers and compare results.

    Uses an optional multi-provider extraction_pipeline.extract_with_consensus()
    for NEEDS_VISION flagged pages. Runs multiple providers, compares,
    and takes the best result.

    Args:
        image_path: Path to the page image.
        providers: List of provider names to use.
        config: Pipeline configuration.

    Returns:
        Dict with consensus extraction result.
    """
    try:
        import asyncio
        from core.extern import load_ocr_module
        # Consensus needs extra OCR providers (extraction_pipeline.py).
        # It is not part of the bundled helpers, so this raises ImportError
        # and the caller falls back to the single-provider result.
        _ep_mod = load_ocr_module("extraction_pipeline")
        ExtractionConfig = _ep_mod.ExtractionConfig
        ExtractionPipeline = _ep_mod.ExtractionPipeline
        skill_consensus = _ep_mod.extract_with_consensus

        pipeline_config = ExtractionConfig(
            ocr_provider=providers[0] if providers else "gemini",
            llm_provider="claude-cli",
            german_mode=True,
        )
        pipeline = ExtractionPipeline(pipeline_config)

        confidence_threshold = config.ocr.get("low_confidence_threshold", 0.85)

        primary_result, consensus_result = asyncio.run(
            skill_consensus(pipeline, str(image_path), confidence_threshold)
        )

        output: dict[str, Any] = {
            "success": primary_result.success,
            "confidence": primary_result.confidence,
            "extraction": primary_result.to_dict(),
            "provider": "consensus",
        }

        if consensus_result:
            output["consensus"] = consensus_result.to_dict()
            output["has_consensus"] = consensus_result.has_consensus
            output["disagreements"] = consensus_result.disagreements

        return output

    except Exception as exc:
        logger.error("Consensus extraction failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


def merge_extractions(
    result_a: dict[str, Any],
    result_b: dict[str, Any],
) -> dict[str, Any]:
    """Merge two extraction results, taking highest-confidence fields.

    For each field, compares field_confidence values from both results
    and picks the one with higher confidence. Logs disagreements as flags.

    Args:
        result_a: First extraction result dict.
        result_b: Second extraction result dict.

    Returns:
        Merged extraction dict with best-of-both fields.
    """
    if not result_a.get("success"):
        return result_b
    if not result_b.get("success"):
        return result_a

    ext_a = result_a.get("extraction", {})
    ext_b = result_b.get("extraction", {})
    conf_a = result_a.get("confidence", 0)
    conf_b = result_b.get("confidence", 0)

    # Start with the higher-confidence result as base
    if conf_b > conf_a:
        merged = dict(ext_b)
        secondary = ext_a
    else:
        merged = dict(ext_a)
        secondary = ext_b

    disagreements: list[str] = []

    # Fill missing fields from secondary
    fill_fields = [
        "absender", "person", "aktenzeichen", "betrag", "iban",
        "datum", "faellig_am", "document_type", "verwendungszweck",
    ]

    for field in fill_fields:
        primary_val = merged.get(field)
        secondary_val = secondary.get(field)

        if not primary_val and secondary_val:
            merged[field] = secondary_val
        elif primary_val and secondary_val and primary_val != secondary_val:
            disagreements.append(
                f"{field}: '{primary_val}' vs '{secondary_val}'"
            )

    # Log disagreements
    if disagreements:
        logger.warning("Extraction merge disagreements: %s", disagreements)

    return {
        "success": True,
        "confidence": max(conf_a, conf_b),
        "extraction": merged,
        "provider": f"merged({result_a.get('provider', '?')}+{result_b.get('provider', '?')})",
        "disagreements": disagreements,
    }


def apply_fallback_strategy(
    page_data: dict[str, Any],
    image_path: Path,
    config: BriefpostConfig,
) -> dict[str, Any]:
    """Apply the appropriate fallback strategy for a page.

    Decision tree:
    1. If NEEDS_VISION flagged -> consensus extraction
    2. If eskalation >= 4 -> extended thinking (court documents)
    3. If confidence < extended_thinking_confidence_threshold -> extended thinking
    4. If confidence < low_confidence_threshold -> Gemini Pro
    5. Otherwise -> return original data unchanged

    Args:
        page_data: Existing page JSON data.
        image_path: Path to the page image.
        config: Pipeline configuration.

    Returns:
        Updated page data dict (may be unchanged if no fallback needed).
    """
    ocr_config = config.ocr
    flags = page_data.get("flags", [])
    confidence = page_data.get("gemini_extraction", {}).get("confidence", 0)
    gemini_text = page_data.get("ocr_text", "")

    low_threshold = ocr_config.get("low_confidence_threshold", 0.85)
    enable_opus = ocr_config.get("enable_opus_fallback", True)
    enable_consensus = ocr_config.get("enable_consensus", True)
    enable_thinking = ocr_config.get("enable_extended_thinking", True)

    # Strategy 1: Consensus for NEEDS_VISION pages
    if enable_consensus and Flag.NEEDS_VISION.value in flags:
        logger.info("Applying consensus extraction for %s", page_data.get("page_id"))
        result = extract_with_consensus(image_path, ["gemini"], config)
        if result.get("success"):
            merged = merge_extractions(
                {"success": True, "confidence": confidence, "extraction": page_data.get("gemini_extraction", {})},
                result,
            )
            extraction = merged.get("extraction", page_data.get("gemini_extraction", {}))
            page_data["gemini_extraction"] = extraction
            return page_data
        return page_data

    # Strategy 2: Extended thinking for court documents
    eskalation = page_data.get("gemini_extraction", {}).get("eskalation", 0)
    gerichtlich_threshold = config.priority.get("gerichtlich_threshold", 4)
    if enable_thinking and eskalation >= gerichtlich_threshold:
        logger.info("Applying extended thinking for %s (eskalation=%d)", page_data.get("page_id"), eskalation)
        result = extract_with_extended_thinking(image_path, config)
        if result.get("success"):
            merged = merge_extractions(
                {"success": True, "confidence": confidence, "extraction": page_data.get("gemini_extraction", {})},
                result,
            )
            extraction = merged.get("extraction", page_data.get("gemini_extraction", {}))
            page_data["gemini_extraction"] = extraction
            return page_data
        return page_data

    # Strategy 2b: Extended thinking for very low confidence pages
    extended_conf_threshold = ocr_config.get("extended_thinking_confidence_threshold", 0.70)
    if enable_thinking and confidence < extended_conf_threshold and confidence > 0:
        logger.info(
            "Applying extended thinking for %s (confidence=%.2f < %.2f)",
            page_data.get("page_id"), confidence, extended_conf_threshold,
        )
        result = extract_with_extended_thinking(image_path, config)
        if result.get("success"):
            merged = merge_extractions(
                {"success": True, "confidence": confidence, "extraction": page_data.get("gemini_extraction", {})},
                result,
            )
            extraction = merged.get("extraction", page_data.get("gemini_extraction", {}))
            page_data["gemini_extraction"] = extraction
            return page_data
        # Fall through to Strategy 3 if extended thinking fails

    # Strategy 3: Gemini Pro for low confidence (dual-provider consensus)
    if enable_opus and confidence < low_threshold:
        logger.info(
            "Applying Gemini Pro for %s (confidence=%.2f < %.2f)",
            page_data.get("page_id"), confidence, low_threshold,
        )
        result = extract_with_gemini_pro(image_path, gemini_text, config)
        if result.get("success"):
            merged = merge_extractions(
                {"success": True, "confidence": confidence, "extraction": page_data.get("gemini_extraction", {})},
                result,
            )
            extraction = merged.get("extraction", page_data.get("gemini_extraction", {}))
            # Add disagreement flags from dual-provider comparison
            disagreements = merged.get("disagreements", [])
            for disagreement in disagreements:
                field = disagreement.split(":")[0].strip()
                flag_val = f"{Flag.CONSENSUS_DISAGREEMENT.value}:{field}"
                if "flags" not in page_data:
                    page_data["flags"] = []
                page_data["flags"].append(flag_val)
            # Boost confidence by 0.10 if providers agree (no disagreements)
            if not disagreements:
                conf = merged.get("confidence", confidence)
                extraction["confidence"] = min(conf + 0.10, 1.0)
            page_data["gemini_extraction"] = extraction
            return page_data
        return page_data

    # No fallback needed
    return page_data
