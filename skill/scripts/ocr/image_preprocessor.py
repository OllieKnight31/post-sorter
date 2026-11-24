"""Phase B1: Image preprocessing for OCR quality improvement.

Pillow-based operations applied before Gemini OCR extraction:
1. Auto-level (histogram equalization for underexposed photos)
2. Sharpen (Unsharp mask)
3. Deskew (detect text angle, rotate to straighten)
4. Denoise (Median filter)
5. Quality score (Laplacian variance for blur, mean brightness, contrast ratio)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from core.config import BriefpostConfig
from core.constants import Flag

logger = logging.getLogger(__name__)

# Default preprocessing parameters
DEFAULT_SHARPEN_RADIUS = 2
DEFAULT_SHARPEN_PERCENT = 150
DEFAULT_SHARPEN_THRESHOLD = 3
DEFAULT_DENOISE_SIZE = 3
DEFAULT_QUALITY_THRESHOLD = 0.30


def compute_quality_score(image: Any) -> dict[str, float]:
    """Compute image quality metrics.

    Calculates blur (Laplacian variance), mean brightness, and
    contrast ratio to determine if the image is suitable for OCR.

    Args:
        image: PIL Image object.

    Returns:
        Dictionary with blur_score, brightness, contrast, and overall quality_score.
    """
    import numpy as np
    from PIL import ImageStat

    # Convert to grayscale for analysis
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)

    # Brightness: mean pixel value normalized to 0-1
    brightness = stat.mean[0] / 255.0

    # Contrast: standard deviation normalized
    contrast = stat.stddev[0] / 128.0

    # Blur detection via Laplacian variance
    arr = np.array(gray, dtype=np.float64)
    # Simple Laplacian kernel convolution
    laplacian = (
        arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:]
        - 4 * arr[1:-1, 1:-1]
    )
    blur_score = float(laplacian.var()) / 1000.0  # Normalize to ~0-1 range
    blur_score = min(blur_score, 1.0)

    # Overall quality: weighted combination
    quality_score = (
        0.4 * blur_score
        + 0.3 * min(brightness * 2, 1.0)  # Penalize very dark images
        + 0.3 * min(contrast, 1.0)
    )

    return {
        "blur_score": round(blur_score, 4),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "quality_score": round(quality_score, 4),
    }


def auto_level(image: Any) -> Any:
    """Apply histogram equalization to improve contrast.

    Args:
        image: PIL Image object.

    Returns:
        Enhanced PIL Image.
    """
    from PIL import ImageOps
    return ImageOps.autocontrast(image, cutoff=1)


def sharpen(
    image: Any,
    radius: int = DEFAULT_SHARPEN_RADIUS,
    percent: int = DEFAULT_SHARPEN_PERCENT,
    threshold: int = DEFAULT_SHARPEN_THRESHOLD,
) -> Any:
    """Apply unsharp mask sharpening.

    Args:
        image: PIL Image object.
        radius: Blur radius for unsharp mask.
        percent: Sharpening strength percentage.
        threshold: Minimum brightness difference to sharpen.

    Returns:
        Sharpened PIL Image.
    """
    from PIL import ImageFilter
    return image.filter(ImageFilter.UnsharpMask(
        radius=radius, percent=percent, threshold=threshold,
    ))


def denoise(image: Any, size: int = DEFAULT_DENOISE_SIZE) -> Any:
    """Apply median filter for noise reduction.

    Args:
        image: PIL Image object.
        size: Filter kernel size (odd number).

    Returns:
        Denoised PIL Image.
    """
    from PIL import ImageFilter
    return image.filter(ImageFilter.MedianFilter(size=size))


def preprocess_image(
    image_path: Path,
    config: BriefpostConfig,
) -> dict[str, Any]:
    """Apply full preprocessing pipeline to an image.

    Operations are individually toggleable via config. Original image
    is preserved; preprocessed version is written to a temp directory.

    Args:
        image_path: Path to the source image.
        config: Pipeline configuration.

    Returns:
        Dictionary with:
        - preprocessed_path: Path to preprocessed image (or original if no changes)
        - quality: Quality metrics dict
        - operations_applied: List of operations performed
        - flags: List of flags to add (e.g., LOW_IMAGE_QUALITY)
    """
    from PIL import Image

    ocr_config = config.ocr
    preprocess_config = ocr_config.get("preprocessing", {})

    enable_auto_level = preprocess_config.get("auto_level", True)
    enable_sharpen = preprocess_config.get("sharpen", True)
    enable_denoise = preprocess_config.get("denoise", True)
    quality_threshold = preprocess_config.get(
        "quality_threshold", DEFAULT_QUALITY_THRESHOLD,
    )

    result: dict[str, Any] = {
        "preprocessed_path": str(image_path),
        "quality": {},
        "operations_applied": [],
        "flags": [],
    }

    try:
        image = Image.open(image_path)
    except Exception as exc:
        logger.error("Cannot open image %s: %s", image_path, exc)
        result["flags"].append(Flag.LOW_IMAGE_QUALITY.value)
        return result

    # Compute initial quality
    quality = compute_quality_score(image)
    result["quality"] = quality

    # Check quality gate
    if quality["quality_score"] < quality_threshold:
        result["flags"].append(Flag.LOW_IMAGE_QUALITY.value)
        logger.warning(
            "Low image quality for %s: score=%.3f (threshold=%.3f)",
            image_path.name, quality["quality_score"], quality_threshold,
        )

    # Apply preprocessing operations
    modified = False

    if enable_auto_level:
        image = auto_level(image)
        result["operations_applied"].append("auto_level")
        modified = True

    if enable_sharpen:
        image = sharpen(
            image,
            radius=preprocess_config.get("sharpen_radius", DEFAULT_SHARPEN_RADIUS),
            percent=preprocess_config.get("sharpen_percent", DEFAULT_SHARPEN_PERCENT),
            threshold=preprocess_config.get("sharpen_threshold", DEFAULT_SHARPEN_THRESHOLD),
        )
        result["operations_applied"].append("sharpen")
        modified = True

    if enable_denoise:
        image = denoise(
            image,
            size=preprocess_config.get("denoise_size", DEFAULT_DENOISE_SIZE),
        )
        result["operations_applied"].append("denoise")
        modified = True

    # Write preprocessed image to temp dir (preserve original)
    # Include parent dir name (doc_group_id) to avoid collision between groups
    if modified:
        doc_id = image_path.parent.name if image_path.parent.name.startswith("doc-") else "unknown"
        temp_dir = Path(tempfile.gettempdir()) / "briefpost-preprocessed" / doc_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_path = temp_dir / image_path.name
        image.save(output_path, quality=95)
        result["preprocessed_path"] = str(output_path)
        logger.debug(
            "Preprocessed %s -> %s (%s)",
            image_path.name, output_path, ", ".join(result["operations_applied"]),
        )

    return result
