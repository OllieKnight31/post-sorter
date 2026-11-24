"""Phase A: Two-Tier Burst Detection + Auto-Rotation for Briefpost photos.

Reads photos (JPG, JPEG, PNG, HEIC) from raw-images/unsorted/, groups them into documents using
timestamp-based burst detection, auto-rotates via EXIF orientation, and
copies rotated photos to grouped/doc-XXX/page-N.jpg with manifest.json.

Algorithm:
  Tier 1: burst_threshold (default 7s) - consecutive photos within threshold = same page burst
  Tier 2: doc_threshold (default 45s) - bursts within threshold = same document
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from core.config import BriefpostConfig
from core.constants import Flag
from core.io import atomic_write_json
from core.interfaces import DocumentGrouper
from core.models import BatchResult, DocumentGroup

logger = logging.getLogger(__name__)

EXIF_ORIENTATION_TAG = 0x0112


PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
HEIC_SUFFIXES = {".heic", ".heif"}

EXIF_IFD_TAG = 0x8769
EXIF_DATETIME_ORIGINAL_TAG = 0x9003
EXIF_DATETIME_TAG = 0x0132

# Timestamp patterns found in camera file names, most specific first.
_FILENAME_TIMESTAMP_PATTERNS = (
    # IMG_20250115_143022.jpg, PXL_20250115_143022123.jpg, 20250115_143022.jpg
    (re.compile(r"(?<!\d)(\d{8})[_-](\d{6})"), "%Y%m%d%H%M%S"),
    # 2025-01-15 14.30.22.jpg (cloud camera uploads), 2025-01-15_14-30-22.jpg
    (re.compile(r"(\d{4}-\d{2}-\d{2})[ _T](\d{2}[.\-:]\d{2}[.\-:]\d{2})"), None),
)


def parse_timestamp(filename: str) -> datetime:
    """Parse a timestamp from a camera file name.

    Understands IMG_YYYYMMDD_HHMMSS.jpg and similar patterns.

    Raises:
        ValueError: If the file name carries no timestamp.
    """
    stem = Path(filename).stem
    for pattern, fmt in _FILENAME_TIMESTAMP_PATTERNS:
        match = pattern.search(stem)
        if not match:
            continue
        try:
            if fmt:
                return datetime.strptime(match.group(1) + match.group(2), fmt)
            digits = re.sub(r"\D", "", match.group(2))
            return datetime.strptime(f"{match.group(1)} {digits}", "%Y-%m-%d %H%M%S")
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp from filename: {filename}")


def photo_timestamp(image_path: Path) -> datetime:
    """Best available capture time for a photo.

    Order: timestamp in the file name, EXIF capture time, file creation or
    modification time. Phones that name photos IMG_1234 are covered by the
    EXIF and file-time fallbacks.

    Args:
        image_path: Path to the photo.

    Returns:
        Capture time as a naive datetime.
    """
    try:
        return parse_timestamp(image_path.name)
    except ValueError:
        pass

    if image_path.suffix.lower() not in HEIC_SUFFIXES:
        try:
            with Image.open(image_path) as img:
                exif = img.getexif()
                raw = None
                try:
                    raw = exif.get_ifd(EXIF_IFD_TAG).get(EXIF_DATETIME_ORIGINAL_TAG)
                except Exception:
                    raw = None
                raw = raw or exif.get(EXIF_DATETIME_TAG)
            if raw:
                return datetime.strptime(str(raw).strip()[:19], "%Y:%m:%d %H:%M:%S")
        except Exception:
            pass  # Fall through to file times

    stat = image_path.stat()
    return datetime.fromtimestamp(getattr(stat, "st_birthtime", None) or stat.st_mtime)


def _open_image(image_path: Path) -> Image.Image:
    """Open a photo, converting HEIC/HEIF (iPhone default) to JPEG first.

    HEIC is converted with the macOS "sips" tool, which ships with the
    system. The converted file is temporary and removed after loading.

    Raises:
        OSError: If the image cannot be opened or converted.
    """
    if image_path.suffix.lower() not in HEIC_SUFFIXES:
        img = Image.open(image_path)
        img.load()  # Force full decode to catch truncated/corrupt files early
        return img

    sips = shutil.which("sips")
    if not sips:
        raise OSError(
            f"{image_path.name} is a HEIC photo and the macOS 'sips' tool was not found. "
            "Export the photos as JPEG and try again."
        )
    with tempfile.TemporaryDirectory(prefix="briefpost-heic-") as tmp:
        converted = Path(tmp) / (image_path.stem + ".jpg")
        proc = subprocess.run(
            [sips, "-s", "format", "jpeg", str(image_path), "--out", str(converted)],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0 or not converted.exists():
            raise OSError(f"sips could not convert {image_path.name}: {proc.stderr.strip()[:200]}")
        img = Image.open(converted)
        img.load()  # Fully decode before the temporary file is removed
        return img


def auto_rotate(image_path: Path) -> Image.Image:
    """Apply EXIF orientation rotation to an image.

    Args:
        image_path: Path to the image file.

    Returns:
        Rotated PIL Image.

    Raises:
        OSError: If the image cannot be opened (corrupt or unreadable).
    """
    try:
        img = _open_image(image_path)
    except (OSError, Image.DecompressionBombError, subprocess.SubprocessError) as exc:
        logger.error("Cannot open image %s: %s", image_path, exc)
        raise OSError(f"Corrupt or unreadable image: {image_path} ({exc})") from exc

    try:
        exif = img.getexif()
        orientation = exif.get(EXIF_ORIENTATION_TAG, 1)
    except Exception:
        logger.warning("Cannot read EXIF from %s, skipping rotation", image_path)
        return img

    if orientation == 3:
        img = img.rotate(180, expand=True)
    elif orientation == 6:
        img = img.rotate(270, expand=True)
    elif orientation == 8:
        img = img.rotate(90, expand=True)

    return img


class PhotoGrouper(DocumentGrouper):
    """Concrete implementation of Phase A grouping."""

    def __init__(self, config: BriefpostConfig) -> None:
        """Initialize grouper with configuration.

        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.burst_threshold = config.grouping.get("burst_threshold_seconds", 7)
        self.doc_threshold = config.grouping.get("doc_threshold_seconds", 45)
        self.oversized_threshold = config.grouping.get("oversized_threshold", 15)
        self.confidence_modifiers = config.grouping.get("confidence_modifiers", {})

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
            config: Grouping configuration (overrides instance config if provided).

        Returns:
            List of DocumentGroup records describing each group.
        """
        burst_threshold = config.get("burst_threshold_seconds", self.burst_threshold)
        doc_threshold = config.get("doc_threshold_seconds", self.doc_threshold)

        logger.info(
            "Starting two-tier burst detection (burst=%ds, doc=%ds)",
            burst_threshold, doc_threshold,
        )

        # 1. Read all photos and sort them by capture time
        if not image_dir.is_dir():
            raise ValueError(
                f"Photo folder not found: {image_dir}. "
                "Put your photos in raw-images/unsorted/ inside the project folder."
            )
        photo_files = [
            f for f in image_dir.iterdir()
            if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in PHOTO_SUFFIXES
        ]
        photo_times = {f: photo_timestamp(f) for f in photo_files}
        jpg_files = sorted(photo_files, key=lambda f: (photo_times[f], f.name))
        total_photos = len(jpg_files)
        logger.info("Found %d photos", total_photos)

        if total_photos == 0:
            raise ValueError(
                f"No photos (JPG, JPEG, PNG, HEIC) found in {image_dir}"
            )

        # RS-032: Warn on large photo sets that may take significant time/memory
        if total_photos > 10000:
            logger.warning(
                "Very large photo set (%d files). "
                "Grouping will proceed but may use significant memory and time.",
                total_photos,
            )
        elif total_photos > 5000:
            logger.info("Large photo set (%d files), grouping may take a while", total_photos)

        timestamps = [photo_times[f] for f in jpg_files]

        # 2. Tier 1: Identify page bursts (gaps <= burst_threshold)
        bursts: list[list[int]] = []
        current_burst: list[int] = [0]

        for i in range(1, total_photos):
            gap = (timestamps[i] - timestamps[i - 1]).total_seconds()
            if gap <= burst_threshold:
                current_burst.append(i)
            else:
                bursts.append(current_burst)
                current_burst = [i]
        bursts.append(current_burst)

        logger.info("Tier 1: %d page bursts identified", len(bursts))

        # 3. Tier 2: Group bursts into documents (inter-burst gaps <= doc_threshold)
        documents: list[list[int]] = []
        current_doc_bursts: list[list[int]] = [bursts[0]]

        for i in range(1, len(bursts)):
            last_photo_current_doc = current_doc_bursts[-1][-1]
            first_photo_this_burst = bursts[i][0]
            gap = (timestamps[first_photo_this_burst] - timestamps[last_photo_current_doc]).total_seconds()

            if gap <= doc_threshold:
                current_doc_bursts.append(bursts[i])
            else:
                doc_indices: list[int] = []
                for burst in current_doc_bursts:
                    doc_indices.extend(burst)
                documents.append(doc_indices)
                current_doc_bursts = [bursts[i]]

        # Append last document
        doc_indices = []
        for burst in current_doc_bursts:
            doc_indices.extend(burst)
        documents.append(doc_indices)

        total_documents = len(documents)
        logger.info("Tier 2: %d document groups formed", total_documents)

        # 4. Calculate inter-document gaps
        doc_gap_before: list[Optional[float]] = [None] * total_documents
        doc_gap_after: list[Optional[float]] = [None] * total_documents

        for d in range(total_documents):
            if d > 0:
                prev_last = documents[d - 1][-1]
                curr_first = documents[d][0]
                doc_gap_before[d] = (timestamps[curr_first] - timestamps[prev_last]).total_seconds()
            if d < total_documents - 1:
                curr_last = documents[d][-1]
                next_first = documents[d + 1][0]
                doc_gap_after[d] = (timestamps[next_first] - timestamps[curr_last]).total_seconds()

        # 5. Create grouped/ directory structure, rotate and copy photos
        # Write to temp directory first, then atomically swap on success
        tmp_dir = Path(str(output_dir) + "-tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        result_groups: list[DocumentGroup] = []
        all_source_files: set[str] = set()

        for d_idx, doc_indices in enumerate(documents):
            doc_id = f"doc-{d_idx + 1:03d}"
            doc_dir = tmp_dir / doc_id
            doc_dir.mkdir()

            photo_count = len(doc_indices)
            burst_count = self._count_bursts_in_doc(doc_indices, timestamps, burst_threshold)

            pages: list[str] = []
            doc_timestamps: list[str] = []

            for page_num, photo_idx in enumerate(doc_indices, start=1):
                src_file = jpg_files[photo_idx]
                all_source_files.add(src_file.name)
                page_name = f"page-{page_num}.jpg"
                dest_path = doc_dir / page_name

                rotated = auto_rotate(src_file)
                if rotated.mode not in ("RGB", "L"):
                    rotated = rotated.convert("RGB")  # PNG with alpha, CMYK, palette
                rotated.save(dest_path, "JPEG", quality=95)
                rotated.close()

                pages.append(page_name)
                doc_timestamps.append(timestamps[photo_idx].strftime("%Y-%m-%dT%H:%M:%S"))

            confidence = self._compute_confidence(
                photo_count, burst_count,
                doc_gap_before[d_idx], doc_gap_after[d_idx],
            )
            flags = self._compute_flags(
                photo_count,
                doc_gap_before[d_idx], doc_gap_after[d_idx],
            )

            group = DocumentGroup(
                doc_id=doc_id,
                pages=pages,
                timestamps=doc_timestamps,
                bursts=burst_count,
                gap_before_seconds=round(doc_gap_before[d_idx], 1) if doc_gap_before[d_idx] is not None else None,
                gap_after_seconds=round(doc_gap_after[d_idx], 1) if doc_gap_after[d_idx] is not None else None,
                confidence=round(confidence, 2),
                flags=flags,
            )
            result_groups.append(group)

            if (d_idx + 1) % 10 == 0 or d_idx == total_documents - 1:
                logger.info("Processed %d/%d documents", d_idx + 1, total_documents)

        # 6. Build and write manifest
        page_counts = [len(doc) for doc in documents]
        manifest = {
            "created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "algorithm": "two-tier-burst-detection",
            "burst_threshold_seconds": burst_threshold,
            "doc_threshold_seconds": doc_threshold,
            "total_photos": total_photos,
            "total_documents": total_documents,
            "groups": [g.to_dict() for g in result_groups],
            "warnings": self._collect_warnings(result_groups),
            "statistics": {
                "min_pages": min(page_counts),
                "max_pages": max(page_counts),
                "median_pages": round(statistics.median(page_counts), 1),
                "avg_pages": round(statistics.mean(page_counts), 1),
                "oversized_count": sum(1 for c in page_counts if c > self.oversized_threshold),
                "singleton_count": sum(1 for c in page_counts if c == 1),
            },
        }

        manifest_path = tmp_dir / "manifest.json"
        atomic_write_json(manifest_path, manifest)

        logger.info("Manifest written: %s", manifest_path)

        # 7. Self-verification (against tmp_dir before swap)
        self._verify(tmp_dir, manifest, total_photos, all_source_files)

        # 8. Atomic swap: replace output_dir with tmp_dir
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.rename(str(tmp_dir), str(output_dir))

        logger.info(
            "Grouping complete: %d photos -> %d documents",
            total_photos, total_documents,
        )

        return result_groups

    def _count_bursts_in_doc(
        self,
        doc_indices: list[int],
        timestamps: list[datetime],
        burst_threshold: float,
    ) -> int:
        """Count how many page bursts exist within a document's photo indices."""
        if len(doc_indices) <= 1:
            return 1
        burst_count = 1
        for j in range(1, len(doc_indices)):
            gap = (timestamps[doc_indices[j]] - timestamps[doc_indices[j - 1]]).total_seconds()
            if gap > burst_threshold:
                burst_count += 1
        return burst_count

    # Default confidence modifiers used when config omits a key.
    # Centralized here so hardcoded fallbacks are explicit and documented.
    _DEFAULT_MODIFIERS: dict[str, float] = {
        "oversized": 0.30,
        "large": 0.50,
        "ambiguous_boundary": 0.80,
        "singleton": 0.60,
        "ideal": 1.10,
    }

    def _compute_confidence(
        self,
        photo_count: int,
        burst_count: int,
        gap_before: Optional[float],
        gap_after: Optional[float],
    ) -> float:
        """Compute confidence score for a document group."""
        confidence = 1.0
        modifiers = self.confidence_modifiers

        def _mod(key: str) -> float:
            return modifiers.get(key, self._DEFAULT_MODIFIERS[key])

        if photo_count > 20:
            confidence *= _mod("oversized")
        elif photo_count > self.oversized_threshold:
            confidence *= _mod("large")

        if gap_before is not None and 30 <= gap_before <= 50:
            confidence *= _mod("ambiguous_boundary")
        if gap_after is not None and 30 <= gap_after <= 50:
            confidence *= _mod("ambiguous_boundary")

        if photo_count == 1:
            confidence *= _mod("singleton")

        if 3 <= photo_count <= 10 and 2 <= burst_count <= 4:
            confidence *= _mod("ideal")

        return min(confidence, 1.0)

    def _compute_flags(
        self,
        photo_count: int,
        gap_before: Optional[float],
        gap_after: Optional[float],
    ) -> list[str]:
        """Determine flags for a document group."""
        flags: list[str] = []
        if photo_count > self.oversized_threshold:
            flags.append(Flag.OVERSIZED_GROUP.value)
        if photo_count == 1:
            flags.append(Flag.SINGLETON.value)
        if gap_before is not None and 40 <= gap_before <= 60:
            flags.append(Flag.AMBIGUOUS_BOUNDARY.value)
        if gap_after is not None and 40 <= gap_after <= 60:
            if Flag.AMBIGUOUS_BOUNDARY.value not in flags:
                flags.append(Flag.AMBIGUOUS_BOUNDARY.value)
        return flags

    def _collect_warnings(self, groups: list[DocumentGroup]) -> list[dict[str, str]]:
        """Collect warnings from groups."""
        warnings: list[dict[str, str]] = []
        for group in groups:
            if Flag.OVERSIZED_GROUP.value in group.flags:
                warnings.append({
                    "doc_id": group.doc_id,
                    "warning": f"{len(group.pages)} pages - OVERSIZED_GROUP, likely multi-doc",
                })
            if Flag.SINGLETON.value in group.flags:
                warnings.append({
                    "doc_id": group.doc_id,
                    "warning": "1 page - SINGLETON",
                })
        return warnings

    def _verify(
        self,
        output_dir: Path,
        manifest: dict[str, Any],
        total_photos: int,
        all_source_files: set[str],
    ) -> None:
        """Run self-verification checks."""
        grouped_count = 0
        for doc_dir in sorted(output_dir.iterdir()):
            if doc_dir.is_dir():
                grouped_count += len(list(doc_dir.glob("*.jpg")))

        if grouped_count != total_photos:
            logger.error(
                "VERIFICATION FAIL: Photos in grouped/ (%d) != raw count (%d)",
                grouped_count, total_photos,
            )

        manifest_photo_count = sum(len(g["pages"]) for g in manifest["groups"])
        if manifest_photo_count != total_photos:
            logger.error(
                "VERIFICATION FAIL: Photos in manifest (%d) != raw count (%d)",
                manifest_photo_count, total_photos,
            )

        if len(all_source_files) != total_photos:
            logger.error(
                "VERIFICATION FAIL: Unique source files (%d) != raw count (%d)",
                len(all_source_files), total_photos,
            )

        empty_groups = [g["doc_id"] for g in manifest["groups"] if len(g["pages"]) == 0]
        if empty_groups:
            logger.error("VERIFICATION FAIL: %d empty groups found", len(empty_groups))

        logger.info("Self-verification complete")


def run_grouping(project_dir: Path, config: BriefpostConfig) -> BatchResult:
    """Entry point called by pipeline.py.

    Args:
        project_dir: Project data directory.
        config: Pipeline configuration.

    Returns:
        BatchResult describing the outcome.
    """
    result = BatchResult(phase="group")

    image_dir = project_dir / "raw-images" / "unsorted"
    output_dir = project_dir / "grouped"

    grouper = PhotoGrouper(config)
    try:
        groups = grouper.group_photos(image_dir, output_dir, config.grouping)
        result.total = len(groups)
        result.succeeded = len(groups)
    except Exception as exc:
        logger.error("Grouping failed: %s", exc, exc_info=True)
        result.total = 1
        result.failed = 1
        result.errors.append({"phase": "group", "error": str(exc)})

    result.complete()
    return result
