"""Main pipeline orchestrator for briefpost-management.

Runs all phases in sequence with verification gates between each phase.
Supports selective phase execution for re-runs.

Usage:
    from pipeline import run_pipeline
    result = run_pipeline(Path("/path/to/project"))
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core.config import BriefpostConfig
from core.io import atomic_write_json
from core.models import BatchResult, PipelineResult

logger = logging.getLogger(__name__)

# Phase names in execution order
PHASE_ORDER = [
    "group",
    "classify",
    "ocr",
    "verify",
    "validate",
    "postprocess",
    "match_pages",
    "match_vorgaenge",
    "reconcile",
    "cross_intelligence",
    "output",
    "qc",
]

# Expected output directories/files per phase (for verification gates)
# DF-013: Check specific files, not just directories, for phases that modify in-place
# DF-035: Output gate also checks for QR code directory
PHASE_OUTPUTS: dict[str, list[str]] = {
    "group": ["grouped/manifest.json"],
    "classify": ["classification/classification-manifest.json"],
    "ocr": ["ocr-pages/", "ocr-letters/"],
    "verify": ["ocr-letters/"],
    "validate": ["ocr-letters/"],
    "postprocess": ["ocr-pages/"],
    "match_pages": ["letters/manifest.json"],
    "match_vorgaenge": ["reconciliation/vorgaenge.json"],
    "reconcile": ["reconciliation/vorgaenge-reconciled.json"],
    "cross_intelligence": ["reconciliation/vorgaenge-cross-intel.json"],
    "output": ["reconciliation/sheet-id.txt", "qr-codes/"],
    "qc": ["reconciliation/qc-report.json"],
}


def _run_low_confidence_fallback(
    project_dir: Path,
    config: BriefpostConfig,
    postprocess_result: BatchResult,
) -> BatchResult:
    """Auto-invoke fallback strategy for pages below low_confidence_threshold.

    Scans all OCR page JSONs after postprocessing. For any page with
    confidence < low_confidence_threshold, calls apply_fallback_strategy()
    and writes back the improved extraction.

    Args:
        project_dir: Project data directory.
        config: Pipeline configuration.
        postprocess_result: Result from the postprocess phase.

    Returns:
        The original postprocess_result (fallback errors are non-fatal).
    """
    ocr_pages_dir = project_dir / "ocr-pages"
    if not ocr_pages_dir.is_dir():
        return postprocess_result

    low_threshold = config.ocr.get("low_confidence_threshold", 0.85)
    enable_opus = config.ocr.get("enable_opus_fallback", True)

    if not enable_opus:
        return postprocess_result

    json_files = sorted(ocr_pages_dir.glob("doc-*.json"))
    fallback_count = 0

    # CQ-046: Import once outside loop
    try:
        from ocr.vision_fallback import apply_fallback_strategy
    except ImportError:
        logger.warning("vision_fallback module not available, skipping low-confidence fallback")
        return postprocess_result

    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            confidence = data.get("gemini_extraction", {}).get("confidence", 0)
            if confidence >= low_threshold:
                continue

            # Find corresponding image
            source_image = data.get("source_image", "")
            if not source_image or not Path(source_image).exists():
                logger.debug(
                    "Skipping fallback for %s: source image not found",
                    json_path.stem,
                )
                continue

            # CQ-022: apply_fallback_strategy returns (was_updated, data) tuple
            fallback_result = apply_fallback_strategy(
                data, Path(source_image), config,
            )

            # Support both old (dict) and new (tuple) return format
            if isinstance(fallback_result, tuple):
                was_updated, updated = fallback_result
            else:
                updated = fallback_result
                was_updated = updated is not data

            if was_updated:
                data["gemini_extraction"] = updated if isinstance(updated, dict) else data.get("gemini_extraction", {})
                atomic_write_json(json_path, data)
                fallback_count += 1

        except Exception as exc:
            logger.warning(
                "Fallback failed for %s: %s", json_path.stem, exc,
            )

    if fallback_count > 0:
        logger.info(
            "Low-confidence fallback applied to %d/%d pages",
            fallback_count, len(json_files),
        )

    return postprocess_result


def verify_phase_output(project_dir: Path, phase: str) -> bool:
    """Check that expected outputs from a phase exist.

    Args:
        project_dir: Project data directory.
        phase: Phase name to verify.

    Returns:
        True if all expected outputs exist.
    """
    expected = PHASE_OUTPUTS.get(phase, [])
    for path_str in expected:
        full_path = project_dir / path_str
        if path_str.endswith("/"):
            if not full_path.is_dir():
                logger.warning("Phase %s output missing: %s", phase, full_path)
                return False
        else:
            if not full_path.exists():
                logger.warning("Phase %s output missing: %s", phase, full_path)
                return False
    return True


def run_phase(phase: str, project_dir: Path, config: BriefpostConfig) -> BatchResult:
    """Execute a single pipeline phase.

    Dynamically imports and runs the appropriate phase module.

    Args:
        phase: Phase name (from PHASE_ORDER).
        project_dir: Project data directory.
        config: Pipeline configuration.

    Returns:
        BatchResult describing the outcome.
    """
    result = BatchResult(phase=phase)
    logger.info("Starting phase: %s", phase)

    try:
        phase_batch_result = None

        if phase == "group":
            from group.grouper import run_grouping
            phase_batch_result = run_grouping(project_dir, config)

        elif phase == "classify":
            from ocr.multi_image_classifier import run_classification
            phase_batch_result = run_classification(project_dir, config)

        elif phase == "ocr":
            from ocr.orchestrator import run_multi_image_ocr
            phase_batch_result = run_multi_image_ocr(project_dir, config)

        elif phase == "verify":
            from ocr.extraction_verifier import run_verification_pass
            phase_batch_result = run_verification_pass(project_dir, config)

        elif phase == "validate":
            from ocr.semantic_validation_wiring import run_semantic_validation
            phase_batch_result = run_semantic_validation(project_dir, config)

        elif phase == "postprocess":
            from ocr.postprocessor import run_postprocess
            phase_batch_result = run_postprocess(project_dir, config)

            # A3: Auto-invoke fallback for low-confidence pages
            if phase_batch_result and phase_batch_result.failed == 0:
                phase_batch_result = _run_low_confidence_fallback(
                    project_dir, config, phase_batch_result,
                )

        elif phase == "match_pages":
            from match.page_matcher import run_page_matching
            phase_batch_result = run_page_matching(project_dir, config)

        elif phase == "match_vorgaenge":
            from match.vorgang_matcher import run_vorgang_matching
            phase_batch_result = run_vorgang_matching(project_dir, config)

        elif phase == "reconcile":
            from reconcile.bank_reconciler import run_reconciliation
            phase_batch_result = run_reconciliation(project_dir, config)

        elif phase == "cross_intelligence":
            from reconcile.cross_intelligence import run_cross_intelligence
            phase_batch_result = run_cross_intelligence(project_dir, config)

        elif phase == "output":
            from output.sheet_builder import run_sheet_builder
            from output.qr_generator import run_qr_generator
            sb_result = run_sheet_builder(project_dir, config)
            qr_result = run_qr_generator(project_dir, config)
            phase_batch_result = sb_result
            # A sheet failure is reported first; a QR failure only if the sheet was fine
            if sb_result.failed == 0 and isinstance(qr_result, BatchResult) and qr_result.failed > 0:
                phase_batch_result = qr_result

        elif phase == "qc":
            from qc.verifier import run_verification
            phase_batch_result = run_verification(project_dir, config)

        else:
            raise ValueError(f"Unknown phase: {phase}")

        # Use returned BatchResult if available
        if isinstance(phase_batch_result, BatchResult) and phase_batch_result.total > 0:
            result = phase_batch_result
            result.phase = phase
        else:
            result.succeeded = 1
            result.total = 1

    except ImportError as exc:
        logger.error("Phase %s: module not found: %s", phase, exc)
        result.failed = 1
        result.total = 1
        result.errors.append({"phase": phase, "error": f"ImportError: {exc}"})

    except Exception as exc:
        logger.error("Phase %s failed: %s", phase, exc, exc_info=True)
        result.failed = 1
        result.total = 1
        result.errors.append({"phase": phase, "error": str(exc)})

    result.complete()

    # CQ-017: Compute actual phase duration from BatchResult timestamps
    duration = 0.0
    if result.started_at and result.completed_at:
        try:
            started = datetime.fromisoformat(result.started_at)
            completed = datetime.fromisoformat(result.completed_at)
            duration = (completed - started).total_seconds()
        except (ValueError, TypeError):
            pass

    logger.info(
        "Phase %s %s (%.1fs)",
        phase,
        "succeeded" if result.failed == 0 else "FAILED",
        duration,
    )
    return result


def _compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        path: File path.

    Returns:
        Hex digest string, or empty string if file not found.
    """
    if not path.exists():
        return ""
    # CQ-053: Use 64KB buffer for better I/O efficiency
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_run_manifest(
    project_dir: Path,
    pipeline_result: PipelineResult,
    config: Any,
) -> None:
    """Write a pipeline run manifest with config hash, timings, and output hashes.

    Args:
        project_dir: Project data directory.
        pipeline_result: Completed pipeline result.
        config: Pipeline configuration.
    """
    try:
        recon_dir = project_dir / "reconciliation"
        recon_dir.mkdir(exist_ok=True)

        # Config hash
        config_dict = config.to_dict() if hasattr(config, "to_dict") else str(config)
        config_json = json.dumps(config_dict, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()

        # Phase timings
        phase_timings = {}
        for phase_result in pipeline_result.phase_results:
            phase_timings[phase_result.phase] = {
                "started_at": phase_result.started_at,
                "completed_at": phase_result.completed_at,
                "total": phase_result.total,
                "succeeded": phase_result.succeeded,
                "failed": phase_result.failed,
            }

        # Output file hashes
        output_files = [
            "vorgaenge.json",
            "vorgaenge-reconciled.json",
            "vorgaenge-cross-intel.json",
            "reconciliation-report.json",
            "qc-report.json",
        ]
        output_hashes = {}
        for fname in output_files:
            fpath = recon_dir / fname
            if fpath.exists():
                output_hashes[fname] = _compute_file_hash(fpath)

        # Status summary
        total_phases = len(pipeline_result.phase_results)
        succeeded = sum(1 for r in pipeline_result.phase_results if r.failed == 0)

        manifest = {
            "timestamp": datetime.now().isoformat(),
            "config_hash": config_hash,
            "phase_timings": phase_timings,
            "output_hashes": output_hashes,
            "status_summary": {
                "total_phases": total_phases,
                "succeeded": succeeded,
                "failed": total_phases - succeeded,
                "all_succeeded": pipeline_result.all_succeeded,
            },
            "total_duration": {
                "started_at": pipeline_result.started_at,
                "completed_at": pipeline_result.completed_at,
            },
        }

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        manifest_path = recon_dir / f"pipeline-run-{ts}.json"
        atomic_write_json(manifest_path, manifest)
        logger.info("Pipeline run manifest: %s", manifest_path)

    except Exception as exc:
        logger.warning("Failed to write pipeline run manifest: %s", exc)


def _load_checkpoint(project_dir: Path) -> dict[str, Any]:
    """Load pipeline checkpoint data if it exists.

    Args:
        project_dir: Project data directory.

    Returns:
        Checkpoint dict with phase entries, or empty dict.
    """
    checkpoint_path = project_dir / "reconciliation" / "pipeline-checkpoint.json"
    if checkpoint_path.exists():
        try:
            # DF-025: Explicit UTF-8 encoding
            with open(checkpoint_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load checkpoint: %s", exc)
    return {}


def _save_checkpoint(
    project_dir: Path,
    checkpoint: dict[str, Any],
) -> None:
    """Save pipeline checkpoint data.

    Args:
        project_dir: Project data directory.
        checkpoint: Checkpoint dict to persist.
    """
    checkpoint_path = project_dir / "reconciliation" / "pipeline-checkpoint.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(checkpoint_path, checkpoint)


def _get_phase_output_hash(project_dir: Path, phase: str) -> str:
    """Compute a combined hash of all expected outputs for a phase.

    Args:
        project_dir: Project data directory.
        phase: Phase name.

    Returns:
        SHA-256 hex digest, or empty string if no outputs found.
    """
    expected = PHASE_OUTPUTS.get(phase, [])
    h = hashlib.sha256()
    found_any = False
    for path_str in expected:
        full_path = project_dir / path_str
        if path_str.endswith("/"):
            continue  # Skip directory checks for hashing
        if full_path.exists():
            h.update(_compute_file_hash(full_path).encode())
            found_any = True
    return h.hexdigest() if found_any else ""


def run_pipeline(
    project_dir: str | Path,
    config: Optional[BriefpostConfig] = None,
    phases: Optional[list[str]] = None,
    dry_run: bool = False,
    resume: bool = False,
) -> PipelineResult:
    """Run the full pipeline or selected phases.

    Args:
        project_dir: Path to the project data directory.
        config: Pipeline configuration. If None, loads defaults with project_dir.
        phases: List of phase names to run. If None, runs all phases in order.
        dry_run: If True, log what would run without executing.
        resume: If True, skip already-completed phases if checkpoint exists
            and outputs have not changed.

    Returns:
        PipelineResult with per-phase results.
    """
    project_path = Path(project_dir).expanduser().resolve()

    if config is None:
        config = BriefpostConfig.from_project_dir(project_path)

    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")

    # People and model IDs come from the config, never from the code
    from core.persons import configure_persons
    from core.llm_helpers import configure_models
    configure_persons(config)
    configure_models(config)

    pipeline_result = PipelineResult(project_dir=str(project_path))

    # Determine phases to run
    phases_to_run = phases if phases else PHASE_ORDER

    # Validate phase names
    invalid_phases = [p for p in phases_to_run if p not in PHASE_ORDER]
    if invalid_phases:
        raise ValueError(f"Unknown phases: {invalid_phases}. Valid: {PHASE_ORDER}")

    # CQ-045: Validate phase ordering when running subset
    if phases:
        phase_indices = [PHASE_ORDER.index(p) for p in phases_to_run]
        if phase_indices != sorted(phase_indices):
            logger.warning(
                "Phases specified out of order: %s. Expected order: %s",
                phases_to_run,
                [p for p in PHASE_ORDER if p in phases_to_run],
            )

    logger.info("Pipeline starting: %d phases to run", len(phases_to_run))
    logger.info("Project dir: %s", project_path)
    logger.info("Phases: %s", ", ".join(phases_to_run))

    if dry_run:
        logger.info("[DRY RUN] Would execute phases: %s", ", ".join(phases_to_run))
        pipeline_result.complete()
        return pipeline_result

    # CQ-004: Load checkpoint for resume support
    checkpoint = _load_checkpoint(project_path) if resume else {}
    checkpoint_frequency = config.get("pipeline.checkpoint_frequency", 5) or 5
    phases_since_checkpoint = 0

    for phase in phases_to_run:
        # CQ-004: Skip completed phases if resuming and outputs unchanged
        if resume and phase in checkpoint:
            saved_hash = checkpoint[phase].get("output_hash", "")
            current_hash = _get_phase_output_hash(project_path, phase)
            if saved_hash and current_hash == saved_hash:
                logger.info(
                    "Phase %s: skipping (checkpoint valid, outputs unchanged)", phase,
                )
                continue

        phase_result = run_phase(phase, project_path, config)
        pipeline_result.phase_results.append(phase_result)
        phases_since_checkpoint += 1

        # CQ-004: Save checkpoint after completed phases
        if phase_result.failed == 0:
            checkpoint[phase] = {
                "phase": phase,
                "completed_at": phase_result.completed_at,
                "output_hash": _get_phase_output_hash(project_path, phase),
            }
            if phases_since_checkpoint >= checkpoint_frequency:
                _save_checkpoint(project_path, checkpoint)
                phases_since_checkpoint = 0

        if phase_result.failed > 0:
            logger.error(
                "Phase %s FAILED. Stopping pipeline. Errors: %s",
                phase,
                phase_result.errors,
            )
            break

        # Verification gate (hard stop if outputs missing)
        if not verify_phase_output(project_path, phase):
            msg = f"Phase {phase} verification gate FAILED: expected outputs missing"
            logger.error(msg)
            gate_result = BatchResult(phase=f"{phase}_verification")
            gate_result.total = 1
            gate_result.failed = 1
            gate_result.errors.append({"phase": phase, "error": msg})
            gate_result.complete()
            pipeline_result.phase_results.append(gate_result)
            break

    # CQ-004: Save final checkpoint
    if checkpoint:
        _save_checkpoint(project_path, checkpoint)

    pipeline_result.complete()

    # Summary
    total_phases = len(pipeline_result.phase_results)
    succeeded = sum(1 for r in pipeline_result.phase_results if r.failed == 0)
    failed = total_phases - succeeded

    logger.info(
        "Pipeline complete: %d/%d phases succeeded, %d failed",
        succeeded, total_phases, failed,
    )

    # A4: Write pipeline run manifest
    _write_run_manifest(project_path, pipeline_result, config)

    return pipeline_result
