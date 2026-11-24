"""Configuration loader for briefpost-management skill.

Loads config/defaults.yaml as base configuration and supports
per-invocation overrides via kwargs or a custom YAML file.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULTS_PATH = Path(__file__).parent.parent.parent / "config" / "defaults.yaml"

# CQ-033m: Sentinel for distinguishing missing keys from explicit None values
_MISSING = object()


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict.

    CQ-032m: Lists are replaced (not appended) to match YAML override semantics.
    Nested dicts are recursively merged.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class BriefpostConfig:
    """Central configuration for the briefpost-management pipeline.

    Loads defaults.yaml as base, then applies any overrides.
    Access config sections as properties: config.ocr, config.matching, etc.
    """

    def __init__(
        self,
        project_dir: str | Path | None = None,
        defaults_path: str | Path | None = None,
        **overrides: Any,
    ) -> None:
        """Initialize configuration.

        Args:
            project_dir: Path to the project data directory. Required for pipeline execution.
            defaults_path: Path to defaults.yaml. Uses built-in default if not specified.
            **overrides: Section-level overrides, e.g. ocr={"model": "gemini-3.8-flash"}.
        """
        base_path = Path(defaults_path) if defaults_path else DEFAULTS_PATH

        if base_path.exists():
            with open(base_path, "r", encoding="utf-8") as f:
                self._data: dict[str, Any] = yaml.safe_load(f) or {}
        else:
            logger.warning("Defaults file not found at %s, using empty config", base_path)
            self._data = {}

        # Apply overrides
        for section, values in overrides.items():
            if isinstance(values, dict) and section in self._data:
                self._data[section] = deep_merge(self._data[section], values)
            else:
                self._data[section] = values

        # Set project_dir (overrides any value in YAML)
        if project_dir is not None:
            self._data.setdefault("pipeline", {})["project_dir"] = str(project_dir)

    # --- Section properties ---

    @property
    def pipeline(self) -> dict[str, Any]:
        return self._data.get("pipeline", {})

    @property
    def ocr(self) -> dict[str, Any]:
        return self._data.get("ocr", {})

    @property
    def persons(self) -> dict[str, Any]:
        return self._data.get("persons", {})

    @property
    def grouping(self) -> dict[str, Any]:
        return self._data.get("grouping", {})

    @property
    def matching(self) -> dict[str, Any]:
        return self._data.get("matching", {})

    @property
    def reconciliation(self) -> dict[str, Any]:
        return self._data.get("reconciliation", {})

    @property
    def priority(self) -> dict[str, Any]:
        return self._data.get("priority", {})

    @property
    def bank(self) -> dict[str, Any]:
        return self._data.get("bank", {})

    @property
    def output(self) -> dict[str, Any]:
        return self._data.get("output", {})

    @property
    def qc(self) -> dict[str, Any]:
        return self._data.get("qc", {})

    @property
    def project_dir(self) -> Path | None:
        """Resolved project directory path, or None if not set."""
        raw = self.pipeline.get("project_dir")
        if raw is None:
            return None
        return Path(raw).expanduser().resolve()

    # --- Access helpers ---

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Get a nested value using dot notation.

        CQ-033m: Uses sentinel to distinguish missing keys from explicit None.

        Example: config.get("ocr.confidence_threshold", 0.50)
        """
        keys = dotted_key.split(".")
        current: Any = self._data
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key, _MISSING)
                if current is _MISSING:
                    return default
            else:
                return default
        return current

    def to_dict(self) -> dict[str, Any]:
        """Return the full configuration as a plain dict."""
        return copy.deepcopy(self._data)

    # --- Validation ---

    def validate(self) -> list[str]:
        """Validate the configuration. Returns list of error messages (empty = valid)."""
        errors: list[str] = []

        if self.project_dir is None:
            errors.append("pipeline.project_dir must be set")

        if self.project_dir is not None and not self.project_dir.exists():
            errors.append(f"pipeline.project_dir does not exist: {self.project_dir}")

        confidence = self.ocr.get("confidence_threshold")
        if confidence is not None and (confidence < 0 or confidence > 1):
            errors.append(f"ocr.confidence_threshold must be 0.0-1.0, got {confidence}")

        low_conf = self.ocr.get("low_confidence_threshold")
        if low_conf is not None and (low_conf < 0 or low_conf > 1):
            errors.append(f"ocr.low_confidence_threshold must be 0.0-1.0, got {low_conf}")

        # CQ-016: Expanded validation for 10+ critical keys
        # Grouping thresholds
        burst = self.grouping.get("burst_threshold_seconds")
        if burst is not None and burst <= 0:
            errors.append(f"grouping.burst_threshold_seconds must be > 0, got {burst}")

        doc_thresh = self.grouping.get("doc_threshold_seconds")
        if doc_thresh is not None and doc_thresh <= 0:
            errors.append(f"grouping.doc_threshold_seconds must be > 0, got {doc_thresh}")

        # Matching
        split_thresh = self.matching.get("split_threshold")
        if split_thresh is not None and split_thresh < 1:
            errors.append(f"matching.split_threshold must be >= 1, got {split_thresh}")

        creditor_conf = self.matching.get("creditor_confidence")
        if creditor_conf is not None and (creditor_conf < 0 or creditor_conf > 1):
            errors.append(f"matching.creditor_confidence must be 0.0-1.0, got {creditor_conf}")

        # Reconciliation tolerances
        amount_tol = self.reconciliation.get("amount_tolerance_eur")
        if amount_tol is not None and amount_tol < 0:
            errors.append(f"reconciliation.amount_tolerance_eur must be >= 0, got {amount_tol}")

        date_tol = self.reconciliation.get("date_tolerance_days")
        if date_tol is not None and date_tol < 0:
            errors.append(f"reconciliation.date_tolerance_days must be >= 0, got {date_tol}")

        ref_conf = self.reconciliation.get("reference_confidence")
        if ref_conf is not None and (ref_conf < 0 or ref_conf > 1):
            errors.append(f"reconciliation.reference_confidence must be 0.0-1.0, got {ref_conf}")

        bezahlt = self.reconciliation.get("bezahlt_threshold_pct")
        if bezahlt is not None and (bezahlt < 0 or bezahlt > 100):
            errors.append(f"reconciliation.bezahlt_threshold_pct must be 0-100, got {bezahlt}")

        # QC thresholds
        pipeline_conf_thresh = self.qc.get("pipeline_confidence_threshold")
        if pipeline_conf_thresh is not None and (pipeline_conf_thresh < 0 or pipeline_conf_thresh > 1):
            errors.append(f"qc.pipeline_confidence_threshold must be 0.0-1.0, got {pipeline_conf_thresh}")

        return errors

    # --- Factory methods ---

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, project_dir: str | Path | None = None) -> BriefpostConfig:
        """Create config from a custom YAML file (merged with defaults).

        Args:
            yaml_path: Path to custom YAML with overrides.
            project_dir: Optional project directory override.
        """
        custom_path = Path(yaml_path)
        if not custom_path.exists():
            raise FileNotFoundError(f"Config file not found: {custom_path}")

        with open(custom_path, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}

        return cls(project_dir=project_dir, **overrides)

    @classmethod
    def from_project_dir(
        cls,
        project_dir: str | Path,
        overrides: Optional[dict[str, Any]] = None,
    ) -> BriefpostConfig:
        """Create config for a specific project directory.

        Looks for a local config.yaml in the project directory. If found,
        merges it with defaults. Otherwise uses defaults only.

        Args:
            project_dir: Path to project data directory.
            overrides: Optional additional overrides.
        """
        project_path = Path(project_dir).expanduser().resolve()
        local_config = project_path / "config.yaml"

        if local_config.exists():
            config = cls.from_yaml(local_config, project_dir=project_path)
        else:
            config = cls(project_dir=project_path)

        if overrides:
            for section, values in overrides.items():
                if isinstance(values, dict) and section in config._data:
                    config._data[section] = deep_merge(config._data[section], values)
                else:
                    config._data[section] = values

        return config
