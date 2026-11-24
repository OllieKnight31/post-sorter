"""Atomic file I/O utilities for briefpost pipeline.

Provides atomic_write_json() that writes to a temp file, fsyncs,
then uses os.replace() for POSIX-atomic rename. Prevents partial
writes from corrupting output files during pipeline runs.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def validate_manifest(data: dict, required_keys: list[str], name: str) -> None:
    """Validate that a loaded manifest dict contains all required keys.

    Args:
        data: Loaded manifest dictionary.
        required_keys: List of keys that must be present.
        name: Human-readable manifest name for error messages.

    Raises:
        ValueError: If any required key is missing.
    """
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Manifest '{name}' missing required keys: {missing}")


def atomic_write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """Atomically write JSON data to a file.

    Writes to a temporary file in the same directory, calls fsync,
    then atomically replaces the target file using os.replace().

    Args:
        path: Target file path.
        data: JSON-serializable data.
        indent: JSON indentation level.

    Raises:
        OSError: If the write or replace fails.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=f".{target.stem}_",
            suffix=".tmp",
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = None  # os.fdopen takes ownership
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, str(target))
        tmp_path = None  # Successful replace, no cleanup needed

        # CQ-042: Directory fsync for durability of the directory entry
        dir_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

        logger.debug("Atomic write: %s", target)

    except Exception:
        # Clean up temp file on failure
        if fd is not None:
            os.close(fd)
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
