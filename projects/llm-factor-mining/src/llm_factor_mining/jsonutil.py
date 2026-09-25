"""Canonical JSON used for hashing, ledgers and run artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    """Recursively convert to JSON-native types; NaN and infinities become ``None``."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        items = list(value) if not isinstance(value, (set, frozenset)) else sorted(value, key=str)
        return [json_safe(item) for item in items]
    if hasattr(value, "value") and hasattr(value, "name"):  # Enum
        return json_safe(value.value)
    return str(value)


def canonical_json(value: Any) -> str:
    """Deterministic compact JSON (sorted keys, no NaN) of :func:`json_safe` output."""

    return json.dumps(
        json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_json(value: Any) -> str:
    """SHA-256 hex digest of :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write pretty, key-sorted JSON (NaN as ``null``) with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


__all__ = ["canonical_json", "json_safe", "sha256_json", "sha256_text", "write_json"]
