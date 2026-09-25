"""Deterministic hashing and canonical serialization used for provenance.

Two hashes matter for reproducibility:

* ``hash_bars`` fingerprints the exact canonical rows a tool consumed.  The
  encoding (format ``bars-sha256/v1``) is bit-exact: prices and volumes are
  hashed as little-endian IEEE-754 float64 bytes and dates as int64
  nanoseconds, so the digest does not depend on float formatting or on the
  pandas version.
* ``result_id`` is a short, stable identifier derived from the tool name,
  canonical arguments, as-of date, and ``data_sha256``.  Identical requests
  against identical data therefore receive identical identifiers across runs,
  which lets answers cite ``[r:<result_id>]`` reproducibly.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from quant_marketdata import CANONICAL_COLUMNS


BARS_HASH_FORMAT = "marketdata-agent/bars-sha256/v1"
RESULT_ID_LENGTH = 12
_NUMERIC = ("open", "high", "low", "close", "volume")
_TEXT = ("symbol", "source", "finality")


def json_safe(value: Any) -> Any:
    """Convert common scientific-Python values to strict JSON values.

    Non-finite floats become ``None`` because canonical JSON forbids NaN.
    """

    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None and value == value.normalize():
            return value.date().isoformat()
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        items = sorted(value) if isinstance(value, (set, frozenset)) else list(value)
        return [json_safe(item) for item in items]
    return str(value)


def canonical_json(value: Any) -> str:
    """Serialize ``value`` as sorted, compact, ASCII-only JSON."""

    return json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of UTF-8 ``text``."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return the SHA-256 hex digest of the canonical JSON form of ``value``."""

    return sha256_text(canonical_json(value))


def hash_bars(frame: pd.DataFrame) -> str:
    """Return a bit-exact SHA-256 fingerprint of canonical bar rows.

    Rows are ordered by ``(symbol, date)`` before hashing so the digest is
    independent of the caller's row order but sensitive to any value change.
    """

    missing = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"cannot hash non-canonical bars; missing {missing}")
    rows = frame.loc[:, list(CANONICAL_COLUMNS)].copy()
    rows["date"] = pd.to_datetime(rows["date"])
    rows["symbol"] = rows["symbol"].astype(str)
    rows = rows.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)

    digest = hashlib.sha256()
    digest.update(f"{BARS_HASH_FORMAT}|rows={len(rows)}|".encode("ascii"))
    digest.update(rows["date"].to_numpy(dtype="datetime64[ns]").astype("<i8").tobytes())
    for column in _NUMERIC:
        digest.update(rows[column].to_numpy(dtype="float64").astype("<f8").tobytes())
    for column in _TEXT:
        digest.update(b"\x1e")
        digest.update("\x1f".join(rows[column].astype(str).tolist()).encode("utf-8"))
    return digest.hexdigest()


def result_id(tool: str, args: Mapping[str, Any], as_of: date, data_sha256: str) -> str:
    """Return the short stable identifier cited as ``[r:<result_id>]``."""

    material = {
        "tool": tool,
        "args": dict(args),
        "as_of": as_of.isoformat(),
        "data_sha256": data_sha256,
    }
    return sha256_json(material)[:RESULT_ID_LENGTH]


def significant(value: float, digits: int = 10) -> float | None:
    """Round ``value`` to ``digits`` significant figures (``None`` if not finite)."""

    number = float(value)
    if not math.isfinite(number):
        return None
    return float(f"{number:.{digits}g}")
