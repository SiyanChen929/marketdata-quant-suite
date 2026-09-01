"""External, checksum-verified Parquet storage for canonical market bars."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

import pandas as pd

try:  # pragma: no cover - all supported deployment targets are POSIX today
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

from .exceptions import CacheIntegrityError, CredentialError, DataContractError
from .schema import (
    CANONICAL_COLUMNS,
    Finality,
    empty_bars,
    normalize_bars,
    normalize_symbols,
    validate_finality,
)


SCHEMA_VERSION = "1.0"
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class MarketDataStore:
    """Year-partitioned canonical store rooted outside the source repository.

    When ``root`` is omitted, the path is read exclusively from
    ``QUANT_DATA_HOME``.  Confirmed and provisional bars have independent roots,
    manifests, and request caches.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        verify_checksums: bool = True,
    ) -> None:
        if root is None:
            configured = os.getenv("QUANT_DATA_HOME", "").strip()
            if not configured:
                raise CredentialError(
                    "QUANT_DATA_HOME is not set; point it to an external data directory"
                )
            root = configured
        self.root = Path(root).expanduser().resolve()
        self.verify_checksums = bool(verify_checksums)

    def write_bars(
        self,
        frame: pd.DataFrame,
        finality: str = "confirmed",
        resolution: str = "D",
    ) -> pd.DataFrame:
        """Atomically upsert canonical bars into the selected finality root."""

        selected = validate_finality(finality)
        selected_resolution = _validate_resolution(resolution)
        bars = normalize_bars(frame, finality=selected)
        if bars.empty:
            return bars
        with _store_lock(self.root):
            return self._write_bars_locked(bars, selected, selected_resolution)

    def _write_bars_locked(
        self,
        bars: pd.DataFrame,
        selected: Finality,
        resolution: str,
    ) -> pd.DataFrame:
        """Write normalized bars while holding the process/file store lock."""

        bars = bars.assign(_year=bars["date"].dt.year.astype("int16"))
        root = self._finality_root(selected)
        manifest = self._read_manifest(selected)
        partitions = dict(manifest.get("partitions", {}))

        for year, incoming in bars.groupby("_year", sort=True):
            path = root / "bars" / f"resolution={resolution}" / f"year={int(year)}" / "bars.parquet"
            clean = incoming.drop(columns="_year").reset_index(drop=True)
            if path.exists():
                self._verify_partition(path, selected, manifest)
                existing = normalize_bars(pd.read_parquet(path), finality=selected)
                clean = _upsert(existing, clean)
            _atomic_parquet(clean, path)
            relative = path.relative_to(root).as_posix()
            partitions[relative] = _partition_metadata(path, clean)

        resolutions = sorted(
            {
                part.split("resolution=", 1)[1].split("/", 1)[0]
                for part in partitions
                if "resolution=" in part
            }
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "finality": selected,
            "columns": list(CANONICAL_COLUMNS),
            "partition_dimensions": ["resolution", "year"],
            "resolutions": resolutions,
            "updated_at": _utc_now(),
            "partitions": dict(sorted(partitions.items())),
        }
        _atomic_json(payload, self._manifest_path(selected))
        return bars.drop(columns="_year").reset_index(drop=True)

    def read_bars(
        self,
        symbols: str | list[str] | tuple[str, ...] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        finality: str = "confirmed",
        resolution: str = "D",
    ) -> pd.DataFrame:
        """Read canonical bars with optional symbol and inclusive date filters."""

        selected = validate_finality(finality)
        selected_resolution = _validate_resolution(resolution)
        wanted = normalize_symbols(symbols)
        start_ts = _date_bound(start, "start")
        end_ts = _date_bound(end, "end")
        if start_ts is not None and end_ts is not None and start_ts > end_ts:
            raise DataContractError("start date is after end date")
        with _store_lock(self.root):
            bars = self._read_bars_locked(wanted, start_ts, end_ts, selected, selected_resolution)
            manifest_path = self._manifest_path(selected)
            if manifest_path.is_file():
                raw_manifest = manifest_path.read_bytes()
                bars.attrs["marketdata_manifest_snapshot"] = json.loads(raw_manifest.decode("utf-8"))
                bars.attrs["marketdata_manifest_sha256"] = hashlib.sha256(raw_manifest).hexdigest()
            bars.attrs["marketdata_resolution"] = selected_resolution
            return bars

    def _read_bars_locked(
        self,
        wanted: list[str] | None,
        start_ts: pd.Timestamp | None,
        end_ts: pd.Timestamp | None,
        selected: Finality,
        resolution: str,
    ) -> pd.DataFrame:
        """Read bars against one internally consistent manifest snapshot."""

        root = self._finality_root(selected)
        paths = self._partition_paths(selected, start_ts, end_ts, resolution)
        if not paths:
            return empty_bars()
        manifest = self._read_manifest(selected)
        if not manifest:
            raise CacheIntegrityError(f"{selected} partitions exist without a manifest")

        frames: list[pd.DataFrame] = []
        for path in paths:
            self._verify_partition(path, selected, manifest)
            frame = pd.read_parquet(path)
            if wanted is not None:
                frame = frame.loc[frame["symbol"].astype(str).str.upper().isin(wanted)]
            if start_ts is not None:
                frame = frame.loc[pd.to_datetime(frame["date"]).ge(start_ts)]
            if end_ts is not None:
                frame = frame.loc[pd.to_datetime(frame["date"]).le(end_ts)]
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return empty_bars()
        return normalize_bars(pd.concat(frames, ignore_index=True), finality=selected)

    def request_hash(self, request: Mapping[str, Any]) -> str:
        """Return a deterministic SHA-256 key for a provider request."""

        encoded = json.dumps(
            _json_safe(dict(request)),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def read_request_cache(
        self,
        request_key: str,
        *,
        finality: str,
    ) -> pd.DataFrame | None:
        """Read and verify one exact-request cache object."""

        selected = validate_finality(finality)
        with _store_lock(self.root):
            data_path, manifest_path = self._request_cache_paths(request_key, selected)
            if not data_path.exists() and not manifest_path.exists():
                return None
            if not data_path.exists() or not manifest_path.exists():
                raise CacheIntegrityError(f"incomplete request cache object: {request_key}")
            manifest = _read_json(manifest_path)
            expected = str(manifest.get("sha256", ""))
            actual = _sha256(data_path)
            if not expected or expected != actual:
                raise CacheIntegrityError(f"request cache checksum mismatch: {request_key}")
            if manifest.get("finality") != selected:
                raise CacheIntegrityError(f"request cache finality mismatch: {request_key}")
            return normalize_bars(pd.read_parquet(data_path), finality=selected)

    def write_request_cache(
        self,
        request_key: str,
        frame: pd.DataFrame,
        *,
        request: Mapping[str, Any],
        finality: str,
    ) -> Path:
        """Atomically persist one exact provider response and safe manifest."""

        selected = validate_finality(finality)
        bars = normalize_bars(frame, finality=selected)
        with _store_lock(self.root):
            data_path, manifest_path = self._request_cache_paths(request_key, selected)
            _atomic_parquet(bars, data_path)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "request_key": request_key,
                "request": _redact_manifest(dict(request)),
                "finality": selected,
                "rows": int(len(bars)),
                "min_date": _date_text(bars["date"].min()) if not bars.empty else None,
                "max_date": _date_text(bars["date"].max()) if not bars.empty else None,
                "sha256": _sha256(data_path),
                "created_at": _utc_now(),
            }
            _atomic_json(payload, manifest_path)
            return data_path

    def read_table_cache(
        self,
        request_key: str,
        *,
        namespace: str,
    ) -> pd.DataFrame | None:
        """Read a checksum-verified exact-request table outside the bar lake."""

        with _store_lock(self.root):
            data_path, manifest_path = self._table_cache_paths(request_key, namespace)
            if not data_path.exists() and not manifest_path.exists():
                return None
            if not data_path.exists() or not manifest_path.exists():
                raise CacheIntegrityError(f"incomplete {namespace} cache object: {request_key}")
            manifest = _read_json(manifest_path)
            expected = str(manifest.get("sha256", ""))
            if not expected or _sha256(data_path) != expected:
                raise CacheIntegrityError(f"{namespace} cache checksum mismatch: {request_key}")
            if manifest.get("namespace") != _safe_namespace(namespace):
                raise CacheIntegrityError(f"{namespace} cache namespace mismatch: {request_key}")
            return pd.read_parquet(data_path)

    def write_table_cache(
        self,
        request_key: str,
        frame: pd.DataFrame,
        *,
        request: Mapping[str, Any],
        namespace: str,
    ) -> Path:
        """Atomically persist an exact-request reference table and provenance."""

        if not isinstance(frame, pd.DataFrame):
            raise DataContractError("cached tables must be pandas DataFrames")
        normalized_namespace = _safe_namespace(namespace)
        with _store_lock(self.root):
            data_path, manifest_path = self._table_cache_paths(
                request_key, normalized_namespace
            )
            _atomic_parquet(frame.reset_index(drop=True), data_path)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "namespace": normalized_namespace,
                "request_key": request_key,
                "request": _redact_manifest(dict(request)),
                "rows": int(len(frame)),
                "columns": [str(column) for column in frame.columns],
                "sha256": _sha256(data_path),
                "created_at": _utc_now(),
            }
            _atomic_json(payload, manifest_path)
            return data_path

    def _finality_root(self, finality: Finality) -> Path:
        if finality == "confirmed":
            return self.root / "lake" / "confirmed"
        return self.root / "staging" / "provisional"

    def _manifest_path(self, finality: Finality) -> Path:
        return self.root / "manifests" / "marketdata" / f"{finality}.json"

    def _read_manifest(self, finality: Finality) -> dict[str, Any]:
        path = self._manifest_path(finality)
        return _read_json(path) if path.exists() else {}

    def _partition_paths(
        self,
        finality: Finality,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
        resolution: str,
    ) -> list[Path]:
        root = self._finality_root(finality) / "bars" / f"resolution={resolution}"
        if start is not None and end is not None:
            paths = [root / f"year={year}" / "bars.parquet" for year in range(start.year, end.year + 1)]
            return [path for path in paths if path.exists()]
        return sorted(root.glob("year=*/bars.parquet"))

    def _verify_partition(
        self,
        path: Path,
        finality: Finality,
        manifest: Mapping[str, Any],
    ) -> None:
        if not self.verify_checksums:
            return
        relative = path.relative_to(self._finality_root(finality)).as_posix()
        metadata = manifest.get("partitions", {}).get(relative)
        if not isinstance(metadata, Mapping):
            raise CacheIntegrityError(f"partition is absent from manifest: {relative}")
        expected = str(metadata.get("sha256", ""))
        if not expected or _sha256(path) != expected:
            raise CacheIntegrityError(f"partition checksum mismatch: {relative}")

    def _request_cache_paths(
        self,
        request_key: str,
        finality: Finality,
    ) -> tuple[Path, Path]:
        if len(request_key) != 64 or any(ch not in "0123456789abcdef" for ch in request_key):
            raise DataContractError("request cache keys must be lowercase SHA-256 hex")
        root = (
            self.root
            / "raw"
            / "marketdata"
            / finality
            / "request-cache"
            / request_key[:2]
        )
        return root / f"{request_key}.parquet", root / f"{request_key}.json"

    def _table_cache_paths(
        self,
        request_key: str,
        namespace: str,
    ) -> tuple[Path, Path]:
        if len(request_key) != 64 or any(ch not in "0123456789abcdef" for ch in request_key):
            raise DataContractError("request cache keys must be lowercase SHA-256 hex")
        root = (
            self.root
            / "raw"
            / "marketdata"
            / _safe_namespace(namespace)
            / "request-cache"
            / request_key[:2]
        )
        return root / f"{request_key}.parquet", root / f"{request_key}.json"


@contextmanager
def _store_lock(root: Path):
    """Serialize partition/manifest snapshots across threads and POSIX processes."""

    key = str(root)
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(key, threading.RLock())
    with thread_lock:
        lock_path = root / ".locks" / "marketdata-store.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _upsert(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    keys = pd.MultiIndex.from_frame(incoming[["date", "symbol"]])
    existing_keys = pd.MultiIndex.from_frame(existing[["date", "symbol"]])
    retained = existing.loc[~existing_keys.isin(keys)]
    if retained.empty:
        return normalize_bars(incoming)
    return normalize_bars(pd.concat([retained, incoming], ignore_index=True))


def _validate_resolution(value: str) -> str:
    normalized = str(value).strip().upper()
    if normalized not in {"D", "W", "M", "H", "1", "5", "15", "30"}:
        raise DataContractError(f"unsupported stored candle resolution: {value!r}")
    return normalized


def _date_bound(value: str | pd.Timestamp | None, label: str) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise DataContractError(f"invalid {label} date: {value!r}")
    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("America/New_York").tz_localize(None)
    date_only = isinstance(value, str) and len(value.strip()) == 10
    if date_only and label == "end":
        return timestamp.normalize() + pd.Timedelta(1, unit="D") - pd.Timedelta(1, unit="ns")
    return timestamp.normalize() if date_only else timestamp


def _partition_metadata(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "sha256": _sha256(path),
        "rows": int(len(frame)),
        "min_date": _date_text(frame["date"].min()),
        "max_date": _date_text(frame["date"].max()),
    }


def _date_text(value: object) -> str:
    return str(pd.Timestamp(value).date())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheIntegrityError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CacheIntegrityError(f"manifest must contain a JSON object: {path}")
    return payload


def _redact_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    sensitive = ("token", "secret", "password", "authorization", "api_key", "apikey")
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        if any(marker in str(key).lower() for marker in sensitive):
            continue
        clean[str(key)] = _json_safe(value)
    return clean


def _safe_namespace(value: str) -> str:
    normalized = str(value).strip().lower()
    if not normalized or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
        for character in normalized
    ):
        raise DataContractError(f"invalid cache namespace: {value!r}")
    return normalized


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
