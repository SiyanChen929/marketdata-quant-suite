"""Atomic, provenance-aware file-cache helpers for reference data."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import threading
from typing import Iterator, Mapping
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import pandas as pd

from quant_system.data.paths import marketdata_cache_dir


_CACHE_LOCKS: dict[str, threading.RLock] = {}
_CACHE_LOCKS_GUARD = threading.Lock()
_REQUEST_FIELDS = {
    "provider",
    "dataset",
    "base_url",
    "endpoint",
    "symbol",
    "start",
    "end",
}


class CacheIntegrityError(RuntimeError):
    """Raised when cached data and its manifest disagree."""


class DataCache:
    """CSV cache with atomic writes, request identity, and checksums."""

    def __init__(self, root: str | Path = "data/cache") -> None:
        self.root = marketdata_cache_dir() if str(root) == "data/cache" else Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        lock_key = str(self.root.resolve())
        with _CACHE_LOCKS_GUARD:
            self._thread_lock = _CACHE_LOCKS.setdefault(lock_key, threading.RLock())

    def path_for(self, key: str) -> Path:
        """Return a safe cache path for key."""

        return self.root / f"{_safe_key(key)}.csv"

    def manifest_path_for(self, key: str) -> Path:
        """Return the provenance-manifest path paired with a cache entry."""

        return self.root / f"{_safe_key(key)}.json"

    def read(self, key: str, *, request: Mapping[str, object] | None = None) -> pd.DataFrame | None:
        """Read and verify a cache entry, or return ``None`` for a cache miss."""

        path = self.path_for(key)
        manifest_path = self.manifest_path_for(key)
        expected_request = _canonical_request(request) if request is not None else None
        with self._locked():
            if not path.exists() and not manifest_path.exists():
                return None
            # Legacy unmanifested files are intentionally treated as misses so
            # the next successful request upgrades them to the current format.
            if path.exists() and not manifest_path.exists():
                return None
            if manifest_path.exists() and not path.exists():
                raise CacheIntegrityError(f"Cache manifest has no data file: {manifest_path}")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise CacheIntegrityError(f"Invalid cache manifest: {manifest_path}") from exc
            _validate_manifest_identity(key, manifest)
            manifest_request = _canonical_request(manifest.get("request", {}))
            if manifest.get("request_sha256") != _request_sha256(manifest_request):
                raise CacheIntegrityError(f"Cache request checksum mismatch: {manifest_path}")
            if expected_request is not None and manifest_request != expected_request:
                return None
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != manifest.get("csv_sha256"):
                raise CacheIntegrityError(f"Cache data checksum mismatch: {path}")
            try:
                frame = pd.read_csv(BytesIO(raw))
            except pd.errors.EmptyDataError:
                frame = pd.DataFrame()
            if len(frame) != int(manifest.get("rows", -1)):
                raise CacheIntegrityError(f"Cache row count mismatch: {path}")
            if list(frame.columns) != list(manifest.get("columns", [])):
                raise CacheIntegrityError(f"Cache columns mismatch: {path}")
            frame.attrs["cache_manifest"] = manifest
            return frame

    def write(
        self,
        key: str,
        frame: pd.DataFrame,
        *,
        request: Mapping[str, object],
        fetched_at: datetime | None = None,
    ) -> None:
        """Atomically write cached data and a redacted provenance manifest."""

        path = self.path_for(key)
        manifest_path = self.manifest_path_for(key)
        canonical_request = _canonical_request(request)
        fetched = fetched_at or datetime.now(timezone.utc)
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        csv_bytes = frame.to_csv(index=False).encode("utf-8")
        manifest = {
            "schema_version": 1,
            "cache_key": key,
            "request": canonical_request,
            "request_sha256": _request_sha256(canonical_request),
            "fetched_at": fetched.astimezone(timezone.utc).isoformat(),
            "rows": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
            "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        }
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        nonce = f"{os.getpid()}-{threading.get_ident()}-{uuid4().hex}"
        csv_tmp = path.with_name(f".{path.name}.{nonce}.tmp")
        manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{nonce}.tmp")
        with self._locked():
            try:
                _write_fsynced(csv_tmp, csv_bytes)
                _write_fsynced(manifest_tmp, manifest_bytes)
                os.replace(csv_tmp, path)
                os.replace(manifest_tmp, manifest_path)
            finally:
                csv_tmp.unlink(missing_ok=True)
                manifest_tmp.unlink(missing_ok=True)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        lock_path = self.root / ".data-cache.lock"
        with self._thread_lock:
            with lock_path.open("a+b") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _safe_key(key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in key)


def _canonical_request(request: Mapping[str, object] | object) -> dict[str, object]:
    if not isinstance(request, Mapping):
        return {}
    return {
        str(key): _request_value(str(key), value)
        for key, value in sorted(request.items(), key=lambda item: str(item[0]))
        if str(key) in _REQUEST_FIELDS
    }


def _request_value(key: str, value: object) -> object:
    if key == "base_url" and value is not None:
        parsed = urlsplit(str(value))
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = hostname
        if parsed.port is not None:
            netloc = f"{hostname}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    if key == "endpoint" and value is not None:
        return urlsplit(str(value)).path
    return _json_value(value)


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _request_sha256(request: Mapping[str, object]) -> str:
    raw = json.dumps(dict(request), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_manifest_identity(key: str, manifest: object) -> None:
    if not isinstance(manifest, dict):
        raise CacheIntegrityError("Cache manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise CacheIntegrityError("Unsupported cache manifest schema")
    if manifest.get("cache_key") != key:
        raise CacheIntegrityError("Cache manifest key mismatch")


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def reference_cache_manifests(*frames: pd.DataFrame | None) -> list[dict[str, object]]:
    """Collect unique cache manifests carried by reference-data frames."""

    manifests: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for frame in frames:
        if not isinstance(frame, pd.DataFrame):
            continue
        candidates: list[object] = []
        if isinstance(frame.attrs.get("cache_manifest"), dict):
            candidates.append(frame.attrs["cache_manifest"])
        carried = frame.attrs.get("reference_cache_manifests", [])
        if isinstance(carried, list):
            candidates.extend(carried)
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            identity = (str(candidate.get("request_sha256", "")), str(candidate.get("csv_sha256", "")))
            if identity in seen:
                continue
            seen.add(identity)
            manifests.append(dict(candidate))
    return sorted(
        manifests,
        key=lambda item: (
            str(item.get("request", {}).get("dataset", "")) if isinstance(item.get("request"), dict) else "",
            str(item.get("request", {}).get("symbol", "")) if isinstance(item.get("request"), dict) else "",
            str(item.get("cache_key", "")),
        ),
    )


def attach_reference_cache_manifests(frame: pd.DataFrame, *sources: pd.DataFrame | None) -> pd.DataFrame:
    """Attach exact consumed cache manifests without altering table columns."""

    manifests = reference_cache_manifests(frame, *sources)
    if manifests:
        frame.attrs["reference_cache_manifests"] = manifests
    return frame
