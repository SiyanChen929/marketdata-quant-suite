"""Safe local market-data consolidation and cache maintenance.

The maintenance layer deliberately separates planning from deletion.  Only
rebuildable research caches are eligible for removal; raw data, the Parquet
lake, configured market-data sources, and saved backtest runs are never part of
a cleanup plan.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from quant_system.utils.validation import OHLCV_COLUMNS, validate_ohlcv


REBUILDABLE_CACHE_DIRS = {
    "signal_cache": "signals",
    "research_bundle_cache": "research_bundles",
}


def bytes_label(value: int | float) -> str:
    """Return a compact binary size label."""

    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def storage_inventory(data_root: str | Path = "data", runs_root: str | Path = "runs") -> pd.DataFrame:
    """Inventory major storage categories without reading file contents."""

    data = Path(data_root)
    runs = Path(runs_root)
    targets = [
        ("signal_cache", data / "cache" / "signals", True),
        ("research_bundle_cache", data / "cache" / "research_bundles", True),
        ("other_cache", data / "cache", False),
        ("parquet_lake", data / "lake", False),
        ("raw_market_data", data / "raw", False),
        ("processed_market_data", data / "processed", False),
        ("backtest_runs", runs, False),
    ]
    rows: list[dict[str, object]] = []
    separately_counted = {
        (data / "cache" / REBUILDABLE_CACHE_DIRS["signal_cache"]).resolve(),
        (data / "cache" / REBUILDABLE_CACHE_DIRS["research_bundle_cache"]).resolve(),
    }
    for category, root, cache in targets:
        if category == "other_cache":
            files = [path for path in _iter_files(root) if not any(parent in path.resolve().parents for parent in separately_counted)]
        else:
            files = list(_iter_files(root))
        total = sum(_safe_size(path) for path in files)
        mtimes = [mtime for path in files if (mtime := _safe_mtime(path)) is not None]
        latest_mtime = max(mtimes, default=None)
        rows.append(
            {
                "category": category,
                "path": str(root),
                "file_count": len(files),
                "size_bytes": total,
                "size": bytes_label(total),
                "latest_modified": latest_mtime,
                "rebuildable_cache": cache,
            }
        )
    return pd.DataFrame(rows)


def build_cleanup_plan(
    cache_root: str | Path = "data/cache",
    *,
    keep_signals: int = 20,
    keep_bundles: int = 5,
    min_age_days: int = 14,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Plan LRU cleanup for rebuildable caches.

    A file is eligible only when it is both older than ``min_age_days`` and
    outside the newest retained files for its cache class.
    """

    root = Path(cache_root)
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    cutoff = reference - timedelta(days=max(0, int(min_age_days)))
    policies = {
        "signal_cache": max(0, int(keep_signals)),
        "research_bundle_cache": max(0, int(keep_bundles)),
    }
    rows: list[dict[str, object]] = []
    for category, keep_count in policies.items():
        directory = root / REBUILDABLE_CACHE_DIRS[category]
        files = sorted(
            (path for path in _iter_files(directory) if path.suffix.lower() == ".pkl"),
            key=lambda path: (_safe_mtime_timestamp(path), path.name),
            reverse=True,
        )
        retained = set(files[:keep_count])
        for path in files:
            modified = _safe_mtime(path)
            if path in retained or modified is None or modified >= cutoff:
                continue
            size = _safe_size(path)
            rows.append(
                {
                    "path": str(path),
                    "category": category,
                    "reason": f"older_than_{min_age_days}d_outside_latest_{keep_count}",
                    "size_bytes": size,
                    "size": bytes_label(size),
                    "modified_at": modified,
                }
            )
    columns = ["path", "category", "reason", "size_bytes", "size", "modified_at"]
    return pd.DataFrame(rows, columns=columns).sort_values("size_bytes", ascending=False, ignore_index=True)


def execute_cleanup(plan: pd.DataFrame, cache_root: str | Path = "data/cache", *, confirmed: bool = False) -> pd.DataFrame:
    """Delete a validated cleanup plan after explicit confirmation."""

    if not confirmed:
        raise ValueError("Cleanup requires confirmed=True")
    root = Path(cache_root).resolve()
    allowed = {(root / name).resolve() for name in REBUILDABLE_CACHE_DIRS.values()}
    rows: list[dict[str, object]] = []
    for record in plan.to_dict("records"):
        path = Path(str(record.get("path", ""))).resolve()
        allowed_path = any(directory in path.parents for directory in allowed)
        status = "skipped"
        error = ""
        size = _safe_size(path)
        if not allowed_path or path.suffix.lower() != ".pkl":
            error = "outside_allowed_rebuildable_cache"
        elif not path.exists():
            status = "already_missing"
        elif not path.is_file() and not path.is_symlink():
            error = "not_a_file"
        else:
            try:
                path.unlink()
                status = "deleted"
            except OSError as exc:
                status = "error"
                error = str(exc)
        rows.append({"path": str(path), "status": status, "size_bytes": size, "error": error})
    return pd.DataFrame(rows, columns=["path", "status", "size_bytes", "error"])


def consolidate_daily_ohlcv(sources: Iterable[str | Path], output_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge daily OHLCV files into one validated, de-duplicated source.

    Later source arguments take precedence for duplicate symbol/date rows.  The
    output format is selected from the destination suffix (CSV or Parquet).
    """

    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for priority, source_like in enumerate(sources):
        source = Path(source_like)
        if not source.exists() or not source.is_file():
            audit_rows.append({"source": str(source), "status": "missing", "input_rows": 0, "valid_rows": 0})
            continue
        try:
            if source.suffix.lower() in {".parquet", ".pq"}:
                raw = pd.read_parquet(source)
            else:
                raw = pd.read_csv(source)
            valid = validate_ohlcv(raw)
        except Exception as exc:
            audit_rows.append(
                {"source": str(source), "status": f"invalid: {exc}", "input_rows": 0, "valid_rows": 0}
            )
            continue
        valid = valid.copy()
        valid["_source_priority"] = priority
        frames.append(valid)
        audit_rows.append({"source": str(source), "status": "ok", "input_rows": len(raw), "valid_rows": len(valid)})
    if not frames:
        raise ValueError("No valid daily OHLCV source was provided")
    combined = pd.concat(frames, ignore_index=True)
    input_rows = len(combined)
    combined = combined.sort_values(["symbol", "date", "_source_priority"])
    combined = combined.drop_duplicates(["symbol", "date"], keep="last")
    combined = combined.drop(columns="_source_priority")
    combined = validate_ohlcv(combined)
    output_columns = [*OHLCV_COLUMNS, *[column for column in ("source", "finality") if column in combined.columns]]
    combined = combined[output_columns]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() in {".parquet", ".pq"}:
        combined.to_parquet(destination, index=False)
    else:
        combined.to_csv(destination, index=False)
    audit = pd.DataFrame(audit_rows)
    audit["duplicates_removed"] = input_rows - len(combined)
    return combined, audit


def _iter_files(root: Path):
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() or path.is_symlink():
            yield path


def _safe_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _safe_mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _safe_mtime_timestamp(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0
