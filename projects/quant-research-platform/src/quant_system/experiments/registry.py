"""Lightweight run registry for backtest experiments."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pandas as pd

from quant_system.config import AppConfig


REGISTRY_COLUMNS = [
    "run_id",
    "run_dir",
    "timestamp",
    "fast",
    "config_hash",
    "data_provider",
    "data_source",
    "data_finality",
    "data_max_date",
    "data_manifest_sha256",
    "start_date",
    "end_date",
    "cagr",
    "sharpe",
    "max_drawdown",
    "trailing_one_year_return",
    "total_return",
    "average_turnover",
    "average_gross_exposure",
]


def record_run(run_dir: str | Path, config: AppConfig, metrics: dict[str, float], fast: bool = False) -> Path:
    """Append a run to ``runs/registry.csv``."""

    run_path = Path(run_dir)
    registry = Path(config.output_dir) / "registry.csv"
    registry.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = run_path / "data_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    row = {
        "run_id": run_path.name,
        "run_dir": str(run_path),
        "timestamp": pd.Timestamp.now().isoformat(),
        "fast": bool(fast),
        "config_hash": config_hash(config),
        "data_provider": provenance.get("data_provider", config.data_provider),
        "data_source": ",".join(provenance.get("sources", [])),
        "data_finality": ",".join(provenance.get("finalities", [])),
        "data_max_date": provenance.get("data_max_date", ""),
        "data_manifest_sha256": provenance.get("data_manifest_sha256", ""),
        "start_date": config.start_date,
        "end_date": config.end_date or "",
        "cagr": metrics.get("cagr"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
        "trailing_one_year_return": metrics.get("trailing_one_year_return"),
        "total_return": metrics.get("total_return"),
        "average_turnover": metrics.get("average_turnover"),
        "average_gross_exposure": metrics.get("average_gross_exposure"),
    }
    existing = pd.read_csv(registry) if registry.exists() else pd.DataFrame(columns=REGISTRY_COLUMNS)
    out = pd.DataFrame([row]) if existing.empty else pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    out = out.drop_duplicates(["run_id"], keep="last")
    for column in REGISTRY_COLUMNS:
        if column not in out:
            out[column] = pd.NA
    out[REGISTRY_COLUMNS].to_csv(registry, index=False)
    return registry


def load_registry(output_dir: str | Path = "runs") -> pd.DataFrame:
    """Load the run registry."""

    path = Path(output_dir) / "registry.csv"
    if not path.exists():
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    return pd.read_csv(path)


def compare_runs(output_dir: str | Path = "runs", limit: int = 20) -> pd.DataFrame:
    """Return recent runs sorted by Sharpe then CAGR."""

    registry = load_registry(output_dir)
    if registry.empty:
        return registry
    for column in ("cagr", "sharpe", "max_drawdown", "trailing_one_year_return", "average_turnover", "average_gross_exposure"):
        registry[column] = pd.to_numeric(registry[column], errors="coerce")
    return (
        registry.sort_values(["sharpe", "cagr", "trailing_one_year_return"], ascending=[False, False, False])
        .head(limit)
        .reset_index(drop=True)
    )


def config_hash(config: AppConfig) -> str:
    """Return a stable hash of an AppConfig."""

    raw = json.dumps(asdict(config), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]
