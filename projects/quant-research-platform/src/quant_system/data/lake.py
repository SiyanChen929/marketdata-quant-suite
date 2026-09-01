"""DuckDB/Parquet lake helpers for quant_system.

CSV remains a portable interchange format, but the research hot path should
prefer columnar Parquet datasets with optional DuckDB predicate pushdown.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from quant_system.data.intraday_provider import INTRADAY_COLUMNS, validate_intraday_ohlcv
from quant_system.data.paths import quant_system_lake_dir
from quant_system.regime.intraday_risk import intraday_daily_symbol_summary
from quant_system.utils.validation import OHLCV_COLUMNS, OHLCV_LINEAGE_COLUMNS, require_columns, validate_ohlcv


DAILY_LAKE_COLUMNS = [*OHLCV_COLUMNS, *OHLCV_LINEAGE_COLUMNS]


class QuantSystemLake:
    """Columnar lake rooted under ``QUANT_DATA_HOME/lake/confirmed/quant_system``."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else quant_system_lake_dir()
        self.daily_path = self.root / "daily_prices"
        self.intraday_root = self.root / "intraday"
        self.manifest_path = self.root / "manifest.json"
        self.root.mkdir(parents=True, exist_ok=True)

    def has_fresh_daily(self, source: str | Path) -> bool:
        """Return true when the daily Parquet dataset exists and matches source."""

        return self.daily_path.exists() and self._manifest_matches("daily_prices", source)

    def has_fresh_intraday(self, source: str | Path, timeframe: str) -> bool:
        """Return true when the intraday Parquet dataset exists and matches source."""

        path = self.intraday_path(timeframe)
        return path.exists() and self._manifest_matches(f"intraday_{timeframe}", source)

    def has_fresh_intraday_summary(self, source: str | Path, timeframe: str) -> bool:
        """Return true when the intraday summary dataset exists and matches source."""

        path = self.intraday_summary_path(timeframe)
        return path.exists() and self._manifest_matches(f"intraday_summary_{timeframe}", source)

    def intraday_path(self, timeframe: str) -> Path:
        return self.intraday_root / f"timeframe={timeframe}"

    def intraday_summary_path(self, timeframe: str) -> Path:
        return self.root / "intraday_daily_summary" / f"timeframe={timeframe}"

    def materialize_daily_from_csv(self, csv_path: str | Path) -> Path:
        """Build a lineage-preserving confirmed daily Parquet dataset."""

        source = Path(csv_path)
        raw = pd.read_csv(source)
        require_columns(raw, OHLCV_LINEAGE_COLUMNS, "Daily lake source")
        frame = validate_ohlcv(raw)
        if not frame["finality"].eq("confirmed").all():
            raise ValueError("daily research lake accepts confirmed rows only")
        frame["year"] = pd.to_datetime(frame["date"]).dt.year.astype("int16")
        _replace_parquet_dataset(frame, self.daily_path, partition_cols=["year"])
        self._update_manifest("daily_prices", source, rows=len(frame), path=self.daily_path)
        return self.daily_path

    def materialize_intraday_from_csv(self, csv_path: str | Path, timeframe: str = "5min") -> Path:
        """Build the intraday Parquet dataset from a CSV cache."""

        source = Path(csv_path)
        frame = validate_intraday_ohlcv(pd.read_csv(source))
        frame["year"] = pd.to_datetime(frame["date"]).dt.year.astype("int16")
        path = self.intraday_path(timeframe)
        _replace_parquet_dataset(frame, path, partition_cols=["year"])
        self._update_manifest(f"intraday_{timeframe}", source, rows=len(frame), path=path)
        return path

    def materialize_intraday_summary_from_csv(self, csv_path: str | Path, timeframe: str = "5min") -> Path:
        """Build symbol/date intraday summaries used by pretrade risk scoring."""

        source = Path(csv_path)
        if self.has_fresh_intraday(source, timeframe) and duckdb_available():
            summary = self._intraday_summary_from_duckdb(timeframe)
        else:
            frame = validate_intraday_ohlcv(pd.read_csv(source))
            summary = intraday_daily_symbol_summary(frame)
        summary["year"] = pd.to_datetime(summary["date"]).dt.year.astype("int16")
        path = self.intraday_summary_path(timeframe)
        _replace_parquet_dataset(summary, path, partition_cols=["year"])
        self._update_manifest(f"intraday_summary_{timeframe}", source, rows=len(summary), path=path)
        return path

    def read_daily(self, symbols: list[str], start: str, end: str | None) -> pd.DataFrame:
        """Read daily OHLCV from the lake using DuckDB when available."""

        if not self.daily_path.exists():
            return pd.DataFrame(columns=DAILY_LAKE_COLUMNS)
        if not symbols:
            return validate_ohlcv(_read_dataset_all(self.daily_path, columns=DAILY_LAKE_COLUMNS))
        symbols = [symbol.upper() for symbol in symbols]
        frame = _read_dataset(
            self.daily_path,
            columns=DAILY_LAKE_COLUMNS,
            symbols=symbols,
            start=start,
            end=end,
            datetime_column="date",
        )
        return validate_ohlcv(frame) if not frame.empty else pd.DataFrame(columns=DAILY_LAKE_COLUMNS)

    def daily_symbols(self) -> list[str]:
        """Return symbols present in the daily lake."""

        if not self.daily_path.exists():
            return []
        frame = _read_dataset_all(self.daily_path, columns=["symbol"])
        return sorted(frame["symbol"].astype(str).str.upper().unique().tolist()) if not frame.empty else []

    def read_intraday(self, symbols: list[str], start: str, end: str | None, timeframe: str = "5min") -> pd.DataFrame:
        """Read intraday OHLCV from the lake using DuckDB when available."""

        path = self.intraday_path(timeframe)
        if not path.exists():
            return pd.DataFrame(columns=INTRADAY_COLUMNS)
        symbols = [symbol.upper() for symbol in symbols]
        frame = _read_dataset(
            path,
            columns=INTRADAY_COLUMNS,
            symbols=symbols,
            start=start,
            end=end,
            datetime_column="date",
        )
        return validate_intraday_ohlcv(frame) if not frame.empty else pd.DataFrame(columns=INTRADAY_COLUMNS)

    def read_intraday_summary(self, symbols: list[str], start: str, end: str | None, timeframe: str = "5min") -> pd.DataFrame:
        """Read precomputed symbol/date intraday summaries."""

        columns = [
            "date",
            "symbol",
            "first_open",
            "last_close",
            "low",
            "high",
            "intraday_volume",
            "minute_intraday_return",
            "minute_low_from_open",
            "minute_range",
            "minute_volume_ratio_20d",
        ]
        path = self.intraday_summary_path(timeframe)
        if not path.exists():
            return pd.DataFrame(columns=columns)
        frame = _read_dataset(
            path,
            columns=columns,
            symbols=[symbol.upper() for symbol in symbols],
            start=start,
            end=end,
            datetime_column="date",
        )
        if frame.empty:
            return pd.DataFrame(columns=columns)
        frame["date"] = pd.to_datetime(frame["date"], format="mixed", errors="coerce").dt.normalize()
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        return frame.sort_values(["date", "symbol"]).reset_index(drop=True)

    def _intraday_summary_from_duckdb(self, timeframe: str) -> pd.DataFrame:
        """Aggregate the intraday Parquet lake to symbol/date summaries in DuckDB."""

        import duckdb

        relation = str(self.intraday_path(timeframe) / "**" / "*.parquet")
        query = """
            WITH ranked AS (
                SELECT
                    date,
                    symbol,
                    datetime,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    ROW_NUMBER() OVER (PARTITION BY date, symbol ORDER BY datetime ASC) AS rn_first,
                    ROW_NUMBER() OVER (PARTITION BY date, symbol ORDER BY datetime DESC) AS rn_last
                FROM read_parquet(?, hive_partitioning = true)
            ),
            daily AS (
                SELECT
                    date,
                    symbol,
                    MAX(CASE WHEN rn_first = 1 THEN open END) AS first_open,
                    MAX(CASE WHEN rn_last = 1 THEN close END) AS last_close,
                    MIN(low) AS low,
                    MAX(high) AS high,
                    SUM(volume) AS intraday_volume
                FROM ranked
                GROUP BY date, symbol
            ),
            scored AS (
                SELECT
                    *,
                    last_close / NULLIF(first_open, 0) - 1.0 AS minute_intraday_return,
                    low / NULLIF(first_open, 0) - 1.0 AS minute_low_from_open,
                    high / NULLIF(low, 0) - 1.0 AS minute_range
                FROM daily
            )
            SELECT
                *,
                intraday_volume / NULLIF(
                    AVG(intraday_volume) OVER (
                        PARTITION BY symbol
                        ORDER BY date
                        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    ),
                    0
                ) AS minute_volume_ratio_20d
            FROM scored
            ORDER BY date, symbol
        """
        return duckdb.connect(database=":memory:").execute(query, [relation]).df()

    def _manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"tables": {}}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"tables": {}}

    def _manifest_matches(self, table: str, source: str | Path) -> bool:
        source_path = Path(source)
        if not source_path.exists():
            return False
        table_meta = self._manifest().get("tables", {}).get(table, {})
        return table_meta.get("source_signature") == _file_signature(source_path)

    def _update_manifest(self, table: str, source: Path, rows: int, path: Path) -> None:
        manifest = self._manifest()
        manifest.setdefault("tables", {})[table] = {
            "path": str(path),
            "rows": int(rows),
            "source_signature": _file_signature(source),
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def duckdb_available() -> bool:
    """Return whether DuckDB can be imported."""

    try:
        import duckdb  # noqa: F401

        return True
    except ImportError:
        return False


def _replace_parquet_dataset(frame: pd.DataFrame, path: Path, partition_cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path)
    frame.to_parquet(path, index=False, partition_cols=partition_cols)


def _read_dataset(
    path: Path,
    *,
    columns: list[str],
    symbols: list[str],
    start: str,
    end: str | None,
    datetime_column: str,
) -> pd.DataFrame:
    if duckdb_available():
        return _read_dataset_duckdb(path, columns=columns, symbols=symbols, start=start, end=end, datetime_column=datetime_column)
    return _read_dataset_pyarrow(path, columns=columns, symbols=symbols, start=start, end=end, datetime_column=datetime_column)


def _read_dataset_all(path: Path, *, columns: list[str]) -> pd.DataFrame:
    if duckdb_available():
        import duckdb

        return duckdb.connect(database=":memory:").execute(
            f"SELECT {', '.join(columns)} FROM read_parquet(?, hive_partitioning = true)",
            [str(path / "**" / "*.parquet")],
        ).df()
    import pyarrow.dataset as ds

    return ds.dataset(path, format="parquet", partitioning="hive").to_table(columns=columns).to_pandas()


def _read_dataset_duckdb(
    path: Path,
    *,
    columns: list[str],
    symbols: list[str],
    start: str,
    end: str | None,
    datetime_column: str,
) -> pd.DataFrame:
    import duckdb

    relation = str(path / "**" / "*.parquet")
    query = f"""
        SELECT {", ".join(columns)}
        FROM read_parquet(?, hive_partitioning = true)
        WHERE symbol IN ({", ".join(["?"] * len(symbols))})
          AND {datetime_column} >= ?
    """
    params: list[Any] = [relation, *symbols, pd.Timestamp(start)]
    if end:
        query += f" AND {datetime_column} <= ?"
        params.append(pd.Timestamp(end))
    query += f" ORDER BY {datetime_column}, symbol"
    return duckdb.connect(database=":memory:").execute(query, params).df()


def _read_dataset_pyarrow(
    path: Path,
    *,
    columns: list[str],
    symbols: list[str],
    start: str,
    end: str | None,
    datetime_column: str,
) -> pd.DataFrame:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    filt = pc.field("symbol").isin(symbols) & (pc.field(datetime_column) >= pd.Timestamp(start).to_datetime64())
    if end:
        filt = filt & (pc.field(datetime_column) <= pd.Timestamp(end).to_datetime64())
    table = dataset.to_table(columns=columns, filter=filt)
    return table.to_pandas()


def _file_signature(path: Path) -> dict[str, int | str | bool]:
    if not path.exists():
        return {"path": str(path), "exists": False, "size": 0, "mtime_ns": 0}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
