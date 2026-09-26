"""Wide date-by-symbol panels built from the suite's canonical confirmed bars.

All prices enter through :mod:`quant_marketdata`.  Panels are daily and
confirmed-only: formal factor evaluation never consumes provisional sessions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from quant_marketdata import CANONICAL_COLUMNS, MarketDataStore, normalize_bars


PANEL_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
CONTENT_HASH_NAMESPACE = "llm-factor-mining/panel-v1"
VALUES_HASH_NAMESPACE = "llm-factor-mining/panel-values-v1"


class PanelError(ValueError):
    """Bars cannot form a confirmed daily research panel."""


@dataclass(frozen=True)
class Panel:
    """Aligned date x symbol OHLCV frames plus provenance.

    ``content_sha256`` is always computed from the panel values; when the bars
    came from :class:`MarketDataStore`, ``manifest_sha256`` also records the
    store manifest snapshot captured with the read.
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    sources: tuple[str, ...]
    content_sha256: str
    manifest_sha256: str | None = None
    finality: str = "confirmed"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.finality != "confirmed":
            raise PanelError("research panels accept confirmed sessions only")
        reference = self.close
        if not isinstance(reference.index, pd.DatetimeIndex):
            raise PanelError("panel index must be a DatetimeIndex")
        if not reference.index.is_monotonic_increasing or reference.index.has_duplicates:
            raise PanelError("panel dates must be strictly increasing")
        if reference.columns.has_duplicates:
            raise PanelError("panel symbols must be unique")
        for name in PANEL_FIELDS:
            frame = getattr(self, name)
            if not frame.index.equals(reference.index) or not frame.columns.equals(reference.columns):
                raise PanelError(f"field {name!r} is not aligned with close")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(str(symbol) for symbol in self.close.columns)

    @property
    def available(self) -> pd.DataFrame:
        """True where a confirmed bar exists for ``(date, symbol)``."""

        return self.close.notna()

    def field(self, name: str) -> pd.DataFrame:
        if name not in PANEL_FIELDS:
            raise KeyError(f"unknown panel field {name!r}")
        return getattr(self, name)

    def provenance(self) -> dict[str, Any]:
        """JSON-safe provenance record for run logs."""

        dates = self.dates
        return {
            "finality": self.finality,
            "sources": list(self.sources),
            "n_dates": int(len(dates)),
            "n_symbols": int(len(self.symbols)),
            "first_date": None if dates.empty else str(dates[0].date()),
            "last_date": None if dates.empty else str(dates[-1].date()),
            "content_sha256": self.content_sha256,
            "manifest_sha256": self.manifest_sha256,
            "metadata": {str(key): _json_safe(value) for key, value in self.metadata.items()},
        }


def panel_from_bars(
    frame: pd.DataFrame,
    *,
    manifest_sha256: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Panel:
    """Build a :class:`Panel` from canonical long bars (confirmed, daily).

    Bars are validated with :func:`quant_marketdata.normalize_bars`.  Any row
    whose finality is not ``confirmed`` is rejected rather than dropped.
    """

    if not isinstance(frame, pd.DataFrame):
        raise PanelError("bars must be a pandas DataFrame")
    attrs = dict(frame.attrs)
    missing = sorted(set(CANONICAL_COLUMNS) - set(frame.columns))
    if missing:
        # normalize_bars would silently fill finality/source; a research panel must not
        raise PanelError(f"bars are missing canonical columns: {missing}")
    labels = frame["finality"].astype(str).str.strip().str.lower()
    if not labels.eq("confirmed").all():
        bad = sorted(set(labels[~labels.eq("confirmed")]))
        raise PanelError(f"panel requires confirmed bars; found finality {bad}")
    resolution = attrs.get("marketdata_resolution")
    if resolution is not None and str(resolution).upper() != "D":
        raise PanelError(f"panel requires daily bars; got resolution {resolution!r}")

    bars = normalize_bars(frame, finality="confirmed")
    if bars.empty:
        raise PanelError("no bars supplied")
    if not bars["date"].eq(bars["date"].dt.normalize()).all():
        raise PanelError("panel requires daily session dates without intraday timestamps")

    wide: dict[str, pd.DataFrame] = {}
    for name in PANEL_FIELDS:
        table = bars.pivot(index="date", columns="symbol", values=name).sort_index().sort_index(axis=1)
        table.index = pd.DatetimeIndex(table.index, name="date")
        table.columns = pd.Index([str(symbol) for symbol in table.columns], name="symbol")
        wide[name] = table.astype("float64")

    sources = tuple(sorted(set(bars["source"].astype(str))))
    manifest = manifest_sha256 or attrs.get("marketdata_manifest_sha256")
    return Panel(
        **wide,
        sources=sources,
        content_sha256=panel_content_hash(wide, sources),
        manifest_sha256=None if manifest is None else str(manifest),
        metadata=dict(metadata or {}),
    )


def load_confirmed_panel(
    store: MarketDataStore,
    symbols: Sequence[str] | str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> Panel:
    """Read confirmed daily bars from the shared store and build a panel."""

    requested = [symbols] if isinstance(symbols, str) else list(symbols)
    bars = store.read_bars(requested, start=start, end=end, finality="confirmed", resolution="D")
    if bars.empty:
        raise PanelError("the confirmed store holds no bars for the requested symbols and dates")
    present = set(bars["symbol"].astype(str))
    missing = sorted({str(symbol).strip().upper() for symbol in requested} - present)
    return panel_from_bars(
        bars,
        metadata={
            "requested_symbols": sorted({str(symbol).strip().upper() for symbol in requested}),
            "missing_symbols": missing,
            "requested_start": str(pd.Timestamp(start).date()),
            "requested_end": str(pd.Timestamp(end).date()),
            "resolution": "D",
        },
    )


def panel_content_hash(fields: Mapping[str, pd.DataFrame], sources: Sequence[str]) -> str:
    """Deterministic SHA-256 over dates, symbols, sources and float64 field values."""

    close = fields["close"]
    digest = hashlib.sha256()
    digest.update(CONTENT_HASH_NAMESPACE.encode("utf-8"))
    digest.update(np.asarray(close.index.asi8, dtype="<i8").tobytes())
    digest.update("\x1f".join(str(symbol) for symbol in close.columns).encode("utf-8"))
    digest.update("\x1f".join(sources).encode("utf-8"))
    for name in PANEL_FIELDS:
        values = np.ascontiguousarray(fields[name].to_numpy(dtype="float64"), dtype="<f8")
        digest.update(name.encode("utf-8"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def panel_values_sha256(panel: Panel) -> str:
    """SHA-256 over the panel's dates, symbols and float64 field values only.

    Unlike :attr:`Panel.content_sha256` it ignores the ``sources`` labels, so a
    panel cut from a longer load (:func:`~llm_factor_mining.protocol.splits.truncate_panel`
    keeps the parent's source set) and a direct load of the shorter range have
    equal values hashes whenever their bars agree, even when a bar source
    appears only later in the sample (for example a vendor switch).
    """

    close = panel.close
    digest = hashlib.sha256()
    digest.update(VALUES_HASH_NAMESPACE.encode("utf-8"))
    digest.update(np.asarray(close.index.asi8, dtype="<i8").tobytes())
    digest.update("\x1f".join(str(symbol) for symbol in close.columns).encode("utf-8"))
    for name in PANEL_FIELDS:
        values = np.ascontiguousarray(panel.field(name).to_numpy(dtype="float64"), dtype="<f8")
        digest.update(name.encode("utf-8"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def align_symbols(panel: Panel, symbols: Sequence[str]) -> Panel:
    """Reindex a panel's columns to ``symbols`` (absent ones become all-NaN) and rehash.

    Used so that panels of one study loaded over different date ranges share
    one column set, which makes their content hashes comparable.
    """

    columns = pd.Index([str(symbol) for symbol in symbols], name="symbol")
    if columns.has_duplicates:
        raise PanelError("symbols must be unique")
    fields = {name: panel.field(name).reindex(columns=columns) for name in PANEL_FIELDS}
    return Panel(
        **fields,
        sources=panel.sources,
        content_sha256=panel_content_hash(fields, panel.sources),
        manifest_sha256=panel.manifest_sha256,
        finality=panel.finality,
        metadata=dict(panel.metadata),
    )


def recompute_content_sha256(panel: Panel) -> str:
    """Content hash recomputed from the panel's current values (not the stored attribute)."""

    return panel_content_hash({name: panel.field(name) for name in PANEL_FIELDS}, panel.sources)


def verify_panel_hash(panel: Panel) -> str:
    """Recompute the content hash and fail if the values changed since construction.

    Panels are frozen dataclasses, but their DataFrames are mutable in place;
    this check turns a silent mutation into a :class:`PanelError`.
    """

    digest = recompute_content_sha256(panel)
    if digest != panel.content_sha256:
        raise PanelError("panel values no longer match its content hash (mutated in place?)")
    return digest


def is_synthetic(panel: Panel) -> bool:
    """True when every bar source of the panel is the synthetic generator."""

    return bool(panel.sources) and all(source == "synthetic" for source in panel.sources)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


__all__ = [
    "CANONICAL_COLUMNS",
    "PANEL_FIELDS",
    "Panel",
    "PanelError",
    "align_symbols",
    "is_synthetic",
    "load_confirmed_panel",
    "panel_content_hash",
    "panel_from_bars",
    "panel_values_sha256",
    "recompute_content_sha256",
    "verify_panel_hash",
]
