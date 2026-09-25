"""Confirmed-only bar sources and the point-in-time view the tools read through.

All US market prices enter through :mod:`quant_marketdata`.  Two sources are
provided:

* :class:`StoreBarSource` reads the suite's external, checksum-verified lake
  with ``finality="confirmed"`` hard-wired; it has no code path that can read
  provisional staging.
* :class:`FrameBarSource` serves an in-memory canonical frame (synthetic data
  and tests), validated with :func:`quant_marketdata.normalize_bars` and
  rejected if any row is not confirmed.

Tools never touch a source directly.  They read through
:class:`PointInTimeBars`, which refuses dates after ``as_of`` and re-validates
every frame a source returns (canonical schema, confirmed finality, requested
symbols and dates only).  The filter is by session date: it cannot undo
vendor adjustments applied after ``as_of`` (see :mod:`marketdata_agent.clock`).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from quant_marketdata import MarketDataStore, empty_bars, normalize_bars

from .clock import AsOfClock
from .errors import (
    ProvisionalDataError,
    SourceContractError,
    ToolInputError,
    UnknownSymbolError,
)
from .provenance import hash_bars


CONFIRMED = "confirmed"


@runtime_checkable
class BarSource(Protocol):
    """Structural interface for confirmed daily bars in canonical long form."""

    def read_bars(
        self,
        symbols: Sequence[str],
        start: date | None,
        end: date | None,
    ) -> pd.DataFrame:
        """Return canonical rows for ``symbols`` with ``start <= date <= end``."""
        ...

    def list_symbols(self) -> list[str]:
        """Return every symbol the source can serve, sorted."""
        ...


def _normalize_symbol_list(symbols: Sequence[str]) -> list[str]:
    if isinstance(symbols, str):
        symbols = [symbols]
    normalized: list[str] = []
    for value in symbols:
        symbol = str(value).strip().upper()
        if symbol and symbol not in normalized:
            normalized.append(symbol)
    return normalized


def _require_confirmed(frame: pd.DataFrame, origin: str) -> None:
    if frame.empty:
        return
    finality = frame["finality"].astype(str).str.strip().str.lower()
    if not finality.eq(CONFIRMED).all():
        labels = sorted(set(finality) - {CONFIRMED})
        raise ProvisionalDataError(
            f"{origin} returned non-confirmed rows ({labels}); only confirmed bars are admissible"
        )


class StoreBarSource:
    """Confirmed daily bars from the suite's external :class:`MarketDataStore`.

    ``universe`` optionally restricts the symbols the agent may see.  The
    store's manifest SHA-256 captured with the most recent read is exposed as
    :meth:`snapshot_id` for provenance.
    """

    def __init__(
        self,
        store: MarketDataStore,
        *,
        universe: Sequence[str] | None = None,
    ) -> None:
        self.store = store
        self.universe = tuple(_normalize_symbol_list(universe)) if universe is not None else None
        self._last_snapshot: str | None = None

    @classmethod
    def from_environment(
        cls,
        *,
        data_home: str | Path | None = None,
        universe: Sequence[str] | None = None,
    ) -> "StoreBarSource":
        """Open the store rooted at ``data_home`` or ``QUANT_DATA_HOME``."""

        return cls(MarketDataStore(root=data_home), universe=universe)

    def read_bars(
        self,
        symbols: Sequence[str],
        start: date | None,
        end: date | None,
    ) -> pd.DataFrame:
        wanted = _normalize_symbol_list(symbols)
        if self.universe is not None:
            wanted = [symbol for symbol in wanted if symbol in self.universe]
        if not wanted:
            return empty_bars()
        frame = self.store.read_bars(
            symbols=wanted,
            start=start.isoformat() if start is not None else None,
            end=end.isoformat() if end is not None else None,
            finality=CONFIRMED,
            resolution="D",
        )
        snapshot = frame.attrs.get("marketdata_manifest_sha256")
        if snapshot:
            self._last_snapshot = f"store-manifest:{snapshot}"
        _require_confirmed(frame, "MarketDataStore")
        return frame

    def list_symbols(self) -> list[str]:
        frame = self.store.read_bars(
            symbols=list(self.universe) if self.universe is not None else None,
            finality=CONFIRMED,
            resolution="D",
        )
        return sorted(frame["symbol"].astype(str).unique().tolist())

    def snapshot_id(self) -> str | None:
        """Return the manifest fingerprint observed by the latest read, if any."""

        return self._last_snapshot


class FrameBarSource:
    """In-memory canonical bars, e.g. synthetic panels for tests and benchmarks.

    The frame must carry explicit ``source`` and ``finality`` columns; missing
    labels are never defaulted to ``confirmed``.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        if "finality" not in frame.columns or "source" not in frame.columns:
            raise SourceContractError("FrameBarSource requires explicit source and finality columns")
        bars = normalize_bars(frame)
        _require_confirmed(bars, "FrameBarSource input")
        self._bars = bars
        self._snapshot = f"frame:{hash_bars(bars)}"

    @property
    def bars(self) -> pd.DataFrame:
        """Return a copy of the validated panel."""

        return self._bars.copy()

    def read_bars(
        self,
        symbols: Sequence[str],
        start: date | None,
        end: date | None,
    ) -> pd.DataFrame:
        wanted = _normalize_symbol_list(symbols)
        mask = self._bars["symbol"].isin(wanted)
        if start is not None:
            mask &= self._bars["date"].ge(pd.Timestamp(start))
        if end is not None:
            mask &= self._bars["date"].le(pd.Timestamp(end))
        return self._bars.loc[mask].reset_index(drop=True)

    def list_symbols(self) -> list[str]:
        return sorted(self._bars["symbol"].astype(str).unique().tolist())

    def snapshot_id(self) -> str:
        """Return a fingerprint of the full in-memory panel."""

        return self._snapshot


class PointInTimeBars:
    """As-of-clamped, re-validating read view used by every tool.

    ``clock`` is the episode's information cutoff *t*. It defines ``"latest"``,
    the session calendar, the known universe and the default end of every
    read. ``cutoff`` is the date after which **explicitly requested** dates are
    refused with :class:`LookaheadViolation`. The two are equal unless the
    no-clock ablation (``Copilot(enforce_clock=False)``) raises ``cutoff`` to
    9999-12-31: then explicitly named later dates and explicitly named
    symbols are served, while everything the tools derive implicitly (the
    meaning of ``"latest"``, the universe listed by ``list_symbols``, proposal
    reference prices) still refers to *t*.

    Frames returned by the source are re-normalized and must be confirmed,
    inside the requested date range, and limited to the requested symbols;
    otherwise :class:`SourceContractError` is raised rather than silently
    repaired.
    """

    def __init__(self, source: BarSource, clock: AsOfClock, *, cutoff: AsOfClock | None = None) -> None:
        if not isinstance(source, BarSource):
            raise TypeError("source must implement read_bars(symbols, start, end) and list_symbols()")
        self.source = source
        self.clock = clock
        self.cutoff = cutoff if cutoff is not None else clock
        if self.cutoff.as_of < self.clock.as_of:
            raise ValueError("the refusal cutoff cannot precede the as-of date")
        self.read_count = 0
        self._sessions: tuple[date, ...] | None = None
        self._available: tuple[str, ...] | None = None
        self._reachable: tuple[str, ...] | None = None
        self._known_panel: pd.DataFrame | None = None

    @property
    def clock_enforced(self) -> bool:
        """True unless the no-clock ablation lifted the refusal of later dates."""

        return self.cutoff.as_of == self.clock.as_of

    def read(
        self,
        symbols: Sequence[str],
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Return validated confirmed rows for ``symbols`` in ``[start, end]``.

        ``end=None`` means ``as_of``; ``start=None`` means the earliest bar.
        Explicit dates after ``cutoff`` raise :class:`LookaheadViolation`.
        """

        if start is not None:
            self.cutoff.check(start, field="start")
        upper = self.clock.as_of if end is None else self.cutoff.check(end, field="end")
        if start is not None and start > upper:
            raise ToolInputError(f"start {start.isoformat()} is after end {upper.isoformat()}")
        wanted = _normalize_symbol_list(symbols)
        self.read_count += 1
        raw = self.source.read_bars(wanted, start, upper)
        return self._validate(raw, wanted, start, upper)

    def _validate(
        self,
        raw: pd.DataFrame,
        wanted: list[str],
        start: date | None,
        end: date,
    ) -> pd.DataFrame:
        if raw is None or raw.empty:
            return empty_bars()
        if "finality" not in raw.columns or "source" not in raw.columns:
            raise SourceContractError("source returned bars without source/finality labels")
        _require_confirmed(raw, type(self.source).__name__)
        bars = normalize_bars(raw)
        if bars["date"].gt(pd.Timestamp(end)).any():
            raise SourceContractError(
                f"source returned rows after the requested end {end.isoformat()}"
            )
        if start is not None and bars["date"].lt(pd.Timestamp(start)).any():
            raise SourceContractError(
                f"source returned rows before the requested start {start.isoformat()}"
            )
        unexpected = sorted(set(bars["symbol"].astype(str)) - set(wanted))
        if unexpected:
            raise SourceContractError(f"source returned unrequested symbols: {unexpected}")
        return bars

    def known_panel(self) -> pd.DataFrame:
        """Return every confirmed bar dated ``<= as_of`` across the source universe."""

        if self._known_panel is None:
            self._known_panel = self.read(self.source.list_symbols(), None, None)
        return self._known_panel.copy()

    def available_symbols(self) -> tuple[str, ...]:
        """Symbols with at least one confirmed bar on or before ``as_of``.

        Symbols that first trade after ``as_of`` are excluded: knowing that a
        future listing exists would itself be look-ahead information.
        """

        if self._available is None:
            panel = self.known_panel()
            self._available = tuple(sorted(panel["symbol"].astype(str).unique().tolist()))
        return self._available

    def sessions(self) -> tuple[date, ...]:
        """Sorted union of confirmed session dates ``<= as_of`` across the universe."""

        if self._sessions is None:
            panel = self.known_panel()
            dates = pd.DatetimeIndex(panel["date"].unique()).sort_values()
            self._sessions = tuple(ts.date() for ts in dates)
        return self._sessions

    def reachable_symbols(self) -> tuple[str, ...]:
        """Symbols with at least one confirmed bar on or before ``cutoff``.

        Equal to :meth:`available_symbols` when the clock is enforced. In the
        no-clock ablation it also contains symbols that list after ``as_of``,
        because naming a symbol explicitly is treated like naming a date.
        """

        if self.clock_enforced:
            return self.available_symbols()
        if self._reachable is None:
            panel = self.read(self.source.list_symbols(), None, self.cutoff.as_of)
            self._reachable = tuple(sorted(panel["symbol"].astype(str).unique().tolist()))
        return self._reachable

    def require_symbols(self, symbols: Sequence[str]) -> list[str]:
        """Normalize ``symbols`` and raise :class:`UnknownSymbolError` for unknown ones."""

        normalized = _normalize_symbol_list(symbols)
        if not normalized:
            raise UnknownSymbolError("no symbol was provided")
        known = set(self.reachable_symbols())
        missing = [symbol for symbol in normalized if symbol not in known]
        if missing:
            raise UnknownSymbolError(
                f"no confirmed bars on or before {self.clock.as_of.isoformat()} for {missing}; "
                "call list_symbols for the available universe"
            )
        return normalized

    def snapshot_id(self) -> str | None:
        """Return the source's snapshot fingerprint when it exposes one."""

        getter = getattr(self.source, "snapshot_id", None)
        return getter() if callable(getter) else None
