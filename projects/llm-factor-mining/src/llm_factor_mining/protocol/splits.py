"""Chronological formation / validation / sealed-test windows with embargo gaps.

The three windows are disjoint, ordered in time and separated by at least
``lag + max_horizon`` sessions, so a forward return that starts in one window
can never end in the next.  (Each window's metrics additionally embargo their
own last ``lag + horizon`` signal dates; see
:func:`llm_factor_mining.evaluate.metrics.embargoed_signal_dates`.)

:func:`truncate_panel` gives the search phase a panel that ends at the
formation window, so every statistic a proposer can receive is computed from
formation data only.  (Whether later prices are *loaded* at all depends on the
caller: :func:`llm_factor_mining.search.run_search` accepts a loader that
reads each window's data only when that phase starts.)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..data import PANEL_FIELDS, Panel, panel_content_hash
from ..evaluate.metrics import EvaluationWindow


WINDOW_NAMES: tuple[str, str, str] = ("formation", "validation", "test")


class SplitError(ValueError):
    """Windows overlap, are out of order, or violate the embargo."""


@dataclass(frozen=True)
class Splits:
    """Three named, chronologically ordered evaluation windows."""

    formation: EvaluationWindow
    validation: EvaluationWindow
    test: EvaluationWindow
    embargo_sessions: int
    lag: int
    max_horizon: int

    def __post_init__(self) -> None:
        for expected, window in zip(WINDOW_NAMES, self.windows()):
            if window.name != expected:
                raise SplitError(f"window {window.name!r} must be named {expected!r}")
            if window.start is None or window.end is None:
                raise SplitError(f"window {window.name!r} needs explicit start and end dates")
        if self.embargo_sessions < self.lag + self.max_horizon:
            raise SplitError(
                f"embargo of {self.embargo_sessions} sessions is shorter than lag + max_horizon "
                f"= {self.lag + self.max_horizon}"
            )

    def windows(self) -> tuple[EvaluationWindow, EvaluationWindow, EvaluationWindow]:
        return (self.formation, self.validation, self.test)

    def check_calendar(self, calendar: Sequence[pd.Timestamp] | pd.DatetimeIndex) -> None:
        """Verify ordering and the embargo gap in *sessions* of ``calendar``."""

        dates = pd.DatetimeIndex(calendar)
        positions = []
        for window in self.windows():
            inside = (dates >= window.start) & (dates <= window.end)
            if not inside.any():
                raise SplitError(f"window {window.name!r} contains no sessions of the calendar")
            idx = inside.nonzero()[0]
            positions.append((int(idx[0]), int(idx[-1])))
        for (_, left_end), (right_start, _), name in zip(
            positions[:-1], positions[1:], WINDOW_NAMES[1:]
        ):
            gap = right_start - left_end - 1
            if gap < self.embargo_sessions:
                raise SplitError(
                    f"only {gap} session(s) separate {name!r} from the previous window; "
                    f"{self.embargo_sessions} required"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "embargo_sessions": self.embargo_sessions,
            "lag": self.lag,
            "max_horizon": self.max_horizon,
            **{
                window.name: {
                    "start": str(pd.Timestamp(window.start).date()),
                    "end": str(pd.Timestamp(window.end).date()),
                }
                for window in self.windows()
            },
        }


def splits_from_dict(payload: Mapping[str, Any]) -> Splits:
    """Inverse of :meth:`Splits.to_dict`."""

    windows = [
        EvaluationWindow(name, pd.Timestamp(payload[name]["start"]), pd.Timestamp(payload[name]["end"]))
        for name in WINDOW_NAMES
    ]
    return Splits(
        *windows,
        embargo_sessions=int(payload["embargo_sessions"]),
        lag=int(payload["lag"]),
        max_horizon=int(payload["max_horizon"]),
    )


def chronological_splits(
    calendar: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    *,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    lag: int = 1,
    max_horizon: int = 1,
    embargo_sessions: int | None = None,
    min_sessions: int = 20,
) -> Splits:
    """Split a trading calendar into formation / validation / test by session counts.

    ``embargo_sessions`` defaults to ``lag + max_horizon`` and may not be
    smaller.  The embargo sessions belong to no window.
    """

    dates = pd.DatetimeIndex(calendar)
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise SplitError("calendar must be strictly increasing")
    if lag < 1 or max_horizon < 1:
        raise SplitError("lag and max_horizon must be at least 1")
    if len(fractions) != 3 or min(fractions) <= 0:
        raise SplitError("fractions must be three positive numbers")
    required = lag + max_horizon
    embargo = required if embargo_sessions is None else int(embargo_sessions)
    if embargo < required:
        raise SplitError(f"embargo_sessions={embargo} is below lag + max_horizon = {required}")
    usable = len(dates) - 2 * embargo
    total = float(sum(fractions))
    n_formation = int(usable * fractions[0] / total)
    n_validation = int(usable * fractions[1] / total)
    n_test = usable - n_formation - n_validation
    if min(n_formation, n_validation, n_test) < min_sessions:
        raise SplitError(
            f"calendar of {len(dates)} sessions is too short for three windows of at least "
            f"{min_sessions} sessions plus two embargoes of {embargo}"
        )
    f_end = n_formation - 1
    v_start = f_end + 1 + embargo
    v_end = v_start + n_validation - 1
    t_start = v_end + 1 + embargo
    splits = Splits(
        formation=EvaluationWindow("formation", dates[0], dates[f_end]),
        validation=EvaluationWindow("validation", dates[v_start], dates[v_end]),
        test=EvaluationWindow("test", dates[t_start], dates[-1]),
        embargo_sessions=embargo,
        lag=lag,
        max_horizon=max_horizon,
    )
    splits.check_calendar(dates)
    return splits


def truncate_panel(panel: Panel, end: pd.Timestamp | str) -> Panel:
    """Return a new panel holding only sessions on or before ``end``.

    The content hash is recomputed; the parent hash and cut date are recorded
    in the metadata.
    """

    cut = pd.Timestamp(end)
    keep = panel.dates <= cut
    if not keep.any():
        raise SplitError(f"no sessions on or before {cut.date()}")
    fields = {name: panel.field(name).loc[keep] for name in PANEL_FIELDS}
    metadata = dict(panel.metadata)
    metadata.update(
        {"truncated_at": str(cut.date()), "parent_content_sha256": panel.content_sha256}
    )
    return Panel(
        **fields,
        sources=panel.sources,
        content_sha256=panel_content_hash(fields, panel.sources),
        manifest_sha256=panel.manifest_sha256,
        metadata=metadata,
    )


__all__ = [
    "SplitError",
    "Splits",
    "WINDOW_NAMES",
    "chronological_splits",
    "splits_from_dict",
    "truncate_panel",
]
