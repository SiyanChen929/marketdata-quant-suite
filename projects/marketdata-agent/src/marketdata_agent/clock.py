"""Point-in-time clock: every date an episode touches is validated against ``as_of``.

Semantics: an episode with ``as_of = t`` operates after the close of session
``t``.  Confirmed bars dated ``<= t`` are knowable; anything later is not.  A
request that names a later date raises :class:`LookaheadViolation` instead of
being clamped, so look-ahead attempts stay observable in the audit log and can
be counted by the benchmark.  Orders proposed at ``t`` may execute no earlier
than the next session (the suite's ``t+1`` rule).

Scope of the guarantee: it is **date-based, not knowledge-time-based**.  The
clock filters rows by their session date.  It cannot know when a row's
*values* became known.  Bars in the suite's store are requested from the
vendor split- and dividend-adjusted as of ingestion, so on real data a bar
dated ``<= t`` can embed corporate actions that happened after ``t``, and a
universe built from symbols ingested later is survivorship-biased.  The
synthetic benchmark panel is unadjusted by construction and unaffected; see
``docs/safety-model.md`` (risk 1) before running on real data.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
import re

import pandas as pd

from .errors import InsufficientDataError, LookaheadViolation, ToolInputError


LATEST = "latest"
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

DateLike = date | str | pd.Timestamp


def parse_date(value: DateLike, *, field: str = "date") -> date:
    """Parse a strict ``YYYY-MM-DD`` string or date-like object into a ``date``."""

    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None or value != value.normalize():
            raise ToolInputError(f"{field} must be a session date without a time component")
        return value.date()
    if isinstance(value, datetime):
        if value.tzinfo is not None or value.time() != datetime.min.time():
            raise ToolInputError(f"{field} must be a session date without a time component")
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and _ISO_DATE.match(value.strip()):
        try:
            return date.fromisoformat(value.strip())
        except ValueError as exc:
            raise ToolInputError(f"{field}={value!r} is not a valid calendar date") from exc
    raise ToolInputError(f"{field} must be a YYYY-MM-DD date string, got {value!r}")


@dataclass(frozen=True)
class AsOfClock:
    """Immutable information cutoff for one episode."""

    as_of: date

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", parse_date(self.as_of, field="as_of"))

    def check(self, value: date, *, field: str) -> date:
        """Return ``value`` or raise :class:`LookaheadViolation` if it is after ``as_of``."""

        if value > self.as_of:
            raise LookaheadViolation(field, value, self.as_of)
        return value

    def validate(self, value: DateLike, *, field: str) -> date:
        """Parse an explicit date and refuse it when it lies after ``as_of``."""

        if isinstance(value, str) and value.strip().lower() == LATEST:
            raise ToolInputError(f"{field} does not accept {LATEST!r}; give a YYYY-MM-DD date")
        return self.check(parse_date(value, field=field), field=field)

    def resolve(
        self,
        value: DateLike,
        *,
        field: str,
        sessions: Iterable[date] | None = None,
    ) -> date:
        """Resolve an end-style date that may be the literal ``"latest"``.

        ``"latest"`` maps to the last confirmed session ``<= as_of`` drawn from
        ``sessions``; explicit dates are parsed and checked against ``as_of``.
        """

        if isinstance(value, str) and value.strip().lower() == LATEST:
            if sessions is None:
                raise ToolInputError(f"{field}={LATEST!r} requires a session calendar")
            return self.resolve_latest(sessions)
        return self.validate(value, field=field)

    def resolve_latest(self, sessions: Iterable[date]) -> date:
        """Return the last confirmed session on or before ``as_of``."""

        eligible = [session for session in sessions if session <= self.as_of]
        if not eligible:
            raise InsufficientDataError(
                f"no confirmed session exists on or before as_of={self.as_of.isoformat()}"
            )
        return max(eligible)
