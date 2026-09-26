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
_MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
_MONTH_NAME = r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_LENIENT_DATES = (
    re.compile(r"(?<!\d)(?P<y>\d{4})[-/.](?P<m>\d{1,2})[-/.](?P<d>\d{1,2})(?!\d)"),  # 2023-07-31, 2023/07/31, 2023-07-31T00:00
    re.compile(r"(?<!\d)(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<y>\d{4})(?!\d)"),  # 07/31/2023
    re.compile(r"(?<!\d)(?P<y>(?:19|20)\d{2})(?P<m>\d{2})(?P<d>\d{2})(?!\d)"),  # 20230731
    re.compile(rf"\b{_MONTH_NAME}\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y>\d{{4}})\b", re.IGNORECASE),
    re.compile(rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+{_MONTH_NAME}\.?,?\s+(?P<y>\d{{4}})\b", re.IGNORECASE),
)

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


def dates_mentioned(value: object) -> list[date]:
    """Calendar dates written in any common form anywhere inside ``value``.

    Used to count look-ahead attempts in calls that are denied before their
    dates are parsed strictly (an unknown tool, a schema failure, a non-ISO
    date string). Strings, integers, lists and objects are searched
    recursively; forms are ISO (with or without a time), ``YYYY/MM/DD``,
    ``MM/DD/YYYY``, ``YYYYMMDD`` and month names (``July 31, 2023``,
    ``31 July 2023``). Impossible dates are skipped.
    """

    found: list[date] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found += dates_mentioned(key) + dates_mentioned(item)
        return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found += dates_mentioned(item)
        return found
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return found
    text = str(value)
    for pattern in _LENIENT_DATES:
        for match in pattern.finditer(text):
            parts = match.groupdict()
            month = _MONTHS[parts["mon"][:3].lower()] if parts.get("mon") else int(parts["m"])
            try:
                found.append(date(int(parts["y"]), month, int(parts["d"])))
            except ValueError:
                continue
    return found


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
