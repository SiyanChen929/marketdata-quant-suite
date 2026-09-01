"""Provider-neutral event parsing and point-in-time symbol resolution.

Licensed index-notice parsers deliberately live outside this public project.
They should emit the validated tabular contract implemented here.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import pandas as pd


EVENT_COLUMNS = (
    "announcement_date",
    "effective_close_date",
    "segment",
    "action",
    "security_name",
    "symbol",
)


class EventSchemaError(ValueError):
    """Raised when event inputs violate the public research contract."""


@dataclass(frozen=True)
class ParsedReview:
    """Provider-neutral representation of one index review."""

    announcement_date: pd.Timestamp
    effective_close_date: pd.Timestamp
    segment: str
    source_id: str
    additions: tuple[str, ...]
    deletions: tuple[str, ...]


CORPORATE_STOPWORDS = {
    "A",
    "B",
    "CLASS",
    "CO",
    "COMPANY",
    "CORP",
    "CORPORATION",
    "GROUP",
    "GRP",
    "HOLDING",
    "HOLDINGS",
    "HLDG",
    "HLDGS",
    "INC",
    "INCORPORATED",
    "INTL",
    "LTD",
    "PLC",
    "THE",
}


def normalize_company_name(value: str) -> str:
    """Normalize a company name for audited fuzzy matching."""

    normalized = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    normalized = normalized.upper().replace("&", " AND ")
    tokens = re.findall(r"[A-Z0-9]+", normalized)
    return " ".join(token for token in tokens if token not in CORPORATE_STOPWORDS)


def name_similarity(left: str, right: str) -> float:
    """Return a transparent similarity score in ``[0, 1]``."""

    a = normalize_company_name(left)
    b = normalize_company_name(right)
    if not a or not b:
        return 0.0
    sequence = SequenceMatcher(None, a, b).ratio()
    left_tokens, right_tokens = set(a.split()), set(b.split())
    token_score = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    containment = min(len(a), len(b)) / max(len(a), len(b)) if (a in b or b in a) else 0.0
    return max(sequence, token_score, containment)


def best_name_match(name: str, candidates: Iterable[str]) -> tuple[str, float]:
    """Return the highest-scoring candidate and its score."""

    best_name = ""
    best_score = 0.0
    for candidate in candidates:
        candidate_name = str(candidate)
        score = name_similarity(name, candidate_name)
        if score > best_score:
            best_name, best_score = candidate_name, score
    return best_name, best_score


def review_to_frame(review: ParsedReview) -> pd.DataFrame:
    """Convert one parsed private review into the public event schema."""

    rows: list[dict[str, object]] = []
    for action, names in (("add", review.additions), ("delete", review.deletions)):
        for name in names:
            rows.append(
                {
                    "announcement_date": review.announcement_date,
                    "effective_close_date": review.effective_close_date,
                    "segment": review.segment,
                    "action": action,
                    "security_name": str(name).strip(),
                    "source_id": review.source_id,
                }
            )
    return pd.DataFrame(rows)


def build_primary_event_frame(
    reviews: Iterable[ParsedReview],
    *,
    primary_segment: str = "primary",
    secondary_segment: str = "secondary",
    migration_threshold: float = 0.92,
) -> pd.DataFrame:
    """Classify primary-segment changes and opposite secondary migrations."""

    frames = [review_to_frame(review) for review in reviews]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "announcement_date",
                "effective_close_date",
                "segment",
                "action",
                "security_name",
                "source_id",
                "change_type",
                "secondary_match",
            ]
        )
    all_events = pd.concat(frames, ignore_index=True)
    primary = all_events.loc[all_events["segment"].eq(primary_segment)].copy()
    secondary = all_events.loc[all_events["segment"].eq(secondary_segment)].copy()

    change_types: list[str] = []
    secondary_matches: list[str] = []
    for row in primary.itertuples(index=False):
        opposite = "delete" if row.action == "add" else "add"
        candidates = secondary.loc[
            secondary["effective_close_date"].eq(row.effective_close_date)
            & secondary["action"].eq(opposite),
            "security_name",
        ]
        matched_name, score = best_name_match(row.security_name, candidates)
        if score >= migration_threshold:
            change_types.append("up_from_secondary" if row.action == "add" else "down_to_secondary")
            secondary_matches.append(matched_name)
        else:
            change_types.append("new_to_primary" if row.action == "add" else "exit_primary")
            secondary_matches.append("")
    primary["change_type"] = change_types
    primary["secondary_match"] = secondary_matches
    return primary.sort_values(["effective_close_date", "action", "security_name"]).reset_index(
        drop=True
    )


def validate_event_frame(events: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an event table without consulting future data."""

    missing = sorted(set(EVENT_COLUMNS) - set(events.columns))
    if missing:
        raise EventSchemaError(f"Missing event columns: {', '.join(missing)}")
    frame = events.copy()
    for column in ("announcement_date", "effective_close_date"):
        try:
            frame[column] = pd.to_datetime(frame[column], errors="raise").dt.normalize()
        except (TypeError, ValueError) as exc:
            raise EventSchemaError(f"Invalid {column}") from exc
    frame["action"] = frame["action"].astype(str).str.lower().str.strip()
    invalid_actions = sorted(set(frame["action"]) - {"add", "delete"})
    if invalid_actions:
        raise EventSchemaError(f"Unsupported actions: {', '.join(invalid_actions)}")
    for column in ("segment", "security_name", "symbol"):
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise EventSchemaError(f"Blank values in {column}")
    frame["symbol"] = frame["symbol"].str.upper()
    if frame["announcement_date"].gt(frame["effective_close_date"]).any():
        raise EventSchemaError("announcement_date must not follow effective_close_date")
    duplicate_key = ["effective_close_date", "symbol", "action"]
    if frame.duplicated(duplicate_key).any():
        raise EventSchemaError(f"Duplicate event key: {', '.join(duplicate_key)}")
    return frame.sort_values(["effective_close_date", "action", "symbol"]).reset_index(drop=True)


def load_event_csv(path: str | Path) -> pd.DataFrame:
    """Load a provider-normalized CSV and apply the public schema gate."""

    return validate_event_frame(pd.read_csv(path))


def resolve_symbols(
    events: pd.DataFrame,
    security_master: pd.DataFrame,
    *,
    overrides: dict[str, str] | None = None,
    minimum_score: float = 0.80,
) -> pd.DataFrame:
    """Resolve symbols with explicit provenance; unresolved rows stay blank.

    A production security master should be point-in-time and identifier based.
    Fuzzy name matching is an auditable fallback, not a production substitute.
    """

    required = {"security_name", "symbol"}
    missing = sorted(required - set(security_master.columns))
    if missing:
        raise EventSchemaError(f"Missing security-master columns: {', '.join(missing)}")
    manual = {normalize_company_name(key): value.upper() for key, value in (overrides or {}).items()}
    names = security_master["security_name"].astype(str).tolist()
    symbol_by_name = dict(
        zip(
            security_master["security_name"].astype(str),
            security_master["symbol"].astype(str).str.upper(),
            strict=True,
        )
    )
    rows: list[dict[str, object]] = []
    for event in events.to_dict("records"):
        name = str(event["security_name"])
        key = normalize_company_name(name)
        if key in manual:
            symbol, matched_name, score, method = manual[key], "manual override", 1.0, "override"
        else:
            matched_name, score = best_name_match(name, names)
            symbol = symbol_by_name.get(matched_name, "") if score >= minimum_score else ""
            method = "name_fuzzy" if symbol else "unresolved"
        event.update(
            {
                "symbol": symbol,
                "symbol_match_method": method,
                "symbol_match_name": matched_name,
                "symbol_match_score": round(float(score), 4),
            }
        )
        rows.append(event)
    return pd.DataFrame(rows)
