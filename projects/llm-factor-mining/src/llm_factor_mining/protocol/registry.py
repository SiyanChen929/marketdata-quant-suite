"""Project-wide study registry: fixed splits, exploration log and reveal register.

One append-only, hash-chained JSON Lines file (a
:class:`~llm_factor_mining.protocol.ledger.TrialLedger`, with timestamps)
records, for every *study* (a data set plus its pre-registered windows):

* ``study`` - the data definition (source, symbols, dates), the split
  fractions, lag and horizon, the resulting windows and the pre-registered
  maximum number of test reveals.  Registering fixes the windows before any
  evaluation; a study cannot be re-registered with other terms;
* ``exploration`` - every evaluation made outside a search run (the CLI
  ``evaluate`` command).  These are forking paths that no run's trial ledger
  counts, so the CLI reports their number (with the commit and reveal
  counts, :meth:`StudyRegistry.summary`) next to every store-data result: in
  the ``search`` and ``reveal`` outputs and in the run's ``provenance.json``
  (``registry_at_commit``, ``registry_at_reveal``);
* ``commit`` - commitments of search runs on the study's data;
* ``reveal`` / ``reveal_refused`` - every test-window reveal, capped by the
  study's pre-registered count.  A reveal must match a commitment recorded
  here beforehand.

Like the trial ledger, the register is self-attested: it makes repeated
reveals visible and countable, not impossible.  Publishing its head hash (or
the commitment hashes) outside the repository is what lets a third party
check it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..jsonutil import json_safe, sha256_json
from .ledger import TrialLedger
from .seal import SealError
from .splits import Splits, splits_from_dict


STUDY_KIND = "study"
EXPLORATION_KIND = "exploration"
COMMIT_KIND = "commit"
REVEAL_KIND = "reveal"
REVEAL_REFUSED_KIND = "reveal_refused"


class RegistryError(ValueError):
    """A study is missing, already registered with other terms, or misused."""


@dataclass(frozen=True)
class StudySpec:
    """Data definition and pre-registered protocol terms of one study."""

    name: str
    source: str
    symbols: tuple[str, ...]
    start: str
    end: str
    fractions: tuple[float, float, float]
    lag: int
    max_horizon: int
    max_reveals: int = 1

    def __post_init__(self) -> None:
        if not self.name or any(char.isspace() for char in self.name):
            raise RegistryError("study names must be non-empty and contain no whitespace")
        if self.source not in ("store", "synthetic"):
            raise RegistryError("study source must be 'store' or 'synthetic'")
        if self.max_reveals < 1:
            raise RegistryError("max_reveals must be at least 1")
        object.__setattr__(self, "symbols", tuple(str(symbol) for symbol in self.symbols))
        object.__setattr__(self, "fractions", tuple(float(value) for value in self.fractions))

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StudySpec":
        return cls(
            name=str(payload["name"]),
            source=str(payload["source"]),
            symbols=tuple(payload["symbols"]),
            start=str(payload["start"]),
            end=str(payload["end"]),
            fractions=tuple(payload["fractions"]),  # type: ignore[arg-type]
            lag=int(payload["lag"]),
            max_horizon=int(payload["max_horizon"]),
            max_reveals=int(payload["max_reveals"]),
        )


@dataclass(frozen=True)
class Study:
    """A registered study: its terms, its windows and the registration record hash."""

    spec: StudySpec
    splits: Splits
    record_hash: str


class StudyRegistry:
    """File-backed register shared by every run and command on a study."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.ledger = TrialLedger(self.path, timestamps=True)

    @property
    def head_hash(self) -> str:
        return self.ledger.head_hash

    def _records(self, kind: str, study: str) -> list[Mapping[str, Any]]:
        self.ledger.sync()
        return [record.payload for record in self.ledger.records if record.kind == kind and record.payload.get("study") == study]

    # ----------------------------------------------------------------- studies

    def study(self, name: str) -> Study | None:
        self.ledger.sync()
        for record in self.ledger.records:
            if record.kind == STUDY_KIND and record.payload.get("study") == name:
                payload = record.payload
                return Study(StudySpec.from_dict(payload["spec"]), splits_from_dict(payload["splits"]), record.record_hash)
        return None

    def require(self, name: str) -> Study:
        found = self.study(name)
        if found is None:
            raise RegistryError(
                f"study {name!r} is not registered in {self.path}; register its windows first "
                "(llm-factor-mining register-study ...)"
            )
        return found

    def register_study(self, spec: StudySpec, splits: Splits) -> Study:
        """Record a study and its windows; refuses to change an existing registration."""

        existing = self.study(spec.name)
        if existing is not None:
            if existing.spec == spec and existing.splits.to_dict() == splits.to_dict():
                return existing
            raise RegistryError(f"study {spec.name!r} is already registered with different terms")
        if (splits.lag, splits.max_horizon) != (spec.lag, spec.max_horizon):
            raise RegistryError("splits were built for a different lag or horizon than the study terms")
        payload = {
            "study": spec.name,
            "spec": spec.to_dict(),
            "splits": splits.to_dict(),
            "splits_sha256": sha256_json(splits.to_dict()),
        }
        record = self.ledger.append(STUDY_KIND, payload)
        return Study(spec, splits, record.record_hash)

    # ------------------------------------------------------------- exploration

    def record_exploration(self, study: str, **details: Any) -> int:
        """Log one out-of-search evaluation; returns the study's exploration count."""

        self.require(study)
        self.ledger.append(EXPLORATION_KIND, {"study": study, **json_safe(details)})
        return self.n_explorations(study)

    def n_explorations(self, study: str) -> int:
        return len(self._records(EXPLORATION_KIND, study))

    # ------------------------------------------------------- commits / reveals

    def record_commit(self, study: str, **details: Any) -> None:
        self.require(study)
        if "commitment" not in details:
            raise RegistryError("a commit record needs the commitment hash")
        self.ledger.append(COMMIT_KIND, {"study": study, **json_safe(details)})

    def commits(self, study: str) -> list[Mapping[str, Any]]:
        return self._records(COMMIT_KIND, study)

    def reveals(self, study: str) -> list[Mapping[str, Any]]:
        return self._records(REVEAL_KIND, study)

    def reveal_authorizer(self, study: str, *, run: str = "") -> Callable[[Mapping[str, Any]], None]:
        """Callable for :meth:`SealedHoldout.reveal`: checks the cap, then records the reveal."""

        registered = self.require(study)

        def authorize(descriptor: Mapping[str, Any]) -> None:
            details = {"study": study, "run": run, **json_safe(dict(descriptor))}
            test = registered.splits.to_dict()["test"]
            window = descriptor.get("window") or {}
            reason = ""
            if (window.get("start"), window.get("end")) != (test["start"], test["end"]):
                reason = "test window differs from the study's registered test window"
            elif not any(commit.get("commitment") == descriptor.get("commitment") for commit in self.commits(study)):
                reason = "commitment was never recorded in the register"
            elif len(self.reveals(study)) >= registered.spec.max_reveals:
                reason = f"the study's pre-registered maximum of {registered.spec.max_reveals} reveal(s) is used up"
            if reason:
                self.ledger.append(REVEAL_REFUSED_KIND, {**details, "reason": reason})
                raise SealError(f"reveal register refused the reveal: {reason}")
            self.ledger.append(REVEAL_KIND, details)

        return authorize

    def summary(self, study: str) -> dict[str, Any]:
        registered = self.require(study)
        return {
            "study": study,
            "registry": str(self.path),
            "registry_head": self.head_hash,
            "max_reveals": registered.spec.max_reveals,
            "n_reveals": len(self.reveals(study)),
            "n_commits": len(self.commits(study)),
            "n_explorations": self.n_explorations(study),
            "splits": registered.splits.to_dict(),
        }


def study_symbols(values: Sequence[str]) -> tuple[str, ...]:
    """Normalized, de-duplicated, sorted symbol list for a study definition."""

    return tuple(sorted({str(value).strip().upper() for value in values if str(value).strip()}))


__all__ = [
    "RegistryError",
    "Study",
    "StudyRegistry",
    "StudySpec",
    "study_symbols",
]
