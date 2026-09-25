"""Append-only, hash-chained trial ledger (JSON Lines).

Every record carries ``prev_hash`` (the previous record's ``record_hash``, or
64 zeros for the first record) and ``record_hash`` = SHA-256 of the canonical
JSON of ``{version, seq, kind, timestamp, payload, prev_hash}``.  Editing,
deleting, inserting or reordering any line breaks the chain and is detected by
:meth:`TrialLedger.verify` / :func:`verify_ledger_file`.  Truncating the *tail*
cannot be detected from the file alone; compare against a separately recorded
head hash (``expected_head``).

Scope of the guarantee: the chain is self-attested.  It detects accidental or
naive edits, but anyone with write access can drop records, rebuild the chain
and rewrite the head stored next to it.  A head hash only proves something to
a third party once it has been published *outside* the run directory (for
example in a pushed Git commit or a public timestamp) before the result it
protects was revealed; see the two-step ``search`` / ``reveal`` CLI flow.

The ledger counts every proposal the search processed.  Records of kind
``"trial"`` (evaluated, invalid, or failed during evaluation) are the family
size used by multiple-testing corrections.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from ..jsonutil import canonical_json, json_safe


LEDGER_VERSION = "llm-factor-mining/ledger-v1"
GENESIS_HASH = "0" * 64
TRIAL_KIND = "trial"


class LedgerIntegrityError(RuntimeError):
    """The ledger's hash chain does not verify."""


@dataclass(frozen=True)
class LedgerRecord:
    """One immutable ledger entry."""

    seq: int
    kind: str
    payload: Mapping[str, Any]
    prev_hash: str
    record_hash: str
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": LEDGER_VERSION,
            "seq": self.seq,
            "kind": self.kind,
            "timestamp": self.timestamp,
            "payload": json_safe(self.payload),
            "prev_hash": self.prev_hash,
            "record_hash": self.record_hash,
        }


def record_digest(
    seq: int, kind: str, payload: Mapping[str, Any], prev_hash: str, timestamp: str | None
) -> str:
    """Hash of a record's content (everything except ``record_hash``)."""

    body = {
        "version": LEDGER_VERSION,
        "seq": int(seq),
        "kind": str(kind),
        "timestamp": timestamp,
        "payload": payload,
        "prev_hash": prev_hash,
    }
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TrialLedger:
    """Hash-chained record of everything a search did, optionally mirrored to disk.

    Opening an existing file verifies it and continues the chain.  There is no
    API to modify or delete a record.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        timestamps: bool = True,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.path = None if path is None else Path(path)
        self.timestamps = bool(timestamps)
        self._clock = clock or _utc_now
        self._records: list[LedgerRecord] = []
        self._size = 0
        if self.path is not None and self.path.exists() and self.path.stat().st_size > 0:
            self._records = verify_ledger_file(self.path)
            self._size = self.path.stat().st_size

    def sync(self) -> None:
        """Adopt records another writer appended to the same file.

        A no-op for in-memory ledgers.  Raises :class:`LedgerIntegrityError`
        when the file no longer extends this writer's chain (edited, truncated
        or rewritten).
        """

        if self.path is None:
            return
        size = self.path.stat().st_size if self.path.exists() else 0
        if size == self._size:
            return
        records = verify_ledger_file(self.path) if size else []
        mine = [record.record_hash for record in self._records]
        if [record.record_hash for record in records[: len(mine)]] != mine:
            raise LedgerIntegrityError(
                "ledger file no longer extends this writer's chain (edited, truncated or rewritten)"
            )
        self._records = records
        self._size = size

    # ------------------------------------------------------------------ access

    @property
    def records(self) -> tuple[LedgerRecord, ...]:
        return tuple(self._records)

    @property
    def head_hash(self) -> str:
        return self._records[-1].record_hash if self._records else GENESIS_HASH

    def __len__(self) -> int:
        return len(self._records)

    def of_kind(self, kind: str) -> list[LedgerRecord]:
        return [record for record in self._records if record.kind == kind]

    @property
    def n_trials(self) -> int:
        """Number of ``"trial"`` records: the multiple-testing family size."""

        return sum(1 for record in self._records if record.kind == TRIAL_KIND)

    # ------------------------------------------------------------------ writes

    def append(self, kind: str, payload: Mapping[str, Any]) -> LedgerRecord:
        """Append one record and (if file-backed) flush it to disk."""

        if not kind or not isinstance(kind, str):
            raise ValueError("record kind must be a non-empty string")
        clean = json.loads(canonical_json(payload))  # detached, JSON-native copy
        self.sync()
        seq = len(self._records)
        prev = self.head_hash
        timestamp = self._clock() if self.timestamps else None
        digest = record_digest(seq, kind, clean, prev, timestamp)
        record = LedgerRecord(seq, kind, clean, prev, digest, timestamp)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(canonical_json(record.to_dict()) + "\n")
                handle.flush()
            self._size = self.path.stat().st_size
        self._records.append(record)
        return record

    # ------------------------------------------------------------ verification

    def verify(self, expected_head: str | None = None) -> None:
        """Re-verify the in-memory chain and, if file-backed, the file itself."""

        self.sync()
        _verify_chain(self._records, expected_head)
        if self.path is not None:
            on_disk = verify_ledger_file(self.path, expected_head)
            if [record.record_hash for record in on_disk] != [
                record.record_hash for record in self._records
            ]:
                raise LedgerIntegrityError("ledger file diverges from the in-memory chain")


def _verify_chain(records: list[LedgerRecord], expected_head: str | None = None) -> None:
    prev = GENESIS_HASH
    for position, record in enumerate(records):
        if record.seq != position:
            raise LedgerIntegrityError(f"record {position} has sequence number {record.seq}")
        if record.prev_hash != prev:
            raise LedgerIntegrityError(f"record {position} does not link to its predecessor")
        digest = record_digest(record.seq, record.kind, record.payload, record.prev_hash, record.timestamp)
        if digest != record.record_hash:
            raise LedgerIntegrityError(f"record {position} content does not match its hash")
        prev = record.record_hash
    if expected_head is not None and prev != expected_head:
        raise LedgerIntegrityError("ledger head does not match the expected head hash")


def verify_ledger_file(path: str | Path, expected_head: str | None = None) -> list[LedgerRecord]:
    """Parse and verify a ledger file; raises :class:`LedgerIntegrityError`."""

    text = Path(path).read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise LedgerIntegrityError("ledger file ends with a partial record")
    records: list[LedgerRecord] = []
    for number, line in enumerate(text.splitlines()):
        if not line.strip():
            raise LedgerIntegrityError(f"blank line {number + 1} in ledger")
        try:
            raw = json.loads(line)
            if raw.get("version") != LEDGER_VERSION:
                raise LedgerIntegrityError(f"line {number + 1} has an unknown ledger version")
            records.append(
                LedgerRecord(
                    seq=int(raw["seq"]),
                    kind=str(raw["kind"]),
                    payload=raw["payload"],
                    prev_hash=str(raw["prev_hash"]),
                    record_hash=str(raw["record_hash"]),
                    timestamp=raw.get("timestamp"),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LedgerIntegrityError(f"line {number + 1} is not a ledger record: {exc}") from exc
    _verify_chain(records, expected_head)
    return records


__all__ = [
    "GENESIS_HASH",
    "LEDGER_VERSION",
    "LedgerIntegrityError",
    "LedgerRecord",
    "TRIAL_KIND",
    "TrialLedger",
    "record_digest",
    "verify_ledger_file",
]
