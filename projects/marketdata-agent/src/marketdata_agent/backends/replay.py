"""Record model turns to JSONL and replay them offline, keyed by request fingerprint.

``RecordingBackend`` wraps any backend. After each live step it appends
``{"request_sha256", "config", "turn"}`` to a JSONL file, where
``request_sha256`` hashes the backend configuration, system prompt, full
history and tool definitions (:func:`~marketdata_agent.backends.base.request_fingerprint`).
With ``reuse=True`` (the default) a request that is already in the recording
is served from it instead of calling the model again, so a retried episode or
a resumed run (``bench run --resume``) pays only for the turns it has not
recorded yet. A request is therefore sampled at most once per recording, and
repetitions of a run use separate recordings. An existing recording is loaded,
never truncated.
``ReplayBackend`` serves recorded turns for identical requests and raises
:class:`ReplayMissError` for any request it has not seen.

Tool results are deterministic functions of the data and the arguments, so an
episode replayed against the same data and prompt issues the same requests in
the same order. A recorded Claude run can therefore be re-scored, or checked
under a new scorer, without network access. The recording is a cache, not an
audit record: the hash-chained audit log is the tamper-evident record.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from ..provenance import canonical_json
from .base import BackendError, BackendTurn, LLMBackend, request_fingerprint, to_plain


RECORD_FORMAT = "marketdata-agent/turn-recording/v1"


class ReplayMissError(BackendError):
    """No recorded turn exists for this request fingerprint."""

    def __init__(self, fingerprint: str) -> None:
        super().__init__("replay_miss", f"no recorded turn for request {fingerprint[:16]}")
        self.fingerprint = fingerprint


def _load_recording(path: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    turns: dict[str, list[dict[str, Any]]] = defaultdict(list)
    configs: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("format") != RECORD_FORMAT:
            raise ValueError(f"{path}:{number}: not a {RECORD_FORMAT} record")
        turns[record["request_sha256"]].append(record["turn"])
        if record["config"] not in configs:
            configs.append(record["config"])
    return turns, configs


class RecordingBackend:
    """Delegate to ``inner`` and persist every live turn for later replay.

    ``reuse=True`` serves requests already present in the recording (for
    example from an interrupted run) instead of calling ``inner``;
    :attr:`reused` and :attr:`live_calls` count both paths.
    """

    def __init__(self, inner: LLMBackend, path: str | Path, *, reuse: bool = True) -> None:
        self.inner = inner
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.name = f"record:{inner.name}"
        self.reuse = bool(reuse)
        self.reused = 0
        self.live_calls = 0
        self._turns: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if self.path.exists() and self.path.stat().st_size > 0:
            self._turns, configs = _load_recording(self.path)
            current = to_plain(inner.describe())
            if any(config != current for config in configs):
                raise ValueError(f"{self.path} was recorded with a different backend configuration")

    def describe(self) -> dict[str, Any]:
        return self.inner.describe()

    def step(
        self,
        system: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> BackendTurn:
        config = self.inner.describe()
        fingerprint = request_fingerprint(system, messages, tools, config)
        if self.reuse and self._turns.get(fingerprint):
            self.reused += 1
            return BackendTurn.from_dict(self._turns[fingerprint][0])
        turn = self.inner.step(system, messages, tools)
        self.live_calls += 1
        record = {
            "format": RECORD_FORMAT,
            "request_sha256": fingerprint,
            "config": to_plain(config),
            "turn": turn.to_dict(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(record) + "\n")
        self._turns[fingerprint].append(record["turn"])
        return turn


class ReplayBackend:
    """Serve recorded turns for requests identical to the recorded ones.

    Identical requests recorded more than once are served in recorded order,
    and the last one is repeated after that.
    """

    def __init__(self, path: str | Path, *, config: Mapping[str, Any] | None = None) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"no recording at {self.path}")
        self._turns, configs = _load_recording(self.path)
        if config is None:
            if len(configs) != 1:
                raise ValueError(f"{self.path}: expected one recorded backend config, found {len(configs)}; pass config=")
            config = configs[0]
        self.config = dict(config)
        self._served: dict[str, int] = defaultdict(int)
        self.name = f"replay:{self.config.get('backend', 'unknown')}"

    def describe(self) -> dict[str, Any]:
        return dict(self.config)

    def step(
        self,
        system: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> BackendTurn:
        fingerprint = request_fingerprint(system, messages, tools, self.config)
        recorded = self._turns.get(fingerprint)
        if not recorded:
            raise ReplayMissError(fingerprint)
        index = min(self._served[fingerprint], len(recorded) - 1)
        self._served[fingerprint] += 1
        return BackendTurn.from_dict(recorded[index])

    def __len__(self) -> int:
        return sum(len(turns) for turns in self._turns.values())
