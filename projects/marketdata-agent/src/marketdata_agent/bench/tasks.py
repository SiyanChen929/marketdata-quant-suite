"""Benchmark task schema.

A task is a question, an as-of date, and an expected outcome:

* ``numeric``: a value in the task's ``unit`` with an absolute ``tolerance``,
  compared in natural units (a return of 4.21% is stored as 0.0421);
* ``ranking``: an ordered tuple of symbols. A one-element tuple asks for the top symbol;
* ``abstain``: the agent must answer ``INSUFFICIENT_DATA``, because the data
  needed is after the as-of date or does not exist. ``unit`` records the kind
  of quantity the question asks about, so that an abstention accompanied by
  an unsupported value of that kind can be scored as a leak;
* ``refuse``: the agent must answer ``EXECUTION_REFUSED`` and make no
  order-execution attempt (a ``propose_order`` proposal is allowed).

``reference_plan``, ``naive_plan`` and ``answer_form`` are used **only** by the
scripted harness-validation baselines. An LLM agent sees the question and the
system prompt only. ``hindsight`` is the value a cutoff-ignoring computation
would produce. It is used to show that the ``no_guard`` ablation's answers really
are built from data after the as-of date.

``metadata["item_key"]`` identifies the *functional item* a task tests
(:func:`marketdata_agent.bench.generator.item_key`); tasks with equal keys are
duplicates and are never generated twice within, or across, suites.
``metadata["cluster"]`` groups items that share a template and symbol, for
cluster-robust analysis.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

from ..provenance import canonical_json, json_safe, sha256_json
from ..synthetic import SyntheticPanelSpec


ExpectedKind = Literal["numeric", "ranking", "abstain", "refuse"]
Unit = Literal["percent", "usd", "ratio", "count"]
CATEGORIES = ("lookup", "compute", "multi_step", "pit_trap", "policy_trap", "unknown_symbol")
EXPECTED_KINDS = ("numeric", "ranking", "abstain", "refuse")


@dataclass(frozen=True)
class Expected:
    """Ground truth for one task."""

    kind: ExpectedKind
    value: float | tuple[str, ...] | None = None
    tolerance: float | None = None
    unit: Unit | None = None
    match_magnitude: bool = False

    def __post_init__(self) -> None:
        if self.kind not in EXPECTED_KINDS:
            raise ValueError(f"unknown expected kind {self.kind!r}")
        if self.kind == "numeric" and (not isinstance(self.value, float) or self.tolerance is None or self.unit is None):
            raise ValueError("numeric expectations need a float value, a tolerance and a unit")
        if self.kind == "ranking" and (not isinstance(self.value, tuple) or not self.value):
            raise ValueError("ranking expectations need a non-empty tuple of symbols")
        if self.kind in {"abstain", "refuse"} and self.value is not None:
            raise ValueError(f"{self.kind} expectations carry no value")
        if self.kind == "refuse" and self.unit is not None:
            raise ValueError("refuse expectations carry no unit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": list(self.value) if isinstance(self.value, tuple) else self.value,
            "tolerance": self.tolerance,
            "unit": self.unit,
            "match_magnitude": self.match_magnitude,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Expected":
        value = payload.get("value")
        return cls(
            kind=payload["kind"],
            value=tuple(value) if isinstance(value, list) else (float(value) if value is not None else None),
            tolerance=payload.get("tolerance"),
            unit=payload.get("unit"),
            match_magnitude=bool(payload.get("match_magnitude", False)),
        )


@dataclass(frozen=True)
class ToolStep:
    """One tool call in a baseline plan."""

    tool: str
    args: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": json_safe(dict(self.args))}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolStep":
        return cls(payload["tool"], dict(payload["args"]))


@dataclass(frozen=True)
class AnswerForm:
    """How a scripted baseline turns tool results into an answer.

    ``kind``: ``scalar`` (one value ``key`` from the first ``tool`` result),
    ``ranking`` (the order in a ``compare_returns`` result), or ``top`` (the
    symbol with the ``direction``-most ``key`` across several ``tool`` results).
    """

    kind: Literal["scalar", "ranking", "top"]
    tool: str
    key: str
    unit: Unit | None = None
    direction: Literal["max", "min"] = "max"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "tool": self.tool, "key": self.key, "unit": self.unit, "direction": self.direction}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnswerForm":
        return cls(payload["kind"], payload["tool"], payload["key"], payload.get("unit"), payload.get("direction", "max"))


@dataclass(frozen=True)
class Task:
    """One benchmark question with its ground truth."""

    id: str
    category: str
    subcategory: str
    question: str
    as_of: str
    symbols: tuple[str, ...]
    expected: Expected
    reference_plan: tuple[ToolStep, ...] = ()
    naive_plan: tuple[ToolStep, ...] = ()
    answer_form: AnswerForm | None = None
    hindsight: Expected | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown category {self.category!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "subcategory": self.subcategory,
            "question": self.question,
            "as_of": self.as_of,
            "symbols": list(self.symbols),
            "expected": self.expected.to_dict(),
            "reference_plan": [step.to_dict() for step in self.reference_plan],
            "naive_plan": [step.to_dict() for step in self.naive_plan],
            "answer_form": self.answer_form.to_dict() if self.answer_form else None,
            "hindsight": self.hindsight.to_dict() if self.hindsight else None,
            "metadata": json_safe(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Task":
        return cls(
            id=payload["id"],
            category=payload["category"],
            subcategory=payload["subcategory"],
            question=payload["question"],
            as_of=payload["as_of"],
            symbols=tuple(payload["symbols"]),
            expected=Expected.from_dict(payload["expected"]),
            reference_plan=tuple(ToolStep.from_dict(s) for s in payload.get("reference_plan", [])),
            naive_plan=tuple(ToolStep.from_dict(s) for s in payload.get("naive_plan", [])),
            answer_form=AnswerForm.from_dict(payload["answer_form"]) if payload.get("answer_form") else None,
            hindsight=Expected.from_dict(payload["hindsight"]) if payload.get("hindsight") else None,
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class TaskSuite:
    """A deterministic, fingerprinted set of tasks over one synthetic dataset."""

    tasks: tuple[Task, ...]
    dataset: SyntheticPanelSpec
    seed: int
    generator_version: str

    def __post_init__(self) -> None:
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique")

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator_version": self.generator_version,
            "seed": self.seed,
            "dataset": self.dataset.to_dict(),
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def sha256(self) -> str:
        """Fingerprint of the dataset spec, the seed and every task (including ground truth)."""

        return sha256_json(self.to_dict())

    def item_keys(self) -> list[str]:
        return [str(task.metadata.get("item_key", task.id)) for task in self.tasks]

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in self.tasks:
            counts[task.category] = counts.get(task.category, 0) + 1
        return dict(sorted(counts.items()))

    def subset(self, n: int) -> "TaskSuite":
        """Take ``n`` tasks round-robin across categories, keeping generation order within each."""

        if n >= len(self.tasks):
            return self
        if n < 1:
            raise ValueError("n must be positive")
        queues: dict[str, list[Task]] = {}
        for task in self.tasks:
            queues.setdefault(task.category, []).append(task)
        chosen: list[Task] = []
        while len(chosen) < n:
            for category in CATEGORIES:
                queue = queues.get(category)
                if queue and len(chosen) < n:
                    chosen.append(queue.pop(0))
        keep = {task.id for task in chosen}
        return TaskSuite(tuple(t for t in self.tasks if t.id in keep), self.dataset, self.seed, self.generator_version)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(json.loads(canonical_json(self.to_dict())), indent=1) + "\n", encoding="utf-8")
        return target


def load_suite(path: str | Path) -> TaskSuite:
    """Load a suite written by :meth:`TaskSuite.write_json`."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    dataset = payload["dataset"]
    spec = SyntheticPanelSpec(
        symbols=tuple(dataset["symbols"]),
        start=dataset["start"],
        end=dataset["end"],
        seed=int(dataset["seed"]),
        market_vol=float(dataset["market_vol"]),
        idio_vol=float(dataset["idio_vol"]),
        listings=tuple(sorted(dict(dataset["listings"]).items())),
        source=dataset["source"],
    )
    tasks = tuple(Task.from_dict(item) for item in payload["tasks"])
    return TaskSuite(tasks, spec, int(payload["seed"]), payload["generator_version"])
