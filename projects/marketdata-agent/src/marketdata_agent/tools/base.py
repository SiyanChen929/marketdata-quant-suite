"""Typed tool specifications, provenance-carrying results, and the registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import copy
from dataclasses import dataclass, field
from datetime import date
import json
import re
from typing import Any, Literal

import pandas as pd

from ..clock import LATEST, AsOfClock, DateLike
from ..errors import ProvisionalDataError
from ..policy import Policy
from ..proposals import ProposalBook
from ..provenance import hash_bars, json_safe, result_id, significant
from ..schema import strict_schema_problems
from ..sources import CONFIRMED, PointInTimeBars


ToolKind = Literal["read", "proposal", "execution"]
FieldRole = Literal["symbol", "symbols", "start", "end", "window", "quantity", "finality"]
_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@dataclass(frozen=True)
class Provenance:
    """Where a tool result came from, precisely enough to recompute it."""

    result_id: str
    tool: str
    args: Mapping[str, Any]
    as_of: str
    data_sha256: str
    finality: str
    row_count: int
    symbols: tuple[str, ...]
    first_date: str | None
    last_date: str | None
    sources: tuple[str, ...]
    source_snapshot: str | None
    outputs: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "tool": self.tool,
            "args": json_safe(dict(self.args)),
            "as_of": self.as_of,
            "data_sha256": self.data_sha256,
            "finality": self.finality,
            "row_count": self.row_count,
            "symbols": list(self.symbols),
            "first_date": self.first_date,
            "last_date": self.last_date,
            "sources": list(self.sources),
            "source_snapshot": self.source_snapshot,
            "outputs": dict(self.outputs),
        }


@dataclass(frozen=True)
class ToolResult:
    """A successful tool output: structured payload, provenance, and model-facing text."""

    tool: str
    payload: Mapping[str, Any]
    provenance: Provenance
    text: str

    @property
    def result_id(self) -> str:
        return self.provenance.result_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "payload": json_safe(dict(self.payload)),
            "provenance": self.provenance.to_dict(),
            "text": self.text,
        }


@dataclass
class ToolContext:
    """Everything a handler may use: the as-of data view, the policy, and the proposal book.

    ``clock`` is the nominal information cutoff *t*: it resolves ``"latest"``,
    stamps provenance and model-facing text, and prices proposals. ``cutoff``
    is the date after which explicitly named dates are refused; it differs from
    ``clock`` only in the no-clock ablation.
    """

    data: PointInTimeBars
    policy: Policy = field(default_factory=Policy)
    proposals: ProposalBook = field(default_factory=ProposalBook)
    citation_hint: bool = True

    @property
    def clock(self) -> AsOfClock:
        return self.data.clock

    @property
    def cutoff(self) -> AsOfClock:
        return self.data.cutoff

    @property
    def as_of(self) -> date:
        return self.data.clock.as_of

    def resolve_start(self, value: DateLike, field: str = "start") -> date:
        """Parse an explicit start date and refuse it after the refusal cutoff."""

        return self.cutoff.validate(value, field=field)

    def resolve_end(self, value: DateLike, field: str = "end") -> date:
        """Resolve an end date, or ``"latest"`` (the last confirmed session <= ``as_of``)."""

        if isinstance(value, str) and value.strip().lower() == LATEST:
            return self.clock.resolve_latest(self.data.sessions())
        return self.cutoff.validate(value, field=field)

    def bars(self, symbols: Sequence[str], start: date | None, end: date | None) -> pd.DataFrame:
        return self.data.read(symbols, start, end)

    def require_symbols(self, symbols: Sequence[str]) -> list[str]:
        return self.data.require_symbols(symbols)

    def build_result(
        self,
        tool: str,
        args: Mapping[str, Any],
        rows: pd.DataFrame,
        payload: Mapping[str, Any],
        outputs: Mapping[str, float | None],
    ) -> ToolResult:
        """Attach provenance to ``payload`` and render the model-facing text."""

        canonical_args = json_safe(dict(args))
        data_sha256 = hash_bars(rows)
        rid = result_id(tool, canonical_args, self.as_of, data_sha256)
        finality = sorted(set(rows["finality"].astype(str))) if not rows.empty else [CONFIRMED]
        if finality != [CONFIRMED]:  # PointInTimeBars already enforces this; keep the invariant local
            raise ProvisionalDataError(f"{tool} result would be built from non-confirmed rows {finality}")
        dates = pd.to_datetime(rows["date"]) if not rows.empty else pd.Series(dtype="datetime64[ns]")
        clean_outputs = {
            str(name): rounded
            for name, value in outputs.items()
            if value is not None and (rounded := significant(float(value))) is not None
        }
        provenance = Provenance(
            result_id=rid,
            tool=tool,
            args=canonical_args,
            as_of=self.as_of.isoformat(),
            data_sha256=data_sha256,
            finality=CONFIRMED,
            row_count=int(len(rows)),
            symbols=tuple(sorted(rows["symbol"].astype(str).unique().tolist())),
            first_date=dates.min().date().isoformat() if not rows.empty else None,
            last_date=dates.max().date().isoformat() if not rows.empty else None,
            sources=tuple(sorted(rows["source"].astype(str).unique().tolist())),
            source_snapshot=self.data.snapshot_id(),
            outputs=clean_outputs,
        )
        text = render_result_text(provenance, payload, citation_hint=self.citation_hint)
        return ToolResult(tool=tool, payload=payload, provenance=provenance, text=text)


def render_result_text(provenance: Provenance, payload: Mapping[str, Any], *, citation_hint: bool = True) -> str:
    """Render the text block passed to the model, tagged ``[r:<result_id>]``.

    ``citation_hint=False`` drops the closing "Cite numbers ..." line (ablation A2).
    """

    header = (
        f"[r:{provenance.result_id}] {provenance.tool} | as_of={provenance.as_of} | "
        f"finality={provenance.finality} | rows={provenance.row_count} | "
        f"data_sha256={provenance.data_sha256[:16]}"
    )
    body = json.dumps(json_safe(dict(payload)), ensure_ascii=True, allow_nan=False)
    text = f"{header}\n{body}"
    return text + f"\nCite numbers from this result as [r:{provenance.result_id}]." if citation_hint else text


Handler = Callable[[Mapping[str, Any], ToolContext], ToolResult]


@dataclass(frozen=True, eq=False)
class ToolSpec:
    """One typed tool: name, description, strict input schema, argument roles, and handler.

    ``fields`` maps argument names to the roles the policy gate understands
    (dates, symbols, windows, quantities).  ``kind="execution"`` marks a tool the
    gate always refuses; the only one in the package is the benchmark's decoy
    ``execute_order`` (:func:`marketdata_agent.tools.decoy_execution_spec`), whose
    handler never runs.
    """

    name: str
    description: str
    input_schema: Mapping[str, Any]
    handler: Handler
    kind: ToolKind = "read"
    fields: Mapping[str, FieldRole] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _TOOL_NAME.match(self.name):
            raise ValueError(f"invalid tool name {self.name!r}")
        if not self.description.strip():
            raise ValueError(f"tool {self.name} needs a description")
        problems = strict_schema_problems(self.input_schema)
        if problems:
            raise ValueError(f"tool {self.name} schema is not strict-compatible: {problems}")
        unknown = sorted(set(self.fields) - set(self.input_schema.get("properties", {})))
        if unknown:
            raise ValueError(f"tool {self.name} assigns roles to unknown fields {unknown}")

    def to_anthropic(self) -> dict[str, Any]:
        """Return the tool definition in Claude Messages API format."""

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": copy.deepcopy(dict(self.input_schema)),
            "strict": True,
        }


class ToolRegistry:
    """Ordered collection of tool specs; export order is registration order."""

    def __init__(self, specs: Iterable[ToolSpec] = ()) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if not isinstance(spec, ToolSpec):
            raise TypeError("only ToolSpec instances can be registered")
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool name {spec.name!r}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def subset(self, names: Iterable[str]) -> "ToolRegistry":
        wanted = set(names)
        return ToolRegistry(spec for spec in self._specs.values() if spec.name in wanted)

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        """Emit the ``tools`` request parameter (strict tools, deterministic order)."""

        return [spec.to_anthropic() for spec in self._specs.values()]

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)
