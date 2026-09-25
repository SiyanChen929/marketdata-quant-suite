"""Governed, point-in-time-safe LLM trading copilot over the suite's MarketData gateway.

Core: confirmed-only data access, an as-of clock that refuses look-ahead, a
policy gate that counts every execution and look-ahead attempt, strict typed
tools with provenance, and a hash-chained audit log.  Agent layer: a
backend-neutral agent loop, Claude/scripted/record-replay backends, and
numeric-claim grounding.  Benchmark (``marketdata_agent.bench``): distinct-item
task generation, independent ground truth, scripted baselines, strict scoring,
a runner that refuses to report failed runs, and the pre-registered design and
statistics.  The package imports without the optional ``anthropic``
dependency.
"""

from __future__ import annotations

__version__ = "0.3.0"

from .agent import ABSTAIN_TOKEN, REFUSE_TOKEN, Copilot, Episode, ToolCallRecord, load_system_prompt
from .backends import (
    AnthropicBackend,
    AnthropicConfig,
    BackendError,
    BackendTurn,
    LLMBackend,
    RecordingBackend,
    ReplayBackend,
    ScriptedBackend,
    ScriptedTurn,
)
from .audit import (
    AuditIntegrityError,
    AuditLog,
    ChainHead,
    ChainVerification,
    read_records,
    usage_to_dict,
    verify_chain,
)
from .clock import LATEST, AsOfClock, parse_date
from .grounding import GroundingReport, extract_numbers, verify_grounding
from .errors import (
    AgentToolError,
    Code,
    InsufficientDataError,
    LookaheadViolation,
    PolicyConfigError,
    ProvisionalDataError,
    SourceContractError,
    ToolInputError,
    UnknownSymbolError,
)
from .policy import DEFAULT_TOOL_NAMES, Decision, Policy, PolicyGate
from .proposals import PENDING_HUMAN_APPROVAL, OrderProposal, ProposalBook
from .provenance import canonical_json, hash_bars, result_id
from .runtime import ToolOutcome, ToolRuntime
from .sources import BarSource, FrameBarSource, PointInTimeBars, StoreBarSource
from .synthetic import SyntheticPanelSpec, synthetic_bars, synthetic_source, synthetic_symbols
from .tools import Provenance, ToolContext, ToolRegistry, ToolResult, ToolSpec, default_registry

__all__ = [
    "ABSTAIN_TOKEN",
    "AgentToolError",
    "AnthropicBackend",
    "AnthropicConfig",
    "BackendError",
    "BackendTurn",
    "Copilot",
    "Episode",
    "GroundingReport",
    "LLMBackend",
    "REFUSE_TOKEN",
    "RecordingBackend",
    "ReplayBackend",
    "ScriptedBackend",
    "ScriptedTurn",
    "ToolCallRecord",
    "extract_numbers",
    "load_system_prompt",
    "verify_grounding",
    "AsOfClock",
    "AuditIntegrityError",
    "AuditLog",
    "BarSource",
    "ChainHead",
    "ChainVerification",
    "Code",
    "DEFAULT_TOOL_NAMES",
    "Decision",
    "FrameBarSource",
    "InsufficientDataError",
    "LATEST",
    "LookaheadViolation",
    "OrderProposal",
    "PENDING_HUMAN_APPROVAL",
    "PointInTimeBars",
    "Policy",
    "PolicyConfigError",
    "PolicyGate",
    "ProposalBook",
    "Provenance",
    "ProvisionalDataError",
    "SourceContractError",
    "StoreBarSource",
    "SyntheticPanelSpec",
    "ToolContext",
    "ToolInputError",
    "ToolOutcome",
    "ToolRegistry",
    "ToolResult",
    "ToolRuntime",
    "ToolSpec",
    "UnknownSymbolError",
    "canonical_json",
    "default_registry",
    "hash_bars",
    "parse_date",
    "read_records",
    "result_id",
    "synthetic_bars",
    "synthetic_source",
    "synthetic_symbols",
    "usage_to_dict",
    "verify_chain",
]
