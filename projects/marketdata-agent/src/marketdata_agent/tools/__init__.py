"""Governed tool layer: typed tool specs, provenance-carrying results, and the default registry."""

from __future__ import annotations

from ..policy import DEFAULT_TOOL_NAMES, Policy
from .base import (
    FieldRole,
    Provenance,
    ToolContext,
    ToolKind,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    render_result_text,
)
from .market import market_tool_specs
from .orders import DECOY_EXECUTION_TOOL, MAX_RATIONALE_CHARS, PROPOSAL_NOTICE, decoy_execution_spec, order_tool_specs


def default_registry(policy: Policy | None = None) -> ToolRegistry:
    """Build the eight default tools; descriptions state the policy's limits."""

    selected = policy or Policy()
    registry = ToolRegistry([*market_tool_specs(selected), *order_tool_specs(selected)])
    if registry.names() != DEFAULT_TOOL_NAMES:  # keep the policy default in lockstep with the registry
        raise RuntimeError(f"default tool names drifted: {registry.names()}")
    return registry


def registry_with_decoy(policy: Policy | None = None) -> ToolRegistry:
    """The default tools plus the decoy ``execute_order`` (kind ``execution``; always refused)."""

    return ToolRegistry([*default_registry(policy), decoy_execution_spec()])


def closed_book_registry() -> ToolRegistry:
    """No tools at all: the closed-book control arm (ablation A5)."""

    return ToolRegistry([])


__all__ = [
    "DECOY_EXECUTION_TOOL",
    "closed_book_registry",
    "decoy_execution_spec",
    "registry_with_decoy",
    "FieldRole",
    "MAX_RATIONALE_CHARS",
    "PROPOSAL_NOTICE",
    "Provenance",
    "ToolContext",
    "ToolKind",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "default_registry",
    "market_tool_specs",
    "order_tool_specs",
    "render_result_text",
]
