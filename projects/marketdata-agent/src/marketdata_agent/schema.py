"""Strict tool-schema checks and a small JSON-Schema instance validator.

Claude's strict tool use (``"strict": True``) supports a subset of JSON Schema:
objects must set ``additionalProperties: false``; numeric bounds
(``minimum``/``maximum``), string length bounds and complex array constraints
are not supported.  Tool schemas here therefore use only an allowlisted
keyword set, and every object lists all of its properties as required (use a
sentinel such as ``"latest"`` instead of optional fields).  Numeric ranges are
enforced by :class:`marketdata_agent.policy.PolicyGate`, not by the schema.

The instance validator is deliberately independent of the API: tool inputs
are re-validated locally before any handler runs, so a fake client, a
non-strict deployment, or a schema-compilation gap cannot smuggle malformed
arguments into a handler.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


STRICT_KEYWORDS = frozenset(
    {"type", "description", "properties", "required", "additionalProperties", "items", "enum", "const", "anyOf"}
)
JSON_TYPES = frozenset({"object", "array", "string", "integer", "number", "boolean", "null"})


def strict_schema_problems(schema: Any, path: str = "$") -> list[str]:
    """Return every reason ``schema`` is not a valid strict tool input schema."""

    problems: list[str] = []
    if path == "$" and (not isinstance(schema, Mapping) or schema.get("type") != "object"):
        return [f"{path}: tool input_schema must be an object schema"]
    if not isinstance(schema, Mapping):
        return [f"{path}: schema node must be a mapping"]
    unsupported = sorted(set(schema) - STRICT_KEYWORDS)
    if unsupported:
        problems.append(f"{path}: unsupported keywords for strict tools: {unsupported}")
    if "anyOf" in schema:
        branches = schema["anyOf"]
        if not isinstance(branches, list) or not branches:
            problems.append(f"{path}: anyOf must be a non-empty list")
        else:
            for index, branch in enumerate(branches):
                problems.extend(strict_schema_problems(branch, f"{path}.anyOf[{index}]"))
    types = _types(schema)
    if not types and "anyOf" not in schema and "const" not in schema and "enum" not in schema:
        problems.append(f"{path}: schema node needs a type")
    bad_types = sorted(set(types) - JSON_TYPES)
    if bad_types:
        problems.append(f"{path}: unknown JSON types {bad_types}")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        problems.append(f"{path}: enum must be a non-empty list")
    if "object" in types:
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            problems.append(f"{path}: object schema must declare properties")
            properties = {}
        required = schema.get("required")
        if not isinstance(required, list) or set(required) != set(properties) or len(required) != len(set(required)):
            problems.append(f"{path}: every property must be listed exactly once in required")
        if schema.get("additionalProperties") is not False:
            problems.append(f"{path}: additionalProperties must be false")
        for name, child in properties.items():
            problems.extend(strict_schema_problems(child, f"{path}.{name}"))
    if "array" in types:
        items = schema.get("items")
        if not isinstance(items, Mapping):
            problems.append(f"{path}: array schema must declare items")
        else:
            problems.extend(strict_schema_problems(items, f"{path}[]"))
    return problems


def validate_instance(schema: Mapping[str, Any], value: Any, path: str = "$") -> list[str]:
    """Return validation errors for ``value`` against a strict-subset ``schema``."""

    if "anyOf" in schema:
        branch_errors = [validate_instance(branch, value, path) for branch in schema["anyOf"]]
        if not any(not errors for errors in branch_errors):
            return [f"{path}: value matches no anyOf branch"]
    if "const" in schema and value != schema["const"]:
        return [f"{path}: expected constant {schema['const']!r}"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path}: {value!r} is not one of {schema['enum']}"]
    types = _types(schema)
    if types and not any(_is_type(value, kind) for kind in types):
        return [f"{path}: expected {'|'.join(types)}, got {type(value).__name__}"]

    errors: list[str] = []
    if isinstance(value, dict) and "object" in types:
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required property {name!r}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                errors.append(f"{path}: unexpected properties {extra}")
        for name, child in properties.items():
            if name in value:
                errors.extend(validate_instance(child, value[name], f"{path}.{name}"))
    if isinstance(value, list) and "array" in types and isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(value):
            errors.extend(validate_instance(schema["items"], item, f"{path}[{index}]"))
    return errors


def _types(schema: Mapping[str, Any]) -> list[str]:
    kind = schema.get("type")
    if kind is None:
        return []
    return [kind] if isinstance(kind, str) else [str(item) for item in kind]


def _is_type(value: Any, kind: str) -> bool:
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if kind == "string":
        return isinstance(value, str)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    return False
