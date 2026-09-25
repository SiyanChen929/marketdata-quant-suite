from __future__ import annotations

import json

import pytest

from marketdata_agent import DEFAULT_TOOL_NAMES, Policy, ToolRegistry, ToolSpec, default_registry
from marketdata_agent.schema import strict_schema_problems, validate_instance


UNSUPPORTED_IN_STRICT_MODE = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "uniqueItems",
    "pattern",
    "default",
    "oneOf",
    "not",
}


def _walk(schema, path="$"):
    yield path, schema
    for name, child in schema.get("properties", {}).items():
        yield from _walk(child, f"{path}.{name}")
    if isinstance(schema.get("items"), dict):
        yield from _walk(schema["items"], f"{path}[]")


def test_default_registry_has_the_eight_tools_in_a_fixed_order():
    registry = default_registry()
    assert registry.names() == DEFAULT_TOOL_NAMES
    assert len(registry) == 8 and "propose_order" in registry


@pytest.mark.parametrize("name", DEFAULT_TOOL_NAMES)
def test_every_schema_is_strict(name):
    schema = default_registry().get(name).input_schema
    assert strict_schema_problems(schema) == []
    for path, node in _walk(schema):
        assert not (set(node) & UNSUPPORTED_IN_STRICT_MODE), path
        if node.get("type") == "object":
            assert node["additionalProperties"] is False, path
            assert sorted(node["required"]) == sorted(node["properties"]), path
        if path != "$":
            assert node.get("description") or path.endswith("[]"), f"{path} lacks a description"


def test_claude_format_export():
    tools = default_registry().to_anthropic_tools()
    assert [tool["name"] for tool in tools] == list(DEFAULT_TOOL_NAMES)
    for tool in tools:
        assert set(tool) == {"name", "description", "input_schema", "strict"}
        assert tool["strict"] is True
        assert tool["input_schema"]["type"] == "object"
        assert len(tool["description"]) > 60
    assert json.dumps(tools) == json.dumps(default_registry().to_anthropic_tools())  # deterministic bytes
    tools[1]["input_schema"]["properties"]["symbol"]["type"] = "integer"
    assert default_registry().to_anthropic_tools()[1]["input_schema"]["properties"]["symbol"]["type"] == "string"


def test_descriptions_state_the_active_policy_limits():
    tools = {t["name"]: t for t in default_registry(Policy(max_rows_per_result=7, max_order_quantity=250)).to_anthropic_tools()}
    assert "At most 7 of the most recent rows" in tools["get_daily_bars"]["description"]
    assert "1 to 250" in tools["propose_order"]["description"]
    assert "never places, routes, or executes" in tools["propose_order"]["description"]


@pytest.mark.parametrize(
    ("schema", "fragment"),
    [
        ({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}, "additionalProperties"),
        ({"type": "object", "properties": {"a": {"type": "string"}}, "required": [], "additionalProperties": False}, "required"),
        (
            {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}, "required": ["n"], "additionalProperties": False},
            "unsupported",
        ),
        ({"type": "object", "properties": {"xs": {"type": "array"}}, "required": ["xs"], "additionalProperties": False}, "items"),
        (
            {
                "type": "object",
                "properties": {"o": {"type": "object", "properties": {"k": {"type": "string"}}, "required": ["k"]}},
                "required": ["o"],
                "additionalProperties": False,
            },
            "$.o: additionalProperties",
        ),
        ({"type": "string"}, "object schema"),
    ],
)
def test_strict_checker_flags_non_strict_schemas(schema, fragment):
    problems = strict_schema_problems(schema)
    assert problems and any(fragment in problem for problem in problems)


def test_tool_spec_and_registry_validation():
    good = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    with pytest.raises(ValueError):
        ToolSpec("bad name!", "desc", good, lambda a, c: None)  # type: ignore[arg-type,return-value]
    with pytest.raises(ValueError):
        ToolSpec("loose", "desc", {"type": "object", "properties": {}}, lambda a, c: None)  # type: ignore[arg-type,return-value]
    with pytest.raises(ValueError):
        ToolSpec("roles", "desc", good, lambda a, c: None, fields={"missing": "symbol"})  # type: ignore[arg-type,return-value]
    spec = ToolSpec("ok", "desc", good, lambda a, c: None)  # type: ignore[arg-type,return-value]
    with pytest.raises(ValueError):
        ToolRegistry([spec, spec])


def test_instance_validator():
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}, "xs": {"type": "array", "items": {"type": "string"}}, "s": {"type": "string", "enum": ["a", "b"]}},
        "required": ["n", "xs", "s"],
        "additionalProperties": False,
    }
    assert validate_instance(schema, {"n": 1, "xs": ["x"], "s": "a"}) == []
    assert validate_instance(schema, {"n": True, "xs": ["x"], "s": "a"})
    assert validate_instance(schema, {"n": 1, "xs": [1], "s": "a"})
    assert validate_instance(schema, {"n": 1, "xs": [], "s": "c"})
    assert validate_instance(schema, {"n": 1, "xs": [], "s": "a", "extra": 0})
    assert validate_instance(schema, {"n": 1, "xs": []})
    nullable = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert validate_instance(nullable, None) == [] and validate_instance(nullable, "x") == []
    assert validate_instance(nullable, 3)


def test_export_keys_are_accepted_by_the_installed_sdk_tool_type():
    anthropic_types = pytest.importorskip("anthropic.types")
    allowed = set(anthropic_types.ToolParam.__required_keys__) | set(anthropic_types.ToolParam.__optional_keys__)
    for tool in default_registry().to_anthropic_tools():
        assert set(tool) <= allowed
