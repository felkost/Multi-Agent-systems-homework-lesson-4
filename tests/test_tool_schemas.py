"""Schema tests for the OpenAI tool-calling definitions in ``tools.py``.

The JSON Schema is written by hand — the hl-4 brief requires seeing the
protocol rather than generating it from a decorator. These tests take over
the job the decorator used to do: ``_schema_from_signature`` is an oracle
that fails as soon as a function signature and its schema drift apart.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from copy import deepcopy
from typing import Any, get_type_hints

import pytest

from tools import TOOL_REGISTRY, TOOL_SCHEMAS

_JSON_TYPES: dict[type, str] = {str: "string"}


def _schema_from_signature(func: Callable[..., Any]) -> dict[str, Any]:
    """Derive the expected ``parameters`` shape from a function's type hints.

    Parameters
    ----------
    func : Callable
        A tool implementation registered in ``TOOL_REGISTRY``.

    Returns
    -------
    dict
        The ``parameters`` object the hand-written schema is expected to
        carry, without the human-authored ``description`` fields.

    Examples
    --------
    >>> def read_url(url: str) -> str: ...
    >>> _schema_from_signature(read_url)["required"]
    ['url']

    Notes
    -----
    Unwrapped first: ``@traceable`` adds a ``config`` keyword of its own to
    every tool it wraps, and that parameter belongs to LangSmith rather than
    to the contract the model is shown.
    """
    implementation = inspect.unwrap(func)
    hints = get_type_hints(implementation)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, parameter in inspect.signature(implementation).parameters.items():
        properties[name] = {"type": _JSON_TYPES[hints[name]]}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _without_descriptions(parameters: dict[str, Any]) -> dict[str, Any]:
    stripped = deepcopy(parameters)
    for property_schema in stripped["properties"].values():
        property_schema.pop("description", None)
    return stripped


def _schema_id(schema: dict[str, Any]) -> str:
    return str(schema["function"]["name"])


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=_schema_id)
def test_schema_envelope_is_complete(schema: dict[str, Any]) -> None:
    assert schema["type"] == "function"

    function = schema["function"]

    assert function["name"]
    assert function["description"].strip()
    assert function["parameters"]["type"] == "object"


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=_schema_id)
def test_every_parameter_has_a_description(schema: dict[str, Any]) -> None:
    properties = schema["function"]["parameters"]["properties"]

    assert properties, "a tool with no parameters cannot be called usefully"
    for name, property_schema in properties.items():
        assert property_schema["description"].strip(), f"{name} has no description"


def test_schema_names_match_registry_keys() -> None:
    schema_names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}

    assert schema_names == set(TOOL_REGISTRY)
    assert len(TOOL_SCHEMAS) == len(TOOL_REGISTRY)


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=_schema_id)
def test_schema_matches_function_signature(schema: dict[str, Any]) -> None:
    func = TOOL_REGISTRY[schema["function"]["name"]]
    parameters = _without_descriptions(schema["function"]["parameters"])

    assert parameters == _schema_from_signature(func)


def test_tool_schemas_are_json_serializable() -> None:
    payload = json.dumps(TOOL_SCHEMAS)

    assert json.loads(payload) == TOOL_SCHEMAS
