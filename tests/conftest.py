from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

import tools  # noqa: E402
from config import Settings  # noqa: E402


def call_tool(name: str, **kwargs: Any) -> Any:
    """Invoke a tool by name through its current LangChain ``@tool`` wrapper.

    The only place that knows tools are still LangChain ``BaseTool``
    objects, so stage 2 (plain functions + ``TOOL_REGISTRY``) only needs
    to change this function, not every test that calls a tool.
    """
    tool_object = getattr(tools, name)
    return tool_object.invoke(kwargs)


@pytest.fixture
def configured_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Settings:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MAX_SEARCH_RESULTS", "3")
    monkeypatch.setenv("MAX_SEARCH_SNIPPET_LENGTH", "100")
    monkeypatch.setenv("MAX_URL_CONTENT_LENGTH", "1000")
    monkeypatch.setenv("HTTP_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))

    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def patch_tool_settings(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    monkeypatch.setattr(
        tools,
        "load_settings",
        lambda: configured_settings,
    )
