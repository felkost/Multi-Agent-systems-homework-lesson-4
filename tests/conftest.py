from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from langsmith.utils import get_env_var  # noqa: E402

from research_agent.settings import Settings  # noqa: E402
from research_agent.tools import TOOL_REGISTRY  # noqa: E402
from research_agent.tools import fetch, report_writer, search  # noqa: E402
from research_agent.tracing import _TRACING_FLAGS  # noqa: E402

_TRACING_ENVIRONMENT = (
    *_TRACING_FLAGS,
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_WORKSPACE_ID",
)


def call_tool(name: str, **kwargs: Any) -> Any:
    """Invoke a tool by name the way the ReAct loop does.

    Dispatching through ``TOOL_REGISTRY`` keeps the characterization tests
    independent of how tools are wired up: stage 2 replaced LangChain
    ``BaseTool`` objects with plain functions by editing only this helper.
    """
    return TOOL_REGISTRY[name](**kwargs)


@pytest.fixture(autouse=True)
def isolate_from_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run every test from an empty working directory.

    ``Settings`` resolves ``env_file=".env"`` relative to the working
    directory, so without this the suite would silently pick up the
    developer's real .env on top of the variables a test sets.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def isolate_tracing_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep one test's tracing configuration out of the next one.

    ``configure_tracing`` assigns to ``os.environ`` directly, and monkeypatch
    can only roll back a name it already knows about, so every tracing
    variable is registered here whether the test touches it or not. The SDK
    caches these lookups (``lru_cache`` on ``get_env_var``), so the cache is
    dropped too: a stale ``true`` would send a later test at a real endpoint.
    """
    for name in _TRACING_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    cast(Any, get_env_var).cache_clear()


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
    monkeypatch.setenv("HTTP_RETRIES", "0")
    monkeypatch.setenv("MAX_DOWNLOAD_BYTES", "100000")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))

    return Settings.model_validate({})


@pytest.fixture
def patch_tool_settings(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    monkeypatch.setattr(search, "load_settings", lambda: configured_settings)
    monkeypatch.setattr(fetch, "load_settings", lambda: configured_settings)
    monkeypatch.setattr(report_writer, "load_settings", lambda: configured_settings)
