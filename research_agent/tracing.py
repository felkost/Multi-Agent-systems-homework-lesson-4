"""What reaches LangSmith, and the one hand-off that makes it possible.

`langsmith` resolves its own configuration from `os.environ` only, while
this project's values arrive through `.env`, which pydantic-settings
parses without exporting anything. `configure_tracing` is the copy that
closes that gap.
"""

import os
from typing import Any, cast

from langsmith.utils import get_env_var

from research_agent.settings import Settings

# The SDK resolves TRACING_V2 before TRACING and accepts both the LANGSMITH_
# and LANGCHAIN_ prefixes, so writing one name leaves three others that can
# outrank it: a leftover LANGCHAIN_TRACING_V2=true kept tracing enabled after
# this module had already decided against it.
_TRACING_FLAGS = (
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING",
)


def configure_tracing(settings: Settings) -> bool:
    """Publish the LangSmith variables its SDK reads, and report the outcome.

    Parameters
    ----------
    settings : Settings
        Configuration for this process.

    Returns
    -------
    bool
        Whether tracing is active. ``False`` also means every ``@traceable``
        in this project is a no-op for the rest of the process.

    Notes
    -----
    `langsmith` resolves its own configuration from `os.environ` only, while
    this project's values arrive through `.env`, which pydantic-settings
    parses without exporting anything. Skipping this copy leaves the SDK with
    no key and no ``LANGSMITH_TRACING``, so it drops every span in silence --
    the failure mode is a UI that stays empty while the run looks fine.

    The flags are written in both directions on purpose, so that what the SDK
    resolves is what `Settings` decided rather than whatever the shell
    happened to export. `Settings` itself still takes the environment ahead
    of ``.env`` (pydantic-settings' own precedence), which is what makes
    ``LANGSMITH_TRACING=false python main.py`` a one-run override.
    """
    api_key = settings.langsmith_api_key
    if api_key is None or not settings.langsmith_tracing:
        active = False
    else:
        active = True
        os.environ["LANGSMITH_API_KEY"] = api_key.get_secret_value()
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        if settings.langsmith_workspace_id is not None:
            os.environ["LANGSMITH_WORKSPACE_ID"] = settings.langsmith_workspace_id
    for flag in _TRACING_FLAGS:
        os.environ[flag] = "true" if active else "false"
    # The SDK reads these through an lru_cached getter, so a value it looked
    # up before this call would otherwise outlive the assignments above. The
    # cast is for mypy only: `get_env_var` is an overloaded def, and its
    # runtime lru_cache attributes are invisible to the type checker.
    cast(Any, get_env_var).cache_clear()
    return active
