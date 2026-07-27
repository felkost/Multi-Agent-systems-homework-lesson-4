"""Runtime configuration: environment, ``.env``, and the per-call output-dir
override.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from the environment and then ``.env``.

    Field names map to upper-case environment variables. The bounds on the
    numeric fields are enforced by pydantic, so an out-of-range value fails
    at startup instead of halfway through a research run.
    """

    api_key: SecretStr = Field(validation_alias="OPENAI_API_KEY")
    model_name: str = Field(default="gpt-4o-mini", validation_alias="MODEL_NAME")
    # Names an entry of SYSTEM_PROMPTS. It rides in every trace's metadata,
    # which is how stage 8 tells a v1 experiment from a v2 one.
    prompt_version: str = "v2"

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    max_search_results: int = Field(default=5, ge=1, le=10)
    max_search_snippet_length: int = Field(default=500, ge=100, le=2000)
    max_url_content_length: int = Field(default=5000, ge=1000, le=10000)
    http_timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)
    http_retries: int = Field(default=1, ge=0, le=5)
    max_download_bytes: int = Field(default=2_000_000, ge=100_000, le=20_000_000)
    max_iterations: int = Field(default=8, ge=1, le=30)
    # Deliberately generous: Anthropic's compaction advice is to start by
    # maximizing recall and tighten only against measured quality (plan I.3),
    # so this number is due for tuning on the stage-8 eval set, not before.
    compact_keep_recent: int = Field(default=6, ge=0, le=50)
    max_consecutive_tool_errors: int = Field(default=3, ge=1, le=10)
    output_dir: str = "output"

    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "research-agent-hl4"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    # Only an organization-scoped key needs this: it names the workspace the
    # traces belong to, and LangSmith rejects such a key without it.
    langsmith_workspace_id: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


_output_dir_override: ContextVar[str | None] = ContextVar(
    "output_dir_override", default=None
)


@contextmanager
def output_directory(path: str) -> Iterator[None]:
    """Send reports written inside this block to `path`.

    Parameters
    ----------
    path : str
        Directory for `tools.write_report` to save into.

    Yields
    ------
    None

    Notes
    -----
    Every tool calls `load_settings` itself on each invocation rather than
    receiving the agent's `Settings`, so a `model_copy` passed into
    `ResearchAgent` never reaches `write_report`. The obvious workaround --
    assigning ``os.environ["OUTPUT_DIR"]`` -- is process-global, which makes
    it unusable the moment two runs proceed at once: one run's report lands
    in another's directory.

    A `ContextVar` is the same redirection scoped to the caller instead of
    the process. `contextvars` values do not leak across threads, and the one
    concurrent caller this exists for (`evals.run_eval`) runs each example
    through `copy_context().run(...)`, so every example sees only what it set
    itself.
    """
    token = _output_dir_override.set(path)
    try:
        yield
    finally:
        _output_dir_override.reset(token)


def load_settings() -> Settings:
    """Build `Settings` from the environment and ``.env``.

    Returns
    -------
    Settings
        Validated configuration, with `output_dir` replaced when the caller
        is inside an `output_directory` block.

    Raises
    ------
    pydantic.ValidationError
        If ``OPENAI_API_KEY`` is missing or a field is out of range.
    """
    # model_validate({}) rather than Settings(): the settings sources still
    # read the environment and .env, but mypy no longer demands api_key as a
    # constructor argument.
    settings = Settings.model_validate({})
    override = _output_dir_override.get()
    if override is None:
        return settings
    return settings.model_copy(update={"output_dir": override})
