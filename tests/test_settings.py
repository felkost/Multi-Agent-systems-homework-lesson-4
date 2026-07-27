from contextvars import copy_context

import pytest
from pydantic import ValidationError

from research_agent.settings import Settings, load_settings, output_directory


def test_settings_reads_environment_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")
    monkeypatch.setenv("MODEL_NAME", "test-model")

    settings = Settings.model_validate({})

    assert settings.api_key.get_secret_value() == "test-secret-key"
    assert settings.model_name == "test-model"


def test_settings_uses_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")
    monkeypatch.delenv("MODEL_NAME", raising=False)

    settings = Settings.model_validate({})

    assert settings.model_name == "gpt-4o-mini"


def test_settings_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings.model_validate({})


def test_secret_is_not_exposed_in_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "never-print-this-secret"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    settings = Settings.model_validate({})
    representation = repr(settings)

    assert secret not in representation
    assert "**********" in representation


def test_settings_caps_download_size_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")
    monkeypatch.delenv("MAX_DOWNLOAD_BYTES", raising=False)

    settings = Settings.model_validate({})

    assert settings.max_download_bytes == 2_000_000


def test_settings_disable_tracing_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    settings = Settings.model_validate({})

    assert settings.langsmith_tracing is False
    assert settings.langsmith_api_key is None
    assert settings.langsmith_project == "research-agent-hl4"
    assert settings.langsmith_endpoint == "https://api.smith.langchain.com"


def test_settings_read_tracing_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_test")
    monkeypatch.setenv("LANGSMITH_PROJECT", "other-project")

    settings = Settings.model_validate({})

    assert settings.langsmith_tracing is True
    assert settings.langsmith_api_key is not None
    assert settings.langsmith_api_key.get_secret_value() == "lsv2_pt_test"
    assert settings.langsmith_project == "other-project"


def test_tracing_key_is_not_exposed_in_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_never_print_this")

    representation = repr(Settings.model_validate({}))

    assert "lsv2_pt_never_print_this" not in representation


def test_settings_default_prompt_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")
    monkeypatch.delenv("PROMPT_VERSION", raising=False)

    settings = Settings.model_validate({})

    assert settings.prompt_version == "v2"


def test_settings_reads_prompt_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")
    monkeypatch.setenv("PROMPT_VERSION", "v2")

    settings = Settings.model_validate({})

    assert settings.prompt_version == "v2"


@pytest.mark.parametrize("value", ["99999", "20000001"])
def test_settings_rejects_out_of_range_download_size(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")
    monkeypatch.setenv("MAX_DOWNLOAD_BYTES", value)

    with pytest.raises(ValidationError):
        Settings.model_validate({})


def test_output_directory_redirects_saved_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")

    with output_directory("somewhere/else"):
        assert load_settings().output_dir == "somewhere/else"


def test_output_directory_restores_the_previous_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")
    monkeypatch.setenv("OUTPUT_DIR", "output")

    with output_directory("somewhere/else"):
        pass

    assert load_settings().output_dir == "output"


def test_output_directory_restores_the_previous_setting_after_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")
    monkeypatch.setenv("OUTPUT_DIR", "output")

    with pytest.raises(RuntimeError):
        with output_directory("somewhere/else"):
            raise RuntimeError("the agent blew up mid-run")

    assert load_settings().output_dir == "output"


def test_output_directory_nests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")

    with output_directory("outer"):
        with output_directory("inner"):
            assert load_settings().output_dir == "inner"
        assert load_settings().output_dir == "outer"


def test_output_directory_does_not_leak_between_copied_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property the whole mechanism exists for.

    `evaluate()` submits every example through `copy_context().run(...)`
    (`langsmith.utils.ContextThreadPoolExecutor`), so one example's redirect
    must be invisible to the next. An `os.environ` assignment would fail this
    test, which is the reason it is not one.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")
    monkeypatch.setenv("OUTPUT_DIR", "output")
    seen: list[str] = []

    def one_example(directory: str) -> None:
        with output_directory(directory):
            seen.append(load_settings().output_dir)

    copy_context().run(one_example, "example-a")
    copy_context().run(one_example, "example-b")

    assert seen == ["example-a", "example-b"]
    assert load_settings().output_dir == "output"
