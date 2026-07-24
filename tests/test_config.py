import pytest
from pydantic import ValidationError

from config import Settings


def _load_without_dotenv() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_settings_reads_environment_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")
    monkeypatch.setenv("MODEL_NAME", "test-model")

    settings = _load_without_dotenv()

    assert settings.api_key.get_secret_value() == "test-secret-key"
    assert settings.model_name == "test-model"


def test_settings_uses_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")
    monkeypatch.delenv("MODEL_NAME", raising=False)

    settings = _load_without_dotenv()

    assert settings.model_name == "gpt-4o-mini"


def test_settings_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        _load_without_dotenv()


def test_secret_is_not_exposed_in_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "never-print-this-secret"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    settings = _load_without_dotenv()
    representation = repr(settings)

    assert secret not in representation
    assert "**********" in representation
