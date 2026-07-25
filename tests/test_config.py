import pytest
from pydantic import ValidationError

from config import Settings


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


@pytest.mark.parametrize("value", ["99999", "20000001"])
def test_settings_rejects_out_of_range_download_size(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")
    monkeypatch.setenv("MAX_DOWNLOAD_BYTES", value)

    with pytest.raises(ValidationError):
        Settings.model_validate({})
