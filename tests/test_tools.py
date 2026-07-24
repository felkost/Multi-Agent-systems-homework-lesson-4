from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

import tools
from config import Settings


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


@pytest.fixture(autouse=True)
def patch_tool_settings(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    monkeypatch.setattr(
        tools,
        "load_settings",
        lambda: configured_settings,
    )


def test_web_search_normalizes_and_removes_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_client = Mock()
    search_client.text.return_value = [
        {
            "title": "First result",
            "href": "https://example.com/first",
            "body": "First snippet",
        },
        {
            "title": "Duplicate result",
            "href": "https://example.com/first",
            "body": "Duplicate snippet",
        },
        {
            "title": "Second result",
            "href": "https://example.com/second",
            "body": "Second snippet",
        },
    ]
    ddgs_class = Mock(return_value=search_client)
    monkeypatch.setattr(tools, "DDGS", ddgs_class)

    result = tools.web_search.invoke(
        {"query": "  RAG retrieval  "},
    )

    assert result == [
        {
            "title": "First result",
            "url": "https://example.com/first",
            "snippet": "First snippet",
        },
        {
            "title": "Second result",
            "url": "https://example.com/second",
            "snippet": "Second snippet",
        },
    ]
    ddgs_class.assert_called_once_with()
    search_client.text.assert_called_once_with(
        "RAG retrieval",
        max_results=3,
    )


def test_web_search_handles_missing_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_client = Mock()
    search_client.text.return_value = [
        {
            "href": "https://example.com/no-fields",
        },
        {
            "title": "Missing URL",
            "body": "This result must be skipped",
        },
    ]
    monkeypatch.setattr(
        tools,
        "DDGS",
        Mock(return_value=search_client),
    )

    result = tools.web_search.invoke({"query": "test"})

    assert result == [
        {
            "title": "Untitled",
            "url": "https://example.com/no-fields",
            "snippet": "",
        }
    ]


def test_web_search_truncates_snippet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_client = Mock()
    search_client.text.return_value = [
        {
            "title": "Long result",
            "href": "https://example.com/long",
            "body": "x" * 150,
        }
    ]
    monkeypatch.setattr(
        tools,
        "DDGS",
        Mock(return_value=search_client),
    )

    result = tools.web_search.invoke({"query": "long text"})

    assert isinstance(result, list)
    assert result[0]["snippet"] == "x" * 100


def test_web_search_returns_safe_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_client = Mock()
    search_client.text.side_effect = RuntimeError(
        "private DNS and system details",
    )
    monkeypatch.setattr(
        tools,
        "DDGS",
        Mock(return_value=search_client),
    )

    result = tools.web_search.invoke({"query": "test"})

    assert result == ("ERROR: Web search is temporarily unavailable.")


def test_web_search_rejects_empty_query() -> None:
    result = tools.web_search.invoke({"query": "   "})

    assert result == "ERROR: Search query cannot be empty."


def test_read_url_rejects_non_http_scheme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_mock = Mock()
    monkeypatch.setattr(tools.httpx, "get", get_mock)

    result = tools.read_url.invoke(
        {"url": "file:///etc/passwd"},
    )

    assert result == ("ERROR: URL must be a valid HTTP or HTTPS address.")
    get_mock.assert_not_called()


def test_read_url_handles_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_mock = Mock(
        side_effect=httpx.TimeoutException("timed out"),
    )
    monkeypatch.setattr(tools.httpx, "get", get_mock)

    result = tools.read_url.invoke(
        {"url": "https://example.com"},
    )

    assert result == "ERROR: The page request timed out."


def test_read_url_handles_empty_extract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Mock()
    response.text = "<html><body></body></html>"
    response.raise_for_status.return_value = None

    get_mock = Mock(return_value=response)
    extract_mock = Mock(return_value=None)

    monkeypatch.setattr(tools.httpx, "get", get_mock)
    monkeypatch.setattr(
        tools.trafilatura,
        "extract",
        extract_mock,
    )

    result = tools.read_url.invoke(
        {"url": "https://example.com"},
    )

    assert result == ("ERROR: No readable text was found on the page.")
    get_mock.assert_called_once_with(
        "https://example.com",
        timeout=2.0,
        follow_redirects=True,
    )
    extract_mock.assert_called_once_with(response.text)


def test_read_url_truncates_extracted_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Mock()
    response.text = "<html>long content</html>"
    response.raise_for_status.return_value = None

    get_mock = Mock(return_value=response)
    extract_mock = Mock(return_value="x" * 1100)

    monkeypatch.setattr(tools.httpx, "get", get_mock)
    monkeypatch.setattr(
        tools.trafilatura,
        "extract",
        extract_mock,
    )

    result = tools.read_url.invoke(
        {"url": "https://example.com/long"},
    )

    expected = f"{'x' * 1000}\n\n" "[Content truncated to 1000 characters.]"
    assert result == expected


@pytest.mark.parametrize(
    ("filename", "expected_name"),
    [
        ("../test-report.txt", "test-report.md"),
        (r"..\..\windows.exe", "windows.md"),
    ],
)
def test_write_report_keeps_path_inside_output(
    configured_settings: Settings,
    filename: str,
    expected_name: str,
) -> None:
    content = "# Тестовий звіт\n\nТекст українською."

    result = tools.write_report.invoke(
        {
            "filename": filename,
            "content": content,
        }
    )

    assert isinstance(result, str)
    assert result.startswith("Report saved to: ")

    report_path = Path(
        result.removeprefix("Report saved to: "),
    )
    output_directory = Path(
        configured_settings.output_dir,
    ).resolve()

    assert report_path == (output_directory / expected_name).resolve()
    assert report_path.parent == output_directory
    assert report_path.suffix == ".md"
    assert (
        report_path.read_text(
            encoding="utf-8",
        )
        == content
    )


def test_write_report_rejects_empty_content() -> None:
    result = tools.write_report.invoke(
        {
            "filename": "empty",
            "content": "   ",
        }
    )

    assert result == "ERROR: Report content cannot be empty."
