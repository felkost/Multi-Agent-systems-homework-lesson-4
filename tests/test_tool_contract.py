"""Characterization tests pinning the current LangChain-``@tool`` contract
of ``web_search``, ``read_url`` and ``write_report`` before the ReAct
rewrite (plan stage 2) replaces the LangChain plumbing around them.

Every assertion is about observable behaviour that reaches the model's
context — exact error strings, output normalization, file-path safety —
not implementation details. ``call_tool`` (conftest.py) is the only place
that still knows tools are LangChain ``BaseTool`` objects.
"""

from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

import tools
from config import Settings

from conftest import call_tool

pytestmark = pytest.mark.usefixtures("patch_tool_settings")


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


def test_web_search_rejects_empty_query() -> None:
    result = call_tool("web_search", query="   ")

    assert result == "ERROR: Search query cannot be empty."


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

    result = call_tool("web_search", query="  RAG retrieval  ")

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
    search_client.text.assert_called_once_with("RAG retrieval", max_results=3)


def test_web_search_handles_missing_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_client = Mock()
    search_client.text.return_value = [
        {"href": "https://example.com/no-fields"},
        {"title": "Missing URL", "body": "This result must be skipped"},
    ]
    monkeypatch.setattr(tools, "DDGS", Mock(return_value=search_client))

    result = call_tool("web_search", query="test")

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
    monkeypatch.setattr(tools, "DDGS", Mock(return_value=search_client))

    result = call_tool("web_search", query="long text")

    assert isinstance(result, list)
    assert result[0]["snippet"] == "x" * 100


def test_web_search_returns_safe_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_client = Mock()
    search_client.text.side_effect = RuntimeError("private DNS and system details")
    monkeypatch.setattr(tools, "DDGS", Mock(return_value=search_client))

    result = call_tool("web_search", query="test")

    assert result == "ERROR: Web search is temporarily unavailable."


# ---------------------------------------------------------------------------
# read_url
# ---------------------------------------------------------------------------


def test_read_url_rejects_non_http_scheme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_mock = Mock()
    monkeypatch.setattr(tools.httpx, "get", get_mock)

    result = call_tool("read_url", url="file:///etc/passwd")

    assert result == "ERROR: URL must be a valid HTTP or HTTPS address."
    get_mock.assert_not_called()


def test_read_url_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tools.httpx,
        "get",
        Mock(side_effect=httpx.TimeoutException("timed out")),
    )

    result = call_tool("read_url", url="https://example.com")

    assert result == "ERROR: The page request timed out."


def test_read_url_handles_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    response = Mock()
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404",
        request=Mock(),
        response=Mock(status_code=404),
    )
    monkeypatch.setattr(tools.httpx, "get", Mock(return_value=response))

    result = call_tool("read_url", url="https://example.com/missing")

    assert result == "ERROR: The page is unavailable."


def test_read_url_handles_empty_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    response = Mock()
    response.text = "<html><body></body></html>"
    response.raise_for_status.return_value = None
    get_mock = Mock(return_value=response)
    extract_mock = Mock(return_value=None)

    monkeypatch.setattr(tools.httpx, "get", get_mock)
    monkeypatch.setattr(tools.trafilatura, "extract", extract_mock)

    result = call_tool("read_url", url="https://example.com")

    assert result == "ERROR: No readable text was found on the page."
    get_mock.assert_called_once_with(
        "https://example.com",
        timeout=2.0,
        follow_redirects=True,
    )
    extract_mock.assert_called_once_with(response.text)


def test_read_url_handles_unexpected_extraction_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Mock()
    response.text = "<html>content</html>"
    response.raise_for_status.return_value = None
    monkeypatch.setattr(tools.httpx, "get", Mock(return_value=response))
    monkeypatch.setattr(
        tools.trafilatura,
        "extract",
        Mock(side_effect=ValueError("unexpected parser failure")),
    )

    result = call_tool("read_url", url="https://example.com")

    assert result == "ERROR: The page could not be read."


def test_read_url_returns_text_within_limit_unmodified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Mock()
    response.text = "<html>short</html>"
    response.raise_for_status.return_value = None
    monkeypatch.setattr(tools.httpx, "get", Mock(return_value=response))
    monkeypatch.setattr(
        tools.trafilatura,
        "extract",
        Mock(return_value="x" * 1000),
    )

    result = call_tool("read_url", url="https://example.com/short")

    assert result == "x" * 1000
    assert "truncated" not in result


def test_read_url_truncates_extracted_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Mock()
    response.text = "<html>long content</html>"
    response.raise_for_status.return_value = None
    monkeypatch.setattr(tools.httpx, "get", Mock(return_value=response))
    monkeypatch.setattr(
        tools.trafilatura,
        "extract",
        Mock(return_value="x" * 1100),
    )

    result = call_tool("read_url", url="https://example.com/long")

    expected = f"{'x' * 1000}\n\n[Content truncated to 1000 characters.]"
    assert result == expected


# ---------------------------------------------------------------------------
# write_report
# ---------------------------------------------------------------------------


def test_write_report_rejects_empty_content() -> None:
    result = call_tool("write_report", filename="empty", content="   ")

    assert result == "ERROR: Report content cannot be empty."


def test_write_report_rejects_filename_with_no_safe_characters() -> None:
    result = call_tool("write_report", filename="...", content="text")

    assert result == "ERROR: Report filename is invalid."


@pytest.mark.parametrize(
    ("filename", "expected_name"),
    [
        ("report", "report.md"),
        ("report.txt", "report.md"),
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

    result = call_tool("write_report", filename=filename, content=content)

    assert isinstance(result, str)
    assert result.startswith("Report saved to: ")

    report_path = Path(result.removeprefix("Report saved to: "))
    output_directory = Path(configured_settings.output_dir).resolve()

    assert report_path == (output_directory / expected_name).resolve()
    assert report_path.parent == output_directory
    assert report_path.suffix == ".md"
    assert report_path.read_text(encoding="utf-8") == content


def test_write_report_handles_unexpected_save_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tools.Path,
        "mkdir",
        Mock(side_effect=OSError("disk full")),
    )

    result = call_tool("write_report", filename="report", content="text")

    assert result == "ERROR: Report could not be saved."
