"""Characterization tests pinning the observable contract of ``web_search``,
``read_url`` and ``write_report`` while the agent around them is rewritten.

Every assertion is about behaviour that reaches the model's context — exact
error strings, output normalization, file-path safety — not implementation
details. ``call_tool`` (conftest.py) is the only place that knows how a tool
name is resolved to a callable.
"""

from collections.abc import Iterator
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

TOO_LARGE_ERROR = (
    "ERROR: The page is too large to read. "
    "Pick a smaller source from your search results."
)
NOT_TEXT_ERROR = (
    "ERROR: The page is not HTML or plain text. "
    "Pick a different source from your search results."
)


class FakeStream:
    """Stand-in for ``httpx.stream``: a context manager that is its own response.

    Parameters
    ----------
    chunks : list of bytes
        Body pieces handed out by `iter_bytes`, one per iteration.
    content_type : str
        Value of the ``content-type`` response header.
    extra_headers : dict of str to str, optional
        Further response headers, used to fake a lying ``content-length``.
    encoding : str or None
        What httpx would report as the response encoding.
    status_error : httpx.HTTPError, optional
        Raised by `raise_for_status` instead of returning.

    Notes
    -----
    `bytes_yielded` counts what the tool actually pulled off the wire, which
    is how a test proves that reading stopped early rather than merely that
    the return value was an error.
    """

    def __init__(
        self,
        chunks: list[bytes],
        content_type: str = "text/html; charset=utf-8",
        extra_headers: dict[str, str] | None = None,
        encoding: str | None = "utf-8",
        status_error: httpx.HTTPError | None = None,
    ) -> None:
        self.headers: dict[str, str] = {"content-type": content_type}
        self.headers.update(extra_headers or {})
        self.encoding = encoding
        self.bytes_yielded = 0
        self.closed = False
        self._chunks = chunks
        self._status_error = status_error

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.closed = True

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error

    def iter_bytes(self) -> Iterator[bytes]:
        for chunk in self._chunks:
            self.bytes_yielded += len(chunk)
            yield chunk


def _patch_stream(
    monkeypatch: pytest.MonkeyPatch,
    stream: FakeStream,
) -> Mock:
    """Install `stream` as ``httpx.stream`` and return the call recorder."""
    stream_mock = Mock(return_value=stream)
    monkeypatch.setattr(tools.httpx, "stream", stream_mock)
    return stream_mock


def test_read_url_rejects_non_http_scheme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_mock = _patch_stream(monkeypatch, FakeStream([b"<html></html>"]))

    result = call_tool("read_url", url="file:///etc/passwd")

    assert result == "ERROR: URL must be a valid HTTP or HTTPS address."
    stream_mock.assert_not_called()


def test_read_url_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tools.httpx,
        "stream",
        Mock(side_effect=httpx.TimeoutException("timed out")),
    )

    result = call_tool("read_url", url="https://example.com")

    assert result == "ERROR: The page request timed out."


def test_read_url_handles_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = FakeStream(
        [b"<html>not found</html>"],
        status_error=httpx.HTTPStatusError(
            "404",
            request=Mock(),
            response=Mock(status_code=404),
        ),
    )
    _patch_stream(monkeypatch, stream)

    result = call_tool("read_url", url="https://example.com/missing")

    assert result == "ERROR: The page is unavailable."
    assert stream.bytes_yielded == 0


def test_read_url_handles_empty_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"<html><body></body></html>"
    stream_mock = _patch_stream(monkeypatch, FakeStream([body]))
    extract_mock = Mock(return_value=None)
    monkeypatch.setattr(tools.trafilatura, "extract", extract_mock)

    result = call_tool("read_url", url="https://example.com")

    assert result == "ERROR: No readable text was found on the page."
    stream_mock.assert_called_once_with(
        "GET",
        "https://example.com",
        timeout=2.0,
        follow_redirects=True,
    )
    extract_mock.assert_called_once_with(body.decode())


def test_read_url_handles_unexpected_extraction_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_stream(monkeypatch, FakeStream([b"<html>content</html>"]))
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
    _patch_stream(monkeypatch, FakeStream([b"<html>short</html>"]))
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
    _patch_stream(monkeypatch, FakeStream([b"<html>long content</html>"]))
    monkeypatch.setattr(
        tools.trafilatura,
        "extract",
        Mock(return_value="x" * 1100),
    )

    result = call_tool("read_url", url="https://example.com/long")

    expected = f"{'x' * 1000}\n\n[Content truncated to 1000 characters.]"
    assert result == expected


def test_read_url_rejects_oversized_page(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    limit = configured_settings.max_download_bytes
    _patch_stream(monkeypatch, FakeStream([b"x" * (limit // 10)] * 30))

    result = call_tool("read_url", url="https://example.com/huge")

    assert result == TOO_LARGE_ERROR


def test_read_url_stops_reading_after_limit(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    limit = configured_settings.max_download_bytes
    chunk_size = limit // 10
    stream = FakeStream([b"x" * chunk_size] * 30)
    _patch_stream(monkeypatch, stream)

    call_tool("read_url", url="https://example.com/huge")

    assert stream.bytes_yielded <= limit + chunk_size
    assert stream.closed


def test_read_url_rejects_non_text_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = FakeStream([b"%PDF-1.7"], content_type="application/pdf")
    _patch_stream(monkeypatch, stream)

    result = call_tool("read_url", url="https://example.com/paper.pdf")

    assert result == NOT_TEXT_ERROR
    assert stream.bytes_yielded == 0


def test_read_url_accepts_page_at_exact_limit(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    limit = configured_settings.max_download_bytes
    stream = FakeStream([b"x" * (limit // 2)] * 2)
    _patch_stream(monkeypatch, stream)
    monkeypatch.setattr(
        tools.trafilatura,
        "extract",
        Mock(return_value="page text"),
    )

    result = call_tool("read_url", url="https://example.com/exact")

    assert result == "page text"
    assert stream.bytes_yielded == limit


def test_read_url_ignores_lying_content_length(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    limit = configured_settings.max_download_bytes
    _patch_stream(
        monkeypatch,
        FakeStream(
            [b"x" * (limit // 10)] * 30,
            extra_headers={"content-length": "1024"},
        ),
    )

    result = call_tool("read_url", url="https://example.com/compressed")

    assert result == TOO_LARGE_ERROR


def test_read_url_replaces_undecodable_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_stream(monkeypatch, FakeStream([b"<html>caf\xe9</html>"]))
    extract_mock = Mock(return_value="café")
    monkeypatch.setattr(tools.trafilatura, "extract", extract_mock)

    result = call_tool("read_url", url="https://example.com/latin1")

    assert result == "café"
    assert "\ufffd" in extract_mock.call_args.args[0]


def test_read_url_accepts_uppercase_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_stream(
        monkeypatch,
        FakeStream([b"<html>ok</html>"], content_type="TEXT/HTML"),
    )
    monkeypatch.setattr(
        tools.trafilatura,
        "extract",
        Mock(return_value="page text"),
    )

    result = call_tool("read_url", url="https://example.com/shouty")

    assert result == "page text"


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
