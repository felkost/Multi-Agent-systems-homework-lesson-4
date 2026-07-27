"""Characterization tests pinning the observable contract of ``read_url``.

Every assertion is about behaviour that reaches the model's context — exact
error strings, output normalization, retry policy — not implementation
details. ``call_tool`` (conftest.py) is the only place that knows how a tool
name is resolved to a callable.
"""

from collections.abc import Iterator
from unittest.mock import Mock

import httpx
import pytest

from research_agent.settings import Settings
from research_agent.tools import fetch

from conftest import call_tool

pytestmark = pytest.mark.usefixtures("patch_tool_settings")

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
    monkeypatch.setattr(fetch.httpx, "stream", stream_mock)
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
        fetch.httpx,
        "stream",
        Mock(side_effect=httpx.TimeoutException("timed out")),
    )

    result = call_tool("read_url", url="https://example.com")

    assert (
        result == "ERROR: The page request timed out. Try a different source "
        "from your search results."
    )


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

    assert (
        result == "ERROR: The page is unavailable. Pick another URL from your "
        "search results."
    )
    assert stream.bytes_yielded == 0


def test_read_url_does_not_retry_client_error(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    configured_settings.http_retries = 2
    sleep_mock = Mock()
    monkeypatch.setattr(fetch.time, "sleep", sleep_mock)
    stream = FakeStream(
        [b"<html>not found</html>"],
        status_error=httpx.HTTPStatusError(
            "404",
            request=Mock(),
            response=Mock(status_code=404),
        ),
    )
    stream_mock = _patch_stream(monkeypatch, stream)

    call_tool("read_url", url="https://example.com/missing")

    stream_mock.assert_called_once()
    sleep_mock.assert_not_called()


def test_read_url_retries_transient_error_once(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    configured_settings.http_retries = 1
    sleep_mock = Mock()
    monkeypatch.setattr(fetch.time, "sleep", sleep_mock)
    good_stream = FakeStream([b"<html>recovered</html>"])
    stream_mock = Mock(side_effect=[httpx.TimeoutException("timed out"), good_stream])
    monkeypatch.setattr(fetch.httpx, "stream", stream_mock)
    monkeypatch.setattr(fetch.trafilatura, "extract", Mock(return_value="Recovered"))

    result = call_tool("read_url", url="https://example.com")

    assert stream_mock.call_count == 2
    assert sleep_mock.call_count == 1
    assert result == "Recovered"


def test_read_url_retries_server_error(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    configured_settings.http_retries = 1
    monkeypatch.setattr(fetch.time, "sleep", Mock())
    bad_stream = FakeStream(
        [b""],
        status_error=httpx.HTTPStatusError(
            "503",
            request=Mock(),
            response=Mock(status_code=503),
        ),
    )
    good_stream = FakeStream([b"<html>recovered</html>"])
    monkeypatch.setattr(
        fetch.httpx, "stream", Mock(side_effect=[bad_stream, good_stream])
    )
    monkeypatch.setattr(fetch.trafilatura, "extract", Mock(return_value="Recovered"))

    result = call_tool("read_url", url="https://example.com")

    assert result == "Recovered"


def test_read_url_gives_up_after_retry_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    configured_settings.http_retries = 1
    monkeypatch.setattr(fetch.time, "sleep", Mock())
    stream_mock = Mock(side_effect=httpx.TimeoutException("timed out"))
    monkeypatch.setattr(fetch.httpx, "stream", stream_mock)

    result = call_tool("read_url", url="https://example.com")

    assert stream_mock.call_count == 2
    assert (
        result == "ERROR: The page request timed out. Try a different source "
        "from your search results."
    )


def test_read_url_handles_empty_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"<html><body></body></html>"
    stream_mock = _patch_stream(monkeypatch, FakeStream([body]))
    extract_mock = Mock(return_value=None)
    monkeypatch.setattr(fetch.trafilatura, "extract", extract_mock)

    result = call_tool("read_url", url="https://example.com")

    assert (
        result == "ERROR: No readable text was found (the page may be "
        "JS-only or a PDF). Try another source."
    )
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
        fetch.trafilatura,
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
        fetch.trafilatura,
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
        fetch.trafilatura,
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
        fetch.trafilatura,
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
    monkeypatch.setattr(fetch.trafilatura, "extract", extract_mock)

    result = call_tool("read_url", url="https://example.com/latin1")

    assert result == "café"
    assert "�" in extract_mock.call_args.args[0]


def test_read_url_accepts_uppercase_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_stream(
        monkeypatch,
        FakeStream([b"<html>ok</html>"], content_type="TEXT/HTML"),
    )
    monkeypatch.setattr(
        fetch.trafilatura,
        "extract",
        Mock(return_value="page text"),
    )

    result = call_tool("read_url", url="https://example.com/shouty")

    assert result == "page text"
