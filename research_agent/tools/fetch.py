"""``read_url``: fetch and extract the readable text of one page."""

import time
from typing import Any
from urllib.parse import urlparse

import httpx
import trafilatura
from langsmith import traceable

from research_agent.settings import Settings, load_settings

# Media types whose body is worth handing to trafilatura. Anything else
# (a PDF, an image, an archive) is rejected on the headers, before a byte
# of the body is read.
READABLE_MEDIA_TYPES: frozenset[str] = frozenset(
    {
        "text/html",
        "text/plain",
        "application/xhtml+xml",
    }
)


def _fetch_page_bytes(url: str, settings: Settings) -> tuple[list[bytes], str] | str:
    """Stream one URL, enforcing the media-type gate and the size cap.

    Returns
    -------
    tuple of (list of bytes, str) or str
        The downloaded chunks and the response encoding on success, or an
        ``"ERROR: "`` message when the page is the wrong media type or too
        large. A transport failure is not caught here -- it propagates to
        the caller, which owns the retry policy.

    Notes
    -----
    ``Content-Length`` is deliberately ignored: it can be absent, wrong, or
    describe a compressed body that expands far past it, so the cap counts
    the bytes that actually arrive.

    A 5xx response or a transport-level failure (timeout, connection error)
    is retried up to `settings.http_retries` times with a one-second pause; a
    4xx is not, since the resource itself is the problem and a retry would
    just get the same rejection again.
    """
    attempt = 0
    while True:
        try:
            with httpx.stream(
                "GET",
                url,
                timeout=settings.http_timeout_seconds,
                follow_redirects=True,
            ) as response:
                response.raise_for_status()

                media_type = response.headers.get("content-type", "").split(";")[0]
                if media_type.strip().lower() not in READABLE_MEDIA_TYPES:
                    return (
                        "ERROR: The page is not HTML or plain text. "
                        "Pick a different source from your search results."
                    )

                chunks: list[bytes] = []
                downloaded_bytes = 0
                for chunk in response.iter_bytes():
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > settings.max_download_bytes:
                        return (
                            "ERROR: The page is too large to read. "
                            "Pick a smaller source from your search results."
                        )
                    chunks.append(chunk)
                return chunks, response.encoding or "utf-8"
        except httpx.HTTPStatusError as exc:
            # Only a 5xx is worth a second try: a 4xx means the resource
            # itself is the problem, and retrying gets the same rejection.
            server_error = exc.response.status_code >= 500
            if not server_error or attempt >= settings.http_retries:
                raise
            time.sleep(1)
            attempt += 1
        except httpx.HTTPError:
            if attempt >= settings.http_retries:
                raise
            time.sleep(1)
            attempt += 1


def _extract_readable_text(
    chunks: list[bytes], encoding: str, settings: Settings
) -> str:
    """Decode, extract and truncate one page's downloaded bytes.

    Returns
    -------
    str
        Extracted page text, truncated to `settings.max_url_content_length`,
        or an ``"ERROR: "`` message when trafilatura finds no readable text.
    """
    # errors="replace" rather than a raised UnicodeDecodeError: a page with
    # a broken declared encoding is still worth extracting text from.
    html = b"".join(chunks).decode(encoding, errors="replace")

    extracted_text = trafilatura.extract(html)
    if not extracted_text:
        return (
            "ERROR: No readable text was found (the page may be "
            "JS-only or a PDF). Try another source."
        )
    text = extracted_text.strip()
    if len(text) <= settings.max_url_content_length:
        return text
    truncated_text = text[: settings.max_url_content_length]
    return (
        f"{truncated_text}\n\n"
        f"[Content truncated to "
        f"{settings.max_url_content_length} characters.]"
    )


@traceable(run_type="tool", name="read_url")
def read_url(url: str) -> str:
    """Read the main text content of an HTTP or HTTPS page.

    Parameters
    ----------
    url : str
        Absolute HTTP or HTTPS URL, normally taken from a previous
        `web_search` result.

    Returns
    -------
    str
        Extracted page text, truncated to `Settings.max_url_content_length`
        characters, or a message starting with ``"ERROR: "`` when the page is
        not readable text, exceeds `Settings.max_download_bytes`, or cannot
        be fetched.

    See Also
    --------
    research_agent.tools.search.web_search : Finds candidate URLs to pass to
        this function.

    Notes
    -----
    The response is streamed and its media type is checked before the body is
    touched, so a PDF or an image costs one request and no memory.
    ``Content-Length`` is deliberately ignored: it can be absent, wrong, or
    describe a compressed body that expands far past it, so the cap counts
    the bytes that actually arrive.

    A 5xx response or a transport-level failure (timeout, connection error)
    is retried up to `Settings.http_retries` times with a one-second pause; a
    4xx is not, since the resource itself is the problem and a retry would
    just get the same rejection again.

    Page content is untrusted input. It is handed to the model as data and is
    never executed as instructions.
    """
    normalized_url = url.strip()
    parsed_url = urlparse(normalized_url)

    if parsed_url.scheme.lower() not in {"http", "https"} or not parsed_url.netloc:
        return "ERROR: URL must be a valid HTTP or HTTPS address."
    try:
        settings = load_settings()
        fetched = _fetch_page_bytes(normalized_url, settings)
        if isinstance(fetched, str):
            return fetched
        chunks, encoding = fetched
        return _extract_readable_text(chunks, encoding, settings)
    except httpx.TimeoutException:
        return (
            "ERROR: The page request timed out. Try a different source "
            "from your search results."
        )
    except httpx.HTTPError:
        return (
            "ERROR: The page is unavailable. Pick another URL from your "
            "search results."
        )
    except Exception:
        return "ERROR: The page could not be read."


READ_URL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_url",
        "description": (
            "Read the main text of one HTTP or HTTPS page. Use it after "
            "web_search when a candidate source has to be examined in "
            "detail, since a snippet is not enough to cite. Only HTML and "
            "plain-text pages can be read: a PDF, an image or an oversized "
            "page returns ERROR:, so pick another source instead of "
            "retrying. Returns the extracted plain text, truncated with an "
            "explicit note when the page is long, or a message starting "
            "with ERROR:."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "Absolute http:// or https:// URL, taken from a "
                        "web_search result rather than invented."
                    ),
                }
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    },
}
