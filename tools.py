"""Research tools and their OpenAI tool-calling schemas.

Every tool reports failure as a string starting with ``"ERROR: "`` instead of
raising: the ReAct loop feeds a tool's return value straight back to the
model, so a failure has to arrive as readable data rather than a traceback.

The JSON Schema next to each function is what the model actually reads. The
mechanics of a tool, meaning what it does, when it applies and what it
returns, belong in that ``description`` and not in the system prompt.
"""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlparse

import httpx
import trafilatura
from ddgs import DDGS

from config import load_settings

SearchResult = TypedDict(
    "SearchResult",
    {
        "title": str,
        "url": str,
        "snippet": str,
    },
)


def web_search(query: str) -> list[SearchResult] | str:
    """Search the web and return compact candidate sources.

    Parameters
    ----------
    query : str
        Focused search query, passed to the search engine after stripping.

    Returns
    -------
    list of SearchResult or str
        Up to `Settings.max_search_results` results with distinct URLs, or a
        message starting with ``"ERROR: "`` when the query is empty or the
        search backend fails.

    See Also
    --------
    read_url : Fetches the full text of one of the returned URLs.

    Notes
    -----
    Snippets are truncated to `Settings.max_search_snippet_length`. They mark
    candidates worth reading, not evidence to cite on their own.
    """
    normalized_query = query.strip()
    if not normalized_query:
        return "ERROR: Search query cannot be empty."

    try:
        settings = load_settings()
        raw_results = DDGS().text(
            normalized_query,
            max_results=settings.max_search_results,
        )
        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for item in raw_results:
            title = str(item.get("title") or "Untitled").strip()
            url = str(item.get("href") or "").strip()
            snippet = str(item.get("body") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet[: settings.max_search_snippet_length],
                }
            )
        return results
    except Exception:
        return "ERROR: Web search is temporarily unavailable."


WEB_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for sources relevant to a research question. "
            "Use it to discover candidate pages, and run it again with a "
            "different query for each distinct angle of the topic. Returns "
            "a list of {title, url, snippet} objects, or a message starting "
            "with ERROR:. Snippets are previews, not full page texts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Focused search query in the language of the sources."
                    ),
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


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
    web_search : Finds candidate URLs to pass to this function.

    Notes
    -----
    The response is streamed and its media type is checked before the body is
    touched, so a PDF or an image costs one request and no memory.
    ``Content-Length`` is deliberately ignored: it can be absent, wrong, or
    describe a compressed body that expands far past it, so the cap counts
    the bytes that actually arrive.

    Page content is untrusted input. It is handed to the model as data and is
    never executed as instructions.
    """
    normalized_url = url.strip()
    parsed_url = urlparse(normalized_url)

    if parsed_url.scheme.lower() not in {"http", "https"} or not parsed_url.netloc:
        return "ERROR: URL must be a valid HTTP or HTTPS address."
    try:
        settings = load_settings()
        with httpx.stream(
            "GET",
            normalized_url,
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

        # errors="replace" rather than a raised UnicodeDecodeError: a page with
        # a broken declared encoding is still worth extracting text from.
        encoding = response.encoding or "utf-8"
        html = b"".join(chunks).decode(encoding, errors="replace")

        extracted_text = trafilatura.extract(html)
        if not extracted_text:
            return "ERROR: No readable text was found on the page."
        text = extracted_text.strip()
        if len(text) <= settings.max_url_content_length:
            return text
        truncated_text = text[: settings.max_url_content_length]
        return (
            f"{truncated_text}\n\n"
            f"[Content truncated to "
            f"{settings.max_url_content_length} characters.]"
        )
    except httpx.TimeoutException:
        return "ERROR: The page request timed out."
    except httpx.HTTPError:
        return "ERROR: The page is unavailable."
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


def write_report(filename: str, content: str) -> str:
    """Save a completed Markdown research report.

    Parameters
    ----------
    filename : str
        Desired report name. Any directory component is dropped and the stem
        keeps only word characters, dots and dashes.
    content : str
        Full Markdown text of the report.

    Returns
    -------
    str
        ``"Report saved to: <path>"`` on success, or a message starting with
        ``"ERROR: "`` when the content is empty, the name sanitizes to
        nothing, or the file cannot be written.

    Notes
    -----
    The report always lands directly in `Settings.output_dir`: a name such as
    ``../escape.md`` is reduced to ``escape.md`` rather than rejected.
    """
    if not content.strip():
        return "ERROR: Report content cannot be empty."
    normalized_name = filename.strip().replace("\\", "/")
    base_name = normalized_name.rsplit("/", maxsplit=1)[-1]
    stem = Path(base_name).stem
    # re.UNICODE keeps Cyrillic and other non-ASCII letters, so a Ukrainian
    # report name survives sanitizing instead of collapsing to nothing.
    safe_stem = re.sub(
        r"[^\w.-]",
        "",
        stem,
        flags=re.UNICODE,
    ).strip(".")
    if not safe_stem:
        return "ERROR: Report filename is invalid."
    try:
        settings = load_settings()
        output_directory = Path(settings.output_dir).resolve()
        output_directory.mkdir(parents=True, exist_ok=True)

        report_path = (output_directory / f"{safe_stem}.md").resolve()
        if report_path.parent != output_directory:
            return "ERROR: Report path is outside the output directory."

        report_path.write_text(content, encoding="utf-8")
        return f"Report saved to: {report_path}"
    except Exception:
        return "ERROR: Report could not be saved."


WRITE_REPORT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_report",
        "description": (
            "Save the finished Markdown report to a file. Call it once, "
            "after the research is done and the full report text is ready. "
            "This is the only way the report reaches the user. Returns "
            "'Report saved to: <path>' on success, or a message starting "
            "with ERROR:."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": (
                        "Short descriptive name derived from the question, "
                        "without a directory or extension; .md is added."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": (
                        "The complete Markdown report, including its "
                        "Sources section."
                    ),
                },
            },
            "required": ["filename", "content"],
            "additionalProperties": False,
        },
    },
}

# Two structures rather than one: TOOL_SCHEMAS is what the model reads,
# TOOL_REGISTRY is what the loop calls. tests/test_tool_schemas.py fails if a
# tool ever appears in one and not the other.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    WEB_SEARCH_SCHEMA,
    READ_URL_SCHEMA,
    WRITE_REPORT_SCHEMA,
]

TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "web_search": web_search,
    "read_url": read_url,
    "write_report": write_report,
}
