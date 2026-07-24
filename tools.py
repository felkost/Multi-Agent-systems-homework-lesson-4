import re  # очищення назви звіту
from pathlib import Path  # безпечна побудова шляху
from typing import TypedDict  # контракт результату пошуку
from urllib.parse import urlparse  # перевірка URL

import httpx  # завантаження сторінки
import trafilatura  # виділення основного текст
from ddgs import DDGS  # пошук
from langchain.tools import tool  # перетворення функції на LangChain tool

from config import load_settings  # отримання лімітів без глобального Settings()

# кожен успішний результат матиме рівно три текстові поля
SearchResult = TypedDict(
    "SearchResult",
    {
        "title": str,
        "url": str,
        "snippet": str,
    },
)


@tool
def web_search(query: str) -> list[SearchResult] | str:
    """Search the web and return compact candidate sources.

    Use this tool to discover pages relevant to a research
    question. Search snippets are not full source texts.
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


@tool
def read_url(url: str) -> str:
    """Read the main text content of an HTTP or HTTPS page.

    Use this tool after web_search when a source needs to be
    examined in detail.
    """
    normalized_url = url.strip()
    parsed_url = urlparse(normalized_url)

    if parsed_url.scheme.lower() not in {"http", "https"} or not parsed_url.netloc:
        return "ERROR: URL must be a valid HTTP or HTTPS address."
    try:
        settings = load_settings()
        response = httpx.get(
            normalized_url,
            timeout=settings.http_timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        extracted_text = trafilatura.extract(response.text)
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


@tool
def write_report(filename: str, content: str) -> str:
    """Save a completed Markdown research report.

    The report is written as a .md file inside the configured
    output directory.
    """
    if not content.strip():
        return "ERROR: Report content cannot be empty."
    normalized_name = filename.strip().replace("\\", "/")
    base_name = normalized_name.rsplit("/", maxsplit=1)[-1]
    stem = Path(base_name).stem
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
