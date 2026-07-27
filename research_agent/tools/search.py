"""``web_search``: find candidate sources for a research question."""

from typing import Any

from ddgs import DDGS
from langsmith import traceable

from research_agent.settings import load_settings
from research_agent.tools.contract import SearchResult


@traceable(run_type="tool", name="web_search")
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
    research_agent.tools.fetch.read_url : Fetches the full text of one of
        the returned URLs.

    Notes
    -----
    Snippets are truncated to `Settings.max_search_snippet_length`. They mark
    candidates worth reading, not evidence to cite on their own.
    """
    normalized_query = query.strip()
    if not normalized_query:
        return (
            "ERROR: Search query cannot be empty. Provide a specific "
            "question or phrase."
        )

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
