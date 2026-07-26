"""Pure functions that turn a ToolStep into the CLI's two log lines.

``print`` itself is not tested here -- only the strings it would receive, per
plan Etap 5: truncating long arguments, hiding query strings, never leaking a
tool's full payload into the console.
"""

import json

from agent import ToolStep
from main import format_tool_call, format_tool_result


def test_format_tool_call_renders_string_arguments() -> None:
    step = ToolStep(
        name="web_search",
        arguments={"query": "naive RAG approach explained"},
        result="[]",
        ok=True,
    )

    assert (
        format_tool_call(step)
        == '🔧 Tool call: web_search(query="naive RAG approach explained")'
    )


def test_format_tool_call_truncates_long_argument() -> None:
    content = "# RAG Comparison\n\n" + "Body text. " * 20
    step = ToolStep(
        name="write_report",
        arguments={"filename": "rag.md", "content": content},
        result="Report saved to: output/rag.md",
        ok=True,
    )

    rendered = format_tool_call(step)

    assert content not in rendered
    assert rendered.count("...") == 1
    assert 'filename="rag.md"' in rendered


def test_format_tool_call_strips_query_string_from_url() -> None:
    step = ToolStep(
        name="read_url",
        arguments={"url": "https://example.com/page?token=super-secret"},
        result="Hello",
        ok=True,
    )

    rendered = format_tool_call(step)

    assert rendered == '🔧 Tool call: read_url(url="https://example.com/page")'
    assert "token" not in rendered
    assert "super-secret" not in rendered


def test_format_tool_call_renders_non_string_argument() -> None:
    step = ToolStep(
        name="web_search", arguments={"max_results": 5}, result="[]", ok=True
    )

    assert format_tool_call(step) == "🔧 Tool call: web_search(max_results=5)"


def test_format_tool_result_summarizes_web_search() -> None:
    results = json.dumps([{"title": "A", "url": "https://a", "snippet": "..."}] * 5)
    step = ToolStep(
        name="web_search", arguments={"query": "RAG"}, result=results, ok=True
    )

    assert format_tool_result(step) == "📎 Result: Found 5 results..."


def test_format_tool_result_survives_malformed_search_json() -> None:
    step = ToolStep(
        name="web_search", arguments={"query": "RAG"}, result="not json", ok=True
    )

    assert format_tool_result(step) == "📎 Result: Found 0 results..."


def test_format_tool_result_previews_read_url() -> None:
    text = "Article about RAG approaches and why they differ in practice." + "x" * 5000
    step = ToolStep(
        name="read_url", arguments={"url": "https://example.com"}, result=text, ok=True
    )

    rendered = format_tool_result(step)

    assert rendered.startswith(f"📎 Result: [{len(text)} chars] Article about RAG")
    assert "x" * 5000 not in rendered


def test_format_tool_result_shows_write_report_path_verbatim() -> None:
    step = ToolStep(
        name="write_report",
        arguments={"filename": "rag.md", "content": "# RAG"},
        result="Report saved to: output/20260726-101010_rag.md",
        ok=True,
    )

    assert (
        format_tool_result(step)
        == "📎 Result: Report saved to: output/20260726-101010_rag.md"
    )


def test_format_tool_result_shows_error_verbatim() -> None:
    error = "ERROR: The page is unavailable. Pick another URL from your search results."
    step = ToolStep(
        name="read_url",
        arguments={"url": "https://example.com/gone"},
        result=error,
        ok=False,
    )

    assert format_tool_result(step) == (
        "📎 Result: ERROR: The page is unavailable. Pick another URL from your "
        "search results."
    )


def test_format_tool_result_never_leaks_full_page_text() -> None:
    secret = "sk-do-not-print-this-api-key"
    text = "Safe preview text that is long enough to push the secret out. " + secret
    step = ToolStep(
        name="read_url", arguments={"url": "https://example.com"}, result=text, ok=True
    )

    assert secret not in format_tool_result(step)


def test_format_tool_call_accepts_an_ascii_icon() -> None:
    step = ToolStep(name="web_search", arguments={"query": "RAG"}, result="[]", ok=True)

    assert format_tool_call(step, "[tool]").startswith("[tool] Tool call:")


def test_format_tool_result_accepts_an_ascii_icon() -> None:
    step = ToolStep(name="web_search", arguments={"query": "RAG"}, result="[]", ok=True)

    assert format_tool_result(step, "[result]").startswith("[result] Result:")
