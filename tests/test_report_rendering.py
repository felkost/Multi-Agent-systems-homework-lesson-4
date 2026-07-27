"""Tests for `render_report`: the deterministic Markdown renderer.

No `ResearchAgent` and no scripted client here -- every case builds a
`ResearchReport` directly and renders it, since numbering, anchors and
citation filtering are pure functions of that input.
"""

import re

from research_agent.prompts.requests import (
    FALLBACK_REPORT_REQUEST,
    FALLBACK_STRUCTURED_REPORT_REQUEST,
)
from research_agent.report import (
    ReportSection,
    ResearchReport,
    SourceRef,
    render_report,
)


def _citation_numbers(rendered: str) -> tuple[set[str], set[str], set[str]]:
    """In-text references, anchors and numbered entries of a rendered report."""
    return (
        set(re.findall(r"\[(\d+)\]\(#source-\d+\)", rendered)),
        set(re.findall(r'<a id="source-(\d+)"></a>', rendered)),
        set(re.findall(r"</a>(\d+)\. \[", rendered)),
    )


def test_rendered_report_citations_are_consistent() -> None:
    report = ResearchReport(
        title="T",
        summary="S",
        sections=[
            ReportSection(
                heading="H",
                body="B",
                source_urls=[
                    "https://a.example",
                    "https://b.example",
                    "https://c.example",
                ],
            )
        ],
        limitations="L",
        sources=[
            SourceRef(url="https://a.example", title="A"),
            SourceRef(url="https://b.example", title="B"),
            SourceRef(url="https://c.example", title="C"),
        ],
    )

    in_text, anchors, numbered_entries = _citation_numbers(render_report(report))

    # The in-text set is the half the stage-7 ladder found missing: the old
    # renderer produced anchors nothing ever referenced.
    assert in_text == anchors == numbered_entries == {"1", "2", "3"}


def test_rendered_report_cites_the_sections_that_used_a_source() -> None:
    report = ResearchReport(
        title="T",
        summary="S",
        sections=[
            ReportSection(
                heading="First", body="Claim one.", source_urls=["https://a.example"]
            ),
            ReportSection(
                heading="Second", body="Claim two.", source_urls=["https://b.example"]
            ),
        ],
        limitations="L",
        sources=[
            SourceRef(url="https://a.example", title="A"),
            SourceRef(url="https://b.example", title="B"),
        ],
    )

    rendered = render_report(report)

    assert "Claim one.\n\n[1](#source-1)" in rendered
    assert "Claim two.\n\n[2](#source-2)" in rendered


def test_rendered_report_numbers_sources_by_first_appearance() -> None:
    """Numbering follows the text, not the order the model listed sources in.

    The prompt's own output contract says "numbered by first appearance",
    so the renderer -- not the model -- is what makes that true.
    """
    report = ResearchReport(
        title="T",
        summary="S",
        sections=[
            ReportSection(
                heading="First",
                body="Uses the second source.",
                source_urls=["https://b.example"],
            ),
            ReportSection(
                heading="Second",
                body="Uses the first source.",
                source_urls=["https://a.example"],
            ),
        ],
        limitations="L",
        sources=[
            SourceRef(url="https://a.example", title="A"),
            SourceRef(url="https://b.example", title="B"),
        ],
    )

    rendered = render_report(report)

    assert "Uses the second source.\n\n[1](#source-1)" in rendered
    assert "Uses the first source.\n\n[2](#source-2)" in rendered
    assert '<a id="source-1"></a>1. [B](https://b.example)' in rendered
    assert '<a id="source-2"></a>2. [A](https://a.example)' in rendered


def test_rendered_report_reuses_one_number_for_a_repeated_url() -> None:
    report = ResearchReport(
        title="T",
        summary="S",
        sections=[
            ReportSection(
                heading="First", body="Claim one.", source_urls=["https://a.example"]
            ),
            ReportSection(
                heading="Second",
                body="Claim two.",
                source_urls=["https://a.example", "https://a.example"],
            ),
        ],
        limitations="L",
        sources=[SourceRef(url="https://a.example", title="A")],
    )

    rendered = render_report(report)

    assert "Claim one.\n\n[1](#source-1)" in rendered
    assert "Claim two.\n\n[1](#source-1)" in rendered
    assert rendered.count('<a id="source-') == 1


def test_rendered_report_ignores_a_url_the_report_never_listed() -> None:
    """A section may only cite sources the report itself lists.

    Without this the renderer would mint an anchor for a URL that has no
    title and, worse, no evidence that any tool ever returned it.
    """
    report = ResearchReport(
        title="T",
        summary="S",
        sections=[
            ReportSection(
                heading="H",
                body="Claim.",
                source_urls=["https://a.example", "https://invented.example"],
            )
        ],
        limitations="L",
        sources=[SourceRef(url="https://a.example", title="A")],
    )

    rendered = render_report(report)

    in_text, anchors, _ = _citation_numbers(rendered)

    assert in_text == anchors == {"1"}
    assert "invented.example" not in rendered


def test_rendered_report_drops_a_source_no_section_cites() -> None:
    report = ResearchReport(
        title="T",
        summary="S",
        sections=[
            ReportSection(heading="H", body="Claim.", source_urls=["https://a.example"])
        ],
        limitations="L",
        sources=[
            SourceRef(url="https://a.example", title="A"),
            SourceRef(url="https://unused.example", title="Unused"),
        ],
    )

    rendered = render_report(report)

    in_text, anchors, _ = _citation_numbers(rendered)

    assert in_text == anchors == {"1"}
    assert "unused.example" not in rendered


def test_rendered_report_drops_sources_that_were_not_read() -> None:
    """R7a: a URL the agent never opened must not reach the Sources list.

    Measured on a real run: a fallback report cited `meilisearch.com`,
    which had only ever appeared in search results. R1 made citations
    internally consistent, which is not the same as true.
    """
    report = ResearchReport(
        title="T",
        summary="S",
        sections=[
            ReportSection(
                heading="H",
                body="Claim.",
                source_urls=["https://read.example", "https://unread.example"],
            )
        ],
        limitations="L",
        sources=[
            SourceRef(url="https://read.example", title="Read"),
            SourceRef(url="https://unread.example", title="Never opened"),
        ],
    )

    rendered = render_report(report, read_urls={"https://read.example"})

    in_text, anchors, _ = _citation_numbers(rendered)

    assert in_text == anchors == {"1"}
    assert "unread.example" not in rendered
    assert "read.example" in rendered


def test_rendered_report_keeps_every_source_when_reads_are_unknown() -> None:
    """Omitting `read_urls` keeps the pre-R7a behaviour, so the renderer
    stays usable by callers that have no read log to check against."""
    report = ResearchReport(
        title="T",
        summary="S",
        sections=[
            ReportSection(heading="H", body="Claim.", source_urls=["https://a.example"])
        ],
        limitations="L",
        sources=[SourceRef(url="https://a.example", title="A")],
    )

    assert "a.example" in render_report(report)


def test_rendered_report_filters_unread_sources_without_tagged_sections() -> None:
    """The untagged-report branch filters too, or it would smuggle back in
    exactly the phantom sources the tagged branch just removed."""
    report = ResearchReport(
        title="T",
        summary="S",
        sections=[ReportSection(heading="H", body="B")],
        limitations="L",
        sources=[
            SourceRef(url="https://read.example", title="Read"),
            SourceRef(url="https://unread.example", title="Never opened"),
        ],
    )

    rendered = render_report(report, read_urls={"https://read.example"})

    assert "read.example" in rendered
    assert "unread.example" not in rendered


def test_rendered_report_without_tagged_sections_still_lists_sources() -> None:
    """An untagged report keeps the pre-R1 rendering rather than losing its
    sources: no in-text references are possible, but the list is still the
    honest record of what was read."""
    report = ResearchReport(
        title="T",
        summary="S",
        sections=[ReportSection(heading="H", body="B")],
        limitations="L",
        sources=[
            SourceRef(url="https://a.example", title="A"),
            SourceRef(url="https://b.example", title="B"),
        ],
    )

    rendered = render_report(report)

    in_text, anchors, numbered_entries = _citation_numbers(rendered)

    assert in_text == set()
    assert anchors == numbered_entries == {"1", "2"}


def test_references_go_on_their_own_line() -> None:
    """R9: appended inline, a reference block attaches itself to whatever
    the body ends with -- measured on real runs as markers landing inside
    the last cell of a Markdown table."""
    report = ResearchReport(
        title="T",
        summary="S",
        sections=[
            ReportSection(
                heading="Comparison",
                body="| a | b |\n|---|---|\n| 1 | 2 |",
                source_urls=["https://a.example"],
            )
        ],
        limitations="L",
        sources=[SourceRef(url="https://a.example", title="A")],
    )

    rendered = render_report(report)

    assert "| 1 | 2 |\n\n[1](#source-1)" in rendered


def test_structured_fallback_request_says_references_are_automatic() -> None:
    """R8: on the structured path `render_report` writes the references, so
    the model hand-writing `[1][3]` markers only creates contradictions."""
    assert "automatic" in FALLBACK_STRUCTURED_REPORT_REQUEST.lower()


def test_plain_fallback_request_does_not_suppress_citations() -> None:
    """The plain-text path has no renderer behind it: telling the model
    there that references are added for it would remove them entirely."""
    assert "automatic" not in FALLBACK_REPORT_REQUEST.lower()


def test_rendered_report_does_not_repeat_a_heading_it_owns() -> None:
    """T2: `render_report` emits `## Limitations` from the dedicated field,
    so a model section carrying the same name produced the heading twice --
    seen in 1 of 5 reports, the two copies saying slightly different
    things."""
    report = ResearchReport(
        title="T",
        summary="S",
        sections=[
            ReportSection(
                heading="Findings", body="Claim.", source_urls=["https://a.example"]
            ),
            ReportSection(heading="Limitations", body="The sources disagree."),
        ],
        limitations="Benchmarks were unavailable.",
        sources=[SourceRef(url="https://a.example", title="A")],
    )

    rendered = render_report(report)

    assert rendered.count("## Limitations") == 1


def test_rendered_report_keeps_the_text_of_a_folded_section() -> None:
    """Folded, not dropped: the model wrote that text and silently losing
    it would be the same class of dishonesty as silently rewriting it."""
    report = ResearchReport(
        title="T",
        summary="S",
        sections=[ReportSection(heading="Limitations", body="The sources disagree.")],
        limitations="Benchmarks were unavailable.",
        sources=[],
    )

    rendered = render_report(report)

    assert "Benchmarks were unavailable." in rendered
    assert "The sources disagree." in rendered


def test_rendered_report_folds_a_colliding_summary_section() -> None:
    report = ResearchReport(
        title="T",
        summary="The field summary.",
        sections=[ReportSection(heading="summary", body="The section summary.")],
        limitations="L",
        sources=[],
    )

    rendered = render_report(report)

    assert rendered.count("## Summary") == 1
    assert "The field summary." in rendered
    assert "The section summary." in rendered
