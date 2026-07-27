"""The structured report shape and the deterministic renderer that turns it
into the Markdown ``write_report`` expects.

A pure leaf module: rendering never reaches settings, tools or the loop, so
it is safe to import from anywhere without risking a cycle.
"""

import re

from pydantic import BaseModel, Field

_MARKDOWN_LINK_URL_RE = re.compile(r"\]\((https?://[^)\s]+)\)")


def _cites_unread_sources(markdown: str | None, read_urls: list[str]) -> bool:
    """Whether `markdown` links to a URL that no `read_url` returned.

    Examples
    --------
    >>> _cites_unread_sources("[A](https://a.example)", ["https://a.example"])
    False
    >>> _cites_unread_sources("[B](https://b.example)", ["https://a.example"])
    True
    """
    if not markdown:
        return False
    linked = set(_MARKDOWN_LINK_URL_RE.findall(markdown))
    return bool(linked - set(read_urls))


def _build_report_filename(question: str) -> str:
    """Build a fallback report filename from the user's question.

    Parameters
    ----------
    question : str
        The user's question for this turn.

    Returns
    -------
    str
        A slug derived from the question, without a directory or extension.
        ``write_report`` sanitizes, caps the length, and timestamps it
        further, so this name only has to be descriptive, not unique,
        already safe, or already short.

    Examples
    --------
    >>> _build_report_filename("What is RAG?")
    'research_what_is_rag'
    """
    slug = re.sub(r"[^\w]+", "_", question.lower(), flags=re.UNICODE).strip("_")
    return f"research_{slug or 'report'}"


class SourceRef(BaseModel):
    """One source the fallback report cites, in citation order."""

    url: str
    title: str


class ReportSection(BaseModel):
    """One body section of a synthesized report."""

    heading: str
    body: str
    # URLs, not numbers: a section is written before any numbering exists,
    # which is precisely why asking the model for `[n](#source-n)` here
    # produced reports with anchors and no references to them (stage 7).
    source_urls: list[str] = Field(
        default_factory=list,
        description=(
            "URLs of the sources this section's claims come from. Use the "
            "URL itself, not a number: citation numbers are assigned when "
            "the report is rendered."
        ),
    )


class ResearchReport(BaseModel):
    """Structured shape the model fills in when it never wrote a report."""

    title: str
    summary: str
    sections: list[ReportSection]
    limitations: str
    sources: list[SourceRef]


# Headings `render_report` writes itself, each from its own schema field.
# A model section carrying one of these names produced the heading twice
# (seen in 1 report of 5), so such a section is folded into the block that
# owns the name instead of printing a second one -- folded rather than
# dropped, since the model wrote that text and losing it silently would be
# the same class of dishonesty as silently rewriting it.
_OWN_HEADINGS = ("summary", "limitations", "sources")


def render_report(report: ResearchReport, read_urls: set[str] | None = None) -> str:
    """Render a structured report into the Markdown ``write_report`` expects.

    Parameters
    ----------
    report : ResearchReport
        Structured content the model produced via ``chat.completions.parse``.
    read_urls : set of str, optional
        URLs the agent actually opened this turn. When given, every source
        outside it is dropped before anything is numbered. Omitting it
        keeps all sources, which is what a caller with no read log wants.

    Returns
    -------
    str
        Markdown text whose in-text ``[n](#source-n)`` references, anchors
        and ``## Sources`` numbering are all derived here, from the URLs
        each section declared -- the model never counts anything itself.

    Notes
    -----
    Numbering follows first appearance in the text, which is what the
    prompt's own output contract promises; making the renderer honour it
    is cheaper than asking the model to. Sources land in the list only
    when a section cites them, so both halves of "every reference has an
    entry, every entry is referenced" hold by construction -- stage 7
    measured 16 of 17 fallback reports failing the second half, because
    anchors were rendered from a list nothing pointed at.

    A report whose sections declare no sources at all keeps the older,
    reference-free rendering: an untagged report should still carry the
    record of what was read.

    Filtering by `read_urls` happens before numbering rather than after,
    so a dropped source leaves no gap in the sequence. Internal
    consistency is not truthfulness: a real run produced a perfectly
    balanced report citing a page the agent had only seen in search
    results, which is the case this parameter exists for.

    Examples
    --------
    >>> report = ResearchReport(
    ...     title="RAG vs long context",
    ...     summary="Both have trade-offs.",
    ...     sections=[
    ...         ReportSection(
    ...             heading="Findings",
    ...             body="RAG wins on cost.",
    ...             source_urls=["https://example.com"],
    ...         )
    ...     ],
    ...     limitations="Only two sources were read.",
    ...     sources=[SourceRef(url="https://example.com", title="Example")],
    ... )
    >>> print(render_report(report))
    # RAG vs long context
    <BLANKLINE>
    ## Summary
    Both have trade-offs.
    <BLANKLINE>
    ## Findings
    RAG wins on cost.
    <BLANKLINE>
    [1](#source-1)
    <BLANKLINE>
    ## Limitations
    Only two sources were read.
    <BLANKLINE>
    ## Sources
    <a id="source-1"></a>1. [Example](https://example.com)
    """
    sources = report.sources
    if read_urls is not None:
        sources = [source for source in sources if source.url in read_urls]
    titles = {source.url: source.title for source in sources}
    cited: list[str] = []
    for section in report.sections:
        for url in section.source_urls:
            # An unlisted URL is dropped rather than numbered: it would get
            # an anchor with no title and no evidence a tool ever returned it.
            if url in titles and url not in cited:
                cited.append(url)
    numbers = {url: index for index, url in enumerate(cited, start=1)}

    def rendered(section: ReportSection) -> str:
        references = " ".join(
            f"[{numbers[url]}](#source-{numbers[url]})"
            for url in dict.fromkeys(section.source_urls)
            if url in numbers
        )
        # Own paragraph, not appended inline: a body ending in a Markdown
        # table swallowed the markers into its last cell (stage-7 A/B).
        return f"{section.body}\n\n{references}" if references else section.body

    body_sections: list[ReportSection] = []
    folded: dict[str, list[ReportSection]] = {name: [] for name in _OWN_HEADINGS}
    for section in report.sections:
        folded.get(section.heading.strip().lower(), body_sections).append(section)

    lines = [f"# {report.title}", "", "## Summary", report.summary]
    lines += [line for s in folded["summary"] for line in ("", rendered(s))]
    for section in body_sections:
        lines += ["", f"## {section.heading}", rendered(section)]
    lines += ["", "## Limitations", report.limitations]
    lines += [line for s in folded["limitations"] for line in ("", rendered(s))]
    lines += ["", "## Sources"]
    lines += [line for s in folded["sources"] for line in (rendered(s), "")]

    if cited:
        entries = [(numbers[url], titles[url], url) for url in cited]
    else:
        entries = [
            (index, source.title, source.url)
            for index, source in enumerate(sources, start=1)
        ]
    for index, title, url in entries:
        lines.append(f'<a id="source-{index}"></a>{index}. [{title}]({url})')
    return "\n".join(lines)
