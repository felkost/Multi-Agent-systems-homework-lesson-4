"""One-off request messages the completion gate sends but never persists.

None of these are appended to ``self._messages``: each belongs to one
turn's budget or one turn's fallback, never to the conversation the next
question will also see.
"""

# Sent only as the last message of the final request, never appended to
# self._messages: a reminder that belongs to one turn's budget, not to the
# conversation the next question will also see.
BUDGET_NUDGE_MESSAGE: dict[str, str] = {
    "role": "system",
    "content": (
        "This is the last iteration of your tool-call budget. Call "
        "write_report now with the best report you can produce from the "
        "evidence already gathered."
    ),
}


# Sent as a one-off user message when the loop ended with sources read but
# no write_report attempt at all. Never appended to self._messages -- like
# BUDGET_NUDGE_MESSAGE, it belongs to one turn's fallback, not to history.
#
# This one goes to the plain-text path, where nothing renders the report
# afterwards, so it must NOT tell the model to skip citations: there they
# are the model's job or nobody's.
FALLBACK_REPORT_REQUEST = (
    "Your research budget is finished. Write the final report now, using "
    "only the sources you already read. Do not add commentary or apologies."
)


# The structured path renders the report through `render_report`, which
# numbers the sources and writes every `[n](#source-n)` itself. Left to
# guess, the model hand-writes its own markers next to the rendered ones
# and the two disagree -- observed in 5 of 29 reports of the stage-7 A/B.
FALLBACK_STRUCTURED_REPORT_REQUEST = (
    f"{FALLBACK_REPORT_REQUEST} Write the section bodies as plain prose "
    "and name each section's sources in its source_urls field: the "
    "citation markers and their numbering are added automatically "
    "afterwards, so writing them yourself only creates duplicates."
)
