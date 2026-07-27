"""``write_report``: save the finished Markdown report."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from langsmith import traceable

from research_agent.settings import load_settings
from research_agent.tools.contract import REPORT_SAVED_PREFIX

_MAX_STEM_LENGTH = 40  # keeps the descriptive part short; the timestamp
# already guarantees uniqueness, so this is purely about readability.


def _timestamp() -> str:
    """Current local time as a filename-safe suffix.

    Notes
    -----
    A separate function so tests can freeze it with ``monkeypatch`` — the
    same pattern ``load_settings`` uses. Naive local time: this is a
    single-user CLI on one machine, and the suffix only has to be readable
    and unique, not timezone-aware.
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


@traceable(run_type="tool", name="write_report")
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
        return (
            "ERROR: Report content cannot be empty. Write the Markdown "
            "report first, then save it."
        )
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
    )[
        :_MAX_STEM_LENGTH
    ].strip(".")
    if not safe_stem:
        return "ERROR: Report filename is invalid."
    try:
        settings = load_settings()
        output_directory = Path(settings.output_dir).resolve()
        output_directory.mkdir(parents=True, exist_ok=True)

        # The timestamp goes first so a plain directory listing sorts
        # chronologically; the topic slug after it is what tells two runs
        # apart at a glance. Together they mean two saves in the same run,
        # or a rerun of the same question, never collide.
        report_path = (output_directory / f"{_timestamp()}_{safe_stem}.md").resolve()
        if report_path.parent != output_directory:
            return "ERROR: Report path is outside the output directory."

        report_path.write_text(content, encoding="utf-8")
        return f"{REPORT_SAVED_PREFIX}{report_path}"
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
