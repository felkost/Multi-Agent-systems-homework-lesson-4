"""The shared contract between a tool's return string and its callers.

Every tool reports failure as a string starting with ``ERROR_PREFIX``
instead of raising: the ReAct loop feeds a tool's return value straight
back to the model, so a failure has to arrive as readable data rather than
a traceback. A successful `write_report` starts with ``REPORT_SAVED_PREFIX``
so the loop and the completion gate can tell success from failure by
matching a prefix, without a second definition of what "saved" means.
"""

from typing import TypedDict

ERROR_PREFIX = "ERROR: "
REPORT_SAVED_PREFIX = "Report saved to: "

SearchResult = TypedDict(
    "SearchResult",
    {
        "title": str,
        "url": str,
        "snippet": str,
    },
)
