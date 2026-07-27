"""Research tools: schemas and the registry the ReAct loop dispatches through.

No re-exports of tool internals live here beyond ``TOOL_SCHEMAS`` and
``TOOL_REGISTRY`` themselves -- those two are genuinely owned by this
module (they aggregate all three tools), not a convenience mirror of
another module's globals.
"""

from collections.abc import Callable
from typing import Any

from research_agent.tools.fetch import READ_URL_SCHEMA, read_url
from research_agent.tools.report_writer import WRITE_REPORT_SCHEMA, write_report
from research_agent.tools.search import WEB_SEARCH_SCHEMA, web_search

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
