"""The ReAct loop: model call, tool calls, tool results, model call again.

The loop is a plain function over a list of messages. ``ResearchAgent`` only
keeps that list between turns, which is all the memory this agent has: no
checkpointer, no graph, no framework state.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, cast

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from langsmith.utils import get_env_var
from langsmith.wrappers import wrap_openai
from openai import BadRequestError, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCallUnion,
)
from pydantic import BaseModel, Field, ValidationError

from config import (
    BUDGET_NUDGE_MESSAGE,
    ERROR_PREFIX,
    FALLBACK_REPORT_REQUEST,
    REPORT_SAVED_PREFIX,
    Settings,
    build_system_prompt,
)
from tools import TOOL_REGISTRY, TOOL_SCHEMAS, write_report

Messages = list[dict[str, Any]]

StopReason = Literal[
    "goal_satisfied",
    "iteration_limit",
    "token_budget",
    "stagnation",
    "tool_failures",
    "refusal",
    "truncated",
]

ReportSource = Literal["tool", "fallback", "none"]


class CompletionsProtocol(Protocol):
    """The single SDK call this agent makes.

    Notes
    -----
    The parameters are named but typed ``Any`` on purpose. Spelling out the
    SDK's own parameter types would make the real client fail this protocol
    check: its ``create`` is overloaded and expects TypedDict unions rather
    than the plain dicts this project builds. ``tools``, ``tool_choice`` and
    ``parallel_tool_calls`` default to ``None``: the completion gate's
    fallback call omits all three on purpose, since a request with no tools
    is the guarantee that the model cannot go search instead of writing.
    """

    def create(
        self,
        *,
        model: Any,
        messages: Any,
        temperature: Any,
        tools: Any = None,
        tool_choice: Any = None,
        parallel_tool_calls: Any = None,
    ) -> ChatCompletion: ...

    def parse(
        self,
        *,
        model: Any,
        messages: Any,
        response_format: Any,
    ) -> Any: ...


class ChatProtocol(Protocol):
    @property
    def completions(self) -> CompletionsProtocol: ...


class LLMClient(Protocol):
    @property
    def chat(self) -> ChatProtocol: ...


# The SDK resolves TRACING_V2 before TRACING and accepts both the LANGSMITH_
# and LANGCHAIN_ prefixes, so writing one name leaves three others that can
# outrank it: a leftover LANGCHAIN_TRACING_V2=true kept tracing enabled after
# this module had already decided against it.
_TRACING_FLAGS = (
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING",
)


def configure_tracing(settings: Settings) -> bool:
    """Publish the LangSmith variables its SDK reads, and report the outcome.

    Parameters
    ----------
    settings : Settings
        Configuration for this process.

    Returns
    -------
    bool
        Whether tracing is active. ``False`` also means every ``@traceable``
        in this project is a no-op for the rest of the process.

    Notes
    -----
    `langsmith` resolves its own configuration from `os.environ` only, while
    this project's values arrive through `.env`, which pydantic-settings
    parses without exporting anything. Skipping this copy leaves the SDK with
    no key and no ``LANGSMITH_TRACING``, so it drops every span in silence --
    the failure mode is a UI that stays empty while the run looks fine.

    The flags are written in both directions on purpose, so that what the SDK
    resolves is what `Settings` decided rather than whatever the shell
    happened to export. `Settings` itself still takes the environment ahead
    of ``.env`` (pydantic-settings' own precedence), which is what makes
    ``LANGSMITH_TRACING=false python main.py`` a one-run override.
    """
    api_key = settings.langsmith_api_key
    if api_key is None or not settings.langsmith_tracing:
        active = False
    else:
        active = True
        os.environ["LANGSMITH_API_KEY"] = api_key.get_secret_value()
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        if settings.langsmith_workspace_id is not None:
            os.environ["LANGSMITH_WORKSPACE_ID"] = settings.langsmith_workspace_id
    for flag in _TRACING_FLAGS:
        os.environ[flag] = "true" if active else "false"
    # The SDK reads these through an lru_cached getter, so a value it looked
    # up before this call would otherwise outlive the assignments above. The
    # cast is for mypy only: `get_env_var` is an overloaded def, and its
    # runtime lru_cache attributes are invisible to the type checker.
    cast(Any, get_env_var).cache_clear()
    return active


def build_client(settings: Settings) -> LLMClient:
    """Build the chat client, traced when tracing is configured.

    Parameters
    ----------
    settings : Settings
        Model credentials and tracing configuration.

    Returns
    -------
    LLMClient
        A plain `OpenAI` client, or the same client wrapped so that every
        completion becomes an LLM span with its own token counts.

    Notes
    -----
    `wrap_openai` patches the client's `create` and `parse` methods, so the
    ReAct loop keeps calling exactly what it called before -- tracing costs
    this project no change to the loop itself.
    """
    client = OpenAI(api_key=settings.api_key.get_secret_value())
    if not configure_tracing(settings):
        return client
    return wrap_openai(client)


@dataclass(slots=True)
class ToolStep:
    """One executed tool call: what the CLI logs and what evaluators read."""

    name: str
    arguments: dict[str, Any]
    result: str
    ok: bool


@dataclass(slots=True)
class RunState:
    """What one turn accumulated while its tools ran."""

    tool_calls_made: int = 0
    read_urls: list[str] = field(default_factory=list)
    consecutive_tool_errors: int = 0
    # Keyed on the raw query string. Lives only as long as this RunState (one
    # run()); a repeat of the same phrase in a later question is not a cache
    # hit, since the web has had a chance to change by then.
    search_cache: dict[str, str] = field(default_factory=dict)
    last_report_markdown: str | None = None
    last_report_filename: str | None = None
    saved_report_path: str | None = None
    # The Markdown that actually reached disk, whichever path wrote it.
    # Kept so a finished turn can be checked against `read_urls` without
    # reading the file back.
    saved_report_markdown: str | None = None

    @property
    def successful_reads(self) -> int:
        """How many `read_url` calls returned a page during this turn."""
        return len(self.read_urls)


@dataclass(slots=True)
class SessionState:
    """What the session accumulated, one `RunState` per finished turn.

    Notes
    -----
    Without it a turn's business state dies with its `RunState` while the
    messages live on, so "what have you already read?" has two answers that
    drift apart (12-factor #5, plan H.3).
    """

    runs: list[RunState] = field(default_factory=list)

    @property
    def all_read_urls(self) -> set[str]:
        """Every URL read successfully so far, deduplicated."""
        return {url for run in self.runs for url in run.read_urls}

    @property
    def all_saved_reports(self) -> list[str]:
        """Paths of the reports saved so far, oldest first."""
        return [run.saved_report_path for run in self.runs if run.saved_report_path]


@dataclass(slots=True)
class StepResult:
    """Outcome of one model call and the tool calls it requested."""

    messages: Messages
    steps: list[ToolStep]
    final_answer: str | None
    stop_reason: StopReason | None


@dataclass(slots=True)
class AgentResult:
    """What one user question produced."""

    final_answer: str | None
    steps: list[ToolStep]
    iterations_used: int
    budget_exhausted: bool
    stop_reason: StopReason
    saved_report_path: str | None
    report_source: ReportSource
    # True when the saved report lists a URL no `read_url` returned. The
    # gate's own path cannot produce this (render_report filters), so it
    # only ever flags a report the model wrote itself -- where hl-4
    # requires free-form Markdown, leaving the code able to report the
    # problem but not to fix it.
    cites_unread_sources: bool = False


def _assistant_message_to_dict(message: ChatCompletionMessage) -> dict[str, Any]:
    """Convert a model message into the plain dict the history stores.

    Parameters
    ----------
    message : ChatCompletionMessage
        Message returned by the model.

    Returns
    -------
    dict
        ``role`` and ``content``, plus ``tool_calls`` when the model asked for
        any.

    Notes
    -----
    Built field by field rather than with ``model_dump()``: the SDK object
    also carries ``annotations``, ``audio`` and the deprecated
    ``function_call``, which the API does not need back and which would make
    the history brittle to SDK changes. The same pass narrows the tool-call
    union to its function variant.
    """
    payload: dict[str, Any] = {"role": "assistant", "content": message.content}
    function_calls = [
        tool_call
        for tool_call in (message.tool_calls or [])
        if tool_call.type == "function"
    ]
    if function_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in function_calls
        ]
    return payload


def _summarize_tool_result(name: str, arguments: dict[str, Any], content: str) -> str:
    """Describe a tool result in one line, without its payload."""
    if content.startswith(ERROR_PREFIX):
        # Already one short line, and the model still needs to know that this
        # source failed: shrinking it would drop the reason, not the bulk.
        return content
    if name == "web_search":
        try:
            count = len(json.loads(content))
        except (json.JSONDecodeError, TypeError):
            count = 0
        query = arguments.get("query", "")
        return f'[web_search: {count} results for "{query}", payload dropped]'
    if name == "read_url":
        url = arguments.get("url", "")
        return f"[read_url: {url} ({len(content)} chars), payload dropped]"
    if name == "write_report":
        return f"[write_report: saved to {content.removeprefix(REPORT_SAVED_PREFIX)}]"
    return f"[{name}: {len(content)} chars, payload dropped]"


def compact_history(messages: Messages, keep_recent: int) -> Messages:
    """Replace the payloads of older tool results with one-line summaries.

    Parameters
    ----------
    messages : list of dict
        Conversation so far. Not mutated.
    keep_recent : int
        How many of the most recent tool results keep their full payload.

    Returns
    -------
    list of dict
        New list of the same shape: same messages in the same order, with only
        the ``content`` of stale tool results replaced.

    Notes
    -----
    Deleting a stale message instead would leave an assistant ``tool_call``
    without its answer, and the next request would fail with HTTP 400
    (invariant A.2). What survives compaction is the fact of the call, its
    arguments and its outcome, so the agent still knows what it has already
    searched and read; only the raw page text is dropped.

    Examples
    --------
    >>> messages = [
    ...     {
    ...         "role": "assistant",
    ...         "content": None,
    ...         "tool_calls": [
    ...             {
    ...                 "id": "call_1",
    ...                 "type": "function",
    ...                 "function": {
    ...                     "name": "read_url",
    ...                     "arguments": '{"url": "https://example.com"}',
    ...                 },
    ...             }
    ...         ],
    ...     },
    ...     {"role": "tool", "tool_call_id": "call_1", "content": "Long page text"},
    ... ]
    >>> compact_history(messages, keep_recent=0)[1]["content"]
    '[read_url: https://example.com (14 chars), payload dropped]'
    """
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call["function"]
            try:
                arguments = json.loads(function["arguments"])
            except json.JSONDecodeError:
                arguments = {}
            calls[tool_call["id"]] = (
                function["name"],
                arguments if isinstance(arguments, dict) else {},
            )

    results = [
        position
        for position, message in enumerate(messages)
        if message.get("role") == "tool"
    ]
    # Sliced by count rather than [:-keep_recent], which silently keeps
    # everything when keep_recent is 0.
    stale = set(results[: len(results) - keep_recent])

    compacted: Messages = []
    for position, message in enumerate(messages):
        if position not in stale:
            compacted.append(message)
            continue
        name, arguments = calls.get(message.get("tool_call_id", ""), ("tool", {}))
        compacted.append(
            {
                **message,
                "content": _summarize_tool_result(name, arguments, message["content"]),
            }
        )
    return compacted


def _record_failure(
    state: RunState, name: str, arguments: dict[str, Any], message: str
) -> ToolStep:
    """Build an ERROR step and count it toward consecutive tool failures."""
    state.consecutive_tool_errors += 1
    return ToolStep(
        name=name,
        arguments=arguments,
        result=f"{ERROR_PREFIX}{message}",
        ok=False,
    )


def _execute_tool_call(
    tool_call: ChatCompletionMessageToolCallUnion,
    state: RunState,
) -> ToolStep:
    """Run one tool call and describe the outcome as a step.

    Parameters
    ----------
    tool_call : ChatCompletionMessageToolCallUnion
        One entry of the model's ``tool_calls``.
    state : RunState
        Updated in place with what the call produced.

    Returns
    -------
    ToolStep
        Executed step, whose ``result`` is the string handed back to the
        model.

    Notes
    -----
    Every failure here is returned as a step, never raised: a broken tool
    call is data the model can act on and recover from, not a reason to
    crash the loop.

    A repeated `web_search` query is served from `state.search_cache` instead
    of hitting the backend again; only a successful search is ever cached, so
    a transient failure still gets a genuine retry.
    """
    state.tool_calls_made += 1

    if tool_call.type != "function":
        return _record_failure(
            state, tool_call.type, {}, "Only function tool calls are supported."
        )

    name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return _record_failure(state, name, {}, "Tool arguments are not valid JSON.")
    if not isinstance(arguments, dict):
        return _record_failure(state, name, {}, "Tool arguments must be a JSON object.")

    function = TOOL_REGISTRY.get(name)
    if function is None:
        return _record_failure(state, name, arguments, f"Unknown tool '{name}'.")

    if name == "web_search":
        query = arguments.get("query")
        if isinstance(query, str) and query in state.search_cache:
            state.consecutive_tool_errors = 0
            return ToolStep(
                name=name,
                arguments=arguments,
                result=state.search_cache[query],
                ok=True,
            )

    if name == "write_report":
        # Recorded before the call: when saving fails, the completion gate
        # still has the markdown the model produced.
        state.last_report_markdown = arguments.get("content")
        state.last_report_filename = arguments.get("filename")

    try:
        result = function(**arguments)
    except TypeError:
        return _record_failure(
            state, name, arguments, f"Invalid arguments for '{name}'."
        )

    # A tool result reaches the model as text, so anything that is not already
    # a string is serialized here. ensure_ascii=False keeps Cyrillic readable
    # and roughly three times cheaper in tokens than its escaped form.
    content = (
        result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    )
    ok = not content.startswith(ERROR_PREFIX)
    state.consecutive_tool_errors = 0 if ok else state.consecutive_tool_errors + 1

    if name == "web_search" and ok and isinstance(arguments.get("query"), str):
        state.search_cache[arguments["query"]] = content
    if name == "read_url" and ok:
        state.read_urls.append(str(arguments["url"]))
    if name == "write_report" and content.startswith(REPORT_SAVED_PREFIX):
        state.saved_report_path = content.removeprefix(REPORT_SAVED_PREFIX).strip()
        state.saved_report_markdown = state.last_report_markdown

    return ToolStep(name=name, arguments=arguments, result=content, ok=ok)


def react_step(
    messages: Messages,
    client: LLMClient,
    settings: Settings,
    state: RunState,
    *,
    force_write_report: bool = False,
    on_step: Callable[[ToolStep], None] | None = None,
) -> StepResult:
    """Call the model once and run whatever tools it asked for.

    Parameters
    ----------
    messages : list of dict
        Conversation so far. Never mutated: the extended history comes back in
        the result.
    client : LLMClient
        Chat-completions client.
    settings : Settings
        Model name and temperature for the request.
    state : RunState
        Updated in place by the tool calls of this step.
    force_write_report : bool, optional
        Requested by the caller on the last iteration. Takes effect only when
        no report has been saved yet and at least one source was read: forcing
        it after a successful save would loop the model into writing forever,
        and forcing it with no evidence makes the model invent its sources.
    on_step : callable, optional
        Called once per executed tool call, immediately after it finishes --
        the hook the CLI uses to print a step while the turn is still running,
        instead of waiting for the whole turn to end.

    Returns
    -------
    StepResult
        Extended history, executed steps, and either a stop reason or ``None``
        when the turn should continue.

    Notes
    -----
    The assistant message is appended before any tool runs. The API rejects a
    request in which a ``tool_call_id`` has no assistant message announcing
    it, so this ordering is part of the protocol, not bookkeeping.
    """
    force = (
        force_write_report
        and state.saved_report_path is None
        # A forced write with nothing read produces a complete-looking report
        # whose sources no tool ever returned -- observed directly in the K.3
        # max_iterations=1 experiment, not hypothesised.
        and state.successful_reads > 0
    )
    # The nudge is a request-only addition: request_messages carries it to
    # the API, but `updated` (the persisted history) is built from the
    # original `messages` a few lines down, so the next question never sees
    # it.
    request_messages = [*messages, BUDGET_NUDGE_MESSAGE] if force else messages
    tool_choice: str | dict[str, Any] = (
        {"type": "function", "function": {"name": "write_report"}} if force else "auto"
    )

    response = client.chat.completions.create(
        model=settings.model_name,
        messages=request_messages,
        tools=TOOL_SCHEMAS,
        tool_choice=tool_choice,
        temperature=settings.temperature,
        parallel_tool_calls=True,
    )

    choice = response.choices[0]
    message = choice.message
    updated: Messages = [*messages, _assistant_message_to_dict(message)]

    if message.refusal:
        return StepResult(updated, [], message.refusal, "refusal")
    if choice.finish_reason == "length":
        # The response was cut mid-token, so any tool arguments in it are
        # almost certainly broken JSON. Stopping beats feeding the model back
        # its own truncated request.
        return StepResult(updated, [], message.content, "truncated")
    if not message.tool_calls:
        return StepResult(updated, [], message.content, "goal_satisfied")

    steps: list[ToolStep] = []
    for tool_call in message.tool_calls:
        step = _execute_tool_call(tool_call, state)
        steps.append(step)
        if on_step is not None:
            on_step(step)
        updated.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": step.result,
            }
        )
    if state.consecutive_tool_errors >= settings.max_consecutive_tool_errors:
        # [12-factor #9]: an unreachable backend would otherwise burn the
        # whole iteration budget on calls that were never going to succeed.
        return StepResult(updated, steps, None, "tool_failures")
    return StepResult(updated, steps, None, None)


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
    RAG wins on cost. [1](#source-1)
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

    lines = [f"# {report.title}", "", "## Summary", report.summary]
    for section in report.sections:
        references = " ".join(
            f"[{numbers[url]}](#source-{numbers[url]})"
            for url in dict.fromkeys(section.source_urls)
            if url in numbers
        )
        body = f"{section.body} {references}" if references else section.body
        lines += ["", f"## {section.heading}", body]
    lines += ["", "## Limitations", report.limitations, "", "## Sources"]

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


class ResearchAgent:
    """Research agent that runs its own ReAct loop.

    Parameters
    ----------
    settings : Settings
        Model, temperature and budget for the session.
    client : LLMClient, optional
        Chat-completions client. Tests pass a scripted one, which is what lets
        the suite run without an API key.

    Notes
    -----
    The session's memory is `messages` and nothing else: one list of dicts
    living on the instance for as long as the process runs.
    """

    def __init__(self, settings: Settings, client: LLMClient | None = None) -> None:
        self._settings = settings
        if client is None:
            client = build_client(settings)
        # Annotated explicitly: without it, mypy narrows the attribute to the
        # concrete OpenAI type through this conditional assignment, and any
        # later call through self._client is then checked against the SDK's
        # own strict overloads instead of this Protocol.
        self._client: LLMClient = client
        system_prompt = build_system_prompt(
            settings.prompt_version, settings.max_iterations
        )
        self._messages: Messages = [{"role": "system", "content": system_prompt}]
        self._session = SessionState()

    @property
    def messages(self) -> Messages:
        """Conversation history of the whole session, oldest message first."""
        return self._messages

    @property
    def session(self) -> SessionState:
        """Sources read and reports saved, accumulated over every turn."""
        return self._session

    @traceable(run_type="chain", name="research_run")
    def run(
        self,
        user_input: str,
        on_step: Callable[[ToolStep], None] | None = None,
    ) -> AgentResult:
        """Answer one question, calling tools until the model stops asking.

        Parameters
        ----------
        user_input : str
            The user's question.
        on_step : callable, optional
            Called once per executed tool call, as soon as it finishes, so a
            caller such as the CLI can print progress during the turn instead
            of only after ``run`` returns.

        Returns
        -------
        AgentResult
            Final answer, executed steps and why the turn ended.

        Notes
        -----
        The metadata is attached from inside the body because the values live
        on the instance: a decorator argument is evaluated once at import
        time, when no `Settings` exists yet. `get_current_run_tree` returns
        ``None`` whenever tracing is off, which is what keeps this block free
        for untraced runs.
        """
        run_tree = get_current_run_tree()
        if run_tree is not None:
            run_tree.add_metadata(
                {
                    "model": self._settings.model_name,
                    "max_iterations": self._settings.max_iterations,
                    "prompt_version": self._settings.prompt_version,
                }
            )

        state = RunState()
        steps: list[ToolStep] = []
        # Compacted here and not inside the loop: within one question the pages
        # just read are the evidence the model is reasoning over, and only a
        # finished turn's payloads have earned their summary (plan H.6).
        history = compact_history(self._messages, self._settings.compact_keep_recent)
        pending: Messages = [*history, {"role": "user", "content": user_input}]
        max_iterations = self._settings.max_iterations

        step = react_step(
            pending,
            self._client,
            self._settings,
            state,
            force_write_report=max_iterations == 1,
            on_step=on_step,
        )
        steps.extend(step.steps)
        iteration = 1

        while step.stop_reason is None and iteration < max_iterations:
            iteration += 1
            step = react_step(
                step.messages,
                self._client,
                self._settings,
                state,
                force_write_report=iteration == max_iterations,
                on_step=on_step,
            )
            steps.extend(step.steps)

        # Assigned once, after the turn survived. An exception mid-turn would
        # otherwise leave a tool_call_id without its tool result, and every
        # later request would fail with HTTP 400.
        self._messages = step.messages
        result = self._finish(step, steps, state, iteration, user_input)
        # The completion gate can still save a report after the loop ended, so
        # the turn's own record is only complete once _finish has run.
        state.saved_report_path = result.saved_report_path
        self._session.runs.append(state)
        return result

    def _finish(
        self,
        step: StepResult,
        steps: list[ToolStep],
        state: RunState,
        iterations_used: int,
        question: str,
    ) -> AgentResult:
        """Assemble the result of a finished turn."""
        stop_reason: StopReason = step.stop_reason or "iteration_limit"
        saved_report_path, report_source = self._ensure_report_saved(state, question)
        return AgentResult(
            final_answer=step.final_answer,
            steps=steps,
            iterations_used=iterations_used,
            budget_exhausted=step.stop_reason is None,
            stop_reason=stop_reason,
            saved_report_path=saved_report_path,
            report_source=report_source,
            cites_unread_sources=_cites_unread_sources(
                state.saved_report_markdown, state.read_urls
            ),
        )

    def _ensure_report_saved(
        self, state: RunState, question: str
    ) -> tuple[str | None, ReportSource]:
        """Decide what report this turn produced, once the loop has ended.

        Parameters
        ----------
        state : RunState
            What the turn's tool calls accumulated.
        question : str
            The user's question, used to name a fallback report file.

        Returns
        -------
        tuple of (str or None, ReportSource)
            Path of the saved report, and how that file came to exist.

        Notes
        -----
        The guarantee lives here, in Python after the loop, rather than in
        the system prompt: an instruction the model is asked to follow is
        not something the code can rely on.

        A turn with no successful read and no attempted report is treated as
        a non-research turn (e.g. "where did you save it?"): a report with
        no evidence behind it has no value, and writing one on every reply
        would litter `output/` with noise.

        When the model attempted `write_report` but the call failed, this
        retries with the same markdown under a filename built from the
        question, deliberately not the model's own filename: reusing it
        would repeat the exact failure whenever the name itself was the
        problem, as happened during this stage's own K.3 probe, where a
        forced write reused a name that collided with an existing report.

        When the model never attempted `write_report` at all, one extra
        model call asks for the report directly -- see
        `_request_report_markdown`.
        """
        if state.saved_report_path is not None:
            return state.saved_report_path, "tool"
        if state.successful_reads == 0 and state.last_report_markdown is None:
            return None, "none"

        markdown = state.last_report_markdown
        if markdown is None:
            markdown = self._request_report_markdown(set(state.read_urls))
        if markdown is None or not markdown.strip():
            return None, "none"

        result = write_report(
            filename=_build_report_filename(question), content=markdown
        )
        if result.startswith(REPORT_SAVED_PREFIX):
            state.saved_report_markdown = markdown
            return result.removeprefix(REPORT_SAVED_PREFIX).strip(), "fallback"
        return None, "none"

    def _request_report_markdown(self, read_urls: set[str]) -> str | None:
        """Ask the model for a final report with no tools available.

        Returns
        -------
        str or None
            The model's markdown, or ``None`` if it returned nothing.

        Notes
        -----
        Called only when the loop read at least one source but the model
        never attempted `write_report`. The request is transient: neither
        this prompt nor the response is appended to `self._messages`, so the
        next question does not open with "your budget is exhausted." Passing
        no `tools` is itself the guarantee that the model cannot go search
        instead of writing -- there is nothing left it could call.

        `parse()` is tried first: a structured `ResearchReport` means the
        renderer builds source numbering and anchors deterministically,
        instead of trusting the model to count correctly. A model or request
        that does not support strict structured output falls back to a plain
        `create()` call with free-form markdown.
        """
        messages = [
            *self._messages,
            {"role": "user", "content": FALLBACK_REPORT_REQUEST},
        ]
        try:
            completion = self._client.chat.completions.parse(
                model=self._settings.model_name,
                messages=messages,
                response_format=ResearchReport,
            )
            report = completion.choices[0].message.parsed
            if report is not None:
                return render_report(report, read_urls=read_urls)
        except (BadRequestError, ValidationError):
            pass
        response = self._client.chat.completions.create(
            model=self._settings.model_name,
            messages=messages,
            temperature=self._settings.temperature,
        )
        return response.choices[0].message.content
