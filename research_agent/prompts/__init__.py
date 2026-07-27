"""Registry of system-prompt versions and the renderer that fills them in.

This ``__init__`` is a genuine composition point, not a facade: it builds
``SYSTEM_PROMPTS`` by combining what ``v1.py``/``v2.py`` each define, and
that combination is real content this module owns, not a re-export of
someone else's global.
"""

from datetime import date

from research_agent.prompts.v1 import SYSTEM_PROMPT_V1
from research_agent.prompts.v2 import (
    SYSTEM_PROMPT_V2,
    SYSTEM_PROMPT_V2_FEW,
    SYSTEM_PROMPT_V2_MIN,
)

# Kept verbatim as the baseline stage 8 measures v2 against, not as a
# fallback: nothing selects a version except Settings.prompt_version.
SYSTEM_PROMPTS: dict[str, str] = {
    "v1": SYSTEM_PROMPT_V1,
    "v2-min": SYSTEM_PROMPT_V2_MIN,
    "v2-few": SYSTEM_PROMPT_V2_FEW,
    "v2": SYSTEM_PROMPT_V2,
}


def get_system_prompt(version: str) -> str:
    """Return the raw prompt template registered under `version`.

    Parameters
    ----------
    version : str
        Key of `SYSTEM_PROMPTS`.

    Returns
    -------
    str
        The template, placeholders still unfilled.

    Raises
    ------
    ValueError
        If no such version is registered. Loud on purpose: a typo in
        PROMPT_VERSION would otherwise run v1 inside an experiment named
        after v2, and stage 8 would be measuring noise.
    """
    try:
        return SYSTEM_PROMPTS[version]
    except KeyError:
        raise ValueError(
            f"Unknown prompt version '{version}'. "
            f"Available: {', '.join(sorted(SYSTEM_PROMPTS))}."
        ) from None


def build_system_prompt(
    version: str,
    max_iterations: int,
    today: date | None = None,
) -> str:
    """Render the system prompt for one session.

    Parameters
    ----------
    version : str
        Key of `SYSTEM_PROMPTS`.
    max_iterations : int
        Tool-call budget, told to the model up front rather than sprung on
        it at the last iteration.
    today : datetime.date, optional
        Date to inject; defaults to the current date. Without it the model
        reads "latest" as its own training cutoff (plan E.2).

    Returns
    -------
    str
        Prompt text with every placeholder filled in.

    Notes
    -----
    `str.format` ignores keywords a template never mentions, so a version
    that needs no date costs nothing here.
    """
    return get_system_prompt(version).format(
        max_iterations=max_iterations,
        today=(today or date.today()).isoformat(),
    )
