"""The research agent: a hand-rolled ReAct loop over three tools.

This package has no re-exports at its top level on purpose. A facade that
mirrored a submodule's globals would let ``monkeypatch.setattr(research_agent,
"load_settings", ...)`` silently patch nothing while the real call still
resolves ``research_agent.settings.load_settings`` -- the same class of
silent-config-miss this project has already hit twice (``OUTPUT_DIR`` via
``model_validate``, LangSmith reading ``os.environ`` instead of ``.env``).
Import the submodule you mean.
"""
