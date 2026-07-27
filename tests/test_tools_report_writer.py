"""Characterization tests pinning the observable contract of ``write_report``.

Every assertion is about behaviour that reaches the model's context — exact
error strings, file-path safety — not implementation details. ``call_tool``
(conftest.py) is the only place that knows how a tool name is resolved to a
callable.
"""

import re
from pathlib import Path
from unittest.mock import Mock

import pytest

from research_agent.settings import Settings
from research_agent.tools import report_writer

from conftest import call_tool

pytestmark = pytest.mark.usefixtures("patch_tool_settings")


def test_write_report_rejects_empty_content() -> None:
    result = call_tool("write_report", filename="empty", content="   ")

    assert (
        result == "ERROR: Report content cannot be empty. Write the Markdown "
        "report first, then save it."
    )


def test_write_report_rejects_filename_with_no_safe_characters() -> None:
    result = call_tool("write_report", filename="...", content="text")

    assert result == "ERROR: Report filename is invalid."


@pytest.mark.parametrize(
    ("filename", "expected_stem"),
    [
        ("report", "report"),
        ("report.txt", "report"),
        ("../test-report.txt", "test-report"),
        (r"..\..\windows.exe", "windows"),
        ("a" * 60, "a" * 40),
    ],
)
def test_write_report_keeps_path_inside_output(
    configured_settings: Settings,
    filename: str,
    expected_stem: str,
) -> None:
    content = "# Тестовий звіт\n\nТекст українською."

    result = call_tool("write_report", filename=filename, content=content)

    assert isinstance(result, str)
    assert result.startswith("Report saved to: ")

    report_path = Path(result.removeprefix("Report saved to: "))
    output_directory = Path(configured_settings.output_dir).resolve()

    assert report_path.parent == output_directory
    assert re.fullmatch(
        rf"\d{{8}}-\d{{6}}_{re.escape(expected_stem)}\.md", report_path.name
    )
    assert report_path.read_text(encoding="utf-8") == content


def test_write_report_handles_unexpected_save_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        report_writer.Path,
        "mkdir",
        Mock(side_effect=OSError("disk full")),
    )

    result = call_tool("write_report", filename="report", content="text")

    assert result == "ERROR: Report could not be saved."
