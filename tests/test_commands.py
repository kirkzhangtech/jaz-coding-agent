"""The command table, completion, and the model browser.

The commands module is pure data plus pure functions, so most of this is
straightforward. The interesting tests are the ones about *consistency* -- help
text, completion and dispatch must all derive from the same table, or they
silently drift.
"""

from __future__ import annotations

import asyncio
import warnings

import pytest

from jaz_agent.commands import (
    COMMANDS,
    MODEL_ARGUMENT,
    canonical,
    complete,
    find,
    help_text,
    hint_text,
)

warnings.simplefilter("ignore")


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------


def test_every_command_has_a_summary_and_signature():
    """A command with no help text is a command nobody will ever find."""
    for command in COMMANDS:
        assert command.summary, f"/{command.name} has no summary"
        assert command.signature.startswith(f"/{command.name}")


def test_command_names_are_unique():
    names = [c.name for c in COMMANDS]
    assert len(names) == len(set(names))


def test_aliases_are_unique_across_the_table():
    """Two commands sharing an alias makes dispatch ambiguous."""
    seen: set[str] = set()
    for command in COMMANDS:
        for alias in command.aliases:
            assert alias not in seen, f"alias {alias!r} is used twice"
            assert alias != command.name, f"/{alias} aliases itself"
            seen.add(alias)


def test_switchmodules_is_registered():
    """The feature this table was built for must be in it."""
    assert find("switchmodules") is not None
    assert find("switchmodule") is not None, "the singular alias should work too"


# --------------------------------------------------------------------------
# lookup and completion
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "typed,expected",
    [
        ("help", "help"),
        ("HELP", "help"),
        ("h", "help"),          # alias
        ("q", "quit"),          # alias
        ("exit", "quit"),       # alias
        ("switchmodules", "switchmodules"),
        ("switchmodule", "switchmodules"),  # singular alias
        ("  model  ", "model"),
        ("nonsense", None),
        ("", None),
    ],
)
def test_canonical_resolves_names_and_aliases(typed, expected):
    assert canonical(typed) == expected


def test_complete_offers_the_whole_table_for_an_empty_prefix():
    """Typing just "/" should offer everything."""
    assert complete("") == list(COMMANDS)
    assert complete("/") == list(COMMANDS)


def test_complete_narrows_as_you_type():
    """The candidate list must shrink with the prefix, or it is not a filter."""
    every = len(complete(""))
    few = complete("mo")
    assert 0 < len(few) < every
    assert [c.name for c in few] == ["model"]

    assert [c.name for c in complete("s")] == ["status", "switchmodules"]


def test_complete_matches_aliases_too():
    """/h must offer /help -- otherwise the alias is invisible."""
    assert [c.name for c in complete("h")] == ["help"]


# --------------------------------------------------------------------------
# rendered text
# --------------------------------------------------------------------------


def test_help_lists_every_command():
    """The table and the help output must not be able to drift apart."""
    rendered = help_text()
    for command in COMMANDS:
        assert command.signature in rendered, f"/{command.name} missing from /help"


def test_help_columns_line_up():
    """Misaligned help is harder to scan than prose; check the padding.

    Each line is ``"  " + signature.ljust(width) + "  " + summary``, so the
    summary starts at a fixed column. Measured by rebuilding the layout rather
    than by searching for spaces -- a signature may itself contain a space
    (``/model [name|backend [name]]``), which makes ``index("  ")`` meaningless.
    """
    rows = [(c.signature, c.summary) for c in COMMANDS]
    width = max(len(sig) for sig, _ in rows)
    expected = 2 + width + 2

    lines = [ln for ln in help_text().splitlines() if ln.startswith("  /")]
    assert len(lines) == len(rows)
    for (signature, summary), line in zip(rows, lines):
        assert line[expected:].strip() == summary, f"misaligned: {line!r}"
        assert line[: expected].rstrip() == f"  {signature}", f"wrong signature: {line!r}"


def test_hint_lists_candidates_for_a_partial_command():
    hint = hint_text("s")
    assert "/status" in hint
    assert "/switchmodules" in hint


def test_hint_expands_the_only_match():
    """One candidate should produce its full signature, not a bare name."""
    hint = hint_text("cle")
    assert hint.startswith("/clear ")
    assert "clear the transcript" in hint


def test_hint_says_so_when_nothing_matches():
    assert "no such command" in hint_text("zzzz")


def test_hint_truncates_a_long_candidate_list():
    """The hint is one line; it must not grow with the table."""
    hint = hint_text("")
    assert len(hint.splitlines()) == 1
    assert len(hint) < 200


# --------------------------------------------------------------------------
# the suggester (lives in tui.py because it subclasses a Textual widget)
# --------------------------------------------------------------------------


def _suggester(models: list[str] | None = None):
    from jaz_agent.tui import CommandSuggester

    s = CommandSuggester()
    if models is not None:
        s.load_models(models)
    return s


async def _suggest(value: str, models: list[str] | None = None):
    return await _suggester(models).get_suggestion(value)


def test_suggester_ignores_ordinary_text():
    """Tab must not interfere with someone writing a normal task."""
    assert asyncio.run(_suggest("add a flag to")) is None
    assert asyncio.run(_suggest("")) is None


def test_suggester_completes_a_command_name():
    assert asyncio.run(_suggest("/mo")) == "/model"
    assert asyncio.run(_suggest("/swit")) == "/switchmodules"


def test_suggester_is_case_insensitive():
    assert asyncio.run(_suggest("/MO")) == "/model"


def test_suggester_advances_past_an_exact_command():
    """An exact name with a trailing space moves Tab into the argument."""
    assert asyncio.run(_suggest("/model")) == "/model "
    # A no-argument command has nowhere to advance to.
    assert asyncio.run(_suggest("/status")) is None


def test_suggester_completes_a_model_id():
    models = ["openai/gpt-5-mini", "openai/gpt-5", "anthropic/claude-sonnet-4.5"]
    assert (
        asyncio.run(_suggest("/model openai/gpt-5-", models))
        == "/model openai/gpt-5-mini"
    )


def test_suggester_falls_back_to_a_substring_match():
    """A partial middle fragment should still complete."""
    models = ["openai/gpt-5-mini", "google/gemini-2.5-flash"]
    assert (
        asyncio.run(_suggest("/model gpt-5-", models))
        == "/model openai/gpt-5-mini"
    )


def test_suggester_does_not_complete_models_for_other_commands():
    """``/status foo`` is a mistake, not a model request."""
    assert asyncio.run(_suggest("/status openai", ["openai/gpt-5"])) is None


def test_suggester_without_a_catalogue_completes_commands_only():
    """Before the fetch lands, command completion must still work."""
    assert asyncio.run(_suggest("/mo")) == "/model"
    assert asyncio.run(_suggest("/model openai")) is None


def test_suggester_invalidate_drops_the_cache():
    """A backend switch must not leave the old provider's ids behind."""
    s = _suggester(["openai/gpt-5-mini"])
    assert s.has_models
    s.invalidate()
    assert not s.has_models
    assert asyncio.run(s.get_suggestion("/model openai")) is None


# --------------------------------------------------------------------------
# the /switchmodules browser
# --------------------------------------------------------------------------


def test_subsequence_filter_handles_missing_punctuation():
    """``gpt5`` should find ``gpt-5`` -- a substring test would miss it."""
    from jaz_agent.tui import _subsequence

    assert _subsequence("gpt5", "openai/gpt-5-mini")
    assert not _subsequence("xyz", "openai/gpt-5-mini")
    # Order matters, so this is not just a sorted-character check.
    assert not _subsequence("5gpt", "gpt-5")


def test_switchmodules_is_documented_as_taking_a_model():
    """The usage string drives both the hint and the completions."""
    assert "switchmodules" in MODEL_ARGUMENT
    assert "[model]" in find("switchmodules").usage