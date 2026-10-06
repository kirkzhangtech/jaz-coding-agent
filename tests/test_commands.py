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

from jaz_agent.bridge import Kind
from jaz_agent.commands import (
    COMMANDS,
    MODEL_ARGUMENT,
    RETIRED,
    canonical,
    candidates,
    complete,
    find,
    help_text,
    hint_text,
)
from jaz_agent.session import AgentSession

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


def test_cost_has_a_command():
    """F5 showed spend with no discoverable equivalent; ``/cost`` is that path."""
    assert find("cost") is not None
    assert find("c") is not None, "/c should be the shorthand"


def test_model_is_not_a_command():
    """``/model`` was removed; ``/switchmodules`` covers browsing and switching.

    Worth pinning because ``/model`` was the older, shorter name and something
    muscle-memory or an old transcript would reach for it. If it ever comes
    back it must come back as a deliberate addition, not a leftover.
    """
    assert find("model") is None
    assert canonical("model") is None
    assert "model" not in {c.name for c in COMMANDS}
    # ...and nothing else quietly kept the name either.
    assert not any(
        line.strip().startswith("/model") for line in help_text().splitlines()
    )


def test_retired_commands_point_at_their_replacement():
    """A removed command redirects rather than shrugging.

    Deleting a command does not delete the muscle memory, so ``/model`` is
    typed long after it stopped existing. ``unknown command`` would be true and
    useless; the redirect is the whole reason the table carries ``RETIRED``.
    """
    assert RETIRED["model"] == "switchmodules"
    # Every redirect must name a command that actually exists, or it is a dead
    # end with extra steps.
    for old, new in RETIRED.items():
        assert find(new) is not None, f"/{old} points at /{new}, which does not exist"
    # And a retired name is not simultaneously a live one.
    assert not (set(RETIRED) & {c.name for c in COMMANDS})


def test_every_command_has_a_handler():
    """A command in the table with no behaviour is worse than an absent one: it
    is advertised in ``/help`` and then reports "not implemented".

    Driven through the real dispatcher on a mounted app, so this stays true
    however the mapping is written.
    """
    import asyncio

    from jaz_agent.tui import CodingAgentApp

    async def run():
        session = AgentSession()
        session.running = False
        app = CodingAgentApp(session)
        unwired: list[str] = []
        async with app.run_test() as pilot:
            await pilot.pause()
            for command in COMMANDS:
                if command.usage:
                    continue  # those take an argument; the browser tests cover them
                app.transcript.clear()
                app._command(f"/{command.name}")
                await pilot.pause()
                blob = "\n".join(app.transcript.dump())
                if "not implemented" in blob:
                    unwired.append(f"/{command.name}")
            # /quit exits, so it goes last and outside the loop.
            app._command("/quit")
        return unwired

    unwired = asyncio.run(run())
    assert unwired == [], f"commands in the table with no handler: {unwired}"


def test_no_action_is_unreachable_from_the_prompt():
    """Every feature is a slash command, so nothing needs a key you must
    remember. F5 was the last holdout and gained ``/cost``; this fails if a
    new ``action_*`` is added without one.

    Scoped to actions defined on *this* class -- ``dir()`` also returns the
    Textual base class actions (``action_focus``, ``action_screenshot``, ...),
    which are framework plumbing rather than features of this app.

    ``switch_modules`` is exempt for a real reason: the command is spelled
    ``/switchmodules``, and it takes an argument, so a plain name match would
    miss it. Everything else is expected to match its command exactly.
    """
    from jaz_agent.tui import CodingAgentApp

    defined = vars(CodingAgentApp)
    actions = {
        name[len("action_") :]
        for name, value in defined.items()
        if name.startswith("action_") and callable(value)
    }
    known = {c.name for c in COMMANDS} | {"switch_modules"}
    orphans = actions - known
    assert orphans == set(), f"actions with no slash command: {sorted(orphans)}"


def test_f2_no_longer_starts_a_new_session():
    """Dropping a session is ``/new`` only -- one path, listed in /help.

    Regression test for the removal: F2 cleared the conversation with no other
    route to it, so the only way back was re-typing the context.
    """
    from jaz_agent.tui import CodingAgentApp

    keys = [
        binding.key if hasattr(binding, "key") else binding[0]
        for binding in CodingAgentApp.BINDINGS
    ]
    assert "f2" not in keys
    assert "f5" in keys, "F5 is the documented shortcut for /cost"


def test_new_session_still_works_as_a_command():
    """The action the binding used to call must still be reachable."""
    from jaz_agent.tui import CodingAgentApp

    session = AgentSession()
    session.context.add("something worth forgetting", "and what came back")
    app = CodingAgentApp.__new__(CodingAgentApp)
    app.session = session
    app.status = None
    app.transcript = None
    app.started = 0.0
    said: list[str] = []
    app._say = lambda text, kind=None: said.append(text)  # type: ignore[method-assign]

    app.action_new()

    assert session.context.turns == []
    assert session.bridge.iterations == 0
    assert any("new session" in text for text in said)


def test_new_session_refuses_while_a_turn_is_running():
    """Clearing history mid-turn would leave jaz iterating over a vanished
    conversation, so it must refuse rather than half-happen."""
    from jaz_agent.tui import CodingAgentApp

    session = AgentSession()
    session.context.add("keep me", "kept too")
    session.running = True  # what `busy` reads

    app = CodingAgentApp.__new__(CodingAgentApp)
    app.session = session
    app.status = None
    app.transcript = None
    app.started = 0.0
    said: list[str] = []
    app._say = lambda text, kind=None: said.append(text)  # type: ignore[method-assign]

    app.action_new()

    assert [t.task for t in session.context.turns] == ["keep me"], (
        "history was cleared while busy"
    )
    assert any("busy" in text.lower() for text in said)


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
        ("  backends  ", "backends"),
        ("nonsense", None),
        ("", None),
    ],
)
def test_canonical_resolves_names_and_aliases(typed, expected):
    assert canonical(typed) == expected


@pytest.mark.parametrize("typed", ["help", "/help", "HELP", " /help ", "//help"])
def test_canonical_accepts_the_text_a_user_actually_types(typed):
    """A leading slash must not change the answer.

    ``complete`` has always stripped it; ``canonical`` did not, so the two
    disagreed about the exact string a user types. A caller that reached for
    the wrong one got ``None`` for a command that plainly exists.
    """
    assert canonical(typed) == "help"


def test_canonical_and_complete_agree_on_slashes():
    """A name offered by the completer must resolve.

    ``complete`` strips a leading slash and ``canonical`` now does too, so a
    candidate's name can be fed straight back in. Before, a compliter result
    and a resolver result could disagree about the same string.
    """
    for prefix in ("", "s", "c", "h"):
        for command in complete(prefix):
            resolved = canonical(command.signature.split()[0])
            assert resolved == command.name, (
                f"complete({prefix!r}) offered {command.name!r} "
                f"but canonical({command.signature.split()[0]!r}) gave {resolved!r}"
            )


def test_complete_offers_the_whole_table_for_an_empty_prefix():
    """Typing just "/" should offer everything."""
    assert complete("") == list(COMMANDS)
    assert complete("/") == list(COMMANDS)


def test_complete_narrows_as_you_type():
    """The candidate list must shrink with the prefix, or it is not a filter."""
    every = len(complete(""))
    few = complete("b")
    assert 0 < len(few) < every
    assert [c.name for c in few] == ["backends"]

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
    (``/switchmodules [model]``), which makes ``index("  ")`` meaningless.
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


def test_hint_lists_every_candidate_vertically():
    """One command per entry, so the list cannot interleave.

    This used to be a single space-joined line, which meant ``/switchmodules
    [model]`` plus its summary overflowed the screen and wrapped wherever the
    terminal width happened to fall -- interleaving one command's summary with
    the next command's name.
    """
    for width in (0, 100):
        lines = hint_text("", width).splitlines()
        for command in COMMANDS:
            assert any(line.strip().startswith(command.signature) for line in lines), (
                f"/{command.name} has no line at width={width}"
            )


def test_hint_never_overflows_at_any_width():
    """No line may exceed the width, for any plausible terminal.

    Regression test: the two-column branch used to be chosen from the *space
    left over*, which said "fits" for a 20-character summary while a
    47-character one still overflowed. That left the whole 56-to-70 column band
    wrapping, which the width-46 test alone could not see.
    """
    for width in range(40, 121):
        for line in hint_text("", width).splitlines():
            assert len(line) <= width, f"line of {len(line)} at width {width}: {line!r}"


def test_hint_uses_two_columns_when_it_fits():
    """A wide terminal gets signature and summary on one line, aligned."""
    lines = hint_text("", 120).splitlines()
    assert len(lines) == len(COMMANDS)
    starts = {line.index(c.summary) for c, line in zip(COMMANDS, lines)}
    assert len(starts) == 1, f"summaries start at different columns: {sorted(starts)}"


def test_hint_stacks_the_summary_when_narrow():
    """A narrow terminal must not wrap mid-summary.

    Letting the terminal wrap is what produced the original unreadable output,
    so the summary moves to its own indented line and is wrapped to fit.
    """
    lines = hint_text("", 46).splitlines()

    # Signatures are indented by two spaces, summaries by six.
    signatures = [line for line in lines if line.startswith("  ") and not line.startswith("      ")]
    summaries = [line for line in lines if line.startswith("      ")]
    assert len(signatures) == len(COMMANDS), "every command keeps its own line"
    assert len(summaries) >= len(COMMANDS), "a summary went missing"
    for command, line in zip(COMMANDS, signatures):
        assert line.strip() == command.signature, f"wrong signature line: {line!r}"


def test_hint_wraps_a_summary_that_cannot_fit():
    """A summary too long for the column is broken, not clipped.

    Asserted against the longest summary in the table rather than against a
    literal: editing a summary should not silently retire this check, which is
    what happened the last time the text was reworded for a real reason.
    """
    longest = max(command.summary for command in COMMANDS)
    lines = hint_text("", 46).splitlines()
    joined = " ".join(line.strip() for line in lines)
    assert longest in joined, "the longest summary was clipped rather than wrapped"


def test_hint_expands_the_only_match():
    """One candidate is a single line: the signature and its summary."""
    hint = hint_text("cle")
    assert len(hint.splitlines()) == 1
    assert hint.startswith("/clear ")
    assert "clear the transcript" in hint


def test_hint_says_so_when_nothing_matches():
    hint = hint_text("zzzz")
    assert len(hint.splitlines()) == 1
    assert "no such command" in hint


def test_hint_has_no_truncation_marker():
    """The old single-line form elided candidates with ``(+N more)``. Now that
    the list is vertical it shows everything, so the marker must be gone --
    otherwise a command exists but can never be discovered."""
    assert "(+" not in hint_text("")


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
    assert asyncio.run(_suggest("/swit")) == "/switchmodules"
    assert asyncio.run(_suggest("/ba")) == "/backends"


def test_suggester_is_case_insensitive():
    assert asyncio.run(_suggest("/BA")) == "/backends"


def test_suggester_advances_past_an_exact_command():
    """An exact name with a trailing space moves Tab into the argument."""
    assert asyncio.run(_suggest("/switchmodules")) == "/switchmodules "
    # A no-argument command has nowhere to advance to.
    assert asyncio.run(_suggest("/status")) is None


def test_suggester_completes_a_model_id():
    models = ["openai/gpt-5-mini", "openai/gpt-5", "anthropic/claude-sonnet-4.5"]
    assert (
        asyncio.run(_suggest("/switchmodules openai/gpt-5-", models))
        == "/switchmodules openai/gpt-5-mini"
    )


def test_suggester_falls_back_to_a_substring_match():
    """A partial middle fragment should still complete."""
    models = ["openai/gpt-5-mini", "google/gemini-2.5-flash"]
    assert (
        asyncio.run(_suggest("/switchmodules gpt-5-", models))
        == "/switchmodules openai/gpt-5-mini"
    )


def test_suggester_does_not_complete_models_for_other_commands():
    """``/status foo`` is a mistake, not a model request."""
    assert asyncio.run(_suggest("/status openai", ["openai/gpt-5"])) is None


def test_suggester_without_a_catalogue_completes_commands_only():
    """Before the fetch lands, command completion must still work."""
    assert asyncio.run(_suggest("/ba")) == "/backends"
    assert asyncio.run(_suggest("/switchmodules openai")) is None


def test_suggester_invalidate_drops_the_cache():
    """A backend switch must not leave the old provider's ids behind."""
    s = _suggester(["openai/gpt-5-mini"])
    assert s.has_models
    s.invalidate()
    assert not s.has_models
    assert asyncio.run(s.get_suggestion("/switchmodules openai")) is None


def test_candidates_expose_the_name_and_the_layout():
    """The keyboard navigation needs the name and the display form separately.

    They differ: ``name`` is ``/switchmodules`` and goes into the input box,
    while ``signature`` is ``/switchmodules [model|@backend]`` and is only ever
    shown. Collapsing them would submit the literal placeholder and be rejected.

    The expected signatures come from the table rather than from literals, so
    that editing a usage string cannot quietly turn this into a test of nothing.
    """
    rows = candidates("/s", 120)
    assert [c.name for c in rows] == ["/status", "/switchmodules"]

    for row in rows:
        command = find(row.name)
        assert command is not None, f"{row.name!r} is not a command"
        assert row.signature == command.signature
        if command.usage:
            # The whole point of carrying both: for a command that takes an
            # argument the shown form is not submittable. ``/status`` keeps
            # this honest -- no argument means the two forms are the same
            # string, so there is nothing to tell apart.
            assert row.signature != row.name, f"{row.name!r} lost its argument shape"

    assert all(c.summary for c in rows), "a row has no summary"
    assert all(c.lines for c in rows), "a row has no rendered text"
    # The rendered text must contain the signature, or the highlight would sit
    # next to nothing.
    assert all(c.signature in "\n".join(c.lines) for c in rows)


def test_candidate_name_never_carries_the_usage_placeholder():
    """The value submitted must be a runnable command name.

    This is the bug the split exists to prevent: ``/switchmodules [model]``
    submitted verbatim is an unknown command, because ``[model]`` is a
    placeholder for the reader, not part of the syntax.
    """
    for row in candidates("", 120):
        assert "[" not in row.name, f"{row.name!r} carries a placeholder"
        # ``canonical`` resolves an alias to its command; the name must already
        # be the canonical spelling, with the slash stripped for the lookup.
        assert canonical(row.name.lstrip("/")) == row.name.lstrip("/"), (
            f"{row.name!r} is not a runnable command name"
        )


def test_candidates_are_empty_when_there_is_nothing_to_navigate():
    """A single row needs no cursor; offering one would look broken."""
    assert candidates("", 120), "an empty prefix matches the whole table"
    assert candidates("cle", 120) == [], "one match is not navigable"
    assert candidates("zzzz", 120) == [], "no match"


def test_candidates_use_the_same_layout_as_hint_text():
    """Both views come from one layout, so they cannot drift.

    The failure mode this prevents is subtle: two layouts agreeing on the total
    line count while disagreeing on which lines belong to which command, which
    would put the highlight on the wrong row.

    Only prefixes with two or more matches are compared -- with one, the hint
    renders a single explanatory line and there is no list to disagree about.
    ``s`` matches status and switchmodules; ``c`` matches clear, cancel, cost.
    """
    for prefix in ("", "s", "c"):
        rows = candidates(prefix, 120)
        assert len(rows) >= 2, f"prefix {prefix!r} is not a list: {len(rows)} row(s)"
        rendered = [line for c in rows for line in c.lines]
        assert rendered == hint_text(prefix, 120).splitlines()


def test_candidate_blocks_stay_together_when_wrapped():
    """A row whose summary wraps keeps every one of its own lines.

    Otherwise the highlight would cover part of one command and part of the
    next, which is worse than no highlight at all.
    """
    rows = candidates("", 46)
    joined = "\n".join(line for c in rows for line in c.lines)
    # Nothing may be lost or duplicated on the way through the block split.
    assert joined == hint_text("", 46)
    for candidate in rows:
        assert candidate.lines[0].strip() == candidate.signature, (
            f"a block does not start with its own signature: {candidate.lines[0]!r}"
        )


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
    """The usage string drives both the hint and the completions.

    It must name the model form *and* the explicit-backend one: the second is
    how a model on another provider is asked for, and it exists precisely
    because the bare id cannot say which provider it means.
    """
    usage = find("switchmodules").usage
    assert "switchmodules" in MODEL_ARGUMENT
    assert "[model" in usage
    assert "@backend" in usage