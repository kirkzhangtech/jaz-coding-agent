"""Headless Textual test: the UI must mount, accept a task, and return to idle.

Textual's ``run_test`` runs the real widget tree without a terminal, so this
exercises the actual compose/render/event path rather than a mock of it.

One trap worth knowing: ``pilot.pause()`` advances Textual's *virtual* clock but
not the wall clock, so an interval timer only fires while you are inside a
pause. The agent, however, runs on a real worker thread. A test that only awaits
``pilot.pause()`` therefore never gives the worker time to finish, and the UI
looks stuck in ``busy`` -- a test bug, not a product bug.
"""

from __future__ import annotations

import asyncio
import warnings

from jaz import MockLLMClient

from jaz_agent.session import AgentSession
from jaz_agent.tui import CodingAgentApp

warnings.simplefilter("ignore")


def _scripted_session() -> tuple[AgentSession, list[int]]:
    """A session whose model calls two canned turns."""
    calls: list[int] = []

    def fake(model, messages, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return "listing = tools.list_dir('.')\nprint('found', len(listing.splitlines()))"
        return "return finish('tui selftest: workspace inspected')"

    session = AgentSession()
    session._llm = MockLLMClient(fn=fake)
    session.max_iterations = 6
    return session, calls


async def _drive(pilot, session: AgentSession, calls: list[int], timeout: float = 30.0) -> None:
    """Submit a task and wait for the turn to finish in real time."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        await pilot.pause(0.05)   # lets the interval timer fire
        await asyncio.sleep(0.01)  # lets the worker thread make progress
        if not session.busy and len(calls) >= 2 and session.queue.empty():
            break
    for _ in range(10):
        await pilot.pause(0.05)
        await asyncio.sleep(0.01)


async def _test_full_cycle():
    session, calls = _scripted_session()
    app = CodingAgentApp(session)

    async with app.run_test() as pilot:
        await pilot.pause()

        # All three widgets exist and the prompt took focus.
        assert app.screen.query_one("#transcript"), "transcript missing"
        assert app.screen.query_one("#status"), "status bar missing"
        box = app.screen.query_one("#task")
        assert box.has_focus, "prompt box did not receive focus"

        box.value = "inspect the workspace"
        await pilot.press("enter")

        await _drive(pilot, session, calls)

        status = app.screen.query_one("#status")
        lines = app.screen.query_one("#transcript").dump()
        blob = "\n".join(lines)

        assert calls, "the agent never ran"
        assert status.state == "ready", f"stuck in {status.state!r}"
        assert status.turns == 2, f"expected 2 turns, got {status.turns}"
        assert not box.disabled, "prompt box left disabled"

        # Every event kind the bridge emits must reach the transcript.
        assert "listing =" in blob, "generated code missing"
        assert "found" in blob, "tool output missing"
        assert "tui selftest" in blob, "final report missing"
        assert "idle" in blob, "idle marker missing"

        return lines, status.render()


def test_tui_full_cycle():
    """End-to-end: mount, submit, render, return to idle."""
    lines, rendered = asyncio.run(_test_full_cycle())
    print("\n--- status line ---")
    print(rendered)
    print("--- transcript ---")
    for line in lines:
        print(repr(line))


async def _test_slash_commands():
    session, calls = _scripted_session()
    app = CodingAgentApp(session)

    async with app.run_test() as pilot:
        await pilot.pause()
        box = app.screen.query_one("#task")

        for cmd in ("/help", "/status", "/clear", "/nonsense"):
            box.value = cmd
            await pilot.press("enter")
            await pilot.pause()

        lines = app.screen.query_one("#transcript").dump()
        blob = "\n".join(lines)

        assert "/help" in blob, "/help produced nothing"
        assert "unknown command" in blob, "unknown commands were not reported"
        assert not calls, "a slash command must not reach the model"
        return lines


def test_tui_slash_commands():
    """Slash commands are handled locally and never hit the model."""
    lines = asyncio.run(_test_slash_commands())
    print("\n--- after slash commands ---")
    for line in lines:
        print(repr(line))


# --------------------------------------------------------------------------
# /model
# --------------------------------------------------------------------------


def _switchable_session() -> tuple[AgentSession, list[int]]:
    """A session with an injected model, so switching needs no network."""
    calls: list[int] = []

    def fake(model, messages, **kwargs):
        calls.append(1)
        return "return finish('ok')"

    session = AgentSession(llm=MockLLMClient(fn=fake))
    session.max_iterations = 4
    return session, calls


async def _run_command(app, pilot, text: str, settle: float = 0.4) -> None:
    """Type a slash command and let the drain timer pick up the result."""
    box = app.screen.query_one("#task")
    box.value = text
    await pilot.press("enter")
    for _ in range(12):
        await pilot.pause(0.05)
        await asyncio.sleep(0.02)
    await pilot.pause(settle)


def test_status_command_reports_the_current_model(monkeypatch):
    """The model commands must remain discoverable from /help."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("jaz_agent.llm_config._list_openrouter_models", lambda b: [])

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _run_command(app, pilot, "/status")
            return app.screen.query_one("#transcript").dump()

    blob = "\n".join(asyncio.run(run()))
    print("\n--- /status ---")
    print(blob)

    assert "model:" in blob
    assert session.model in blob


def _banner_text(app) -> str:
    """Plain text of the banner.

    ``Static.render()`` returns a rich ``Content`` object, not a str, so an
    ``in`` assertion against it silently fails. Converting here keeps the
    assertions below readable.
    """
    rendered = app.banner.render()
    return rendered.plain if hasattr(rendered, "plain") else str(rendered)


def test_switchmodules_switches_to_a_named_model(monkeypatch):
    """``/switchmodules <name>`` commits the switch and says which model is live."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "jaz_agent.tui.list_models",
        lambda backend: ["stealth/space-bunny-alpha", "openai/gpt-5-mini"],
    )

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _run_command(app, pilot, "/switchmodules openai/gpt-5-mini")
            return app.screen.query_one("#transcript").dump(), _banner_text(app)

    lines, banner = asyncio.run(run())
    blob = "\n".join(lines)
    print("\n--- /switchmodules openai/gpt-5-mini ---")
    print(blob)
    print("--- banner ---")
    print(banner)

    assert session.model == "openai/gpt-5-mini"
    assert "switched to" in blob
    # The header must not keep claiming the old model.
    assert "gpt-5-mini" in banner


def test_failed_switch_is_reported_and_leaves_state_alone(monkeypatch):
    """A rejected switch must say so *and* keep the old model.

    Both halves matter: silence leaves the user unsure, and a silent state
    change would break the next turn. The session is built *without* an injected
    model so the real probe path runs -- the probe itself is stubbed, which is
    what a provider rejection looks like from the caller's side.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "jaz_agent.llm_config.list_models",
        lambda backend: ["stealth/space-bunny-alpha"],
    )
    monkeypatch.setattr(
        "jaz_agent.session.probe", lambda llm: "model not found, or not available on this backend"
    )

    session = AgentSession()
    before = (session.backend.name, session.model)
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            # The id is in the catalogue, so this takes the switch branch rather
            # than the filter branch, and the probe is what rejects it.
            await _run_command(app, pilot, "/switchmodules stealth/space-bunny-alpha")
            return app.screen.query_one("#transcript").dump()

    blob = "\n".join(asyncio.run(run()))
    print("\n--- failed switch ---")
    print(blob)

    assert "not switched" in blob
    assert "not found" in blob
    assert (session.backend.name, session.model) == before


def test_backends_command_lists_providers(monkeypatch):
    """``/backends`` names each provider and whether it has a key."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _run_command(app, pilot, "/backends")
            return app.screen.query_one("#transcript").dump()

    blob = "\n".join(asyncio.run(run()))
    print("\n--- /backends ---")
    print(blob)

    assert "openrouter" in blob
    assert "anthropic" in blob
    assert "current" in blob


def test_help_advertises_every_model_command(monkeypatch):
    """A feature nobody can discover is not a feature."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _run_command(app, pilot, "/help")
            return app.screen.query_one("#transcript").dump()

    blob = "\n".join(asyncio.run(run()))
    print("\n--- /help ---")
    print(blob)

    assert "/switchmodules" in blob
    assert "/backends" in blob
    assert "/model\n" not in blob, "/model was removed but is still advertised"


# --------------------------------------------------------------------------
# /switchmodules
# --------------------------------------------------------------------------


def test_switchmodules_lists_the_catalogue(monkeypatch):
    """``/switchmodules`` must show the current model and mark it."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "jaz_agent.tui.list_models",
        lambda backend: ["stealth/space-bunny-alpha", "openai/gpt-5-mini", "google/gemini-2.5-flash"],
    )

    session, calls = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_model_picker(app, pilot)
            return (
                app.screen.query_one("#transcript").dump(),
                _picker_rows(app),
                _hint_text(app),
            )

    lines, rows, hint = asyncio.run(run())
    blob = "\n".join(lines)
    print("\n--- /switchmodules ---")
    print(blob)
    print("rows:", rows)
    print(hint)

    assert "3 model(s)" in blob
    assert set(rows) == {
        "stealth/space-bunny-alpha",
        "openai/gpt-5-mini",
        "google/gemini-2.5-flash",
    }
    assert not calls, "browsing the catalogue must not invoke the LLM"


def test_switchmodules_filters(monkeypatch):
    """A filter narrows the list instead of dumping all of it."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "jaz_agent.tui.list_models",
        lambda backend: [
            "openai/gpt-5-mini",
            "openai/gpt-5",
            "anthropic/claude-sonnet-4.5",
            "google/gemini-2.5-flash",
        ],
    )

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_model_picker(app, pilot, "gpt")
            return (
                app.screen.query_one("#transcript").dump(),
                _picker_rows(app),
            )

    lines, rows = asyncio.run(run())
    blob = "\n".join(lines)
    print("\n--- /switchmodules gpt ---")
    print(blob)
    print("rows:", rows)

    assert "openai/gpt-5-mini" in rows
    assert not any("claude" in r for r in rows), "the filter did not exclude non-matches"
    assert "matching 'gpt'" in blob


def test_switchmodules_filter_with_no_match_says_so(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("jaz_agent.tui.list_models", lambda backend: ["openai/gpt-5-mini"])

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _run_command(app, pilot, "/switchmodules zzzz")
            return app.screen.query_one("#transcript").dump()

    blob = "\n".join(asyncio.run(run()))
    print("\n--- /switchmodules zzzz ---")
    print(blob)

    assert "no model matches" in blob


def test_switchmodules_paginates_a_large_catalogue(monkeypatch):
    """463 models cannot be shown; the list must page and say so."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "jaz_agent.tui.list_models",
        lambda backend: [f"vendor/model-{i}" for i in range(463)],
    )

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_model_picker(app, pilot)
            return (
                app.screen.query_one("#transcript").dump(),
                _picker_rows(app),
            )

    lines, rows = asyncio.run(run())
    blob = "\n".join(lines)
    print("\n--- /switchmodules (463 models) ---")
    print(blob)
    print(f"picker holds {len(rows)} row(s), first={rows[0]!r} last={rows[-1]!r}")

    assert "463 model(s)" in blob
    # Regression test for the real bug: this list used to be truncated to the
    # first twenty with an "and 443 more" line, which is not pagination -- the
    # other 443 rows had no way to be reached by any means. Every row must be
    # held, and the viewport does the rest.
    assert len(rows) == 463, f"the whole catalogue must be reachable, got {len(rows)}"
    assert rows[0] == "vendor/model-0"
    assert rows[-1] == "vendor/model-462"


def test_every_model_is_reachable_by_scrolling(monkeypatch):
    """↓ must be able to walk past the first screen and reach the last row.

    The windowing is only a rendering strategy; the underlying list has to be
    complete and the cursor has to reach all of it, or the scroll is a scroll
    over twenty rows that happens to be numbered.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "jaz_agent.tui.list_models",
        lambda backend: [f"vendor/model-{i:03d}" for i in range(464)],
    )
    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test(size=(100, 44)) as pilot:
            await pilot.pause()
            await _open_model_picker(app, pilot)
            for _ in range(200):
                await pilot.press("down")
            for _ in range(8):
                await pilot.pause(0.05)
            middle = app._selected
            text = _hint_text(app)
            # Walk the rest of the way to the very last row.
            for _ in range(464 - 1 - 200):
                await pilot.press("down")
            for _ in range(8):
                await pilot.pause(0.05)
            end = app._selected
            end_text = _hint_text(app)
            return len(app._candidates), middle, text, end, end_text

    total, middle, middle_text, end, end_text = asyncio.run(run())
    print(f"\n--- scrolling 464 models ---\nheld={total} after 200 ↓={middle}")
    print(middle_text)
    print(f"after ↓ to the end={end}")
    print(end_text)

    assert total == 464
    assert middle == 200, "↓ stopped early"
    assert "model-200" in middle_text, "the selected row is not on screen"
    assert "model-000" not in middle_text, "the window did not scroll"
    assert end == total - 1, "the last row was never reached"
    assert "model-463" in end_text, "the last row is not on screen"
    assert "464 of 464" in end_text, "the end must be stated as such"


def test_switchmodules_says_when_the_current_model_is_filtered_out(monkeypatch):
    """The model you are on must be accounted for, one way or the other.

    With the list no longer paginated the cursor can always reach the current
    model in the unfiltered browser. A *filtered* list may not contain it, and
    then the header has to say which model is actually live -- otherwise the
    browser appears to be describing the session's model when it is not.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "jaz_agent.tui.list_models",
        lambda backend: ["openai/gpt-5", "openai/gpt-5-mini", "stealth/space-bunny-alpha"],
    )

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_model_picker(app, pilot, "gpt")
            return (
                app.screen.query_one("#transcript").dump(),
                _picker_rows(app),
            )

    lines, rows = asyncio.run(run())
    blob = "\n".join(lines)
    print("\n--- /switchmodules gpt, current model filtered out ---")
    print(blob)
    print("rows:", rows)

    assert "space-bunny-alpha" in blob, "the current model is not accounted for"
    assert "not in these results" in blob, "the reason must be stated"
    assert not any("bunny" in r for r in rows), "the filter did not exclude it"


def test_switchmodules_does_not_repeat_the_current_model(monkeypatch):
    """When it is already on the page, no extra mention is needed."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "jaz_agent.tui.list_models",
        lambda backend: ["stealth/space-bunny-alpha", "openai/gpt-5-mini"],
    )

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_model_picker(app, pilot)
            return (
                app.screen.query_one("#transcript").dump(),
                _picker_rows(app),
                app._selected,
            )

    lines, rows, selected = asyncio.run(run())
    blob = "\n".join(lines)
    print("\n--- /switchmodules, current model on page ---")
    print(blob)
    print(f"rows={rows} selected={selected}")

    assert "not on this page" not in blob
    assert rows.count("stealth/space-bunny-alpha") == 1, "it appears once in the list"
    assert rows[selected] == "stealth/space-bunny-alpha", "it is the selected row"


def test_switchmodules_with_an_exact_model_switches(monkeypatch):
    """Naming a model outright should switch, not browse."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _run_command(app, pilot, "/switchmodules openai/gpt-5-mini")
            return app.screen.query_one("#transcript").dump()

    blob = "\n".join(asyncio.run(run()))
    print("\n--- /switchmodules openai/gpt-5-mini ---")
    print(blob)

    assert session.model == "openai/gpt-5-mini"
    assert "switched to" in blob


# --------------------------------------------------------------------------
# the inline hint
# --------------------------------------------------------------------------


def _hint_text(app) -> str:
    """Rendered hint text, or "" when the hint is collapsed."""
    hint = app.screen.query_one("#hint")
    if not hint.display:
        return ""
    rendered = hint.render()
    return rendered.plain if hasattr(rendered, "plain") else str(rendered)


async def _type_slash(app, pilot, text: str, settle: float = 0.15) -> str:
    box = app.screen.query_one("#task")
    box.value = text
    for _ in range(4):
        await pilot.pause(settle)
    return _hint_text(app)


async def _press_tab(app, pilot) -> str:
    """Press Tab and return the prompt's value afterwards."""
    box = app.screen.query_one("#task")
    await pilot.press("tab")
    for _ in range(6):
        await pilot.pause(0.05)
    return box.value


def _picker_rows(app) -> list[str]:
    """The model ids in the navigable list, top to bottom."""
    return [c.name for c in app._candidates]


async def _open_model_picker(app, pilot, prefix: str = "") -> None:
    """Run ``/switchmodules <prefix>`` and wait for the list to arrive.

    The fetch is a worker thread, so this has to wait on the real clock rather
    than only on Textual's virtual one.

    The space after the command name is part of the argument and must actually
    be pressed: typing the prefix straight on gives ``/switchmodulesgpt``,
    which is a different command and fails.
    """
    box = app.screen.query_one("#task")
    box.focus()
    for ch in "/switchmodules":
        await pilot.press(ch)
    if prefix:
        # The space is part of the argument and must actually be pressed:
        # typing the prefix straight on gives ``/switchmodulesgpt``, which is a
        # different command and fails. With no prefix the trailing space would
        # instead select the "takes an argument" branch and open nothing.
        await pilot.press("space")
        for ch in prefix:
            await pilot.press(ch)
    await pilot.press("enter")
    for _ in range(25):
        await pilot.pause(0.05)
        await asyncio.sleep(0.02)
    assert app._candidates, f"the model list never arrived for {prefix!r}"
    for _ in range(4):
        await pilot.pause(0.05)


def test_tab_completes_a_command(monkeypatch):
    """Tab must complete, not move focus.

    Regression test for the real behaviour: Textual 8's ``Input`` binds no Tab
    key at all (a suggestion is accepted with the right arrow) and ``Screen``
    claims Tab for ``app.focus_next``, so before ``PromptInput`` existed Tab
    moved focus off the prompt and completed nothing.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            box = app.screen.query_one("#task")
            box.focus()
            await pilot.pause()
            for ch in "/swi":
                await pilot.press(ch)
            for _ in range(6):
                await pilot.pause(0.05)
            after = await _press_tab(app, pilot)
            return after, box.has_focus

    value, still_focused = asyncio.run(run())
    print("\n--- Tab on '/swi' ---")
    print(f"value={value!r} focused={still_focused}")

    assert value == "/switchmodules", "Tab did not complete the command"
    assert still_focused, "Tab moved focus instead of completing"


def test_tab_on_a_whole_name_opens_the_argument(monkeypatch):
    """A name that is already complete completes to the space that starts its
    argument, so Tab is never a dead key.

    Typing ``/switchmodules`` character by character means the suggester sees
    ``/switchm``, ``/switchmodu``... the moment the name is whole it returns
    the space, so one Tab is enough. Asserting the intermediate state would be
    asserting a race with the async suggester, which is not the contract.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            box = app.screen.query_one("#task")
            box.focus()
            for ch in "/switchmodules":
                await pilot.press(ch)
            for _ in range(6):
                await pilot.pause(0.05)
            return await _press_tab(app, pilot)

    value = asyncio.run(run())
    print("\n--- Tab on the whole name '/switchmodules' ---")
    print(f"value={value!r}")

    assert value == "/switchmodules ", "Tab should open the argument"


def test_tab_on_ordinary_text_is_inert(monkeypatch):
    """Tab must not nudge the caret while a task is being typed.

    ``cursor_right`` would move the cursor when there is no suggestion, so
    ``action_accept_completion`` checks for a pending one instead.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            box = app.screen.query_one("#task")
            box.focus()
            box.value = "add a flag"
            box.cursor_position = 3
            for _ in range(6):
                await pilot.pause(0.05)
            await pilot.press("tab")
            for _ in range(6):
                await pilot.pause(0.05)
            return box.value, box.cursor_position

    value, cursor = asyncio.run(run())
    print("\n--- Tab on ordinary text ---")
    print(f"value={value!r} cursor={cursor}")

    assert value == "add a flag", "Tab inserted something into a task"
    assert cursor == 3, "Tab moved the caret"


def test_hint_is_vertical(monkeypatch):
    """The candidate list must be one command per line."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            return await _type_slash(app, pilot, "/s")

    hint = asyncio.run(run())
    print("\n--- vertical hint for '/s' ---")
    print(hint)

    lines = [line for line in hint.splitlines() if line.strip()]
    assert len(lines) == 2, f"expected /status and /switchmodules on separate lines: {lines}"
    assert "/status" in lines[0] and "/switchmodules" in lines[1]


# --------------------------------------------------------------------------
# keyboard navigation over the hint list
# --------------------------------------------------------------------------


async def _navigate(app, pilot, keys: str, start: str = "/") -> tuple[int, str, str]:
    """Type *start*, press the keys named in *keys*, return (sel, hint, input).

    *keys* is a space-separated list of key *names*, not a string of characters
    to type -- iterating a bare string would send "down" as the four characters
    d, o, w, n, which is a silent and very confusing test failure.
    """
    box = app.screen.query_one("#task")
    box.focus()
    for ch in start:
        await pilot.press(ch)
    for _ in range(5):
        await pilot.pause(0.05)
    for key in keys.split():
        await pilot.press(key)
        for _ in range(4):
            await pilot.pause(0.05)
    return app._selected, _hint_text(app), box.value


def _nav_app(monkeypatch) -> CodingAgentApp:
    """An app with a small stubbed catalogue.

    The navigation tests type ``/switchmodules`` and must not reach the
    network: without a stub the command falls through to the real OpenRouter
    fetch, the picker opens seconds later with 464 rows, and the assertions end
    up testing the API instead of the navigation.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "jaz_agent.tui.list_models",
        lambda backend: [
            "aion-labs/aion-2.0",
            "aion-labs/aion-3.0",
            "openai/gpt-5",
            "openai/gpt-5-mini",
        ],
    )
    session, _ = _switchable_session()
    return CodingAgentApp(session)


def test_arrow_keys_move_the_hint_selection(monkeypatch):
    """↓ then ↑ must move the cursor over the rows and back."""
    app = _nav_app(monkeypatch)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            down, hint_down, _ = await _navigate(app, pilot, "down", "/s")
            up, hint_up, _ = await _navigate(app, pilot, "down up", "/s")
            return down, hint_down, up, hint_up

    down, hint_down, up, hint_up = asyncio.run(run())
    print("\n--- arrow keys on '/s' ---")
    print(f"after ↓ selected={down}\n{hint_down}")
    print(f"after ↓↑ selected={up}\n{hint_up}")

    assert down == 1, "↓ did not move the selection"
    assert up == 0, "↑ did not move it back"


def test_selection_wraps_at_both_ends(monkeypatch):
    """Clamping would make a keypress at the end look dropped."""
    app = _nav_app(monkeypatch)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            # ↑ from the first row must land on the last.
            wrapped, _, _ = await _navigate(app, pilot, "up", "/s")
            return wrapped, len(app._candidates)

    wrapped, total = asyncio.run(run())
    print(f"\n--- ↑ from the first of {total} rows ---\nselected={wrapped}")

    assert total == 2, f"expected 2 rows for '/s', got {total}"
    assert wrapped == total - 1, "↑ from the top did not wrap to the bottom"


def test_a_single_model_row_can_still_be_chosen(monkeypatch):
    """A one-row list has no row to move to, so Enter must take row 0.

    Regression test for a real defect found by the live run: the command list
    only honours the highlight after the user *moves* it, which is right -- the
    list appears the instant a slash is typed, so a highlight means nothing on
    its own. Applying that same gate to the model browser made a single-match
    filter unusable: ``/switchmodules space-bunny`` showed one row, there was
    nowhere to arrow, and Enter submitted nothing.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "jaz_agent.tui.list_models",
        lambda backend: ["stealth/space-bunny-alpha", "openai/gpt-5", "openai/gpt-5-mini"],
    )
    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_model_picker(app, pilot, "space-bunny")
            assert len(app._candidates) == 1, "this fixture needs exactly one row"
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause(0.05)
                await asyncio.sleep(0.02)
            return app.session.model, app._candidates

    model, rows = asyncio.run(run())
    print(f"\n--- Enter on a one-row model list ---\nmodel={model!r} rows={rows}")

    assert model == "stealth/space-bunny-alpha", "the only row could not be chosen"
    assert not rows, "the picker stayed open"


def test_model_picker_navigates_and_switches(monkeypatch):
    """↑/↓ move over the models and Enter switches to the highlighted one.

    The whole point of the feature: browsing is no longer "read the list, then
    type the id". Selection goes through the same validated switch as the typed
    command, so the probe still runs.
    """
    app = _nav_app(monkeypatch)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_model_picker(app, pilot)
            opened_on = app._candidates[app._selected].name
            await pilot.press("down")
            for _ in range(5):
                await pilot.pause(0.05)
            target = app._candidates[app._selected].name
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause(0.05)
                await asyncio.sleep(0.02)
            return opened_on, target, app.session.model, app._candidates, app._picker

    opened_on, target, after, rows, picker = asyncio.run(run())
    print(f"\n--- ↓ then Enter in the model list ---\nopened on {opened_on!r}, chose {target!r}")
    print(f"session model now: {after!r} | picker closed: {not rows and picker is None}")

    assert target != opened_on, "↓ did not move off the current model"
    assert after == target, f"switched to {after!r}, not {target!r}"
    assert not rows and picker is None, "the picker stayed open after switching"


def test_model_picker_opens_on_the_current_model(monkeypatch):
    """The list must start where the session already is.

    Opening on row 0 while the session is on a model near the end of the
    catalogue means ↓ picks a model the user was not looking at. Pinning the
    current model to the top is worse: it puts the selection at an index
    unrelated to the page order.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "jaz_agent.tui.list_models",
        lambda backend: [
            "aion-labs/aion-2.0",
            "openai/gpt-5",
            "openai/gpt-5-mini",
            "stealth/space-bunny-alpha",
        ],
    )
    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_model_picker(app, pilot)
            return (
                _picker_rows(app),
                app._selected,
                app._candidates[app._selected].name,
                session.model,
            )

    rows, selected, chosen, current = asyncio.run(run())
    print(f"\n--- picker start ---\nrows={rows}\nselected={selected} -> {chosen!r}")

    assert rows.index(current) == selected, "the cursor is not on the current model"
    assert chosen == current
    assert selected == len(rows) - 1, "this fixture expects the current model last"


def test_model_picker_marks_the_current_model(monkeypatch):
    """The current model must be visible in the list, not just in the header."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "jaz_agent.tui.list_models",
        lambda backend: ["openai/gpt-5", "openai/gpt-5-mini"],
    )
    session, _ = _switchable_session()
    session.model = "openai/gpt-5-mini"
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_model_picker(app, pilot)
            return _hint_text(app)

    hint = asyncio.run(run())
    print("\n--- picker with the current model marked ---")
    print(hint)

    marked = [line for line in hint.splitlines() if "✓" in line]
    assert len(marked) == 1, f"exactly one row is marked current: {marked}"
    assert "gpt-5-mini" in marked[0], "the wrong row is marked"


def test_escape_closes_the_model_picker(monkeypatch):
    """Escape must dismiss the model list too, not only the command list."""
    app = _nav_app(monkeypatch)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_model_picker(app, pilot)
            await pilot.press("escape")
            for _ in range(6):
                await pilot.pause(0.05)
            return (
                _hint_text(app),
                app._candidates,
                app._picker,
                app.session.model,
            )

    hint, rows, picker, model = asyncio.run(run())
    print(f"\n--- Escape in the model list ---\nhint={hint!r} model={model!r}")

    assert hint == "", "Escape did not close the picker"
    assert not rows and picker is None
    # Dismissing must not switch anything.
    assert model == "stealth/space-bunny-alpha"


def test_typing_replaces_the_model_picker(monkeypatch):
    """A non-empty keystroke means the user has moved on; the list gets out.

    The empty case is different: submitting the picker leaves the box empty,
    and a programmatic clear must not look like the user starting a new task.
    """
    app = _nav_app(monkeypatch)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_model_picker(app, pilot)
            assert app._picker == "model"
            await pilot.press("a")
            for _ in range(6):
                await pilot.pause(0.05)
            return app._candidates, app._picker

    rows, picker = asyncio.run(run())
    print(f"\n--- typing while the picker is open ---\nrows={len(rows)} picker={picker!r}")

    assert not rows, "the model list survived a keystroke"
    assert picker is None


def test_model_picker_rejects_a_model_the_provider_refuses(monkeypatch):
    """Picking a row is not a licence to skip validation.

    The row came from a catalogue fetch, but a catalogue is a menu, not a
    promise -- and this path must go through the same probe the typed command
    does, or a broken model would be committed without ever being tried.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "jaz_agent.tui.list_models",
        lambda backend: ["openai/gpt-5", "openai/gpt-5-mini"],
    )
    monkeypatch.setattr(
        "jaz_agent.session.probe",
        lambda llm: "model not found, or not available on this backend",
    )
    session = AgentSession()
    before = (session.backend.name, session.model)
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _open_model_picker(app, pilot)
            await pilot.press("down")
            for _ in range(5):
                await pilot.pause(0.05)
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause(0.05)
                await asyncio.sleep(0.02)
            return app.screen.query_one("#transcript").dump()

    blob = "\n".join(asyncio.run(run()))
    print("\n--- picking a model the provider rejects ---")
    print("\n".join(blob.splitlines()[-6:]))

    assert "not switched" in blob
    assert "not found" in blob
    assert (session.backend.name, session.model) == before, "the model moved anyway"


def test_enter_runs_the_highlighted_command(monkeypatch):
    """↓ then Enter must run what was highlighted, not what was typed."""
    app = _nav_app(monkeypatch)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _navigate(app, pilot, "down", "/s")
            assert app._chosen_candidate() == "/switchmodules", "the wrong row is selected"
            await pilot.press("enter")
            for _ in range(12):
                await pilot.pause(0.05)
                await asyncio.sleep(0.02)
            return (
                app.screen.query_one("#transcript").dump(),
                [c.name for c in app._candidates],
            )

    lines, rows = asyncio.run(run())
    blob = "\n".join(lines)
    print("\n--- ↓ then Enter on '/s' ---")
    print(blob)
    print("picker now holds:", rows)

    # /s lists status and switchmodules; ↓ highlights switchmodules, and
    # submitting it must run that -- browse the catalogue -- rather than error
    # on the literal "/s".
    assert "unknown command" not in blob, "the typed prefix was submitted instead"
    assert "model(s) on openrouter" in blob, "the highlighted command did not run"
    assert rows, "the model list did not open"


def test_enter_without_moving_runs_what_was_typed(monkeypatch):
    """An untouched list must not hijack the submission.

    This is the regression risk of "Enter runs the highlight": with the
    selection pinned at row 0, an untouched ``/s`` would silently become
    ``/status``.
    """
    app = _nav_app(monkeypatch)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _navigate(app, pilot, "", "/s")
            await pilot.press("enter")
            for _ in range(10):
                await pilot.pause(0.05)
                await asyncio.sleep(0.02)
            return app.screen.query_one("#transcript").dump()

    blob = "\n".join(asyncio.run(run()))
    print("\n--- Enter on '/s' with no arrow key ---")
    print(blob)

    assert "unknown command" in blob, "the typed text was replaced"


def test_arrow_keys_move_the_caret_when_no_list_is_open(monkeypatch):
    """Ordinary task text keeps working arrow keys.

    The prompt forwards ↑/↓ to the app, which hands them back when there is no
    list. Swallowing them would leave no way to edit a mistyped line.
    """
    app = _nav_app(monkeypatch)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            box = app.screen.query_one("#task")
            box.focus()
            box.value = "add a flag"
            box.cursor_position = 10
            for _ in range(5):
                await pilot.pause(0.05)
            await pilot.press("left")
            for _ in range(6):
                await pilot.pause(0.05)
            return box.cursor_position, box.value

    cursor, value = asyncio.run(run())
    print(f"\n--- ← on ordinary text ---\ncursor={cursor} value={value!r}")

    assert cursor == 9, "the caret did not move"
    assert value == "add a flag", "the text changed"


def test_hint_shows_the_usage_string(monkeypatch):
    """Brackets in a usage string must survive rendering.

    Regression test for a real defect: the hint is Rich markup, so
    ``/switchmodules [model]`` rendered as ``/switchmodules`` and the argument
    shape was never shown at all.
    """
    app = _nav_app(monkeypatch)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            return await _type_slash(app, pilot, "/switchmodules")

    hint = asyncio.run(run())
    print("\n--- hint for '/switchmodules' ---")
    print(repr(hint))

    assert "[model]" in hint, "the usage string was eaten as markup"


def test_hint_appears_when_a_slash_is_typed(monkeypatch):
    """Typing / must immediately show what is available."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            return await _type_slash(app, pilot, "/")

    hint = asyncio.run(run())
    print("\n--- hint for '/' ---")
    print(hint)

    assert "/help" in hint
    assert "/switchmodules" in hint


def test_hint_narrows_with_the_prefix(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            return await _type_slash(app, pilot, "/sw")

    hint = asyncio.run(run())
    print("\n--- hint for '/sw' ---")
    print(hint)

    assert "/switchmodules" in hint
    assert "/help" not in hint, "the hint did not narrow"


def test_hint_expands_a_unique_prefix(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            return await _type_slash(app, pilot, "/cle")

    hint = asyncio.run(run())
    print("\n--- hint for '/cle' ---")
    print(hint)

    assert "/clear" in hint
    assert "clear the transcript" in hint


def test_hint_is_hidden_for_ordinary_text(monkeypatch):
    """The hint must not occupy a line while a task is being typed."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            return await _type_slash(app, pilot, "add a flag to the CLI")

    hint = asyncio.run(run())
    print("\n--- hint for ordinary text ---")
    print(repr(hint))

    assert hint == "", "the hint should be collapsed for non-slash input"


def test_hint_flags_an_unknown_command(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            return await _type_slash(app, pilot, "/zzz")

    hint = asyncio.run(run())
    print("\n--- hint for '/zzz' ---")
    print(hint)

    assert "no such command" in hint


def test_hint_collapses_after_submitting(monkeypatch):
    """Submitting a command must not leave its hint on screen."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _type_slash(app, pilot, "/cle")
            await _run_command(app, pilot, "/clear")
            return _hint_text(app)

    hint = asyncio.run(run())
    print("\n--- hint after submit ---")
    print(repr(hint))

    assert hint == ""


def test_retired_command_redirects_to_its_replacement(monkeypatch):
    """``/model`` must name the replacement, not just fail.

    The user typed something that used to work; "unknown command" is true and
    useless, and this is the whole reason ``commands.RETIRED`` exists.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _run_command(app, pilot, "/model")
            return app.screen.query_one("#transcript").dump()

    blob = "\n".join(asyncio.run(run()))
    print("\n--- /model after removal ---")
    print(blob)

    assert "gone" in blob
    assert "/switchmodules" in blob, "it must name the replacement"
    assert "unknown command" not in blob, "a removed command is not a typo"


def test_retired_command_redirects_in_the_inline_hint(monkeypatch):
    """The hint should redirect while typing, before Enter is pressed."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            return await _type_slash(app, pilot, "/model")

    hint = asyncio.run(run())
    print("\n--- hint for '/model' ---")
    print(repr(hint))

    assert "gone" in hint
    assert "/switchmodules" in hint


def test_escape_dismisses_the_hint_and_the_next_keystroke_reopens_it(monkeypatch):
    """Escape closes the list for the current value, not for good.

    The failure this prevents: a stray ``/`` with no way to get the space back.
    The list returns as soon as the value changes, so nothing is lost.
    """
    app = _nav_app(monkeypatch)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            box = app.screen.query_one("#task")
            box.focus()
            for ch in "/s":
                await pilot.press(ch)
            for _ in range(5):
                await pilot.pause(0.05)
            before = _hint_text(app)
            await pilot.press("escape")
            for _ in range(5):
                await pilot.pause(0.05)
            hidden = _hint_text(app)
            value_after_escape = box.value
            # One more keystroke: the list is relevant again.
            await pilot.press("t")
            for _ in range(5):
                await pilot.pause(0.05)
            return before, hidden, value_after_escape, _hint_text(app), box.value

    before, hidden, value, after, value_after = asyncio.run(run())
    print("\n--- Escape on '/s' ---")
    print(f"before   : {before.splitlines()[0]!r}")
    print(f"escaped  : {hidden!r} (input kept {value!r})")
    print(f"after '/st': {after.splitlines()[0]!r} (input {value_after!r})")

    assert before, "the list was not shown to begin with"
    assert hidden == "", "Escape did not close the list"
    assert value == "/s", "Escape must not alter what was typed"
    assert after, "the list did not come back on the next keystroke"


def test_escape_does_not_swallow_the_key_for_the_input(monkeypatch):
    """Escape must still reach the input when no list is open.

    Escape is the input's own cancel in Textual; taking it unconditionally
    would break that for ordinary task text.
    """
    app = _nav_app(monkeypatch)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            box = app.screen.query_one("#task")
            box.focus()
            box.value = "add a flag"
            for _ in range(5):
                await pilot.pause(0.05)
            await pilot.press("escape")
            for _ in range(5):
                await pilot.pause(0.05)
            return app._candidates

    candidates_after = asyncio.run(run())
    print(f"\n--- Escape on ordinary text ---\ncandidates={candidates_after}")

    assert candidates_after == [], "a list was built for non-slash input"


def test_mistyped_command_suggests_the_closest(monkeypatch):
    """A transposition should offer the fix, not just reject."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _run_command(app, pilot, "/switchmoduls")
            return app.screen.query_one("#transcript").dump()

    blob = "\n".join(asyncio.run(run()))
    print("\n--- /switchmoduls ---")
    print(blob)

    assert "did you mean /switchmodules" in blob