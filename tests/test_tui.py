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


def test_model_command_shows_current_and_alternatives(monkeypatch):
    """``/model`` with no argument must report the current model."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("jaz_agent.llm_config._list_openrouter_models", lambda b: [])

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _run_command(app, pilot, "/model")
            return app.screen.query_one("#transcript").dump()

    blob = "\n".join(asyncio.run(run()))
    print("\n--- /model ---")
    print(blob)

    assert "current:" in blob
    assert session.model in blob
    # The menu must also offer somewhere to go.
    assert "/model <name>" in blob


def _banner_text(app) -> str:
    """Plain text of the banner.

    ``Static.render()`` returns a rich ``Content`` object, not a str, so an
    ``in`` assertion against it silently fails. Converting here keeps the
    assertions below readable.
    """
    rendered = app.banner.render()
    return rendered.plain if hasattr(rendered, "plain") else str(rendered)


def test_model_command_switches(monkeypatch):
    """``/model <name>`` commits the switch and says which model is live."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _run_command(app, pilot, "/model openai/gpt-5-mini")
            return app.screen.query_one("#transcript").dump(), _banner_text(app)

    lines, banner = asyncio.run(run())
    blob = "\n".join(lines)
    print("\n--- /model openai/gpt-5-mini ---")
    print(blob)
    print("--- banner ---")
    print(banner)

    assert session.model == "openai/gpt-5-mini"
    assert "switched to" in blob
    # The header must not keep claiming the old model.
    assert "gpt-5-mini" in banner


def test_model_command_switches_backend(monkeypatch):
    """A bare backend name moves the session to that provider."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _run_command(app, pilot, "/model anthropic")
            return app.screen.query_one("#transcript").dump()

    blob = "\n".join(asyncio.run(run()))
    print("\n--- /model anthropic ---")
    print(blob)

    assert session.backend.name == "anthropic"
    assert "anthropic" in blob


def test_failed_switch_is_reported_and_leaves_state_alone(monkeypatch):
    """A rejected switch must say so *and* keep the old model.

    Both halves matter: silence leaves the user unsure, and a silent state
    change would break the next turn. The session is built *without* an injected
    model so the real probe path runs -- the probe itself is stubbed, which is
    what a provider rejection looks like from the caller's side.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "jaz_agent.session.probe", lambda llm: "model not found, or not available on this backend"
    )

    session = AgentSession()
    before = (session.backend.name, session.model)
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _run_command(app, pilot, "/model nope/does-not-exist")
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


def test_help_lists_the_model_commands(monkeypatch):
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

    assert "/model" in blob
    assert "/backends" in blob
    assert "/switchmodules" in blob


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
            await _run_command(app, pilot, "/switchmodules")
            return app.screen.query_one("#transcript").dump()

    blob = "\n".join(asyncio.run(run()))
    print("\n--- /switchmodules ---")
    print(blob)

    assert "3 model(s)" in blob
    assert "gpt-5-mini" in blob
    assert "gemini-2.5-flash" in blob
    # The current model must be marked, and it must not reach the model.
    assert "✓ stealth/space-bunny-alpha" in blob
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
            await _run_command(app, pilot, "/switchmodules gpt")
            return app.screen.query_one("#transcript").dump()

    blob = "\n".join(asyncio.run(run()))
    print("\n--- /switchmodules gpt ---")
    print(blob)

    assert "gpt-5-mini" in blob
    assert "claude" not in blob, "the filter did not exclude non-matches"
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
    """463 models cannot be printed; the list must page and say so."""
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
            await _run_command(app, pilot, "/switchmodules")
            return app.screen.query_one("#transcript").dump()

    blob = "\n".join(asyncio.run(run()))
    print("\n--- /switchmodules (463 models) ---")
    print(blob[:1200])

    assert "463 model(s)" in blob
    assert "model-0" in blob
    assert "and 443 more" in blob, "the truncation must be stated"


def test_switchmodules_shows_the_current_model_even_off_page(monkeypatch):
    """The model you are on must be visible, not buried at position 420.

    Regression test for a real find: OpenRouter returns 463 ids sorted, so an
    alphabetical first page left `stealth/space-bunny-alpha` -- the model this
    project defaults to -- off screen entirely.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "jaz_agent.tui.list_models",
        lambda backend: [f"aaa/vendor-{i}" for i in range(30)]
        + ["zzz/space-bunny-alpha"],
    )

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _run_command(app, pilot, "/switchmodules")
            return app.screen.query_one("#transcript").dump()

    blob = "\n".join(asyncio.run(run()))
    print("\n--- /switchmodules with an off-page current model ---")
    print(blob[:900])

    assert "space-bunny-alpha" in blob, "the current model is not shown"
    assert "← current" in blob, "it is not marked as the current one"


def test_switchmodules_does_not_repeat_the_current_model(monkeypatch):
    """When it already fits on the page, the extra header line must not repeat it."""
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
            await _run_command(app, pilot, "/switchmodules")
            return app.screen.query_one("#transcript").dump()

    blob = "\n".join(asyncio.run(run()))
    print("\n--- /switchmodules, current model on page ---")
    print(blob)

    assert "← current" not in blob
    assert blob.count("stealth/space-bunny-alpha") == 1


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


def test_mistyped_command_suggests_the_closest(monkeypatch):
    """A transposition should offer the fix, not just reject."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    session, _ = _switchable_session()
    app = CodingAgentApp(session)

    async def run():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _run_command(app, pilot, "/modle")
            return app.screen.query_one("#transcript").dump()

    blob = "\n".join(asyncio.run(run()))
    print("\n--- /modle ---")
    print(blob)

    assert "did you mean /model" in blob