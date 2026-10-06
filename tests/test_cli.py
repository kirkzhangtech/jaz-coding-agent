"""The command-line surface: one-shot runs must report what happened.

Small on purpose -- the TUI is the product, and ``-p`` exists so a task can be
scripted. The property worth pinning is that a *failed* one-shot run is
distinguishable from a successful one, which it was not: the failure reached the
event queue, nothing drained it, and the process printed an empty line and
exited 0.
"""

from __future__ import annotations

import warnings

from jaz import MockLLMClient

from jaz_agent import __main__ as cli
from jaz_agent.session import AgentSession
from jaz_agent.tools import ROOT

warnings.simplefilter("ignore")


def _scripted(monkeypatch, fn) -> AgentSession:
    """A session whose only model call is *fn*, handed to ``cli.main``.

    ``main`` builds its own session, so the factory is replaced rather than the
    class: an injected model needs no key and no network, which is what makes
    these two tests run anywhere.
    """
    session = AgentSession(llm=MockLLMClient(fn=fn))
    session.max_iterations = 2
    monkeypatch.setattr(cli, "AgentSession", lambda **kwargs: session)
    return session


def test_prompt_reports_a_failed_turn(monkeypatch, capsys):
    """A turn that never finishes must not look like a silent success.

    The failure is scripted as "the agent never calls finish" rather than as a
    provider exception: an exception is retried by the agent loop with a backoff,
    which made this test take five minutes. The path under test -- an ERROR event
    with no report to print -- is the same one either way.
    """

    def never_finishes(model, messages, **kwargs):
        return "print('still working')"

    _scripted(monkeypatch, never_finishes)

    code = cli.main(["-p", "do something", "-w", str(ROOT)])
    captured = capsys.readouterr()

    assert code != 0, "a failed one-shot run exited successfully"
    assert "IterationLimit" in captured.err


def test_prompt_prints_the_report(monkeypatch, capsys):
    """The normal path still prints the agent's summary and exits 0."""

    def finish(model, messages, **kwargs):
        return "return finish('all good')"

    _scripted(monkeypatch, finish)

    code = cli.main(["-p", "do something", "-w", str(ROOT)])
    captured = capsys.readouterr()

    assert code == 0
    assert "all good" in captured.out
