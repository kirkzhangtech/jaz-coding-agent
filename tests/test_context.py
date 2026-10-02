"""Conversation memory: what is kept, what is condensed, what is sent.

The policy in :mod:`jaz_agent.context` is the one part of the agent that
silently loses information, so these tests are mostly about the failure modes
that would be invisible from the outside: a condensed turn that drops the
current task, a budget trim that eats the newest turn, a report whose alignment
lies about what is being sent.
"""

from __future__ import annotations

import pytest

from jaz_agent.context import (
    DIGEST_FRACTION,
    DIGEST_HEADER,
    DIGEST_MAX_TURNS,
    GRADE_NEWEST,
    GRADE_OLDEST,
    RECENT_TURNS,
    TASK_HEAD_CHARS,
    WORTHWHILE,
    Context,
    Turn,
    _grade,
    _is_digest,
    report,
)


def filled(n: int, *, size: int = 60, report_size: int = 900) -> Context:
    """A context with *n* turns of predictable content.

    The defaults are realistic on purpose. Real reports from a coding agent run
    several hundred to a couple of thousand characters; the break-even against a
    digest sits around 420, so a fixture smaller than that would make every
    test below pass or fail on the wrong side of a threshold that does not
    matter.
    """
    ctx = Context()
    for i in range(n):
        ctx.add(f"task {i} " + "t" * size, f"report {i} " + "r" * report_size)
    return ctx


# -- the record ------------------------------------------------------------


def test_record_keeps_every_turn_forever():
    """Compression must never touch the record -- only the prompt."""
    ctx = filled(50)
    assert len(ctx.turns) == 50
    assert ctx.turns[0].task.startswith("task 0")


def test_clear_drops_the_record():
    ctx = filled(3)
    ctx.clear()
    assert ctx.turns == []
    assert ctx.prompt() == []


def test_size_counts_task_and_report():
    ctx = Context()
    ctx.add("abc", "de")
    assert ctx.size == 5


def test_add_returns_the_turn_it_recorded():
    ctx = Context()
    turn = ctx.add("q", "a")
    assert isinstance(turn, Turn)
    assert ctx.turns[0] is turn


# -- the prompt ------------------------------------------------------------


def test_empty_context_sends_nothing():
    """An empty `prior_turns` would still be rendered into the prompt."""
    assert Context().prompt() == []


def test_turns_below_the_window_are_sent_verbatim():
    ctx = filled(RECENT_TURNS - 1)
    messages = ctx.prompt()
    assert [m["content"] for m in messages if m["role"] == "user"] == [
        f"task {i} " + "t" * 60 for i in range(RECENT_TURNS - 1)
    ]
    assert all("condensed" not in m["content"] for m in messages)


def test_old_turns_become_one_digest_message():
    ctx = filled(10)
    messages = ctx.prompt()
    digests = [m for m in messages if _is_digest(m)]
    assert len(digests) == 1, "old turns should collapse into a single message"
    assert f"[{10 - RECENT_TURNS} earlier turn(s), condensed" in digests[0]["content"]


def test_the_newest_turn_is_never_condensed():
    """A summary of the current task is a summary of the wrong thing."""
    ctx = filled(10)
    last = ctx.prompt()[-2:]
    assert last[0] == {"role": "user", "content": ctx.turns[-1].task}
    assert last[1]["content"] == ctx.turns[-1].report


def test_recent_window_is_configurable():
    ctx = filled(6)
    ctx.recent = 2
    digest = next(m for m in ctx.prompt() if _is_digest(m))
    assert "[4 earlier turn(s), condensed" in digest["content"]
    assert "task 5" not in digest["content"], (
        "turn 5 is inside a window of 2 and must stay verbatim"
    )


def test_zero_window_condenses_everything():
    ctx = filled(4)
    ctx.recent = 0
    messages = ctx.prompt()
    assert len(messages) == 1
    assert "[4 earlier turn(s), condensed" in messages[0]["content"]


# -- the budget ------------------------------------------------------------


def test_prompt_respects_the_budget():
    ctx = filled(40, size=500, report_size=900)
    ctx.budget = 1500
    cost = Context._cost(ctx.prompt())
    assert cost <= 1500, f"sent {cost} chars against a 1500 budget"


def test_the_budget_is_not_a_turn_count():
    """A generous budget must not shrink -- four short turns fit easily."""
    ctx = filled(4)
    ctx.budget = 100_000
    assert Context._cost(ctx.prompt()) < 10_000


def test_a_tiny_budget_still_keeps_the_newest_turn():
    ctx = filled(20)
    ctx.budget = 10
    messages = ctx.prompt()
    assert messages, "an over-budget prompt must not become an empty one"
    assert ctx.turns[-1].task in "".join(m["content"] for m in messages)


def test_an_impossible_budget_degrades_rather_than_crashes():
    ctx = filled(10)
    ctx.budget = 0
    assert isinstance(ctx.prompt(), list)


# -- compression fidelity --------------------------------------------------


def test_tiny_turns_are_left_alone():
    """A digest of one-liners costs more than the one-liners.

    This is the case that would otherwise ship: a short session whose context is
    *worse* than no compression at all, because the summary only removed detail.
    """
    ctx = filled(10, size=5, report_size=20)
    assert not any(_is_digest(m) for m in ctx.prompt())


def test_a_digest_that_saves_nothing_is_declined():
    """The other side of :func:`test_a_long_history_is_condensed`: real reports
    condense, stubs do not."""
    ctx = filled(10, size=5, report_size=100)
    assert not any(_is_digest(m) for m in ctx.prompt())


def test_the_worthwhile_check_is_strict():
    """A digest that breaks even is not used -- it saves nothing and costs detail."""
    assert WORTHWHILE == 1.0


def test_a_long_history_is_condensed():
    ctx = filled(10)
    assert any(_is_digest(m) for m in ctx.prompt())


def test_the_digest_header_constant_is_what_is_matched():
    """`_is_digest` keys off the constant; a divergence would make `/context`
    mislabel every turn."""
    ctx = filled(10)
    digest = next(m for m in ctx.prompt() if _is_digest(m))
    assert _is_digest(digest) == digest["content"].startswith(DIGEST_HEADER)


# -- grading detail by age -------------------------------------------------


def test_older_condensed_turns_keep_less_detail():
    """The digest grades detail by age, which is what makes it scale."""
    ctx = filled(10)
    digest = next(m for m in ctx.prompt() if _is_digest(m))["content"]
    head, _, tail = digest.partition("condensed; oldest first")
    assert len(tail) > 0

    entries = [ln for ln in tail.splitlines() if ln.startswith("- asked:")]
    assert len(entries) >= 2
    sizes = [len(ln) for ln in entries]
    assert sizes == sorted(sizes), f"detail should grow with recency: {sizes}"


def test_grading_runs_from_oldest_to_newest():
    assert _grade(0, 5) == GRADE_OLDEST
    assert _grade(4, 5) == GRADE_NEWEST
    assert _grade(2, 5) == pytest.approx((GRADE_OLDEST + GRADE_NEWEST) / 2)


def test_a_single_condensed_turn_gets_full_detail():
    assert _grade(0, 1) == GRADE_NEWEST


def test_grade_never_reaches_zero():
    """A grade of zero would elide the whole turn, which is dropping it with
    extra steps."""
    assert GRADE_OLDEST > 0
    assert 0 < _grade(0, 1000)


# -- the digest is itself bounded -----------------------------------------


def test_a_very_long_history_still_produces_one_fitting_digest():
    """The measured failure: 26 fat turns graded by age came to 8,893 chars
    against a 3,000 cap, so the digest was rejected and 24 turns were dropped
    outright. Capping the number of turns described is what fixed it."""
    ctx = filled(200)
    messages = ctx.prompt()
    digest = next(m for m in messages if _is_digest(m))
    assert Context._cost([digest]) <= ctx.budget * DIGEST_FRACTION
    assert ctx.describe().dropped == 0


def test_turns_beyond_the_cap_are_counted_rather_than_summarised():
    ctx = filled(200)
    digest = next(m for m in ctx.prompt() if _is_digest(m))["content"]
    assert "summarised to a count" in digest


def test_the_cap_is_not_reached_when_the_history_is_short():
    ctx = filled(RECENT_TURNS + DIGEST_MAX_TURNS)
    digest = next(m for m in ctx.prompt() if _is_digest(m))["content"]
    assert "summarised to a count" not in digest


# -- the report ------------------------------------------------------------


def test_a_trim_never_leaves_an_orphaned_assistant_reply():
    """A transcript starting mid-exchange reads as the model talking to itself."""
    ctx = filled(30)
    ctx.budget = 6000
    messages = ctx.prompt()
    assert messages, "an over-budget prompt must still send something"
    assert messages[0]["role"] == "user", f"starts with {messages[0]['role']}"


def test_trimming_drops_from_the_oldest_end():
    ctx = filled(30)
    ctx.budget = 4000
    text = "".join(m["content"] for m in ctx.prompt())
    assert "task 29" in text, "the newest turn must survive"
    assert "task 0" not in text, "the oldest turn is what gets dropped first"


def test_an_enormous_single_turn_exceeds_the_budget_rather_than_vanishing():
    """Truncating the current task is worse than sending a long prompt."""
    ctx = Context()
    ctx.add("x" * 50_000, "y" * 50_000)
    ctx.budget = 10
    messages = ctx.prompt()
    assert len(messages) == 2
    assert "x" * 50_000 in messages[0]["content"]


def test_a_long_task_is_clipped_not_dropped():
    """A pasted spec is the part most worth keeping verbatim."""
    ctx = Context()
    ctx.add("S" * (TASK_HEAD_CHARS * 3), "r" * 2000)
    ctx.recent = 0
    digest = ctx.prompt()[0]["content"]
    assert "S" * TASK_HEAD_CHARS in digest
    assert "chars elided" in digest


def test_a_long_report_keeps_both_ends():
    """The claim is at the front and the verification at the back; the middle
    is a file listing and is the cheapest thing to lose."""
    turn = Turn(task="t", report="HEAD" + "m" * 5000 + "TAIL")
    compressed = turn.condensed()
    assert "HEAD" in compressed
    assert "TAIL" in compressed
    assert "chars elided" in compressed


def test_a_short_turn_is_not_annotated():
    """Elision markers on turns that lost nothing are noise."""
    assert "elided" not in Turn(task="t", report="r").condensed()


def test_a_turn_with_no_report_says_so():
    """A cancelled turn has no report; silence would read as a lost one."""
    assert "no report" in Turn(task="t").condensed()


def test_a_turn_is_rendered_as_prose_not_a_repr():
    """jaz reprs `prior_turns` into the prompt. A nested repr inside that is
    both harder to read and larger than the same facts in English."""
    ctx = filled(10)
    digest = next(m for m in ctx.prompt() if _is_digest(m))
    assert "{'role'" not in digest["content"]
    assert "- asked: task 0" in digest["content"]


# -- the report ------------------------------------------------------------


def test_report_on_an_empty_context():
    assert "empty" in report(Context()).lower()


def test_report_contrasts_record_with_prompt():
    ctx = filled(12)
    text = report(ctx)
    assert f"{ctx.size:,}" in text, "the full record size should be shown"
    assert "next prompt" in text
    assert "12 turn(s) recorded" in text


def test_report_lists_every_turn_with_its_state():
    ctx = filled(6)
    lines = report(ctx).splitlines()
    body = [ln for ln in lines if ". [" in ln]
    assert len(body) == 6
    assert body[-1].count("verbatim") == 1
    assert body[0].count("condensed") == 1


def test_report_flags_an_over_budget_prompt():
    ctx = filled(30, size=500, report_size=900)
    ctx.budget = 100
    assert "over budget" in report(ctx)


def test_report_shows_the_history_sent_whole_when_it_is():
    ctx = filled(2)
    assert "sent whole" in report(ctx)


def test_report_counts_dropped_turns_rather_than_guessing():
    """`_fit` can shed turns the policy never mentioned; the report has to say so."""
    ctx = filled(30)
    ctx.budget = 100
    text = report(ctx)
    assert "dropped" in text
    assert "[  dropped]" in text, "dropped turns must be labelled as such"


def test_dropped_turns_are_the_oldest():
    """Losing the newest turn would mean the agent cannot see what it just did."""
    ctx = filled(30)
    ctx.budget = 100
    states = [ln.split("[")[1].split("]")[0].strip() for ln in report(ctx).splitlines()
              if ". [" in ln]
    assert states[-1] == "verbatim"
    assert set(states[:-1]) == {"dropped"}


def test_a_condensed_turn_is_not_reported_as_dropped():
    """The bug this guards: counting assistant messages makes every condensed
    turn look dropped, so a 30-turn session reports "26 dropped" when in fact
    all 30 are being sent."""
    ctx = filled(30)
    plan = ctx.describe()
    assert plan.dropped == 0
    assert plan.condensed + plan.verbatim == 30
    header = report(ctx).split("Turns, oldest first:")[0]
    assert "dropped" not in header


def test_report_survives_a_multiline_task():
    """The task preview takes the first line; a leading blank must not crash it."""
    ctx = Context()
    ctx.add("\n\n  \nsecond line", "r")
    report(ctx)


@pytest.mark.parametrize("count", [0, 1, 2, 5, 17, 40])
def test_report_holds_up_whatever_the_history_length(count):
    ctx = filled(count)
    assert report(ctx)


def test_prompt_is_stable_across_calls():
    """`/context` must not perturb what the next turn sends."""
    ctx = filled(10)
    assert ctx.prompt() == ctx.prompt()


# -- wiring into the session and the command -------------------------------


def test_the_session_records_turns_in_the_context(monkeypatch):
    """`run_sync` is the path both the TUI and `--prompt` take."""
    from jaz import MockLLMClient

    from jaz_agent.session import AgentSession

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    session = AgentSession(llm=MockLLMClient(fn=lambda *a, **k: "return finish('done')"))
    session.max_iterations = 4

    assert session.run_sync("first task") == "done"
    assert [t.task for t in session.context.turns] == ["first task"]
    assert session.context.turns[0].report == "done"


def test_prior_turns_reaches_jaz_as_a_bounded_list(monkeypatch):
    """The integration point: what `_run_turn` passes is what `prompt` returns.

    The mock records the rendered messages, which is a stronger check than a
    model-level test: it shows what the model is actually shown, rather than
    what it did with it.
    """
    import re

    from jaz import MockLLMClient

    from jaz_agent.session import AgentSession

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    seen: list[str] = []

    def capture(model, messages, **kwargs):
        seen.append(messages[-1]["content"])
        return "return finish('ok')"

    session = AgentSession(llm=MockLLMClient(fn=capture))
    session.max_iterations = 4
    for i in range(RECENT_TURNS + 4):
        session.run_sync(f"task {i}")

    block = re.search(r"<prior_turns.*?</prior_turns>", seen[-1], re.S)
    assert block, "prior_turns must be rendered into the prompt"
    assert "task 0" in block.group(0), "condensed history must survive into the prompt"
    assert session.context.turns[-1].task in seen[-1], "the live task is passed as `task`"


def test_the_context_command_is_registered_and_dispatched(monkeypatch):
    """A command in the table with no handler says "not implemented" -- worse
    than not listing it at all."""
    from jaz_agent.commands import find
    from jaz_agent.session import AgentSession
    from jaz_agent.tui import CodingAgentApp

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert find("/context") is not None

    session = AgentSession()
    session.context.add("do a thing", "did the thing")

    app = CodingAgentApp.__new__(CodingAgentApp)
    app.session = session
    app.status = None
    app.transcript = None
    app.started = 0.0
    said: list[str] = []
    app._say = lambda text, kind=None: said.append(text)  # type: ignore[method-assign]

    app.action_context()

    assert said, "/context must say something"
    assert "do a thing" in said[0]