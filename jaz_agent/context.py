"""Conversation memory: what the agent remembers, and how much of it.

jaz's ``invoke`` is stateless -- every call is a fresh loop with a fresh
context. A conversational agent has to supply the memory itself, and that code
lives here rather than in :mod:`jaz_agent.session` because it is a policy, and a
policy can be tested without a model.

Two ideas carry the module.

**The record is unbounded; only the prompt is bounded.** Every turn appends to
:attr:`Context.turns` for the life of the session -- ``/new`` is the only thing
that clears it. What is capped is :meth:`Context.prompt`, which is what reaches
the model. Losing the record and losing the ability to recall it are different
failures, and only the second is a bug, so they are deliberately not the same
object.

**Compression must be worth it.** Condensing is not automatically an
improvement: a digest has fixed overhead, and a session of short turns comes out
*larger* condensed than verbatim. That case is declined, because a summary that
saves nothing while removing detail is strictly worse than no summary. The
decision is made in one place, :meth:`Context.build`, which returns both the
messages and a :class:`Plan` describing what it did -- so ``/context`` reports
the prompt's real behaviour instead of predicting it and being wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Turns kept verbatim before older ones are condensed. Four covers "what did I
#: just ask, what came back" twice over, which is the window in which a
#: follow-up like "now do the same for the parser" is actually made. Below that
#: the agent starts forgetting the file it just edited; far above it, the tail
#: of the prompt is mostly history the model will not attend to.
RECENT_TURNS = 4

#: Characters kept from the head of a condensed *task*. A task statement is
#: short by nature; anything longer was a paste, and the paste is the one part
#: worth keeping verbatim because it is often the actual specification.
TASK_HEAD_CHARS = 600

#: Characters kept from a condensed *report*, at each end. A report is the
#: agent's own markdown summary of its work: the head is the claim, the tail is
#: the verification. The middle is where a long file listing sits, and it is the
#: cheapest thing to lose.
#:
REPORT_HEAD_CHARS = 240
REPORT_TAIL_CHARS = 120

#: How much of a condensed turn survives, for a turn at each end of the
#: condensed range. The newest keeps all of it; the oldest keeps a quarter.
#:
#: A flat clip does not actually compress. At 240/120 a 900-character report
#: still keeps 360 characters, so a digest of twenty turns costs more than half
#: of what it replaces and cannot fit the budget it exists to satisfy -- the
#: measured ratio was 12386 chars of digest against 25408 of history, against a
#: 3000-char cap. Grading the clip by age makes the digest *scale*: recency is
#: exactly what gives an old turn its value, so the newest condensed turn stays
#: nearly whole and the oldest is reduced to a bare claim.
GRADE_NEWEST = 1.0
GRADE_OLDEST = 0.25

#: First line of the message standing in for condensed turns. Doubles as the
#: marker used to recognise a digest, so the two cannot drift apart.
DIGEST_HEADER = "Earlier work in this session, in order:"

#: A digest is used only when it costs strictly less than the turns it replaces.
#: The strictness is the point: a digest that breaks even saves no characters
#: while still removing detail the model could have used.
WORTHWHILE = 1.0

#: Largest share of the budget the digest may occupy. The digest has to be
#: bounded in its own right: left uncapped it grows with the session, and a
#: digest of thirty fat turns is larger than the whole budget it was meant to
#: fit inside. Half leaves room for the verbatim window beside it.
DIGEST_FRACTION = 0.5

#: Most condensed turns a digest will describe individually. Past this the
#: oldest are folded into a count rather than summarised, because there is a
#: floor below which a condensed turn stops being readable -- roughly 250
#: characters of task and report no matter how aggressively it is graded -- and
#: 40 turns at that floor is already more than the budget. A summary of the
#: first twenty turns is worth more than a garbled thirty-first, because by then
#: what a turn said is far less important than that work happened at all.
#:
#: Measured: 26 turns graded by age came to 8,893 characters against a 3,000 cap;
#: capped at 12 turns it is 4,272; at 8 it is ~2,900 and fits. Truncating the
#: list rather than the entries is the only thing that fits.
DIGEST_MAX_TURNS = 8


def _grade(index: int, total: int) -> float:
    """Detail multiplier for the *index*-th of *total* condensed turns.

    Linear from :data:`GRADE_OLDEST` at the oldest turn to :data:`GRADE_NEWEST`
    at the newest, so the digest's size grows sub-linearly with the session
    instead of tracking it one-for-one.
    """
    if total <= 1:
        return GRADE_NEWEST
    position = index / (total - 1)  # 0.0 = oldest, 1.0 = newest
    return GRADE_OLDEST + (GRADE_NEWEST - GRADE_OLDEST) * position


def _is_digest(message: dict[str, str]) -> bool:
    """Whether *message* is a condensed digest rather than a real turn."""
    return message["content"].startswith(DIGEST_HEADER)


def _clip(text: str, head: int, tail: int = 0) -> str:
    """Shorten *text* to *head* chars, optionally keeping a *tail* as well.

    Head and tail rather than head alone, because the useful part of a report is
    not all at the front.
    """
    text = text.strip()
    if len(text) <= head + tail:
        return text
    lost = len(text) - head - tail
    if tail:
        return f"{text[:head]}\n[... {lost:,} chars elided ...]\n{text[-tail:]}"
    return f"{text[:head]}[... {lost:,} chars elided ...]"


@dataclass(frozen=True, slots=True)
class Turn:
    """One completed exchange: what was asked, and what came back."""

    task: str
    report: str = ""

    def as_messages(self) -> list[dict[str, str]]:
        """The chat-message form jaz renders into ``prior_turns``."""
        return [
            {"role": "user", "content": self.task},
            {"role": "assistant", "content": self.report},
        ]

    def condensed(self, grade: float = GRADE_NEWEST) -> str:
        """A short stand-in for this turn.

        Rendered as prose rather than as data on purpose. The prompt is a
        transcript the model reads as a transcript; a Python ``repr`` in the
        middle of it -- which is exactly what jaz does to ``prior_turns`` --
        wastes characters and is harder to read than the same facts in English.

        *grade* scales how much survives, from 1.0 for a nearly-current turn down
        to :data:`GRADE_OLDEST` for an ancient one. See :func:`_grade`.
        """
        head = max(1, int(TASK_HEAD_CHARS * grade))
        report_head = max(1, int(REPORT_HEAD_CHARS * grade))
        report_tail = max(1, int(REPORT_TAIL_CHARS * grade))

        task = _clip(self.task, head)
        if not self.report:
            return f"- asked: {task} (no report; the turn did not finish)"
        body = _clip(self.report, report_head, report_tail)
        return f"- asked: {task}\n  did: {body}"


@dataclass(frozen=True, slots=True)
class Plan:
    """What a prompt build did with each recorded turn.

    Carried out of :meth:`Context.build` so that ``/context`` can state what
    happened instead of re-deriving it. An earlier version counted assistant
    messages and reported 26 turns "dropped" on a prompt that in fact contained
    all 30 of them, condensed -- a summary of the context that lied about the
    context, which is the one failure this module cannot have.
    """

    #: Per-turn state, oldest first: ``condensed``, ``verbatim`` or ``dropped``.
    states: tuple[str, ...]
    #: Characters actually sent.
    cost: int = 0

    def state_of(self, index: int) -> str:
        """State of 1-based turn *index*."""
        return self.states[index - 1] if 0 < index <= len(self.states) else "unknown"

    def count(self, state: str) -> int:
        return sum(1 for s in self.states if s == state)

    @property
    def dropped(self) -> int:
        return self.count("dropped")

    @property
    def condensed(self) -> int:
        return self.count("condensed")

    @property
    def verbatim(self) -> int:
        return self.count("verbatim")

    @property
    def summary(self) -> str:
        """One line describing the split, e.g. ``"26 condensed, 4 verbatim"``."""
        parts = [
            f"{n} {state}"
            for state, n in (
                ("dropped", self.dropped),
                ("condensed", self.condensed),
                ("verbatim", self.verbatim),
            )
            if n
        ]
        return ", ".join(parts) or "nothing to send"


@dataclass(slots=True)
class Context:
    """The conversation so far, and how much of it to send.

    A character budget rather than a turn count, because turn sizes vary by two
    orders of magnitude -- "run the tests" and a pasted stack trace are both one
    turn -- so a turn budget is not a cost budget.

    The default budget is sized against real measurements rather than taste. A
    verbatim window of four ordinary turns -- a ~60-character task and a
    ~900-character report -- already occupies about 3,900 characters, so a 6,000
    budget left no room for any history at all and the digest was dropped on
    nearly every turn that had one. 12,000 comfortably holds the window plus a
    digest, and against a 1M-token window it is still a rounding error.
    """

    #: Characters of context to send, across the digest and the verbatim turns.
    budget: int = 12000
    #: Verbatim turns kept at the tail. See :data:`RECENT_TURNS`.
    recent: int = RECENT_TURNS
    #: Every turn of the session, oldest first. Unbounded on purpose.
    turns: list[Turn] = field(default_factory=list)

    # -- the record -------------------------------------------------------

    def add(self, task: str, report: str = "") -> Turn:
        """Record a completed turn and return it."""
        turn = Turn(task=task, report=report)
        self.turns.append(turn)
        return turn

    def clear(self) -> None:
        """Drop the record. ``/new`` calls this."""
        self.turns.clear()

    @property
    def size(self) -> int:
        """Characters the full record occupies, for reporting."""
        return sum(len(t.task) + len(t.report) for t in self.turns)

    # -- the prompt -------------------------------------------------------

    def build(self) -> tuple[list[dict[str, str]], Plan]:
        """Build the prompt and describe it, in one pass.

        Returns the message list for ``prior_turns`` alongside a :class:`Plan`.
        The two are produced together so they cannot disagree; ``/context``
        reports the plan rather than predicting it from the policy.

        Turns are grouped into *blocks* first -- one block per condensed group
        and one per verbatim turn -- so trimming the prompt to fit the budget
        sheds whole blocks. Trimming a flat message list instead can land
        between a turn's ``user`` and ``assistant`` halves, producing a
        transcript that opens with the model apparently answering itself.
        """
        if not self.turns:
            return [], Plan(states=())

        split = len(self.turns) - self.recent if self.recent > 0 else len(self.turns)
        split = max(0, min(split, len(self.turns)))
        older, keep = self.turns[:split], self.turns[split:]

        blocks: list[tuple[list[dict[str, str]], list[str]]] = []

        if older:
            # The digest describes the most recent slice of `older` and counts
            # the rest, so `raw` is measured against the same slice. Comparing
            # against all of `older` would credit the digest with savings it
            # never made.
            described = older[-DIGEST_MAX_TURNS:]
            raw = self._cost([m for t in described for m in t.as_messages()])
            digest = [self._digest_message(older)]
            # Two independent reasons to skip the digest:
            #
            # * It does not pay for itself. Its fixed overhead -- the header,
            #   plus "asked:/did:" per turn -- exceeds the content of a short
            #   turn, so an unconditional digest would make every early
            #   conversation strictly worse than sending it whole.
            # * It does not fit. Left uncapped the digest grows with the
            #   session and can end up larger than the budget it exists to fit
            #   inside, at which point the newest turn -- the one being
            #   continued -- is the thing that gets pushed out.
            #
            # In both cases the turns go through as verbatim blocks instead, and
            # the budget is what decides how many survive. That is strictly
            # better than a digest nobody can afford.
            if self._cost(digest) < min(raw * WORTHWHILE, self.budget * DIGEST_FRACTION):
                blocks.append((digest, ["condensed"] * len(older)))
            else:
                for turn in older:
                    blocks.append((turn.as_messages(), ["verbatim"]))

        for turn in keep:
            blocks.append((turn.as_messages(), ["verbatim"]))

        return self._fit(blocks)

    def prompt(self) -> list[dict[str, str]]:
        """The bounded message list to hand jaz as ``prior_turns``.

        Verbatim for the recent turns, condensed for the rest where that is
        worth it, and trimmed whole-turns at the oldest end if the budget is
        still exceeded. The newest turn is never condensed or dropped: it is
        the one the model is being asked to continue, and a summary of it would
        be a summary of the wrong thing.
        """
        return self.build()[0]

    def describe(self) -> Plan:
        """What :meth:`prompt` will do. See :meth:`build`."""
        return self.build()[1]

    def _fit(
        self, blocks: list[tuple[list[dict[str, str]], list[str]]]
    ) -> tuple[list[dict[str, str]], Plan]:
        """Drop whole blocks until the budget is met, keeping the newest turns.

        Two rules, and the second is the subtle one.

        * Blocks come off from the oldest end, so what survives is always a
          contiguous run of the most recent turns. Nothing in the middle is
          discarded while older history is kept -- that would produce a
          transcript that starts mid-story with no indication of what came
          before.

        * When the budget still cannot be met, the *recent* turns go, not the
          digest. Keeping one condensed line for thirty old turns while throwing
          away the turn the user just finished is backwards: the recent turns
          are the ones a follow-up refers to. So the digest is dropped first,
          then the oldest recent turns, and only the newest turn is guaranteed.

        The last block is always kept whatever it costs. A single turn can be an
        enormous paste that no budget could hold, and truncating the current
        task is worse than sending a long prompt: going over is visible and
        recoverable, silently shortening the history is neither.
        """
        if not blocks:
            return [], Plan(states=())

        # Candidate suffixes, longest first: keep everything, then drop the
        # oldest block, then the next. What survives is always a contiguous run
        # of the most recent turns -- nothing in the middle is discarded while
        # older history is kept, which would produce a transcript that starts
        # mid-story with no indication of what came before.
        #
        # The first block is the digest, so it goes first. Keeping one condensed
        # line for thirty old turns while throwing away the turn the user just
        # finished is backwards: the recent turns are the ones a follow-up
        # refers to.
        for start in range(len(blocks)):
            kept = set(range(start, len(blocks)))
            messages = [m for i in sorted(kept) for m in blocks[i][0]]
            cost = self._cost(messages)
            if cost <= self.budget or start == len(blocks) - 1:
                marked: list[str] = []
                for i, (_, group) in enumerate(blocks):
                    marked.extend(group if i in kept else ["dropped"] * len(group))
                return messages, Plan(states=tuple(marked), cost=cost)

        # Unreachable: the loop returns on its final iteration, where ``start``
        # leaves only the newest block and the guarantee above takes over.
        return [], Plan(states=())

    def _digest_message(self, older: list[Turn]) -> dict[str, str]:
        """One message standing in for every condensed turn.

        A single message rather than one block per turn, so the model reads
        "these happened, then this happened next" instead of wading through
        uniformly-detailed entries whose entire purpose is that the detail is
        gone.

        Only the most recent :data:`DIGEST_MAX_TURNS` are described. Anything
        older is counted instead of summarised: there is a per-turn floor below
        which a condensed turn stops being readable, so past a certain point the
        only way to fit the budget is to stop describing turns one by one. The
        count is not a compromise -- by then *that* the work happened is the fact
        worth carrying, and the detail has long since stopped being actionable.
        """
        described, elided = older[-DIGEST_MAX_TURNS:], older[:-DIGEST_MAX_TURNS]
        total = len(described)

        header = f"[{len(older)} earlier turn(s), condensed; oldest first, detail fades with age]"
        lines = [header]
        if elided:
            lines.append(
                f"- ({len(elided)} turn(s) before that, summarised to a count; "
                "work was done but its detail is no longer recoverable)"
            )
        lines.extend(turn.condensed(_grade(i, total)) for i, turn in enumerate(described))
        return {"role": "user", "content": f"{DIGEST_HEADER}\n\n" + "\n".join(lines)}

    @staticmethod
    def _cost(messages: list[dict[str, str]]) -> int:
        return sum(len(m["content"]) for m in messages)


def report(ctx: Context) -> str:
    """Human-readable snapshot of the context, for the ``/context`` command.

    Contrasts the whole record against what the next turn will send, which is
    the question a user actually has: *what does it still remember?* The two
    numbers differ, sometimes by a lot, and that gap is the feature.
    """
    if not ctx.turns:
        return "Context is empty - this is a fresh session."

    messages, plan = ctx.build()
    cost = ctx._cost(messages)

    lines = [
        f"Context: {len(ctx.turns)} turn(s) recorded, {plan.summary}.",
        "",
        f"  record          {ctx.size:>8,} chars   full history, always kept",
        f"  next prompt     {cost:>8,} chars   what gets sent to the model",
        f"  budget          {ctx.budget:>8,} chars",
        f"  verbatim window {min(len(ctx.turns), ctx.recent):>8,} turns",
    ]

    if plan.dropped:
        lines.append(f"  ! {plan.dropped} turn(s) dropped - over budget, not sent at all")
    if cost > ctx.budget:
        lines.append("  -> over budget; the newest turn is kept even so")
    elif ctx.size > cost:
        lines.append(f"  -> {ctx.size - cost:,} chars condensed away")
    else:
        lines.append("  -> nothing condensed yet; the history is sent whole")

    lines += ["", "Turns, oldest first:"]
    for i, turn in enumerate(ctx.turns, 1):
        lines.append(f"  {i:>3}. [{plan.state_of(i):>9}] {_one_line(turn.task)}")

    return "\n".join(lines)


def _one_line(text: str) -> str:
    """First non-blank line of *text*, truncated to fit a table cell."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:60]
    return "(empty)"