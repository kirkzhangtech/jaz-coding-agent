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
#: These have to be small. A digest's entire value is that it is smaller than
#: what it replaces: at 400/200 a 400-character report is copied essentially
#: whole, the digest comes out *larger* than the original, and
#: :data:`WORTHWHILE` then correctly refuses to condense anything at all. The
#: numbers are therefore sized so that a typical report loses its middle, and
#: the fixed cost per turn ("- asked:" / "  did:") stays a small fraction of
#: what a real report occupies.
REPORT_HEAD_CHARS = 240
REPORT_TAIL_CHARS = 120

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

#: Characters reserved for the digest's two header lines while deciding how many
#: turns fit. The header names the count, which is not known until the count is,
#: so a fixed allowance stands in for it rather than iterating.
_HEADER_SLACK = 80


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

    def condensed(self) -> str:
        """A short stand-in for this turn.

        Rendered as prose rather than as data on purpose. The prompt is a
        transcript the model reads as a transcript; a Python ``repr`` in the
        middle of it -- which is exactly what jaz does to ``prior_turns`` --
        wastes characters and is harder to read than the same facts in English.
        """
        task = _clip(self.task, TASK_HEAD_CHARS)
        if not self.report:
            return f"- asked: {task} (no report; the turn did not finish)"
        body = _clip(self.report, REPORT_HEAD_CHARS, REPORT_TAIL_CHARS)
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
    """

    #: Characters of context to send, across the digest and the verbatim turns.
    budget: int = 6000
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
            raw = self._cost([m for t in older for m in t.as_messages()])
            digest = [self._digest_message(older)]
            # Condense only when it actually saves something. The fixed overhead
            # of a digest -- its header, plus "asked:/did:" per turn -- exceeds
            # the content of a short turn, so an unconditional digest would make
            # every early conversation strictly worse than sending it whole.
            if self._cost(digest) < raw * WORTHWHILE:
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
        """Drop whole blocks from the oldest end until the budget is met.

        The last block is always kept, whatever it costs. That block is the
        newest turn -- the one the model is being asked to continue -- and a
        single turn can be an enormous paste that no budget could hold. Sending
        it long is recoverable and visible; silently dropping or truncating the
        current task is neither, so the budget yields instead.
        """
        states = [state for _, group in blocks for state in group]
        offsets = [0]
        for _, group in blocks:
            offsets.append(offsets[-1] + len(group))

        # Candidate tail lengths, from "everything" down to "the last block".
        for kept in range(len(blocks), 0, -1):
            messages = [m for block, _ in blocks[len(blocks) - kept :] for m in block]
            if kept == 1 or self._cost(messages) <= self.budget:
                dropped = offsets[len(blocks) - kept]
                plan = Plan(
                    states=tuple(["dropped"] * dropped + states[dropped:]),
                    cost=self._cost(messages),
                )
                return messages, plan

        # Unreachable: the loop always returns on its final iteration, where
        # ``kept == 1``. Expressed so a future edit cannot fall off the end.
        return [], Plan(states=())

    def _digest_message(self, older: list[Turn]) -> dict[str, str]:
        """One message standing in for every condensed turn.

        A single message rather than one block per turn, so the model reads
        "these happened, then this happened next" instead of wading through
        uniformly-detailed entries whose entire purpose is that the detail is
        gone.
        """
        lines = [f"[{len(older)} earlier turn(s), condensed]"]
        lines.extend(turn.condensed() for turn in older)
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