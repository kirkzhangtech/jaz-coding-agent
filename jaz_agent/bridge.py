"""Bridges jaz's synchronous hook events onto an async UI.

The problem this solves: jaz drives the agent with plain blocking calls, and its
hooks fire on the agent's own thread. Textual owns a single event loop and its
widgets are not thread-safe. Pushing events straight at a widget from the agent
thread is a data race.

The fix is deliberately dumb: the hook only ever does a non-blocking
``put_nowait`` onto a ``queue.SimpleQueue``. A drain coroutine on the Textual
thread empties the queue and touches widgets. No locks, no ``call_from_thread``,
no ordering hazards -- the queue does the handoff.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from queue import SimpleQueue
from typing import Any, Callable

from jaz.hooks import Hook


class Kind(str, Enum):
    """What kind of thing happened, so the UI can style it without sniffing text."""

    STATUS = "status"
    THINKING = "thinking"
    CODE = "code"
    OUTPUT = "output"
    ERROR = "error"
    RESULT = "result"
    COST = "cost"
    RETRY = "retry"


@dataclass(slots=True)
class Event:
    """One thing the agent did, on its way to the UI."""

    kind: Kind
    text: str
    iteration: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


def _fmt_usage(response: Any) -> str:
    """Pull token counts off an LLMResponse without assuming its exact shape.

    jaz returns its own response type; attribute names differ across versions,
    so every read is guarded.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return ""
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    total = getattr(usage, "total_tokens", None)
    parts = []
    if prompt is not None:
        parts.append(f"{prompt:,} in")
    if completion is not None:
        parts.append(f"{completion:,} out")
    if not parts and total is not None:
        parts.append(f"{total:,} tokens")
    return ", ".join(parts)


def _fmt_cost(response: Any) -> str | None:
    """Cost in USD if jaz priced the call; ``None`` for unpriced models."""
    cost = getattr(response, "cost_usd", None)
    if cost is None:
        cost = getattr(response, "cost", None)
    if cost is None:
        return None
    try:
        return f"${float(cost):.4f}"
    except (TypeError, ValueError):
        return None


class EventBridge(Hook):
    """Turns the agent's lifecycle into :class:`Event` objects on a queue.

    Subclassing ``Hook`` and overriding the typed ``on_*`` handlers means jaz
    dispatches by method name -- no ``isinstance`` chain to keep in sync.
    """

    def __init__(self, queue: SimpleQueue, *, verbose: bool = False) -> None:
        """Bind the bridge to *queue*; ``verbose`` also surfaces REPL output."""
        super().__init__()
        self.queue = queue
        self.verbose = verbose
        self.started = time.monotonic()
        self.total_cost = 0.0
        self.iterations = 0
        self._tokens_in = 0
        self._tokens_out = 0

    # -- helpers ---------------------------------------------------------

    def emit(self, kind: Kind, text: str, iteration: int | None = None, **payload: Any) -> None:
        """Post one event. Never blocks, never raises into the agent.

        The keyword arguments land in ``Event.payload``. The field is named
        ``payload`` rather than ``data`` on purpose: a caller writing
        ``emit(kind, text, data={...})`` would otherwise have its dictionary
        captured by that keyword and nested one level deeper, so
        ``event.data["idle"]`` would silently read ``None``.
        """
        try:
            self.queue.put_nowait(Event(kind, text, iteration, payload))
        except Exception:  # pragma: no cover - the UI should never kill a run
            pass

    def elapsed(self) -> str:
        """Wall-clock duration since the bridge was created, as ``M:SS``."""
        secs = int(time.monotonic() - self.started)
        return f"{secs // 60}:{secs % 60:02d}"

    # -- invoke lifecycle ------------------------------------------------

    def on_invoke_enter(self, event: Any) -> list:
        """Note nested sub-agent invocations so the UI can show recursion depth."""
        depth = 1 if event.parent_invoke_id else 0
        if depth:
            self.emit(Kind.STATUS, f"sub-agent (depth {depth})", depth=depth)
        return []

    def on_invoke_exit(self, event: Any) -> list:
        """Nothing to clean up: the UI derives completion from the outcome widget."""
        return []

    # -- LLM turns -------------------------------------------------------

    def on_llm_query_enter(self, event: Any) -> list:
        """Announce a model round-trip is starting."""
        self.iterations = max(self.iterations, event.iteration + 1)
        self.emit(
            Kind.THINKING,
            f"thinking (turn {event.iteration + 1})",
            iteration=event.iteration,
            model=event.model,
        )
        return []

    def on_llm_query_complete(self, event: Any) -> list:
        """Report tokens and cost for the turn that just finished."""
        response = event.response
        usage = _fmt_usage(response)
        cost = _fmt_cost(response)
        if cost:
            try:
                self.total_cost += float(cost.lstrip("$"))
            except ValueError:
                pass
        else:
            cost = None

        parts = [p for p in (usage, cost) if p]
        detail = f"  ({'  '.join(parts)})" if parts else ""
        self.emit(Kind.COST, f"turn {event.iteration + 1} done{detail}", iteration=event.iteration)
        if cost:
            self.emit(Kind.COST, f"total {cost}", total_cost=cost)
        return []

    def on_llm_query_retry(self, event: Any) -> list:
        """Surface retries loudly -- silent retries look like a hang."""
        self.emit(
            Kind.RETRY,
            f"retry {event.attempt_number} after {event.wait_seconds:.1f}s: {event.exception}",
            iteration=getattr(event, "iteration", None),
        )
        return []

    # -- REPL turns ------------------------------------------------------

    def on_repl_exec_enter(self, event: Any) -> list:
        """Show the Python the model just wrote, before it runs."""
        self.emit(Kind.CODE, event.code, iteration=event.iteration)
        return []

    def on_repl_exec_complete(self, event: Any) -> list:
        """Render whatever the REPL produced, trimmed for readability."""
        result = event.exec_result
        text = _result_text(result)
        if not text and not self.verbose:
            return []
        self.emit(Kind.OUTPUT, text or "(no output)", iteration=event.iteration)
        return []


def _result_text(result: Any) -> str:
    """Best-effort extraction of readable text from an ``ExecResult``.

    ``ExecResult`` is a union: a terminal ``return``/``raise`` carries a value,
    a ``continue`` carries nothing, and either may carry printed output. Guard
    every attribute because the union variants differ.
    """
    chunks: list[str] = []
    for attr in ("output", "stdout", "printed", "value", "result", "error"):
        value = getattr(result, attr, None)
        if value is None:
            continue
        text = value if isinstance(value, str) else repr(value)
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()


def make_bridge(queue: SimpleQueue, *, verbose: bool = False) -> EventBridge:
    """Factory so callers do not import the Hook base directly."""
    return EventBridge(queue, verbose=verbose)


class QueueDrain:
    """Callable that empties the bridge queue and hands events to *sink*.

    Kept separate from :class:`EventBridge` so the queue's writer (the agent
    thread) and reader (the UI thread) share no state beyond the queue itself.
    """

    def __init__(self, queue: SimpleQueue, sink: Callable[[Event], None], limit: int = 200) -> None:
        self.queue = queue
        self.sink = sink
        self.limit = limit

    def __call__(self) -> list[Event]:
        """Pop up to ``limit`` events, newest-kept, and forward each to the sink."""
        drained: list[Event] = []
        for _ in range(self.limit):
            try:
                ev = self.queue.get_nowait()
            except Exception:
                break
            drained.append(ev)
            try:
                self.sink(ev)
            except Exception:  # a rendering bug must not abort the drain loop
                continue
        return drained