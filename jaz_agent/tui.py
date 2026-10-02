"""The Textual UI: a transcript pane, a live status bar, and a prompt box.

Design notes worth stating, because they are the non-obvious parts:

* The agent runs on a worker thread, so *nothing* in ``on_event`` may block.
  Every incoming event is appended and the widget is refreshed on a timer
  instead, which keeps the render path single-threaded by construction.
* ``log`` is the transcript. It is append-only and capped; a long agent run
  would otherwise grow without bound and eventually stall the UI.
* Cost and turn counters live in the footer, not the transcript, so the log
  stays a readable record of what happened rather than a dashboard.
"""

from __future__ import annotations

import threading
import time
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Callable

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.suggester import Suggester
from textual.widgets import Footer, Header, Input, RichLog, Static

from .bridge import Event, Kind, QueueDrain
from .commands import (
    COMMANDS,
    RETIRED,
    Candidate,
    candidates,
    complete,
    find,
    help_text,
    hint_text,
    picker_rows,
)
from .context import report as context_report
from .llm_config import available_backends, describe_model, list_models
from .session import AgentSession
from .tools import ROOT, tool_catalog


class PromptInput(Input):
    """The prompt box, with Tab bound to completion and ↑/↓ to the hint list.

    Textual 8's ``Input`` has **no** Tab binding: a suggestion is accepted with
    the right-arrow key, which is right for a single-line text field and wrong
    for a command prompt. ``Screen`` claims Tab for ``app.focus_next``, so
    before this existed Tab moved focus off the prompt and completed nothing.

    The arrow keys are bound here too, but only to forward to the app: the hint
    list is the app's state, and ``Input`` is the only widget that sees the key
    press first. The app turns the message into a move, or -- when no list is
    open -- back into cursor movement, so ordinary task text still gets working
    arrow keys rather than nothing.
    """

    BINDINGS = [
        *Input.BINDINGS,
        Binding("tab", "accept_completion", "Accept completion", show=False),
        Binding("up", "hint_up", "Previous suggestion", show=False),
        Binding("down", "hint_down", "Next suggestion", show=False),
        Binding("escape", "dismiss_hint", "Dismiss hint", show=False),
    ]

    #: Sent to the app, which owns the list and the selection index.
    class Navigate(Message):
        """↑ or ↓ was pressed in the prompt."""

        def __init__(self, direction: int) -> None:
            super().__init__()
            self.direction = direction

    class Dismiss(Message):
        """Escape was pressed in the prompt."""

    def action_accept_completion(self) -> None:
        """Apply the pending suggestion, or do nothing.

        Deliberately not ``cursor_right``: that action moves the cursor when
        there is no suggestion, so Tab on ordinary task text would nudge the
        caret one character along for no visible reason. Tab means "complete",
        and when there is nothing to complete it should be inert.
        """
        suggestion = self._suggestion
        if suggestion and len(suggestion) > len(self.value):
            self.value = suggestion
            self.cursor_position = len(self.value)

    def action_hint_up(self) -> None:
        self.post_message(self.Navigate(-1))

    def action_hint_down(self) -> None:
        self.post_message(self.Navigate(1))

    def action_dismiss_hint(self) -> None:
        """Close the hint without touching what was typed.

        The list is a convenience, not a modal: Escape has to leave the prompt
        exactly as it was, because the user may simply want the space back after
        a stray ``/``. Without this the only way to dismiss it was to keep
        typing or submit something.
        """
        self.post_message(self.Dismiss())


class CommandSuggester(Suggester):
    """Tab completion for the prompt box.

    Two levels, because the useful completion changes with what has been typed:

    * ``/mo`` → ``model``
    * ``/switchmodules openai/gpt-5-mini`` → completes to the full id

    The model half needs a catalogue, which is a network call. It is fetched in
    the background by the app and seeded through :meth:`load_models`; nothing
    here ever blocks on the network, because a completion that freezes the UI
    is worse than no completion at all.

    This class lives in ``tui`` rather than its own module because it subclasses
    a Textual widget -- which is exactly the layering rule the architecture
    tests enforce: only this module may import Textual.
    """

    def __init__(self, case_sensitive: bool = False) -> None:
        super().__init__(case_sensitive=case_sensitive)
        self._models: list[str] = []
        self._loaded = False

    def load_models(self, models: list[str]) -> None:
        """Seed the cache from outside. Called once a background fetch lands."""
        self._models = list(models)
        self._loaded = True

    def invalidate(self) -> None:
        """Drop the cache. Called after a backend switch, since the previous
        backend's model ids mean nothing against the new one."""
        self._loaded = False
        self._models = []

    @property
    def has_models(self) -> bool:
        """Whether a catalogue is cached -- lets the app skip the warm-up."""
        return self._loaded and bool(self._models)

    async def get_suggestion(self, value: str) -> str | None:
        """Completion for *value*, or ``None`` for none.

        Non-slash input returns ``None`` so Tab stays out of the way while the
        user writes an ordinary task.
        """
        if not value.startswith("/"):
            return None

        head, sep, tail = value.partition(" ")
        prefix = head.lstrip("/")

        if not sep:
            return self._complete_command(value, prefix)

        # Only a command that declares a model argument gets model completion.
        command = find(prefix)
        if command is None or command.target != "model":
            return None
        return self._complete_model(value, tail)

    def _complete_command(self, value: str, prefix: str) -> str | None:
        """Complete the command name, or the space that starts its argument."""
        lowered = prefix.lower()

        exact = [c for c in complete(lowered) if c.name.lower() == lowered]
        if exact:
            # The name is already complete: offer the space that begins the
            # argument, so Tab advances instead of doing nothing.
            return f"{value} " if exact[0].usage else None

        matches = complete(lowered)
        return f"/{matches[0].name}" if matches else None

    def _complete_model(self, value: str, fragment: str) -> str | None:
        """Complete a model id from the cached catalogue."""
        if not fragment or not self._models:
            return None

        needle = fragment.lower()
        hits = [m for m in self._models if m.lower().startswith(needle)]
        if not hits:
            # A partial middle fragment still completes: "gpt-5" finds
            # "openai/gpt-5-mini".
            hits = [m for m in self._models if needle in m.lower()]
        if not hits:
            return None

        head = value[: len(value) - len(fragment)]
        return f"{head}{hits[0]}"

CSS = """
Screen {
    layout: vertical;
}

/* The transcript gets the room; the prompt is fixed at the bottom. */
#transcript {
    height: 1fr;
    border: round $accent 40%;
    padding: 0 1;
    background: $surface;
}

#prompt {
    dock: bottom;
    height: auto;
    border: round $accent 40%;
}

#status {
    height: auto;
    padding: 0 1;
    color: $text-muted;
    background: $panel;
}

/* The picker: a fixed-height scrolling viewport under the prompt.

   `height: auto` was wrong for anything long. The command list is at most ten
   rows, but the model browser is OpenRouter's 464, and a list that grows to
   464 lines pushes the transcript off the screen entirely -- there was no
   way to see the agent while browsing.

   A fixed viewport with `overflow-y: auto` keeps the layout stable and lets
   the list scroll. Only the visible window is rendered, so moving through
   hundreds of rows costs the same as moving through ten. */
#hint {
    height: 20;
    overflow-y: auto;
    padding: 0 1;
    color: $text-accent;
    display: none;
}

#banner {
    height: auto;
    padding: 0 1;
    color: $text-accent;
}

.busy { color: $warning; }
.error { color: $error; }
.ok { color: $success; }
"""

#: Plain-text prefixes, one per event kind. The transcript runs with rich markup
#: disabled, so these are literal characters rather than style tags -- the
#: variety of glyphs is what carries the distinction on screen.
MARKERS = {
    Kind.STATUS: "·",
    Kind.THINKING: "»",
    Kind.CODE: "│",
    Kind.OUTPUT: " ",
    Kind.ERROR: "✗",
    Kind.RESULT: "✓",
    Kind.COST: " ",
    Kind.RETRY: "↻",
}


def _subsequence(needle: str, haystack: str) -> bool:
    """True if every char of *needle* appears in *haystack*, in order.

    Lets ``/switchmodules gpt5`` match ``openai/gpt-5-mini``: the characters are
    all there but the hyphen is missing. Plain ``in`` would miss that, and a
    fuzzy score would also match unrelated models.
    """
    it = iter(haystack)
    return all(char in it for char in needle)


def _row_offsets(rows: list[Candidate]) -> list[int]:
    """The display line at which each row starts.

    Rows vary in height -- a command whose summary wraps occupies two lines --
    so a scrolling window has to be measured in lines while the selection is
    tracked in rows. Measuring the window in rows instead would scroll by the
    wrong amount and drift further off the longer the list gets.
    """
    offsets: list[int] = []
    line = 0
    for row in rows:
        offsets.append(line)
        line += row.height
    return offsets


class Transcript(RichLog):
    """Append-only, size-capped view of everything the agent did.

    ``RichLog`` takes ``max_lines`` and drops the oldest lines once it is
    exceeded, so the cap is enforced by the widget rather than by reimplementing
    a ring here. Markup and highlighting are both off at construction: agent
    output -- file contents, tracebacks, shell output -- routinely contains
    square brackets that rich would otherwise try to parse as tags.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(max_lines=2000, markup=False, highlight=False, wrap=True, **kwargs)

    def append_event(self, event: Event) -> None:
        """Render one :class:`Event`, prefixed by a marker that encodes its kind.

        The marker is a plain ASCII/plain-text prefix rather than rich markup:
        this widget runs with ``markup=False`` precisely so untrusted agent output
        cannot inject styling or fake tags.
        """
        marker = MARKERS.get(event.kind, " ")
        for line in str(event.text).splitlines() or [""]:
            self.write(f"{marker} {line}")
        self.write("")

    def dump(self) -> list[str]:
        """Best-effort read of the rendered lines, for tests and debugging.

        RichLog does not expose its line buffer publicly, so this walks the
        renderable chunks. Returns ``[]`` rather than raising when the internal
        shape changes -- it is a diagnostic helper, never load-bearing.
        """
        out: list[str] = []
        for attr in ("lines", "_lines", "_line_cache"):
            buf = getattr(self, attr, None)
            if buf is None:
                continue
            try:
                for item in buf:
                    text = getattr(item, "text", item)
                    out.append(text.plain if hasattr(text, "plain") else str(text))
                return out
            except Exception:
                continue
        return out


class StatusBar(Static):
    """Single-line footer: state, model, turns, cost."""

    DEFAULT_CSS = ""

    state: reactive[str] = reactive("ready")
    detail: reactive[str] = reactive("")
    turns: reactive[int] = reactive(0)
    cost: reactive[str] = reactive("")
    elapsed: reactive[str] = reactive("0:00")

    def render(self) -> str:
        """Compose the status line; rich markup only, never user text."""
        colour = "warning" if self.state == "busy" else ("error" if self.state == "error" else "green")
        bits = [f"[{colour}]{self.state}[/]", f"[dim]{self.elapsed}[/]"]
        if self.turns:
            bits.append(f"[dim]turn {self.turns}[/]")
        if self.cost:
            bits.append(f"[dim]{self.cost}[/]")
        if self.detail:
            bits.append(f"[dim]{self.detail}[/]")
        return "  ".join(bits)


class CodingAgentApp(App[None]):
    """A TUI coding agent backed by jaz and OpenRouter's Space Bunny Alpha."""

    CSS = CSS
    # ``/new`` is the way to drop a session. It was also F2, which made two
    # ways to reach one action: one listed in ``/help`` and the inline hint, one
    # only in the footer. The command is the discoverable path, so the key is
    # gone rather than kept as a shortcut nobody finds. ``/cancel`` and
    # ``/clear`` are bound for the same reason -- a key you have to remember
    # while mid-turn is worse than typing two characters.
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+c", "cancel", "Cancel"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("f5", "cost", "Cost"),
    ]

    def __init__(self, session: AgentSession | None = None, *, prompt: str | None = None) -> None:
        """Build the app around *session*, optionally auto-submitting *prompt*."""
        super().__init__()
        self.session = session or AgentSession()
        self.banner: Static | None = None
        self.transcript: Transcript | None = None
        self.prompt_box: Input | None = None
        self.status: StatusBar | None = None
        self.hint: Static | None = None
        self.suggester = CommandSuggester()
        self.drain = QueueDrain(self.session.queue, self._on_agent_event)
        self.started = time.monotonic()
        self._pending_prompt = prompt
        self._busy_since: float | None = None
        #: Rows the ↑/↓ keys can move over, and which one is selected.
        self._candidates: list[Candidate] = []
        self._selected = 0
        #: Set by Escape; cleared on the next edit, so the list is dismissible
        #: for the current value rather than for the rest of the session.
        self._hint_dismissed = False
        #: What Enter does with the highlighted row: ``None`` for the slash
        #: command list (the row is a command name), ``"model"`` for the model
        #: browser (the row is a model id to switch to). One list, two owners --
        #: the alternative is a second almost-identical widget and navigation
        #: path, which is exactly the duplication that let the command list and
        #: the browser drift apart before.
        self._picker: str | None = None
        #: Cached line offsets for the scrolling window; see :meth:`_offsets`.
        self._offset_cache: list[int] | None = None
        self._offset_source: list[Candidate] | None = None

    # -- layout ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Widget tree: header, transcript, status, prompt, footer."""
        yield Header(show_clock=True)
        self.banner = Static(id="banner")
        yield self.banner
        self.transcript = Transcript(id="transcript")
        yield self.transcript
        self.status = StatusBar(id="status")
        yield self.status
        with Horizontal(id="prompt"):
            self.prompt_box = PromptInput(
                placeholder="Describe a task, or / for commands",
                id="task",
                suggester=self.suggester,
            )
            yield self.prompt_box
        self.hint = Static(id="hint")
        yield self.hint
        yield Footer()

    #: Textual 8 dropped the per-widget ``autofocus`` flag; initial focus is
    #: requested by naming a widget id here.
    AUTO_FOCUS = "#task"

    def on_mount(self) -> None:
        """Start the drain timer, render the banner and print the greeting."""
        assert self.transcript and self.status and self.prompt_box
        self._refresh_banner()
        self._say(
            "Ready. Describe a coding task and press Enter.\n"
            "Ctrl+C cancel · Ctrl+L clear · Ctrl+Q quit · type / for commands",
            Kind.STATUS,
        )
        self.set_interval(0.05, self._tick)
        # Prime the completion cache in the background: Tab should offer model
        # ids from the first keystroke, and the fetch is a network call.
        threading.Thread(
            target=self._warm_suggester, name="jaz-models-warm", daemon=True
        ).start()
        if self._pending_prompt:
            text, self._pending_prompt = self._pending_prompt, None
            self.call_after_refresh(self._send, text)

    def _refresh_banner(self) -> None:
        """Redraw the header line.

        Called on mount and after every successful switch, so the
        banner never claims a model the session is no longer using.
        """
        if self.banner is None:
            return
        self.banner.update(
            f"[bold]jaz coding agent[/]  [dim]{self.session.model_name}[/]\n"
            f"[dim]workspace: {ROOT}[/]\n"
            f"[dim]tools: {tool_catalog()}[/]"
        )

    # -- transcript helpers ---------------------------------------------

    def _say(self, text: str, kind: Kind = Kind.STATUS) -> None:
        """Write to the transcript if it exists yet."""
        if self.transcript:
            self.transcript.append_event(Event(kind, text))

    # -- the pump --------------------------------------------------------

    def _tick(self) -> None:
        """Timer callback: drain agent events and refresh the status bar.

        Everything here runs on the Textual thread, which is the only thread
        allowed to touch widgets.
        """
        events = self.drain()
        if not events:
            # Nothing new: still advance the clock so a long turn looks alive.
            self._refresh_elapsed()
            return

        assert self.status
        redraw = False
        for ev in events:
            if ev.kind is Kind.COST and "total_cost" in ev.payload:
                self.status.cost = ev.payload["total_cost"]
            # A committed model switch changes the header; the worker thread
            # cannot redraw it, so it flags the event and the UI acts here.
            if ev.payload.get("redraw_banner"):
                redraw = True
            # The idle marker is the last thing a turn emits, so it is the
            # authoritative end-of-turn signal. Inferring the end from "no more
            # events" is not workable: a slow model looks identical to a finished
            # one.
            if ev.payload.get("idle") or ev.payload.get("fatal"):
                self._end_turn()

        if redraw:
            self._refresh_banner()
        self.status.turns = self.session.bridge.iterations
        if self.session.busy:
            self._mark_busy()
        self._refresh_elapsed()

    def _mark_busy(self) -> None:
        """Flip the status bar into its working state."""
        if not self.status:
            return
        if self.status.state != "busy":
            self._busy_since = time.monotonic()
        self.status.state = "busy"
        self.status.detail = "agent working"
        assert self.prompt_box
        self.prompt_box.disabled = True

    def _end_turn(self) -> None:
        """Return the UI to idle after a turn finishes."""
        assert self.status and self.prompt_box
        self.status.state = "ready"
        self.status.detail = ""
        self.prompt_box.disabled = False
        self.prompt_box.focus()
        self._busy_since = None
        self._refresh_elapsed()

    def _refresh_elapsed(self) -> None:
        """Update the clock; during a run it shows the turn's duration."""
        assert self.status
        base = self._busy_since if self._busy_since is not None else self.started
        secs = int(time.monotonic() - base)
        self.status.elapsed = f"{secs // 60}:{secs % 60:02d}"

    # -- the agent-event sink -------------------------------------------

    def _on_agent_event(self, event: Event) -> None:
        """Sink for :class:`QueueDrain`; runs on the UI thread, must not block."""
        if event.kind is Kind.ERROR:
            self._say(event.text, Kind.ERROR)
            if event.payload.get("fatal"):
                assert self.status
                self.status.state = "error"
                self._end_turn()
        elif event.kind is Kind.RESULT:
            self._say(event.text, Kind.RESULT)
        elif event.kind is Kind.RETRY:
            self._say(event.text, Kind.RETRY)
        else:
            self._say(event.text, event.kind)

        # A model list arrives as ids in the payload rather than as text, so the
        # worker stays off the widgets and the rows become a navigable list here
        # on the Textual thread.
        models = event.payload.get("models")
        if models:
            self._open_model_picker(list(models))

    # -- the inline hint ---------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update the hint line as the user types a slash command.

        The hint is shown only for slash input. For ordinary task text it is
        hidden, which is why the CSS gives ``#hint`` ``display: none`` by
        default -- showing an empty line would push the prompt around while
        someone is typing a long task.
        """
        if self.hint is None:
            return

        # Any edit re-opens a list that Escape closed: the dismissal was about
        # the value on screen, and the value is no longer the one dismissed.
        self._hint_dismissed = False

        value = event.value

        # The model browser owns the list, not the prompt. Typing while it is
        # open means the user has decided to do something else, so the browser
        # gets out of the way -- an empty value is not that: submitting the
        # model list leaves the box empty, and a programmatic clear would
        # otherwise be indistinguishable from the user starting a new task.
        if self._picker == "model" and not value:
            return

        if not value.startswith("/"):
            self._clear_candidates()
            self.hint.display = False
            return

        head, sep, _ = value.partition(" ")
        prefix = head.lstrip("/")

        if sep:
            # After the command name: hint about the argument, if it takes one.
            # Nothing to navigate -- the argument is free text, or a model id
            # with its own completion path.
            self._clear_candidates()
            command = find(prefix)
            if command is None:
                self._set_hint(f"unknown command /{prefix}", error=True)
                return
            if not command.usage:
                self._set_hint(f"/{command.name} takes no arguments — {command.summary}")
                return
            self._set_hint(f"{command.signature} — {command.summary}")
            return

        retired = self._retired_hint(prefix)
        if retired:
            self._clear_candidates()
            self._set_hint(retired, error=True)
            return

        rows = candidates(prefix, self._hint_width())
        self._candidates = rows
        # A fresh list starts at the top. Keeping the old index would point at
        # an unrelated row as soon as the prefix narrowed the list.
        self._selected = 0
        if rows and not self._hint_dismissed:
            self._render_candidates()
        else:
            self._set_hint(hint_text(prefix, self._hint_width()))

        self._set_hint(hint_text(prefix, self._hint_width()))

    def _retired_hint(self, prefix: str) -> str | None:
        """A redirect for a command that used to exist, if *prefix* names one."""
        replacement = RETIRED.get(prefix.lower())
        return f"/{prefix} is gone — use /{replacement}" if replacement else None

    def _hint_width(self) -> int:
        """Usable columns for the hint.

        The screen width is the best available proxy, minus the border the
        prompt draws around this area. ``on_input_changed`` only fires once the
        app is running, so ``screen`` is always mounted here; the guard is for
        a headless render with no terminal geometry rather than a real case.
        """
        size = getattr(self.screen, "size", None)
        if size is None or not size.width:
            return 80
        return max(40, size.width - 2)

    def _set_hint(self, text: str, *, error: bool = False) -> None:
        """Show one line of hint text, styled as an error when appropriate.

        The text is passed to ``Static.update`` as *markup*, so square brackets
        in it are read as Rich tags. That silently mangled every usage string:
        ``/switchmodules [model]`` rendered as ``/switchmodules`` with the
        argument shape gone, and the model completion for that command was
        undiscoverable. The text is escaped first and the style added around
        the result, so the message can contain brackets safely.
        """
        if self.hint is None:
            return
        self.hint.display = True
        body = escape(text)
        self.hint.update(f"[red]{body}[/]" if error else body)

    def _render_candidates(self) -> None:
        """Draw the picker, showing only the lines around the selection.

        Windowed rather than rendering every row. OpenRouter returns 464 ids,
        and writing them all into the widget pushed the transcript off the
        screen -- there was no way to watch the agent while browsing -- and the
        first twenty were the only ones reachable at all.

        The window follows the cursor, so ↓ walks the whole catalogue and the
        list scrolls the way a scrolling list should, with no separate "next
        page" concept to learn. Cost is the same at row 400 as at row 1.

        Rows are never split across the window edge. A half-visible highlighted
        row reads as a rendering bug rather than as a row that continues.
        """
        if self.hint is None or not self._candidates:
            return

        offsets = self._offsets()
        viewport = self._picker_viewport()

        first, last = self._window(offsets, viewport)
        blocks: list[str] = []
        for index in range(first, last + 1):
            candidate = self._candidates[index]
            body = escape("\n".join(candidate.lines)) if candidate.lines else ""
            marker = "▸" if index == self._selected else " "
            style = "[reverse bold]" if index == self._selected else ""
            close = "[/]" if style else ""
            blocks.append(f"{style}{marker}{body}{close}")

        verb = "switch" if self._picker == "model" else "run"
        total = len(self._candidates)
        position = f"[dim] {first + 1}-{last + 1} of {total}[/]"
        self.hint.display = True
        self.hint.update(
            "\n".join(
                [
                    *blocks,
                    position,
                    f"[dim]↑/↓ choose · Enter {verb} · Esc dismiss[/]",
                ]
            )
        )

    def _picker_viewport(self) -> int:
        """How many lines of rows the picker can show.

        Two lines are reserved for the position line and the key hint so they
        stay put while the list scrolls behind them. The floor matters: before
        the first layout pass ``size.height`` can be small or zero, and a
        three-row viewport that then jumps to eighteen as the terminal settles
        would make the list appear to scroll on its own.
        """
        if self.hint is None:
            return 12
        return max(10, self.hint.size.height - 2)

    def _offsets(self) -> list[int]:
        """Display line of each row's start, cached against the row list.

        Recomputing on every keystroke is cheap for ten commands and wasteful
        for 464 models, and the answer only changes when the list does. The
        cache is keyed on the *identity* of the row list rather than its
        length: two different lists can easily be the same length, and reusing
        the wrong offsets would scroll the window to an arbitrary place.
        """
        rows = self._candidates
        if self._offset_cache is None or self._offset_source is not rows:
            self._offset_cache = _row_offsets(rows)
            self._offset_source = rows
        return self._offset_cache

    def _window(self, offsets: list[int], viewport: int) -> tuple[int, int]:
        """The inclusive row range to draw, chosen to contain the selection.

        The selection is aimed about a third of the way down rather than glued
        to an edge, so ↓ shows what is coming instead of only what was passed.

        Binary search rather than a scan: the offsets are sorted, and a linear
        walk over 464 rows on every keystroke is exactly the kind of thing that
        makes a list feel sluggish. ``bisect_right`` finds the row whose start
        is at or after the desired top edge directly.
        """
        sel_line = offsets[self._selected]
        desired = max(0, sel_line - viewport // 3)

        first = bisect_right(offsets, desired) - 1
        # The row before the boundary may still be the one that contains the
        # top edge, since a row can span several lines.
        if first > 0 and offsets[first] > desired:
            first -= 1
        first = max(0, min(first, len(offsets) - 1))

        limit = offsets[first] + viewport
        # The last row is the one starting before the limit; walk back one to
        # include whichever row actually straddles it.
        last = bisect_left(offsets, limit) - 1
        if last < first:
            last = first
        last = min(last, len(offsets) - 1)
        return first, last

    def _open_model_picker(self, models: list[str]) -> None:
        """Turn a fetched model page into the navigable list.

        Starts on the current model rather than row 0. The alternative is worse
        than a preference: ↓ followed by Enter is the natural way to pick "the
        one after this one", and if the list opens on ``aion-labs/aion-2.0``
        while the session is on ``stealth/space-bunny-alpha``, the first ↓ picks
        a model the user was not looking at and Enter switches to it.
        """
        if not models or self.hint is None:
            return
        current = self.session.model
        marks = {m: ("✓" if m == current else " ") for m in models}
        self._candidates = picker_rows(models, marks=marks)
        self._picker = "model"
        self._selected = models.index(current) if current in models else 0
        self._render_candidates()

    def _select_model(self, model: str) -> None:
        """Switch to *model*, on a worker, through the same validated path.

        Routed back through ``/switchmodules <model>`` rather than applied
        directly so that the live probe, the "not switched" wording and the
        catalogue-rejected case are exactly the ones the typed command gets.
        Two paths to the same switch would drift, and the probe is the part that
        must not be skipped.
        """
        self._say(f"switching to {model} …", Kind.THINKING)
        threading.Thread(
            target=self._try_switch_then_filter,
            args=(model,),
            name="jaz-switch-try",
            daemon=True,
        ).start()

    def _clear_candidates(self) -> None:
        """Drop the navigable rows. Called whenever the list cannot apply."""
        self._candidates = []
        self._selected = 0
        self._picker = None
        self._offset_cache = None
        self._offset_source = None

    def _move_selection(self, direction: int) -> bool:
        """Move the highlight, wrapping at both ends. True if it moved.

        Wrapping rather than clamping: the list is short and every row is a
        real command, so there is nothing to gain from a dead end at the top or
        bottom. Clamping would make pressing ↑ at the first row look like a
        dropped keypress.
        """
        if not self._candidates:
            return False
        count = len(self._candidates)
        self._selected = (self._selected + direction) % count
        self._render_candidates()
        return True

    def on_prompt_input_navigate(self, event: PromptInput.Navigate) -> None:
        """Move the hint selection, or fall through to cursor movement.

        The prompt owns the key press but not the list, so the decision lives
        here. When no list is open the key is handed back to the input, because
        swallowing ↑/↓ on ordinary task text would leave no way to move the
        caret back over text that was mistyped.
        """
        if self._move_selection(event.direction):
            event.stop()
            return
        assert self.prompt_box is not None
        if event.direction < 0:
            self.prompt_box.action_cursor_left()
        else:
            self.prompt_box.action_cursor_right()

    def on_prompt_input_dismiss(self, event: PromptInput.Dismiss) -> None:
        """Close the hint, leaving the typed text alone.

        Escape closes it until the value changes again. That is the useful
        behaviour for "I typed ``/`` by accident" -- the list returns on the
        next keystroke, so nothing is permanently lost and nothing has to be
        retyped.
        """
        self._hint_dismissed = True
        self._hide_hint()

    def _hide_hint(self) -> None:
        """Collapse the hint. Called when a command is submitted."""
        self._clear_candidates()
        if self.hint is not None:
            self.hint.display = False

    # -- input handling --------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle a submitted prompt: slash commands, else start a turn.

        A highlighted row wins over whatever was typed, and what it *means*
        depends on which list is open: a command name to run, or a model id to
        switch to. Pressing ↓ and Enter is how every shell completion menu
        works, and treating the highlight as decoration would make it a lie.

        The two lists gate the highlight differently, and both gates are needed:

        * For the command list, only a row the user actually moved to counts.
          The selection sits at row 0 and the list appears the instant a slash
          is typed, so "a row is highlighted" is true from the first keystroke.
          Without the gate, an untouched ``/s`` would submit as ``/status``.
        * For the model browser, row 0 is always the right answer. The list was
          opened deliberately, it opens on the current model, and there may be
          only one row -- a one-item list has no row to move to, so a "did the
          user move?" gate would make it impossible to use.
        """
        text = event.value.strip()

        if self._picker == "model" and self._candidates:
            # A model id in the prompt is not a task and not a command; it can
            # only mean "switch to this". The browser owns the switch, including
            # the live probe, so the row is routed back through it rather than
            # being applied here.
            event.input.value = ""
            chosen = self._candidates[self._selected].name
            self._hide_hint()
            self._select_model(chosen)
            return

        chosen = self._chosen_candidate()

        if not text:
            self._hide_hint()
            return
        event.input.value = ""

        if chosen is not None:
            text = chosen

        self._hide_hint()

        if text.startswith("/"):
            self._command(text)
            return

        self._say(f"› {text}", Kind.STATUS)
        self._send(text)

    def _chosen_candidate(self) -> str | None:
        """The name of the moved-to row, or ``None`` if the list is untouched.

        The bare name (``/switchmodules``) rather than the signature
        (``/switchmodules [model]``): the argument is the user's to type, and
        submitting the placeholder would be rejected as an unknown command.
        """
        if not self._candidates or self._selected == 0:
            return None
        return self._candidates[self._selected].name

    def _send(self, text: str) -> None:
        """Kick off a background turn, rolling the session over if needed."""
        if self.session.busy:
            self._say("busy — Ctrl+C to cancel the running turn", Kind.ERROR)
            return
        self._mark_busy()
        try:
            self.session.submit(text)
        except Exception as exc:
            self._say(f"could not start turn: {exc}", Kind.ERROR)
            self._end_turn()

    def _command(self, text: str) -> None:
        """Dispatch a slash command.

        The name is resolved through :mod:`jaz_agent.commands` so an alias, the
        help text and the completions all agree; this method only decides what
        each canonical command *does*.
        """
        name, _, rest = text[1:].partition(" ")
        rest = rest.strip()

        command = find(name)
        if command is None:
            replacement = RETIRED.get(name.lower())
            if replacement:
                # A command that used to exist is a different failure from a
                # typo: the user knows exactly what they typed and is being told
                # no. Point them at what replaced it rather than making them
                # read /help to find out.
                self._say(
                    f"/{name} is gone — use /{replacement}",
                    Kind.ERROR,
                )
                return
            suggestion = self._closest(name)
            extra = f" — did you mean /{suggestion}?" if suggestion else ""
            self._say(f"unknown command /{name}{extra} — /help lists them all", Kind.ERROR)
            return

        handler = {
            "quit": lambda: self.exit(),
            "help": lambda: self._say(help_text(), Kind.STATUS),
            "status": self._show_status,
            "clear": self.action_clear,
            "new": self.action_new,
            "cancel": self.action_cancel,
            "cost": self.action_cost,
            "context": self.action_context,
            "backends": self.action_backends,
            "switchmodules": lambda: self.action_switch_modules(rest),
        }.get(command.name)

        if handler is None:
            # A command in the table with no behaviour: visible in help, but it
            # would be a lie to accept it silently.
            self._say(f"/{command.name} is not implemented yet", Kind.ERROR)
            return
        handler()

    def _closest(self, name: str) -> str | None:
        """Best-effort "did you mean" for a mistyped command.

        ``difflib`` rather than a hand-rolled distance: it already knows about
        transpositions, which is the common typo (``/modle``).
        """
        import difflib

        options = [c.name for c in COMMANDS]
        matches = difflib.get_close_matches(name.lower(), options, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def _show_status(self) -> None:
        """Print the current model, workspace and tool list."""
        self._say(
            f"model: {self.session.model_name}\n"
            f"workspace: {ROOT}\n"
            f"tools: {tool_catalog()}",
            Kind.STATUS,
        )

    # -- model switching --------------------------------------------------

    #: How many rows the picker's viewport shows, and therefore how many the
    #: window draws. Was a hard cap on the *data* too, which is what made the
    #: other 444 rows unreachable; now it is only a window size and the list
    #: scrolls. Kept as a constant because it is a layout decision rather than a
    #: number the code depends on, and the tests assert against it.
    MODEL_PAGE_SIZE = 18

    def action_switch_modules(self, argument: str = "") -> None:
        """``/switchmodules`` — browse and switch models on the current backend.

        Without an argument this lists the catalogue; with one it switches. The
        argument accepts a filter, so ``/switchmodules gpt-5`` narrows the list
        rather than jumping straight to a switch — OpenRouter currently offers
        463 models, so an unfiltered dump would be unreadable.

        The list is fetched on a worker: it is a network call, and OpenRouter's
        catalogue can be slow.
        """
        query = argument.strip()

        # No argument: browse. With one: try a switch first, then fall back to
        # filtering. Trying the switch first is right because it is validated --
        # a name that is not a real model fails the probe cheaply and lands in
        # the filter path instead, with nothing committed.
        #
        # The switch attempt has to be distinguished from a *successful* one, and
        # a worker reports failures by emitting rather than by raising, so the
        # decision is made here on the session state, not on an exception.
        if query:
            threading.Thread(
                target=self._try_switch_then_filter,
                args=(query,),
                name="jaz-switch-try",
                daemon=True,
            ).start()
            return

        self._say(f"loading models from {self.session.backend.name} …", Kind.THINKING)
        threading.Thread(
            target=self._models_worker,
            args=("",),
            name="jaz-models",
            daemon=True,
        ).start()

    def _try_switch_then_filter(self, query: str) -> None:
        """Switch if *query* names a real model, otherwise browse with it as a filter.

        Runs on a worker: both the catalogue fetch and the probe are network
        calls.

        Membership is decided by the catalogue rather than by attempting the
        switch and catching the failure. Relying on the exception would be
        fragile in a specific way: with an injected backend there is no probe to
        fail, so any string would "succeed" and the filter path would never be
        reached. Checking the catalogue first makes the branch depend only on
        what actually exists.
        """
        backend = self.session.backend
        bare = backend.bare(query)

        try:
            models = list_models(backend)
        except Exception as exc:
            self.session.bridge.emit(
                Kind.ERROR, f"could not load the model list — {type(exc).__name__}: {exc}"
            )
            return

        # Exact match on the bare id: an exact match on the routed id. Anything
        # else is a filter, and the filter path reports its own "no match".
        if models and bare not in models and query not in models:
            self._emit_models(models, query)
            return

        try:
            note = self.session.switch_model(bare)
        except Exception as exc:
            # It was in the catalogue but the provider refused it -- a key
            # problem, a rate limit, a model that just went away.
            self.session.bridge.emit(Kind.ERROR, f"not switched — {exc}")
            return

        self.suggester.invalidate()
        self.session.bridge.emit(Kind.RESULT, note, redraw_banner=True)
        threading.Thread(
            target=self._warm_suggester, name="jaz-models-warm", daemon=True
        ).start()

    def _models_worker(self, query: str) -> None:
        """Body of the browse thread: fetch the catalogue, then report.

        Emits onto the queue like every other worker -- it must not touch a
        widget, which is why the list is formatted here rather than in the drain.
        """
        try:
            models = list_models(self.session.backend)
        except Exception as exc:
            self.session.bridge.emit(
                Kind.ERROR, f"could not load the model list — {type(exc).__name__}: {exc}"
            )
            return
        self._emit_models(models, query)

    def _emit_models(self, models: list[str], query: str = "") -> None:
        """Filter, then hand the whole result set to the UI thread.

        Shared by the browse path and the "that was not a model, here is what
        matches" fall-back, so the two cannot format differently.

        Nothing is truncated here. This used to take the first twenty and print
        "and 443 more", which was not pagination -- the other 444 rows had no
        way to be reached at all, by arrow or by command. The picker scrolls
        instead, so the whole set goes across and the viewport decides what is
        visible.
        """
        backend = self.session.backend

        if not models:
            self.session.bridge.emit(
                Kind.ERROR,
                f"no models available from {backend.name}; check the network, "
                f"or name a model directly with /switchmodules <model>",
            )
            return

        if query:
            needle = query.lower()
            # Substring first -- "gpt" finds "openai/gpt-5-mini". Then a
            # subsequence match so "gpt5" finds "gpt-5" as well.
            substring = [m for m in models if needle in m.lower()]
            fuzzy = [
                m for m in models if needle not in m.lower() and _subsequence(needle, m.lower())
            ]
            models = substring or fuzzy

        if not models:
            self.session.bridge.emit(
                Kind.ERROR, f"no model matches {query!r} on {backend.name}"
            )
            return

        total = len(models)
        header = f"{total} model(s) on {backend.name}"
        if query:
            header += f" matching {query!r}"

        current = self.session.model
        lines = [f"{header}:"]

        # Open on the current model. With 464 ids sorted it is near the end, so
        # starting at row 0 would mean ↓ picks a model the user was not looking
        # at. The picker scrolls to it; when the model was filtered out of the
        # result set there is nothing to scroll to, so say which one is live.
        if query and current not in models:
            lines.append(f" current: {current}  (not in these results)")

        self.session.bridge.emit(
            Kind.STATUS,
            "\n".join(lines),
            models=list(models),
            current=current,
        )

    def action_backends(self) -> None:
        """List the configured backends and whether each has a key."""
        rows = []
        for name, has_key in available_backends():
            mark = "✓" if has_key else "·"
            current = " (current)" if name == self.session.backend.name else ""
            rows.append(f"{mark} {name:12s}{current}")
        self._say("backends:\n" + "\n".join(rows), Kind.STATUS)

    def _warm_suggester(self) -> None:
        """Fetch the current backend's models for the completion cache.

        Fire-and-forget: a failure here leaves completion working on commands
        only, which is a perfectly good state to be in.
        """
        try:
            self.suggester.load_models(list_models(self.session.backend))
        except Exception:
            pass

    # -- actions ---------------------------------------------------------

    def action_clear(self) -> None:
        """Clear the transcript."""
        if self.transcript:
            self.transcript.clear()

    def action_new(self) -> None:
        """Drop conversation history so the next turn starts cold."""
        if self.session.busy:
            self._say("busy — cancel the current turn first", Kind.ERROR)
            return
        self.session.context.clear()
        self.session.bridge.iterations = 0
        self.session.bridge.started = time.monotonic()
        if self.status:
            self.status.turns = 0
            self.status.cost = ""
        self.started = time.monotonic()
        self._say("new session — history cleared", Kind.STATUS)

    def action_cancel(self) -> None:
        """Ask the running turn to stop."""
        if not self.session.busy:
            self._say("nothing running", Kind.STATUS)
            return
        self.session.cancel()

    def action_cost(self) -> None:
        """Show accumulated usage for this session."""
        bridge = self.session.bridge
        self._say(
            f"turns: {bridge.iterations} · cost: ${bridge.total_cost:.4f} · elapsed: {self.status.elapsed if self.status else '0:00'}",
            Kind.COST,
        )

    def action_context(self) -> None:
        """Show what the session remembers and what the next turn will send.

        Safe to render raw: the transcript runs with ``markup=False``, so the
        square brackets in the turn list cannot be read as Rich tags.
        """
        self._say(context_report(self.session.context), Kind.STATUS)


def run_tui(session: AgentSession | None = None, prompt: str | None = None) -> None:
    """Entry point used by ``python -m jaz_agent``."""
    app = CodingAgentApp(session, prompt=prompt)
    app.run()