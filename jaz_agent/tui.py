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
from pathlib import Path
from typing import Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.suggester import Suggester
from textual.widgets import Footer, Header, Input, RichLog, Static

from .bridge import Event, Kind, QueueDrain
from .commands import COMMANDS, complete, find, help_text, hint_text
from .llm_config import (
    LLMConfigError,
    available_backends,
    describe_model,
    list_models,
    resolve_backend,
)
from .session import AgentSession
from .tools import ROOT, tool_catalog


class CommandSuggester(Suggester):
    """Tab completion for the prompt box.

    Two levels, because the useful completion changes with what has been typed:

    * ``/mo`` → ``model``
    * ``/model openai/gpt-5-mini`` → completes to the full id in the catalogue

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

/* The inline hint: one line under the prompt, shown only while typing a
   slash command. Collapses to nothing (display: none) otherwise so it does not
   leave a gap in the layout. */
#hint {
    height: auto;
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
    # Dropping a session is ``/new``. It used to also be F2, but a bare function
    # key that clears your conversation is not discoverable and not reversible,
    # and the command is listed in ``/help`` while a function key is not.
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
            self.prompt_box = Input(
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

        Called on mount and after every successful ``/model`` switch, so the
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

        value = event.value
        if not value.startswith("/"):
            self.hint.display = False
            return

        head, sep, _ = value.partition(" ")
        prefix = head.lstrip("/")

        if sep:
            # After the command name: hint about the argument, if it takes one.
            command = find(prefix)
            if command is None:
                self._set_hint(f"unknown command /{prefix}", error=True)
                return
            if not command.usage:
                self._set_hint(f"/{command.name} takes no arguments — {command.summary}")
                return
            self._set_hint(f"{command.signature} — {command.summary}")
            return

        self._set_hint(hint_text(prefix))

    def _set_hint(self, text: str, *, error: bool = False) -> None:
        """Show one line of hint text, styled as an error when appropriate."""
        if self.hint is None:
            return
        self.hint.display = True
        self.hint.update(f"[red]{text}[/]" if error else text)

    def _hide_hint(self) -> None:
        """Collapse the hint. Called when a command is submitted."""
        if self.hint is not None:
            self.hint.display = False

    # -- input handling --------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle a submitted prompt: slash commands, else start a turn."""
        text = event.value.strip()
        if not text:
            self._hide_hint()
            return
        event.input.value = ""
        self._hide_hint()

        if text.startswith("/"):
            self._command(text)
            return

        self._say(f"› {text}", Kind.STATUS)
        self._send(text)

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
            suggestion = self._closest(name)
            extra = f" — did you mean /{suggestion}?" if suggestion else ""
            self._say(f"unknown command /{name}{extra} — /help lists them all", Kind.ERROR)
            return

        handler = {
            "quit": lambda: self.exit(),
            "help": lambda: self._say(help_text(), Kind.STATUS),
            "status": self._show_status,
            "clear": self.action_clear,
            "new": self.action_new_session,
            "cancel": self.session.cancel,
            "backends": self.action_backends,
            "switchmodules": lambda: self.action_switch_modules(rest),
            "model": lambda: self.action_model(rest),
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

    #: How many models one page of ``/switchmodules`` shows. Small enough to fit
    #: a screen without scrolling the transcript into uselessness.
    MODEL_PAGE_SIZE = 20

    def action_switch_modules(self, argument: str = "") -> None:
        """``/switchmodules`` — browse and switch models on the current backend.

        Without an argument this lists the catalogue; with one it switches, the
        same as ``/model <name>``. The argument accepts a filter, so
        ``/switchmodules gpt-5`` narrows the list rather than jumping straight
        to a switch — OpenRouter currently offers 463 models, so an unfiltered
        dump would be unreadable.

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
        # ``_switch_worker`` reports failures by emitting rather than by raising,
        # so the decision is made here on the session state, not on an exception.
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
        """Filter, page and emit a model list. Shared by browse and fall-back.

        Split out from the worker so the "not a model, here is what matches"
        path and the plain ``/switchmodules`` path cannot format differently.
        """
        backend = self.session.backend

        if not models:
            self.session.bridge.emit(
                Kind.ERROR,
                f"no models available from {backend.name}; check the network, "
                f"or name a model directly with /model <name>",
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

        shown = models[: self.MODEL_PAGE_SIZE]
        header = f"{len(models)} model(s) on {backend.name}"
        if query:
            header += f" matching {query!r}"

        current = self.session.model
        lines = [f"{header}:"]

        # When browsing the whole catalogue, page alphabetically but state the
        # current model up front. OpenRouter lists 463 ids sorted, and a model
        # like `stealth/space-bunny-alpha` sits at position ~420 -- so an
        # alphabetical first page can leave the one model the user is actually
        # on off-screen entirely. A filter is different: if the current model
        # matched, it is already in the result set.
        if not query and current not in shown:
            lines.append(f" ✓ {current}   ← current")

        lines.extend(f" {'✓' if m == current else ' '} {m}" for m in shown)
        if len(models) > len(shown):
            lines.append(f" … and {len(models) - len(shown)} more")
        lines.append("")
        lines.append("switch with  /switchmodules <model>   or  /model <model>")
        self.session.bridge.emit(Kind.STATUS, "\n".join(lines))

    def action_backends(self) -> None:
        """List the configured backends and whether each has a key."""
        rows = []
        for name, has_key in available_backends():
            mark = "✓" if has_key else "·"
            current = " (current)" if name == self.session.backend.name else ""
            rows.append(f"{mark} {name:12s}{current}")
        self._say("backends:\n" + "\n".join(rows), Kind.STATUS)

    def action_model(self, argument: str) -> None:
        """``/model`` shows the current model, ``/model <name>`` switches.

        With no argument this lists what the current backend offers rather than
        printing just the name: the useful answer to "which model am I on" also
        answers "what else could I pick".

        Three forms are accepted, resolved here so ``switch_model`` stays a plain
        setter with one meaning:

        * ``/model <model>``            -- same backend, named model
        * ``/model <backend>``          -- that provider, its default model
        * ``/model <backend> <model>``  -- both
        """
        if not argument:
            self._say(self._model_menu(), Kind.STATUS)
            return

        parts = argument.split()
        if len(parts) == 1:
            # A single word is a model id unless it names a backend, in which
            # case the user means "move to that provider". An empty model is
            # passed through as-is: switch_model resolves the backend's default,
            # so the TUI does not need its own copy of that rule.
            try:
                resolve_backend(parts[0])
            except LLMConfigError:
                backend_name, model_name = None, parts[0]
            else:
                backend_name, model_name = parts[0], ""
        else:
            backend_name, model_name = parts[0], " ".join(parts[1:])

        label = model_name or f"{backend_name} default"
        self._say(f"switching to {label} …", Kind.THINKING)
        # The probe is a live network call, so it runs on a worker and reports
        # back through the same queue as the agent's events.
        threading.Thread(
            target=self._switch_worker,
            args=(backend_name, model_name),
            name="jaz-switch",
            daemon=True,
        ).start()

    def _switch_worker(self, backend_name: str | None, model_name: str) -> None:
        """Body of the switch thread: probe, then commit or report.

        Runs off the UI thread because the probe is a live network call. It must
        therefore not touch a widget -- results go onto the agent's queue as
        events and are rendered by the same drain as everything else. The
        ``banner`` flag rides along because a committed switch changes the
        header, and only the UI thread may redraw that.
        """
        try:
            note = self.session.switch_model(model_name or None, backend=backend_name)
        except LLMConfigError as exc:
            # The old model is still in force. Say so explicitly: "it did not
            # switch" and "it switched and then failed" mean very different
            # things to the user.
            self.session.bridge.emit(Kind.ERROR, f"not switched — {exc}")
        except Exception as exc:  # never let a switch kill the UI
            self.session.bridge.emit(Kind.ERROR, f"not switched — {type(exc).__name__}: {exc}")
        else:
            # The cached model ids belong to the old backend; drop them so Tab
            # does not offer models the new provider does not have.
            self.suggester.invalidate()
            self.session.bridge.emit(Kind.RESULT, note, redraw_banner=True)
            # Warm the new backend's catalogue so Tab completion works straight
            # away rather than after the user's first /model attempt.
            threading.Thread(
                target=self._warm_suggester,
                name="jaz-models-warm",
                daemon=True,
            ).start()

    def _warm_suggester(self) -> None:
        """Fetch the current backend's models for the completion cache.

        Fire-and-forget: a failure here leaves completion working on commands
        only, which is a perfectly good state to be in.
        """
        try:
            self.suggester.load_models(list_models(self.session.backend))
        except Exception:
            pass

    def _model_menu(self) -> str:
        """Current model plus alternatives, for the no-argument ``/model``."""
        session = self.session
        lines = [f"current: {session.model_name}"]

        models = list_models(session.backend)
        others = [m for m in models if m != session.model]
        if others:
            lines.append("")
            lines.append(f"available on {session.backend.name} (showing {min(8, len(others))}):")
            lines.extend(f"  {m}" for m in others[:8])
            if len(others) > 8:
                lines.append(f"  … and {len(others) - 8} more")
        else:
            lines.append("")
            lines.append("model discovery unavailable; type a model id directly")

        lines.append("")
        lines.append("  /model <name>            switch model")
        lines.append("  /model <backend> <model>  switch backend and model")
        lines.append("  /model <backend>         switch backend, keep its default model")
        lines.append("  /backends                list backends and their keys")
        return "\n".join(lines)

    # -- actions ---------------------------------------------------------

    def action_clear(self) -> None:
        """Clear the transcript."""
        if self.transcript:
            self.transcript.clear()

    def action_new_session(self) -> None:
        """Drop conversation history so the next turn starts cold."""
        if self.session.busy:
            self._say("busy — cancel the current turn first", Kind.ERROR)
            return
        self.session.history.clear()
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


def run_tui(session: AgentSession | None = None, prompt: str | None = None) -> None:
    """Entry point used by ``python -m jaz_agent``."""
    app = CodingAgentApp(session, prompt=prompt)
    app.run()