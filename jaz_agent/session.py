"""Runs a single jaz invoke on a worker thread and publishes progress events.

The agent is a blocking, synchronous library call. The TUI is single-threaded
and async. :class:`AgentSession` is the seam between them: ``submit`` starts the
run in a daemon thread, the bridge's queue carries events out, and the UI drains
it. Nothing in here touches a widget.

Two jaz details drive the design here, both learned the hard way:

* **The REPL is a configured component, not a hook.** It is installed with
  ``jaz.configure(repl=...)``; passing it to ``invoke()`` positionally raises
  ``TypeError``.
* **A coding agent needs the sandbox opened up.** jaz's Python REPL denies
  imports and file access by default, which makes a file-editing agent
  impossible. The allow-lists below are the point where that policy is set, so
  they are spelled out rather than hidden behind a stray ``["*"]``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from queue import SimpleQueue
from typing import Any

import jaz
from jaz.hooks import FileLogger, IterationLimit
from jaz.repl.python_repl import PythonREPL

from .bridge import EventBridge, Kind
from .context import Context
from .llm_config import (
    Backend,
    LLMConfigError,
    build_llm,
    default_model_for,
    describe_model,
    probe,
    resolve_backend,
)
from .tools import ROOT, Tools

# The working-style instructions. jaz already describes the REPL and renders the
# tool catalog; this decides whether the agent edits one file or rewrites the
# world, so it stays short and concrete.
SYSTEM_GUIDANCE = """
You are a focused software-engineering agent working inside a real repository.

How to work:
- Investigate before you edit. Use tools.list_dir, tools.search_files and
  tools.read_file to understand the existing code and its conventions first.
- Make the smallest change that solves the task. Match the surrounding style,
  naming and comment density rather than importing preferences from elsewhere.
- Verify your work: run the project's tests or build with tools.run_shell and
  read the output. Fix what you broke and re-run until it passes.
- Prefer tools.edit_file over tools.write_file for existing files -- it fails
  loudly when your match is wrong, which is what you want.
- Do not commit, push, install global packages or delete files unless asked.

When you are done, call finish(summary) with a short markdown summary of what you
changed and what you verified. That ends the task -- do not call it early.
""".strip()


def make_repl(workspace: Path, *, exec_timeout: float = 120.0) -> PythonREPL:
    """Build the REPL the agent runs its generated Python in.

    jaz ships a locked-down REPL: no imports, no reads, no writes. That is the
    right default for a calculator agent and the wrong one for a coding agent,
    so every gate is opened deliberately here:

    * ``allowed_attributes`` -- lets the model call ``tools.read_file(...)``.
      jaz enforces attribute access against this list, and the default omits
      arbitrary method calls on a passed-in object.
    * ``allowed_imports`` -- the agent needs ``os``, ``json``, ``re`` and
      friends to inspect a codebase.
    * ``allowed_read_paths`` / ``allowed_write_paths`` -- scoped to the
      workspace so a stray absolute path cannot wander across the filesystem.
      Both the directory and its contents are listed, because a bare directory
      does not match its own contents under a glob.
    """
    ws = Path(workspace).resolve()
    return PythonREPL(
        exec_timeout=exec_timeout,
        allow_raise=True,
        allow_timeout_pragma=True,
        allowed_attributes=["*"],
        allowed_imports=["*"],
        allowed_read_paths=[str(ws), str(ws / "**")],
        allowed_write_paths=[str(ws), str(ws / "**")],
    )


class AgentSession:
    """One conversation with the agent.

    jaz's ``invoke`` is stateless across calls -- each call is a fresh loop. For
    a conversational agent that is wrong: turn 2 needs to know what turn 1 did.
    ``jaz.scope`` is the intended mechanism for exactly that, so each turn binds
    the prior turns into scope before invoking.
    """

    def __init__(
        self,
        *,
        workspace: Path | None = None,
        model: str | None = None,
        backend: str | None = None,
        max_iterations: int = 40,
        verbose: bool = False,
        log_file: str | None = None,
        llm: Any | None = None,
        exec_timeout: float = 120.0,
    ) -> None:
        """Configure the session. ``llm`` overrides the backend built from
        ``backend``/``model``; ``--selftest`` uses it to inject a mock."""
        self.workspace = Path(workspace or ROOT).resolve()
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.log_file = log_file
        self.exec_timeout = exec_timeout
        self._llm = llm

        # The backend/model pair is session state rather than construction state,
        # so /switchmodules can change it later. Resolved eagerly so a bad name or a
        # missing key fails at startup rather than on the first turn.
        self.backend = resolve_backend(backend)
        self.model = self.backend.bare(model) if model else default_model_for(self.backend)

        self.queue: SimpleQueue = SimpleQueue()
        self.bridge = EventBridge(self.queue, verbose=verbose)
        self.tools = Tools()

        self._thread: threading.Thread | None = None
        self._cancelled = threading.Event()
        self.context = Context()
        self.running = False

    # -- model selection -------------------------------------------------

    @property
    def model_name(self) -> str:
        """``model via backend`` for the header and ``/status``."""
        return f"{self.model} via {self.backend.name}"

    @property
    def busy(self) -> bool:
        """True while a turn is running.

        The TUI uses this to lock the prompt, and ``switch_model`` uses it to
        refuse a swap that would split one history across two models.
        """
        return self.running

    def is_current(self, model: str, backend: Backend | None = None) -> bool:
        """True if *model* on *backend* is the pair already in use.

        One definition of "already there", because two callers need it for
        different reasons and must not disagree. ``switch_model`` uses it to
        skip a probe that would prove nothing; the browser uses it to recognise
        the row it shows for the live model, which the catalogue need not
        contain -- so falling through to a catalogue lookup for it would report
        "no model matches" for the model on screen.
        """
        target = backend or self.backend
        return target is self.backend and target.bare(model) == self.model

    def switch_model(
        self, model: str | None = None, *, backend: str | None = None
    ) -> str:
        """Point the session at a different model or backend, after proving it works.

        The new backend is built and probed with one live request *before*
        anything is committed, so a rejected key or a nonexistent model leaves
        the session exactly where it was. This mirrors
        ``jaz.console.switch_model``: validate, then commit, never the reverse.

        Returns a short confirmation for the UI. Raises
        :class:`~jaz_agent.llm_config.LLMConfigError` when the target is
        unusable -- again without having changed anything.

        Naming a *backend* with no model switches to that provider's default
        model. Resolving that here rather than in the UI keeps one definition of
        "the default", so the TUI and any programmatic caller agree.

        Refuses while a turn is running: the worker is mid-``invoke`` against the
        current backend, and swapping under it would produce a run whose history
        mixes two models.
        """
        if self.running:
            raise LLMConfigError("cannot switch model while a turn is running — Ctrl+C first")

        target_backend = resolve_backend(backend) if backend else self.backend
        bare = (model or "").strip()
        # A bare backend name means "that provider's default model". That is a
        # sensible request even when it names the backend already in use --
        # ``/switchmodules openrouter`` reads as "put me back on the default",
        # not as a forgotten model -- so the only thing that is an error is
        # naming neither.
        if not bare:
            if backend is None:
                raise LLMConfigError(
                    "no model given — pass a model name, or a backend to switch to"
                )
            bare = default_model_for(target_backend)
        bare = target_backend.bare(bare)

        # Already there. Re-selecting the live model is not a switch: the
        # browser deliberately lists it so the user can see where they are, and
        # Enter on that row must neither spend a probe proving something
        # already proven nor be refused for not being in a catalogue it is not
        # in. Reported rather than ignored, so the UI can say so.
        if self.is_current(bare, target_backend):
            return f"already on {bare} — nothing to switch"

        candidate = target_backend.build(bare)

        # An injected llm (tests, --selftest) has no endpoint to probe, and there is
        # nothing to validate: the injection exists precisely to avoid a network
        # call. The commit still happens so the state machine is exercised
        # offline; the injected backend keeps serving turns because dropping it
        # would make the next turn demand a real API key.
        if self._llm is None:
            failure = probe(candidate)
            if failure:
                # Name the backend that answered, not just the model we asked
                # about. ``/switchmodules`` cannot change *provider* -- that is
                # ``-b/--backend`` -- so asking for a model that belongs to
                # another provider probes the one you are already on. Without
                # the name, "the API key was rejected" sends the user to the
                # wrong account; with it, "on openrouter" says where to look.
                raise LLMConfigError(
                    f"{bare} is unusable on {target_backend.name}: {failure}"
                )

        previous_backend = self.backend
        self.backend, self.model = target_backend, bare

        where = (
            "same backend"
            if target_backend is previous_backend
            else f"backend {target_backend.name}"
        )
        return f"switched to {bare} ({where})"

    def cancel(self) -> None:
        """Ask the current turn to stop.

        jaz exposes no cancellation token, so this is advisory: it records the
        request and the UI reports that the current step will finish first.
        """
        self._cancelled.set()
        self.bridge.emit(Kind.ERROR, "cancellation requested; stopping after the current step")

    # -- the actual run --------------------------------------------------

    def submit(self, user_input: str) -> None:
        """Start a turn in the background. Returns immediately."""
        if self.running:
            raise RuntimeError("a turn is already in flight")

        self._cancelled.clear()
        self.running = True
        self._thread = threading.Thread(
            target=self._run_turn,
            args=(user_input,),
            name="jaz-agent",
            daemon=True,
        )
        self._thread.start()

    def _turn_hooks(self) -> list[Any]:
        """Build the hook stack for one turn.

        A ``finish`` callable is passed as a tool rather than a return type being
        enforced: ``ReturnType`` would require the agent to import a sentinel
        class to build its final value, and imports are exactly what the sandbox
        gates. A function the agent can simply call is always available.
        """
        hooks: list[Any] = [IterationLimit(max_iterations=self.max_iterations), self.bridge]
        if self.log_file:
            hooks.append(FileLogger(self.log_file))
        return hooks

    def _run_turn(self, user_input: str) -> None:
        """Body of the worker thread: run one invoke, translate the outcome."""
        try:
            # Read the backend/model at turn start, not at construction: a
            # /switchmodules switch between turns must take effect on the next one.
            llm = self._llm or build_llm(self.model, backend=self.backend.name)
            jaz.configure(
                llm=llm,
                repl=make_repl(self.workspace, exec_timeout=self.exec_timeout),
            )

            prior = self.context.prompt()

            def finish(summary: str) -> str:
                """Call this when the task is complete, passing a short markdown
                summary of what you changed and what you verified."""
                return summary

            inputs: dict[str, Any] = {
                "task": user_input,
                "guidance": SYSTEM_GUIDANCE,
                "tools": self.tools,
                "workspace": str(self.workspace),
                "finish": finish,
            }
            if prior:
                inputs["prior_turns"] = prior

            report = jaz.invoke(*self._turn_hooks(), **inputs)

            summary = report if isinstance(report, str) else str(report)
            self.context.add(user_input, summary)
            self.bridge.emit(Kind.RESULT, summary, final=True)

        except Exception as exc:  # the UI must survive any agent failure
            self.bridge.emit(Kind.ERROR, f"{type(exc).__name__}: {exc}", fatal=True)
        finally:
            self.running = False
            self.bridge.emit(Kind.STATUS, "idle", idle=True)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the worker finishes. Returns True if it did."""
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def run_sync(self, user_input: str) -> str:
        """Run a turn on this thread and return the final summary.

        For tests and ``--prompt`` mode, where a TUI is not involved.
        """
        if self.running:
            raise RuntimeError("a turn is already in flight")
        self.running = True
        try:
            self._run_turn(user_input)
        finally:
            self.running = False
        if not self.context.turns:
            return ""
        return self.context.turns[-1].report