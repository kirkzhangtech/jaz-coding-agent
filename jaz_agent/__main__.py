"""CLI wrapper: argument parsing, then either the TUI or a one-shot run."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from queue import Empty

from .bridge import Event, Kind
from .llm_config import LLMConfigError
from .session import AgentSession
from .tools import ROOT


def _drain(session: AgentSession) -> list[Event]:
    """Everything the finished turn queued, in order.

    The TUI drains this queue on a timer; the one-shot path has to do it by
    hand, because it is the only place the failure of a turn is reported.
    """
    events: list[Event] = []
    while True:
        try:
            events.append(session.queue.get_nowait())
        except Empty:
            return events


def _failures(events: list[Event]) -> list[str]:
    """The error texts in *events*, newest last. Empty when the turn was clean."""
    return [event.text for event in events if event.kind is Kind.ERROR]


def build_parser() -> argparse.ArgumentParser:
    """Command-line surface. Kept small on purpose -- the TUI is the product."""
    p = argparse.ArgumentParser(
        prog="jaz-agent",
        description="A TUI coding agent built on the jaz framework, using OpenRouter's Space Bunny Alpha.",
    )
    p.add_argument("-p", "--prompt", help="Run one task and exit, instead of opening the TUI.")
    p.add_argument("-w", "--workspace", default=str(ROOT), help="Directory the agent may touch (default: cwd).")
    p.add_argument("-m", "--model", default=None, help="Override the model id.")
    p.add_argument(
        "-b",
        "--backend",
        default=None,
        help="LLM backend to use (default: openrouter). See /backends in the TUI.",
    )
    p.add_argument(
        "-n",
        "--max-iterations",
        type=int,
        default=40,
        help="Cap on agent turns per task (default: 40).",
    )
    p.add_argument("--log", default=None, help="Write a jaz event log to this file.")
    p.add_argument("-v", "--verbose", action="store_true", help="Show raw REPL output for every step.")
    p.add_argument("--selftest", action="store_true", help="Run offline checks with a mock model and exit.")
    return p


def _selftest(session: AgentSession) -> int:
    """Exercise the whole loop against a scripted model. No network, no key.

    The script deliberately makes one wrong call before the correct one, so the
    run also proves that a tool traceback reaches the model and the loop keeps
    going -- which is the behaviour the whole tool design rests on.
    """
    from jaz import MockLLMClient

    calls: list[str] = []

    def fake(model, messages, **kwargs):
        """One scripted turn per call, keyed off how many have happened."""
        calls.append("turn")
        n = len(calls)
        print(f"  (scripted turn {n})", flush=True)
        if n == 1:
            # A deliberate mistake: read a file that does not exist. The tool
            # raises, jaz shows the traceback, the loop must continue.
            return "bad = tools.read_file('definitely-missing.txt')"
        if n == 2:
            # Now do something real, using a tool.
            return "listing = tools.list_dir('.')\nprint(len(listing.splitlines()), 'entries')"
        return "return finish('selftest ok: %d entries under the workspace' % len(listing.splitlines()))"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session._llm = MockLLMClient(fn=fake)

    session.max_iterations = 8

    print("running selftest with a scripted model...")
    report = session.run_sync("list the workspace, then report")

    seen: list[str] = []
    texts: dict[str, list[str]] = {}
    while True:
        try:
            ev = session.queue.get_nowait()
        except Empty:
            break
        seen.append(ev.kind.value)
        texts.setdefault(ev.kind.value, []).append(ev.text)

    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        if not cond:
            ok = False
            print(f"  FAIL: {msg}")

    print(f"  model calls : {len(calls)}")
    print(f"  events      : {', '.join(seen) or '(none)'}")
    print(f"  report      : {report!r}")

    check(len(calls) == 3, f"expected 3 scripted turns, got {len(calls)}")
    check("code" in seen, "no CODE event captured")
    check("result" in seen, "no RESULT event captured")
    check("selftest ok" in report, "report did not carry the agent's summary")

    # The deliberate failure must have surfaced as a traceback, then been
    # recovered from -- that is the property under test.
    outs = texts.get("output", [])
    check(any("Error" in o or "Traceback" in o or "no such file" in o for o in outs),
          "the intentional tool error never reached the transcript")
    check(len(calls) == 3, "loop did not recover from the tool error")

    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """Parse args and dispatch. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        print(f"error: workspace is not a directory: {workspace}", file=sys.stderr)
        return 2

    # The session resolves its backend eagerly, so an unknown --backend or a
    # missing key fails here rather than after the UI opens.
    try:
        session = AgentSession(
            workspace=workspace,
            model=args.model,
            backend=args.backend,
            max_iterations=args.max_iterations,
            verbose=args.verbose,
            log_file=args.log,
        )
    except LLMConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    if args.selftest:
        return _selftest(session)

    if args.prompt:
        try:
            report = session.run_sync(args.prompt)
        except LLMConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
            return 130

        # ``run_sync`` reports a failed turn through the event queue rather than
        # by raising -- the agent loop swallows everything so that one bad turn
        # cannot kill the UI. Nothing drains that queue here, so a failure used
        # to print an empty line and exit 0: a one-shot run that says nothing is
        # worse than one that says what went wrong, and it is indistinguishable
        # from a task that legitimately finished.
        failures = _failures(_drain(session))
        if failures and not report:
            for text in failures:
                print(f"error: {text}", file=sys.stderr)
            return 1

        print(report)
        return 0

    try:
        from .tui import run_tui

        run_tui(session, prompt=args.prompt)
        return 0
    except LLMConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())