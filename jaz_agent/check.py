"""Live smoke test against OpenRouter. Run with a real key:

    $env:OPENROUTER_API_KEY = "sk-or-v1-..."
    .\\.venv\\Scripts\\python.exe -m jaz_agent.check

Kept separate from ``--selftest`` on purpose: the selftest must never need a
network or a key, so it stays useful in CI and as a smoke test.
"""

from __future__ import annotations

import sys
import time
from queue import Empty

from .bridge import Kind
from .llm_config import LLMConfigError, build_llm, describe_model, list_models, resolve_backend
from .session import AgentSession
from .tools import ROOT


def main() -> int:
    """Run one trivial task end to end and print what happened."""
    print(f"model    : {describe_model()}")
    try:
        llm = build_llm()
    except LLMConfigError as exc:
        print(f"config   : FAILED\n{exc}")
        return 3
    print("config   : ok (api key found)")
    print(f"workspace: {ROOT}\n")

    # Show what /switchmodules would offer, since that is the next thing a user asks.
    models = list_models(resolve_backend(None))
    print(f"models   : {len(models)} available on this backend, e.g. {', '.join(models[:3])}\n")

    session = AgentSession(max_iterations=12)
    task = sys.argv[1] if len(sys.argv) > 1 else (
        "List the files in the current directory, then call finish() with a "
        "one-line summary of what you see."
    )

    started = time.monotonic()
    session.submit(task)

    # The worker runs on its own thread; this loop only prints.
    while True:
        try:
            ev = session.queue.get(timeout=0.5)
        except Empty:
            if not session.busy and session._thread and not session._thread.is_alive():
                break
            continue

        if ev.kind is Kind.CODE:
            print("--- agent code ---")
            print(ev.text)
        elif ev.kind is Kind.OUTPUT:
            print("--- output ---")
            print(ev.text[:800])
        elif ev.kind is Kind.ERROR:
            print(f"!!! {ev.text}")
        elif ev.kind is Kind.RESULT:
            print("\n=== RESULT ===")
            print(ev.text)
        elif ev.kind is Kind.COST and ev.payload.get("total_cost"):
            print(f"  [cost] {ev.payload['total_cost']}")
        elif ev.kind is Kind.RETRY:
            print(f"  [retry] {ev.text}")

    print(f"\nelapsed: {time.monotonic() - started:.1f}s  turns: {session.bridge.iterations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())