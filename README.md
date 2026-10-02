# jaz coding agent

A terminal coding agent built on the [jaz](https://github.com/jaz-lang/jaz) framework,
running on **Space Bunny Alpha** (`stealth/space-bunny-alpha`) via OpenRouter.

jaz gives you a loop where the model writes Python, sees what it evaluated to,
and iterates. The tools below are what it writes calls to. A Textual front end
shows the loop happening.

```
· Ready. Describe a coding task and press Enter.
· › add a --verbose flag to the CLI parser

» thinking (turn 1)
│ files = tools.search_files('ArgumentParser', glob='*.py')
│ print(files[:200])

  src/cli.py:14:def build_parser():
  tests/test_cli.py:31:    parser = build_parser()

» thinking (turn 2)
│ src = tools.read_file('src/cli.py')
│ print(src[:1500])

 1	import argparse
 2
 3	def build_parser() -> argparse.ArgumentParser:
 ...

✓ Added `--verbose` to `build_parser`, wired through to `main()` in
  `src/cli.py:22`, with a test in `tests/test_cli.py:44`. `pytest` passes: 12 passed.
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

$env:OPENROUTER_API_KEY = "sk-or-v1-..."   # https://openrouter.ai/keys
```

Get a key at <https://openrouter.ai/keys>.

> **Note on terminals.** A terminal opened *before* you set the variable will not
> see it. Either open a new terminal, or put the key in a `key` file in the project
> root and export it yourself:
> ```powershell
> $env:OPENROUTER_API_KEY = (Get-Content .\key).Trim()
> ```
> `key` is listed in `.gitignore` — do not commit it.

## Run

```powershell
# interactive TUI
.\.venv\Scripts\python.exe -m jaz_agent

# one-shot, prints the report and exits
.\.venv\Scripts\python.exe -m jaz_agent -p "add a --verbose flag to the CLI"

# offline check: no key, no network
.\.venv\Scripts\python.exe -m jaz_agent --selftest

# live smoke test against the real model
.\.venv\Scripts\python.exe -m jaz_agent.check
```

Inside the TUI:

| Key       | Action                    |
| --------- | ------------------------- |
| `Enter`   | run the task              |
| `/`       | hint the matching commands while typing |
| `Tab`     | complete the current slash command |
| `Ctrl+C`  | cancel the running turn   |
| `Ctrl+L`  | clear the transcript      |
| `F2`      | new session               |
| `F5`      | show cost                 |
| `Ctrl+Q`  | quit                      |

Commands: `/help` `/status` `/switchmodules [model]` `/model [name|backend [name]]`
`/backends` `/new` `/clear` `/cancel` `/quit`. `/help` prints the full list.

Typing `/` opens an inline hint of the commands that match what you have typed
so far, and `Tab` completes the first one:

```
/swi⏎   hint ── switchmodules  browse and switch the backend's models
         [ switchmodules ]  browse and switch models on the current backend
```

Both are driven by the same table in `commands.py`, so a command cannot exist in
the UI without existing in `/help`, and the three views cannot drift apart.

## Browsing models

`/switchmodules` is the catalogue browser. With no argument it prints the first
page of the current backend's models:

```
» /switchmodules
 463 model(s) on openrouter:
 ✓ stealth/space-bunny-alpha   ← current
   aion-labs/aion-2.0
   aion-labs/aion-3.0
   …
 … and 443 more

 switch with  /switchmodules <model>   or  /model <model>
```

With an argument it does two things, in order. If the argument is exactly a
model id, it switches. Otherwise it filters, so a typo does not fail but shows
you what you probably meant:

```
» /switchmodules gpt-5
 47 model(s) on openrouter matching 'gpt-5':
   openai/gpt-5
   openai/gpt-5-mini
   openai/gpt-5-pro
   …
```

Matching is **substring first, then subsequence**. Substring alone is the
obvious choice and the wrong one — nobody types `gpt-5` when the catalogue calls
it `gpt-5.1-codex-mini`, and nobody can guess `stealth/space-bunny-alpha` from
memory. The subsequence fallback reads `gpt5` as `g-p-t-5` and finds it. Order
is respected, so `5gpt` matches nothing rather than everything.

Two details the real API forced:

- **The current model is stated up front, not just marked in place.** OpenRouter
  returns 463 ids sorted, and `stealth/space-bunny-alpha` sits around position
  420 — an alphabetical first page left the one model you are actually on off
  screen. It is now printed on its own line above the page when it does not
  already fit. A filtered list does not need this: if the current model matched,
  it is in the results.
- **An unknown name is treated as a search, not an error.** `/switchmodules
  gpt-5-minni` lists the three near-misses instead of failing, because the most
  likely cause of an unknown id is a typo rather than a model that does not
  exist. The switch itself still validates before committing (below), so a
  listing can never leave you on a model that does not work.

## Switching models

`/model` with no argument prints the current model *and* what else the backend
offers, because the useful answer to "which model am I on" also answers "what
else could I pick". OpenRouter's catalogue is fetched live; the other backends
fall back to a short built-in list when offline.

Three forms are accepted:

```
/model openai/gpt-5-mini            # same backend, named model
/model anthropic                    # that provider, its default model
/model anthropic claude-sonnet-4-5  # both
/backends                           # which providers have a key right now
```

**A switch is validated before it is committed.** The candidate backend is built
and sent one minimal live request; only if that succeeds does the session adopt
it. A wrong key, a nonexistent model or an unreachable provider leaves you on the
model you already had. This is the same contract `jaz.console.switch_model`
gives, and it is the whole reason the probe exists — finding out a model is
broken *after* a full agent turn has burned is much worse than a one-line error.

Two more rules keep a session's history coherent:

- **No switch during a turn.** The worker is mid-`invoke` against the current
  backend; swapping under it would produce a run whose history mixes two models.
  `Ctrl+C` first.
- **No API key is ever written down.** Only the backend and model ids are state.

Backends on offer: `openrouter` (default), `anthropic`, `openai`, `google`. Each
lists the environment variables it accepts in `llm_config.BACKENDS`.

### Verified live

The switch path was run against the real API, not only against mocks:

```
live catalogue: 463 models          (fetched from OpenRouter)
PASS  default model is in the catalogue
PASS  counts the catalogue
PASS  marks the current model
PASS  pages the list
PASS  finds gpt-5 models            (/switchmodules gpt-5 → 47 matches)
PASS  reports the filter
PASS  finds the bunny by substring
PASS  says so plainly               (no model matches)
PASS  model committed -- openai/gpt-5-mini via openrouter
PASS  bogus model rejected -- definitely/not-real-xyz is unusable:
      model not found, or not available on this backend
PASS  state unchanged after failure
PASS  missing key named in the error -- no API key for backend 'anthropic'.
      Set one of: ANTHROPIC_API_KEY
PASS  restored -- stealth/space-bunny-alpha via openrouter
PASS  agent produced a report          (a real agent turn, 2 model turns)
PASS  switch after a turn
PASS  subsequence helper -- gpt5 matches openai/gpt-5-mini
PASS  order is respected -- 5gpt matches nothing
```

Three real defects surfaced only because the live run existed, all now covered
by tests:

- **A bare `switch_model(backend=...)` failed.** The "use that backend's default
  model" rule lived in the TUI layer, so the session API and the UI disagreed
  about what a bare backend name means. The rule now lives in `switch_model`,
  and the UI just passes the name through.
- **OpenRouter leaks the account's `user_id` in its error payloads.** An
  unrecognised provider error was being shown verbatim, which would have put
  that id in the transcript and in any `--log` file. `_redact` now blanks
  identifier-shaped JSON values, and the wording matcher learned OpenRouter's
  actual phrasing (`is not a valid model ID`) instead of only `model not found`.
- **The current model could be off the visible page.** 463 models sorted, current
  one at ~420, and browsing told you neither where you were nor how far in you
  were. It is now printed above the page as `← current`, with a test that pins
  the off-page case specifically.

Two of the browser's own bugs were subtler. The first "is this a switch or a
search?" check tested for the absence of `/` and spaces in the argument — which
is true of *no* real model id, since every one contains a `/`. It filtered
everything, including exact names. The check is now catalogue membership. And
the switch worker reported failures by emitting an error rather than raising, so
the caller's fall-back path waited on an exception that never arrived; the
worker and the fall-back now agree.

## Architecture

Seven modules, ~2,600 lines. The shape is dictated by one fact: **jaz is
synchronous and blocking, Textual is single-threaded and async.** Everything
else follows from bridging that gap without locks.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Textual UI  (tui.py)                                                │
│  CodingAgentApp · Transcript · StatusBar                             │
│  ─ runs on the Textual thread ─                                      │
│         ▲                                          │ put_nowait      │
│         │ events (drained on a 20 Hz timer)        ▼                 │
│  ─────────────────────── thread boundary ────────────────────────    │
│  SimpleQueue  ◄── EventBridge (bridge.py) ──►  jaz Hook callbacks    │
└──────────────────────────────────────────────────────────────────────┘
                                    │
┌──────────────────────────────────────────────────────────────────────┐
│  Agent loop  (session.py)  ─ worker thread ─                         │
│                                                                       │
│    AgentSession.submit()  →  threading.Thread(daemon)                 │
│         └─ jaz.invoke(IterationLimit, EventBridge, task=…, tools=…)  │
│              │                                    │                   │
│              │                                    ▼                   │
│              │                        make_repl() → PythonREPL        │
│              │                          (sandbox opened here)        │
│              ▼                                                        │
│    LLM  ◄── LiteLLM ──► OpenRouter ──► stealth/space-bunny-alpha     │
│                                                                       │
│    Tools  (tools.py)  ◄── the model calls these as tools.foo(...)     │
└──────────────────────────────────────────────────────────────────────┘
```

### Module map

| Module            | Lines | Responsibility                                                        |
| ----------------- | ----: | --------------------------------------------------------------------- |
| `tui.py`          |   903 | Widgets, event loop, status bar, slash commands, model browser. The only module that touches widgets. |
| `tools.py`        |   366 | The 8 tools the agent calls. Path confinement lives in `_resolve`.     |
| `llm_config.py`   |   341 | The backend registry: routes, keys, model discovery, the switch probe, error redaction. |
| `session.py`      |   311 | One `invoke` per turn on a worker thread. Owns the REPL sandbox and the current model. |
| `bridge.py`       |   242 | `jaz` `Hook` → `Event` objects on a queue. Never blocks, never raises. |
| `commands.py`     |   150 | The command table. One source for dispatch, `/help`, hints and Tab completion. |
| `__main__.py`     |   165 | CLI parsing and dispatch.                                               |
| `check.py`        |    76 | Live end-to-end smoke test.                                             |

Dependencies point one way: `tui` → `session` → `bridge` + `tools` +
`llm_config`, with `tui` → `commands`. Nothing below `tui` imports Textual,
which is what makes the lower two thirds testable without a terminal.

`commands.py` is deliberately pure — a tuple of frozen dataclasses and four
functions, no imports. The table drives the dispatcher, the help text, the
inline hint and the suggester, so a command cannot be reachable in the UI while
missing from `/help`. `CommandSuggester` lives in `tui.py` rather than beside it
because it subclasses Textual's `Suggester`, which would drag the framework into
the module that everything else imports from.

### One turn, end to end

1. `on_input_submitted` fires on the Textual thread → `_send`.
2. `AgentSession.submit` appends to `history`, spawns a **daemon thread**, returns.
3. The worker calls `jaz.invoke(...)` with `task`, `tools`, `guidance`, and
   `prior_turns`. jaz configures the LLM and REPL, then loops:
   - **LLM turn** — the model replies with Python code.
   - **REPL turn** — jaz executes it in the sandbox; stdout and tracebacks go
     back to the model.
   - repeat until the model calls `finish(summary)` or `IterationLimit` trips.
4. `EventBridge` — a jaz `Hook` — receives `on_llm_query_enter`,
   `on_repl_exec_enter`, `on_repl_exec_complete`, … on the **worker** thread, and
   does exactly one thing: `queue.put_nowait(Event(...))`.
5. The UI's 20 Hz timer drains the queue. It is the only thing that writes to a
   widget, so no lock or `call_from_thread` is needed anywhere.
6. The `idle` marker on the last event flips the status bar back to `ready` and
   re-enables the prompt.

### Why a queue and not `call_from_thread`

`call_from_thread` marshals a callable onto the UI loop. That works, but it
means the agent thread blocks until the UI is free, and any exception in a
widget handler surfaces as an agent failure. A `SimpleQueue` gives the same
ordering guarantee with none of that coupling: the producer only ever appends,
and a rendering bug cannot abort the drain loop because `QueueDrain` swallows
per-event exceptions.

### Switching models is a threaded operation too

The probe is a live network call, so `/model` cannot run on the UI thread. It
follows the same rule as the agent: the worker (`_switch_worker`) emits onto the
queue and the drain renders. A committed switch sets `redraw_banner` on its
event, because only the UI thread may redraw the header — the same reason the
header could not be updated inside the worker directly.

```
/model <name>
   │
   ├─ UI thread   resolve the form → which backend, which model
   │
   └─ worker      build candidate → probe (live) → commit or raise
                     │
                     └─ emits RESULT + redraw_banner, or ERROR
   │
   └─ UI thread   drain → transcript + header
```

### Where the boundaries are

- `tools.py` knows nothing about jaz. Every tool is an ordinary function; the
  `Tools` class is a thin namespace so they can be passed as one input.
- `bridge.py` knows nothing about Textual. It emits `Event` dataclasses and never
  imports a widget.
- `session.py` knows nothing about Textual. It exposes `submit()` / `run_sync()`
  and a queue.

That is why `tests/test_agent.py` can drive a full agent loop with a scripted
model and no terminal, and `tests/test_tui.py` can render the real widget tree
headlessly.

These are enforced, not just documented. Three tests in `test_agent.py` parse
the import graph and assert that only `tui.py` imports Textual, that the
internal graph is acyclic, and that `__main__.py` imports the UI *lazily* (so
`-p` one-shot mode does not pay Textual's import cost). A refactor that quietly
breaks the layering fails there rather than in a review.

## Design notes

Four things were not obvious from the jaz documentation and shaped the code.

**The REPL is a configured component, not a hook.** `invoke()` accepts hooks
positionally; `PythonREPL` is not one. Passing it raises
`TypeError: ... not a hook`. It goes in through `jaz.configure(repl=...)`.

**The default REPL sandbox forbids being a coding agent.** Out of the box jaz
denies imports and all file access. `make_repl()` opens `allowed_attributes`,
`allowed_imports` and the read/write path allow-lists, scoping the last two to
the workspace so a stray absolute path cannot wander. This is the one place
where the blast radius is widened, so it is a named function rather than a
`["*"]` inline at the call site.

**Tool methods must be real methods.** Assigning a module function as a class
attribute leaves it unbound, so `self` arrives as the first positional argument
and `tools.read_file(path)` passes the catalog instance where a filename was
expected:

```
TypeError: argument should be a str or an os.PathLike object where __fspath__
returns a str, not 'Tools'
```

The module-level functions stay for direct use; the class delegates to them.

**`invoke()` is stateless across calls.** A conversational agent needs memory,
so each turn passes the prior turns in as a `prior_turns` input, which jaz
renders into the prompt. `tests/test_agent.py` asserts the earlier turn's text
actually reaches the next call's messages.

### Threading

jaz is synchronous; Textual is single-threaded and async. The agent therefore
runs on a daemon thread, and `EventBridge` -- a jaz `Hook` -- does nothing but
`put_nowait` onto a `queue.SimpleQueue` from that thread. A timer on the UI
thread drains the queue and is the only thing that touches widgets. No locks, no
`call_from_thread`, no ordering hazards: the queue is the entire handoff.

### Finishing a turn

The agent ends a task by calling `finish(summary)`, a plain function passed in
as a tool. `ReturnType` would have been the obvious choice, but enforcing it
means the agent must import a sentinel class to build its final value -- and
imports are exactly what the sandbox gates. A function it can just call is
always available.

## The tools

Eight tools, all in `tools.py`. In jaz a tool *is* its signature and docstring —
that text is the only description the model gets, so each docstring is one
imperative line stating when to reach for it.

| Tool                                     | Returns                     | Notes                                                         |
| ---------------------------------------- | --------------------------- | ------------------------------------------------------------- |
| `read_file(path, start_line?, num_lines?)` | numbered text             | range reads keep large files out of the context               |
| `write_file(path, content)`              | confirmation                | creates parent dirs; overwrites                                |
| `edit_file(path, old, new, replace_all?)` | confirmation               | **fails loudly** on a missing or ambiguous match              |
| `list_dir(path, pattern?)`               | one entry per line          | not recursive; explains itself when nothing matches            |
| `search_files(pattern, path?, glob?)`    | `path:line: text`           | regex over contents; skips files > 2 MB                       |
| `run_shell(cmd, timeout?)`               | stdout+stderr+exit code     | hard timeout; always reports the exit code                     |
| `git_status()`                           | porcelain output            |                                                                 |
| `git_diff(staged?)`                      | diff                        |                                                                 |

Three conventions are worth stating because they are what make the loop
recoverable rather than brittle:

**Tools raise `ToolError`, they do not return error strings.** jaz shows the
traceback to the model, which then corrects itself in the next turn. That is the
intended loop, so an error must *interrupt*, not be quietly absorbed.

**`edit_file` refuses an ambiguous match.** If `old_string` appears twice it
raises rather than picking the first. A wrong guess silently corrupts a file; a
raised error costs one turn.

**Output is clipped before it reaches the model.** `run_shell` caps at 30k
chars, `read_file` at 40k, and files over 2 MB are skipped by `search_files`. A
build log or a minified bundle would otherwise consume the context window in a
single turn.

**Paths are confined to the workspace.** `_resolve` anchors relative paths to
the workspace root and rejects anything that escapes it, so `../..` in a
generated path cannot reach outside. Absolute paths are allowed (editors need
them) but still normalized.

### The `finish` contract

The agent ends a task with `finish("summary")`. This is the *only* way a turn
completes — jaz also accepts `return`, but that produces an opaque result the
UI cannot distinguish from an intermediate value. The `guidance` input tells the
model to call `finish` with markdown summarizing what changed and what was
verified.

## Configuration

| Variable              | Default                                  | Meaning                        |
| --------------------- | ---------------------------------------- | ------------------------------ |
| `OPENROUTER_API_KEY`  | --                                       | required for the default backend (or `OR_API_KEY`) |
| `ANTHROPIC_API_KEY`   | --                                       | required for the `anthropic` backend |
| `OPENAI_API_KEY`      | --                                       | required for the `openai` backend |
| `GEMINI_API_KEY`      | --                                       | required for the `google` backend |
| `JAZ_BACKEND`         | `openrouter`                             | starting backend               |
| `JAZ_MODEL`           | `stealth/space-bunny-alpha`              | starting model                 |
| `JAZ_API_BASE`        | per-backend                              | override the API root          |
| `JAZ_HTTP_TIMEOUT`    | `180`                                    | per-request timeout, seconds   |
| `JAZ_MAX_RETRIES`     | `3`                                      | transport retries              |
| `JAZ_AGENT_ROOT`      | cwd                                      | default workspace              |

Command-line flags: `-p/--prompt`, `-w/--workspace`, `-m/--model`,
`-b/--backend`, `-n/--max-iterations`, `--log FILE`, `-v/--verbose`,
`--selftest`.

## Verified against the live model

A real run against `stealth/space-bunny-alpha`, task: *list the `*.py` files in
the current directory*.

```
>>> thinking (turn 1)
│ result = tools.list_dir('.', '*.py')
│ print(result)

  no entries in C:\...\jaz-practice match '*.py'. list_dir does not recurse --
  10 entries exist here: .pytest_cache/, .venv/, jaz_agent/, tests/, tools/,
  .gitignore, key, README.md, requirements.txt

>>> thinking (turn 2)
│ return finish("Listed `*.py` files in the current directory; found 0.")

=== RESULT ===
Listed `*.py` files in the current directory; found 0.
```

Two things this confirmed. The tool chain, the sandbox and the event bridge all
work against the real API. And the answer is *correct* — the project root has no
`.py` files, they all live in subdirectories.

It also caught a real defect. The first version replied
`"<dir> is empty (no entries matching '*.py')"`, and the model believed it,
reported zero files and finished. It never considered that `list_dir` does not
recurse. The reply now names the siblings that do exist, which is what led the
model to check before answering. See
`test_list_dir_empty_result_explains_non_recursion`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

98 tests, no network and no API key. They cover the tool catalog, path
confinement, the bridge payload, the agent loop (including recovery from a tool
error), conversation memory, the architectural layering, the command table
(usage strings, alias resolution, hint and completion output), model switching
(including that a failed probe leaves the session untouched, and that provider
errors are redacted), the model browser (pagination, substring and subsequence
matching, and that the current model is shown even when it falls off the page),
and a headless Textual run that renders the real widget tree and returns to idle.

The browser tests are worth calling out because they were written *after* the
live run found the off-page bug, and they pin the specific case rather than the
general behaviour:

```python
def test_switchmodules_shows_the_current_model_even_off_page(monkeypatch):
    """The model you are on must be visible, not buried at position 420."""
```

A test that only asserted "the model list is paginated" would have passed both
before and after the fix.

## Requirements

Python 3.12+, `jaz-lang==0.2.0a4`, `textual>=8`.