# 还不能用
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
| `Enter`   | run the task, or the highlighted command |
| `/`       | show the matching commands |
| `↑` `↓`   | move the selection in that list |
| `Tab`     | complete the current slash command |
| `Esc`     | dismiss the list without touching the input, or drop the text selection |
| drag      | select text in the transcript -- `Ctrl+C` then copies it |
| `Ctrl+C`  | copy the selection, or cancel the running turn when there is none |
| `Ctrl+L`  | clear the transcript      |
| `F5`      | show cost                 |
| `Ctrl+Q`  | quit                      |

Commands: `/help` `/status` `/switchmodules [model|@backend]` `/backends` `/new` `/copy [all]`
`/clear` `/cancel` `/cost` `/context` `/quit`. `/help` prints the full list.

### Copying text out of the TUI

Selecting with the mouse works, and `Ctrl+C` copies what is selected and does
nothing else; with nothing selected it still cancels the running turn, which is
the only way to stop one mid-flight. The status bar says how much is selected
while the selection is live, and how much went to the clipboard afterwards -- a
copy that quietly took nothing looks exactly like a copy that worked.

The mouse cannot reach most of what is worth keeping: only the visible window
can be dragged over, and the transcript holds up to 2000 lines. So there is a
command for the rest:

* `/copy` puts the last reply on the clipboard. It sends the text the agent
  finished the turn with, not a read-back of the screen, so it arrives without
  the `✓` marker and without the line breaks wrapping added.
* `/copy all` sends the whole transcript, markers and all.

Two mechanisms are used, because the two situations need different ones.
`Ctrl+C` writes an OSC 52 escape sequence, which is the only thing that can
work when the terminal is on another machine -- over ssh, in tmux -- and the
only thing a process can do without touching the OS. It is also a silent no-op
in most terminals on Windows, so the local clipboard is written directly as well
(`jaz_agent/clipboard.py`, Win32 through ctypes). Whichever applies, the copy is
reported in the status bar.

That the transcript can be selected at all is a fix, not a given: `RichLog` is
the one scrolling text widget in Textual that does not implement selection. Its
predecessor `Log` answers `get_selection`, tags every rendered strip with the
coordinates of each character and paints the selected span itself, and the
general path does the same for widgets that render a `Visual`. `RichLog` renders
its own strips, so a drag produced no highlight, `Screen.get_selected_text()`
returned the empty string, and `Ctrl+C` copied nothing *while also* no longer
cancelling -- Textual could see that something was selected, so it had no reason
to fall through to the next binding. `Transcript` implements the missing three
pieces.

Every action has a slash command, and the shortcut keys are only duplicates of
one. That was not true when a function key could clear the conversation on its
own: F2 dropped the session, but the only way to learn that was to read the
footer, and once pressed there was nothing to undo. `Ctrl+C`, `Ctrl+L` and
`Ctrl+Q` kept their keys because you press them mid-turn, when typing is
awkward; everything else is a command. A test walks the app's `action_*` methods
and fails if one has no command, so the next binding added without one will not
pass review.

`/model` was removed. It overlapped `/switchmodules` for browsing and for
switching within a backend, and the two entry points meant two things to keep
in sync. Its one distinct job -- naming a *provider* -- moved into the surviving
command as an argument: `/switchmodules @anthropic`, or a bare backend name
(`/switchmodules anthropic`). Nothing was lost, and `-b/--backend` stays for
scripting.

The name redirects rather than disappearing silently:

```
» /model
✗ /model is gone — use /switchmodules
```

`commands.RETIRED` holds that mapping, and the same redirect appears in the
inline hint while typing. It is not the alias mechanism: an alias still exists,
a retired name is gone and the user has to be told where it went. Deleting a
command does not delete the muscle memory, and answering a command that used to
work with `unknown command` would be true and useless.
`test_model_is_not_a_command` pins the removal, and
`test_retired_commands_point_at_their_replacement` checks every redirect names
a command that exists — a redirect to a dead command is a dead end with extra
steps.

Typing `/` opens an inline list of the commands that match what you have typed
so far, `↑`/`↓` move the selection, and `Enter` runs whatever is highlighted:

```
» /swi
▸  /status                           current model, workspace and tool list
   /switchmodules [model|@backend]   browse and switch models, across backends
 1-2 of 2
↑/↓ choose · Enter run · Esc dismiss
```

Both are driven by the same table in `commands.py`, so a command cannot exist in
the UI without existing in `/help`, and the three views cannot drift apart.

Four details that the first version got wrong:

- **The list is vertical, one command per line.** It was a single space-joined
  line, which ran past the screen width and wrapped wherever the terminal
  happened to break it — interleaving one command's summary with the next
  command's name. A column cannot do that.
- **`Tab` completes.** Textual 8's `Input` binds no Tab key at all; a
  suggestion is accepted with the right arrow, which suits a single-line text
  field and not a command prompt. `Screen` claims Tab for `app.focus_next`, so
  before `PromptInput` existed Tab moved focus off the prompt and completed
  nothing. `PromptInput` binds it to `action_accept_completion`, which — unlike
  `cursor_right` — does nothing when there is no suggestion, so Tab on ordinary
  task text will not nudge the caret.
- **`↑`/`↓` fall through to the input.** The prompt forwards them to the app,
  which hands them back when no list is open. Swallowing them would leave no
  way to move the caret over text that was mistyped.
- **`Enter` runs the highlight, but only if you moved it.** With the selection
  pinned at row 0, submitting an untouched `/s` would silently become
  `/status`. The chosen value is the command *name*, never the signature:
  `/switchmodules [model]` submitted verbatim would be rejected, because
  `[model]` is a placeholder for the reader and not part of the syntax.

On a terminal too narrow for the summary column, the summary moves to its own
indented line and wraps to fit. The longest summary is 47 characters, so this
branch is not hypothetical.

### The markup trap

`Static.update` renders its argument as Rich **markup**, so square brackets in
it are read as tags. That silently ate every usage string: `/switchmodules
[model]` displayed as `/switchmodules`, with the argument shape gone and the
model completion for that command undiscoverable. Everything passed to the hint
is now escaped first and the style added around the result.
`test_hint_shows_the_usage_string` pins it.

## Build the exe

The deliverable is a single file that runs on a machine with no Python:
`dist\jaz-agent.exe`, built from `jaz_agent.spec`.

```powershell
.\build-exe.ps1
```

The script runs the offline suite, builds, deploys, and smoke-tests the deployed
copy -- in that order, stopping at the first failure. Switches: `-SkipTests`,
`-SkipDeploy`, `-DeployTo <dir>`.

What it runs, if you would rather type it yourself:

```powershell
# 1. the offline suite (a stub key keeps it away from the live catalogue)
$env:OPENROUTER_API_KEY = 'sk-or-test'
.\.venv\Scripts\python.exe -m pytest -q -p no:warnings

# 2. the exe
.\.venv\Scripts\python.exe -m PyInstaller jaz_agent.spec --noconfirm

# 3. deploy -- the copy on PATH is the one that actually runs
Copy-Item .\dist\jaz-agent.exe C:\Users\zxc74\bin\jaz-agent.exe -Force

# 4. prove the deployed copy starts
C:\Users\zxc74\bin\jaz-agent.exe --selftest
```

**`dist\` is staging, not the app.** `jaz-agent` on PATH resolves to
`C:\Users\zxc74\bin\jaz-agent.exe`, so a rebuild that stops at `dist\` leaves the
old build in place -- which is indistinguishable from "the fix did not work".
Deploy, then test the *PATH* copy. Compare timestamps to be sure:

```powershell
Get-Item .\dist\jaz-agent.exe, C:\Users\zxc74\bin\jaz-agent.exe |
  Select-Object FullName, LastWriteTime, Length
```

Onefile compresses its payload, so a stale exe cannot be spotted by grepping it
for a new string or a new function name. Timestamps are the only cheap check.

### What the spec carries, and why

The spec exists because the interesting part of the build is the list of things
static analysis cannot see. A `--onefile --hidden-import=...` command line would
be shorter and would rot: nobody can diff it against the reason each entry is
there.

* `collect_data_files("litellm")` -- its pricing and provider tables are read at
  runtime, not imported. A build without them starts, prints a traceback about a
  missing `model_prices.json` and dies.
* every litellm provider adapter (`collect_submodules`) -- `/backends` can reach
  four providers at runtime and the frozen build cannot know which one the user
  picks; plus `litellm.llms.openrouter.chat` explicitly, since the default path
  must survive even if litellm's own scan misses it.
* `collect_data_files("textual")` -- built-in CSS and glyphs; without them the
  app renders every box blank.
* `tiktoken_ext.openai_public` in `hiddenimports` -- **the one that actually
  bit.** tiktoken resolves that plugin *by name at runtime*, so no import
  statement anywhere points at it and PyInstaller's graph never sees it. Without
  the entry the frozen app raises

  ```
  Unknown encoding cl100k_base. Plugins found: []
  ```

  and because litellm counts tokens while *preparing* the request, it wraps that
  in `APIConnectionError` -- so the UI said "could not reach the provider" on a
  machine whose network was fine. If a frozen build reports a network fault the
  interpreter does not, suspect this list before the network.

## Browsing models

`/switchmodules` is the catalogue browser, and the list is navigable: `↑`/`↓`
move the selection, `Enter` switches to whatever is highlighted, `Esc` closes
it without changing anything.

```
» /switchmodules
 464 model(s) on openrouter:

▸    aion-labs/aion-2.0
     aion-labs/aion-3.0
     aion-labs/aion-3.0-mini
     …
 55-72 of 464
↑/↓ choose · Enter switch · Esc dismiss
```

**The whole catalogue is in the list; the viewport is not.** This was not true
once. The browser took the first twenty ids and printed `… and 443 more`, which
is not pagination — the other 444 rows could not be reached by arrow, by
command, or by any other means, because there was no way to ask for page two.
A count of what is hidden is not access to it.

So nothing is truncated. `#hint` is a fixed-height scrolling viewport and only
the lines around the cursor are drawn, which means `↓` walks all 464 the way
`↓` walks a shell history — the list scrolls, with no separate "next page"
concept to learn, and moving to row 400 costs the same as moving to row 1. The
`55-72 of 464` line is not decoration: a scrolling window gives no other sign
of how much is left, so without it a filtered three-row list is
indistinguishable from the first screen of four hundred and sixty-four.

**The list opens on the model you are already on.** With 464 ids returned
sorted, `stealth/space-bunny-alpha` sits near the end, so opening on row 0 would
mean the first ↓ picks a model the user was not looking at. Pinning the current
model to the top of the list is worse — it puts the selection at an index
unrelated to the page order, so ↓ lands somewhere arbitrary. Instead the cursor
lands on the current model. A *filtered* list may not contain it, and then the
header says so, because otherwise the browser appears to be describing the
model you are on when it is not.

**Choosing a row is not a licence to skip validation.** The row came from a
catalogue fetch, but a catalogue is a menu, not a promise. Enter routes back
through `/switchmodules <model>`, so the live probe, the "not switched" wording
and the catalogue-rejected case are exactly the ones the typed command gets.
Two paths to the same switch would drift, and the probe is the part that must
not be skipped.

The rows are built on the **UI thread**, not in the worker that fetched them.
The worker sends the ids through the event payload and the drain turns them into
a list. That is not incidental: `RichLog` is append-only, and a list you can
arrow through cannot be built out of an append-only log.

With an argument the browser filters instead of switching, so a typo shows you
what you probably meant:

```
» /switchmodules gpt-5
 47 model(s) on openrouter matching 'gpt-5':

▸    openai/gpt-5
     openai/gpt-5-image
     openai/gpt-5-mini
     …
 1-10 of 47
↑/↓ choose · Enter switch · Esc dismiss
```

Matching is **substring first, then subsequence**. Substring alone is the
obvious choice and the wrong one — nobody types `gpt-5` when the catalogue calls
it `gpt-5.1-codex-mini`, and nobody can guess `stealth/space-bunny-alpha` from
memory. The subsequence fallback reads `gpt5` as `g-p-t-5` and finds it. Order
is respected, so `5gpt` matches nothing rather than everything.

The exact name still switches rather than browsing, because that is almost
always what was meant:

```
/switchmodules gpt-5              → filter
/switchmodules openai/gpt-5-mini  → switch
```

The switch itself validates before committing (below), so a listing can never
leave you on a model that does not work.

## Switching models

Switching happens through `/switchmodules`:

```
/switchmodules                     # browse the catalogue
/switchmodules gpt-5               # filter
/switchmodules openai/gpt-5-mini   # switch, validated live first
/switchmodules @deepseek           # switch provider, to its default model
/switchmodules @deepseek deepseek-v4-pro   # provider *and* model
/backends                          # which providers have a key right now
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

Switching *provider* is part of the same command: `@<backend>` marks one
explicitly, and a bare backend name means the same thing.

```
/switchmodules @deepseek                    # that provider's default model
/switchmodules deepseek                     # the same request
/switchmodules @deepseek deepseek-v4-pro    # a specific model on it
```

The marker is needed rather than inferring the provider from the model id,
because a routed id is genuinely ambiguous: `deepseek/deepseek-v4-pro` is one of
OpenRouter's models *and* what DeepSeek's own `deepseek-v4-pro` routes to. A rule
like "the prefix names the backend" would silently move existing
`/switchmodules openai/gpt-5-mini` -- today a switch *within* OpenRouter -- onto
the OpenAI backend and its key. `@` is the marker because no provider uses one in
a model id. Rows belonging to another backend are tagged with it in the browser
(`@deepseek deepseek-v4-pro`), so what is displayed is what the command accepts;
rows on the backend you are already on stay bare.

`-b/--backend` and `JAZ_BACKEND` remain for scripting, and
`session.switch_model(model, backend=...)` remains the API-level way in.

Backends on offer: `openrouter` (default), `anthropic`, `openai`, `google`,
`deepseek`. Each lists the environment variables it accepts in
`llm_config.BACKENDS`, and `/backends` shows which ones have a key *in this
process* right now.

### Verified live

The switch path was run against the real API, not only against mocks -- captured
2026-10-06, when the catalogue still carried the 73 `:batch` rows that are
filtered out now:

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

Four real defects surfaced only because the live run existed, all now covered
by tests:

- **A bare `switch_model(backend=...)` failed.** The "use that backend's default
  model" rule lived in the TUI layer, so the session API and the UI disagreed
  about what a bare backend name means. The rule now lives in `switch_model`.
  (The command that used it, `/model`, has since been removed; the session API
  still takes a backend and is still tested.)
- **OpenRouter leaks the account's `user_id` in its error payloads.** An
  unrecognised provider error was being shown verbatim, which would have put
  that id in the transcript and in any `--log` file. `_redact` now blanks
  identifier-shaped JSON values, and the wording matcher learned OpenRouter's
  actual phrasing (`is not a valid model ID`) instead of only `model not found`.
- **The current model could be off the visible page.** 464 models sorted, current
  one near the end, and browsing told you neither where you were nor how far in
  you were. It was first printed above the page as `← current`; when the list
  became navigable that became a cursor that starts on it, plus a
  `not on this page` line in the header when it is not there to select.
  the off-page case specifically.

Two of the browser's own bugs were subtler. The first "is this a switch or a
search?" check tested for the absence of `/` and spaces in the argument — which
is true of *no* real model id, since every one contains a `/`. It filtered
everything, including exact names. The check is now catalogue membership. And
the switch worker reported failures by emitting an error rather than raising, so
the caller's fall-back path waited on an exception that never arrived; the
worker and the fall-back now agree.

A third came out of removing `/model`, which shortened the longest command
signature and shifted the hint's width thresholds. The two-column layout was
chosen from the *space left over* rather than from the longest summary, so at
56–70 columns it said "fits" for a short summary while the long one overflowed
and wrapped — a band that neither the wide nor the narrow test could see. The
branch now checks the longest summary against the width, and
`test_hint_never_overflows_at_any_width` sweeps 40–120.

Making the list navigable then exposed two more, both of which had been there
all along:

- **The usage strings were never actually displayed.** The hint is Rich markup,
  so `/switchmodules [model]` lost its `[model]` to the tag parser and rendered
  as `/switchmodules`. The argument shape — and the hint that model ids
  complete there — had been invisible since the hint was first written.
- **`canonical("/help")` returned `None` while `complete("/s")` matched.** One
  stripped a leading slash and the other did not. Nothing hit it until the
  navigation code tried to feed a candidate's name back through the resolver.

`test_candidate_name_never_carries_the_usage_placeholder` exists because of a
near-miss here: the first version submitted the *signature* when a row was
chosen, which would have run `/switchmodules [model]` and been rejected as an
unknown command. The row type now carries `name` and `signature` separately,
and the test asserts the submitted value resolves to a real command.

Extending the same navigation to the model browser then surfaced one more,
which only the live run could reach because it needs a filter matching exactly
one model:

- **A one-row list was unusable.** The command list honours its highlight only
  after the user *moves* it — correct, because the list appears the instant a
  slash is typed and a highlight alone means nothing. That same gate applied to
  the browser made `/switchmodules space-bunny` a dead end: one row, nowhere to
  arrow, Enter submitted nothing. The browser was opened deliberately and always
  starts on the current model, so its row 0 is always the right answer.
  `test_a_single_model_row_can_still_be_chosen` pins the difference.

And one that was not a defect but a missing feature wearing its clothes:

- **"showing 20 of 464" was not pagination.** It read like it was, which is what
  made it survive review: it states the count and states the truncation, so
  every check passed while the other 444 models had no route to them at all.
  There was no page two, no offset, no filter that would find them — the 20-row
  cap was a slice of the data rather than a window onto it. Counting what is
  hidden is not access to it. The fix is not to add paging commands; it is to
  stop truncating and let the viewport scroll, which also removes the
  "the first twenty, alphabetically" problem entirely.
  `test_every_model_is_reachable_by_scrolling` walks to row 464 and asserts the
  last row is actually reachable and drawn.

## Architecture

Seven modules, ~3,000 lines. The shape is dictated by one fact: **jaz is
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
| `tui.py`          |  1050 | Widgets, event loop, status bar, slash commands, the scrolling model browser. The only module that touches widgets. |
| `tools.py`        |   294 | The 8 tools the agent calls. Path confinement lives in `_resolve`.     |
| `llm_config.py`   |   274 | The backend registry: routes, keys, model discovery, the switch probe, error redaction. |
| `session.py`      |   260 | One `invoke` per turn on a worker thread. Owns the REPL sandbox and the current model. |
| `context.py`      |   367 | Conversation memory: what is recorded, what is condensed, and what is sent. `/context` renders it. |
| `bridge.py`       |   197 | `jaz` `Hook` → `Event` objects on a queue. Never blocks, never raises. |
| `commands.py`     |   261 | The command table and the navigable row type, shared by the command list and the model browser. One source for dispatch, `/help`, the vertical list and Tab completion. |
| `__main__.py`     |   135 | CLI parsing and dispatch.                                               |
| `check.py`        |    61 | Live end-to-end smoke test.                                             |

Dependencies point one way: `tui` → `session` → `bridge` + `tools` +
`context` + `llm_config`, with `tui` → `commands`. `context.py` imports
nothing but the standard library, so the memory policy can be tested without a
model or a terminal. Nothing below `tui` imports Textual, which is what makes
the lower two thirds testable without a terminal.

`commands.py` is deliberately pure — a tuple of frozen dataclasses, a
navigation row type, and a few functions, no imports. The table drives the
dispatcher, the help text, the inline list and the suggester, so a command
cannot be reachable in the UI while missing from `/help`.

`candidates()` and `hint_text()` both render from the same `_blocks()`, split
into one block per command. That split is what the highlight needs: a row whose
summary wraps occupies several display lines, and marking only its first line
would cover part of one command and part of the next. Deriving both views from
one function means they cannot disagree about which lines belong to whom —
`test_candidates_use_the_same_layout_as_hint_text` asserts exactly that, because
the failure mode is two layouts agreeing on the line count while disagreeing on
the ownership.

`CommandSuggester` and `PromptInput` live in `tui.py` rather than beside the
table because they subclass Textual classes, which would drag the framework
into the module that everything else imports from.

### One turn, end to end

1. `on_input_submitted` fires on the Textual thread → `_send`.
2. `AgentSession.submit` spawns a **daemon thread**, returns. (The turn is
   recorded in `Context` once it completes, not on submit — see
   [Context and compression](#context-and-compression).)
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

### Context and compression

`jaz.invoke` is stateless — every call is a fresh loop with a fresh context. A
conversational agent has to supply the memory itself, and `context.py` is where
that happens. The input is `prior_turns`, which jaz renders into the user
message as a single `<prior_turns type="list">` block.

Two decisions carry the module.

**The record is unbounded; only the prompt is bounded.** Every completed turn
appends to `Context.turns` for the life of the session, and `/new` is the only
thing that clears it. What is capped is `Context.prompt()` — the 12,000
characters that actually reach the model. Losing the record and losing the
ability to recall it are different failures, and only the second is a bug.

**Compression has to be worth it.** Condensing is not automatically an
improvement. A digest carries fixed overhead — its header, plus `asked:`/`did:`
per turn — so a session of short turns comes out *larger* condensed than
verbatim. `WORTHWHILE` refuses the digest in that case, because a summary that
saves nothing while removing detail is strictly worse than no summary. Measured
break-even is around 420 characters of report; real reports are 800–2,000.

What survives a prompt:

| Turn age | Treatment |
| -------- | -------- |
| last 4 (`RECENT_TURNS`) | verbatim, always |
| next 8 (`DIGEST_MAX_TURNS`) | condensed, detail graded by age — newest keeps all of it, oldest a quarter |
| older than that | counted, not described |
| still over budget | dropped from the oldest end; the newest turn is kept even so |

Grading detail by age is what makes the digest *scale*. A flat clip does not
compress: at a fixed 240/120 characters a 900-character report keeps a third of
itself, and 26 turns came to 8,893 characters against a 6,000 cap — the digest
was rejected outright and 24 turns dropped. Capping the number of turns
described, rather than the size of each, is the only thing that fits.

Two structural choices matter for correctness. Prompts are assembled from
*blocks* rather than a flat message list, so trimming sheds a whole turn rather
than landing between a turn's `user` and `assistant` halves and producing a
transcript that opens with the model answering itself. And `build()` returns the
messages and a `Plan` describing what it did **together**, so `/context` reports
the prompt's real behaviour instead of re-deriving it — an earlier version
counted assistant messages and cheerfully reported 26 turns "dropped" on a
prompt that contained all 30, condensed. A summary of the context that lies
about the context is the one failure this module cannot have.

`/context` renders the result:

```
Context: 8 turn(s) recorded, 4 condensed, 4 verbatim.

  record              8,140 chars   full history, always kept
  next prompt         5,140 chars   what gets sent to the model
  budget             12,000 chars
  verbatim window         4 turns
  -> 3,000 chars condensed away

Turns, oldest first:
    1. [condensed] refactor the parser in module 0 so it handles nested
    2. [condensed] refactor the parser in module 1 so it handles nested
    3. [condensed] refactor the parser in module 2 so it handles nested
    4. [condensed] refactor the parser in module 3 so it handles nested
    5. [ verbatim] refactor the parser in module 4 so it handles nested
    6. [ verbatim] refactor the parser in module 5 so it handles nested
    7. [ verbatim] refactor the parser in module 6 so it handles nested
    8. [ verbatim] refactor the parser in module 7 so it handles nested
```

### Why a queue and not `call_from_thread`

`call_from_thread` marshals a callable onto the UI loop. That works, but it
means the agent thread blocks until the UI is free, and any exception in a
widget handler surfaces as an agent failure. A `SimpleQueue` gives the same
ordering guarantee with none of that coupling: the producer only ever appends,
and a rendering bug cannot abort the drain loop because `QueueDrain` swallows
per-event exceptions.

### Switching models is a threaded operation too

The probe is a live network call, so a switch cannot run on the UI thread. It
follows the same rule as the agent: the worker (`_try_switch_then_filter`)
emits onto the queue and the drain renders. A committed switch sets
`redraw_banner` on its event, because only the UI thread may redraw the header —
the same reason the header could not be updated inside the worker directly.

```
/switchmodules <model>
   │
   ├─ UI thread   dispatch: catalogue hit → switch, miss → filter
   │
   └─ worker      fetch catalogue → emit the page as *ids*
                     │
                     └─ probe (live) → commit, or ERROR
   │
   └─ UI thread   drain → transcript header + navigable list
                     │
                     ├─ ↑/↓ move the cursor over the ids
                     ├─ Enter → route back through the same worker
                     └─ Esc  → close
```

The last arrow is the point. Enter does not apply the model directly; it calls
`/switchmodules <model>` again, so a row picked with the arrow takes exactly the
same path as one typed by hand — including the probe.

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
| `DEEPSEEK_API_KEY`    | --                                       | required for the `deepseek` backend |
| `JAZ_BACKEND`         | `openrouter`                             | starting backend               |
| `JAZ_MODEL`           | `stealth/space-bunny-alpha`              | starting model                 |
| `JAZ_API_BASE`        | per-backend                              | override the API root          |
| `JAZ_HTTP_TIMEOUT`    | `180`                                    | per-request timeout, seconds   |
| `JAZ_MAX_RETRIES`     | `3`                                      | transport retries              |
| `JAZ_MAX_TOKENS`      | model's own maximum                      | cap on requested output; set it when the account cannot *afford* the model's full `max_tokens` (OpenRouter refuses such a request with a 402) |
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

241 tests, no network and no API key. They cover the tool catalog, path
confinement, the bridge payload, the agent loop (including recovery from a tool
error), conversation memory, the architectural layering, the command table
(usage strings, alias resolution, help output, the vertical list at every width
from 40 to 120, and that the removed `/model` has not crept back), keyboard
navigation of both lists (↑/↓ move, wrap and scroll a viewport over hundreds of
rows, that the command list submits only after the highlight was moved while the
model browser always takes it, Escape closes either list and the command one
reopens on the next keystroke, that the arrow keys still edit ordinary text, and
that typing displaces an open model list), Tab completion (that it completes
rather than moving focus, that it opens the argument once the name is whole, and
that it is inert on ordinary text), that every command has a handler and every
action has a command, model switching (including that a failed probe leaves the
session untouched — whether the model was named or picked — and that provider
errors are redacted), the model browser (that all 464 rows are held and the last
one is reachable by scrolling, opening on the current model, marking it,
pagination, substring and subsequence matching, and that the current model is
accounted for when a filter excludes it, and that `:batch`-only ids are never
offered), that the model shown in the header and the status line follows the
session rather than the event that changed it, copying (that a drag names the
characters under it and not the whole widget, that the selected span is the span
that gets painted, that `Ctrl+C` copies it and still reaches `cancel` when there
is nothing selected, and what `/copy` and `/copy all` send), and a headless
Textual run that renders the real widget tree and returns to idle.

The clipboard is the one part the suite cannot fake entirely, so it does both:
the TUI tests stub the native write and assert it was called with the selected
text, and `tests/test_clipboard.py` does a round trip through the real Windows
clipboard -- reading it back as code units, so that "did the `\n` become
`\r\n`?" is a real question -- and puts back whatever was there first.

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