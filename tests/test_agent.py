"""Offline checks. No network, no API key.

Run everything:

    .\\.venv\\Scripts\\python.exe -m pytest tests/ -q

The point of these tests is the two seams that are easy to get wrong and
expensive to debug live: the jaz tool catalog (does the agent actually get a
bound method?) and the event bridge (does a turn end cleanly?).
"""

from __future__ import annotations

import ast
import time
import warnings
from pathlib import Path
from queue import Empty

import pytest

from jaz import MockLLMClient

from jaz_agent.bridge import Event, EventBridge, Kind, QueueDrain
from jaz_agent.session import AgentSession, make_repl
from jaz_agent.tools import ROOT, ToolError, Tools, search_files, tool_catalog

warnings.simplefilter("ignore")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def scripted(*turns: str):
    """Return a mock LLM that replies with *turns* in order, then repeats."""
    seen: list[int] = []

    def fn(model, messages, **kwargs):
        seen.append(1)
        return turns[min(len(seen) - 1, len(turns) - 1)]

    fn.seen = seen  # type: ignore[attr-defined]
    return MockLLMClient(fn=fn), fn


def drain(session: AgentSession) -> list[Event]:
    """Empty a session's event queue."""
    out: list[Event] = []
    while True:
        try:
            out.append(session.queue.get_nowait())
        except Empty:
            return out


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


def test_tool_methods_are_bound():
    """The catalog must expose bound methods.

    Assigning a plain function as a class attribute leaves it unbound, so
    ``self`` arrives as the first positional argument -- ``read_file(path)``
    would then receive the Tools instance where it expects a filename. This is
    the bug that made every tool raise ``TypeError`` inside the agent loop.
    """
    tools = Tools()
    for name in tool_catalog().split(", "):
        bound = getattr(tools, name)
        assert bound.__self__ is tools, f"{name} is not a bound method"


def test_tool_catalog_is_populated():
    """Every advertised tool exists and is callable."""
    tools = Tools()
    names = [n for n in tool_catalog().split(", ") if n]
    assert {"read_file", "write_file", "edit_file", "list_dir", "search_files", "run_shell"} <= set(names)
    for name in names:
        assert callable(getattr(tools, name))


def test_list_dir_lists_the_workspace():
    """A listing should mention the package directory we know exists."""
    out = Tools().list_dir(".")
    assert "jaz_agent" in out


def test_list_dir_reports_sizes_barely(tmp_path, monkeypatch):
    """Entries must be one per line, so the agent can parse them."""
    monkeypatch.setattr("jaz_agent.tools.ROOT", tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()

    out = Tools().list_dir(".")
    lines = out.splitlines()[1:]  # drop the path header
    assert "a.py" in " ".join(lines)
    assert "sub/" in " ".join(lines)
    # Exactly one entry per line: no prose mixed in.
    assert all("  " not in ln for ln in lines), lines


def test_list_dir_empty_result_explains_non_recursion(tmp_path, monkeypatch):
    """A pattern that matches nothing must not read as "the directory is empty".

    Regression test with a real failure behind it: the old reply was
    "<dir> is empty (no entries matching '*.py')". The model believed it,
    reported zero files for a workspace that had several, and finished the task
    on a wrong answer. list_dir is not recursive, so the reply now names the
    siblings that do exist.
    """
    monkeypatch.setattr("jaz_agent.tools.ROOT", tmp_path)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "README.md").write_text("hi", encoding="utf-8")
    (tmp_path / "pkg" / "mod.py").write_text("y = 2\n", encoding="utf-8")

    out = Tools().list_dir(".", "*.py")

    assert "is empty" not in out, "must not claim the directory is empty"
    assert "does not recurse" in out
    # The siblings that do exist must be visible, including the subdirectory
    # that actually holds the .py files.
    assert "pkg/" in out
    assert "README.md" in out


def test_list_dir_docstring_warns_about_recursion():
    """The docstring is the tool description the model reads; keep the warning."""
    assert "NOT recurse" in (Tools.list_dir.__doc__ or "")


def test_write_then_read_roundtrip(tmp_path, monkeypatch):
    """write_file then read_file must agree on the content."""
    monkeypatch.setattr("jaz_agent.tools.ROOT", tmp_path)
    monkeypatch.setattr("jaz_agent.tools.MAX_READ_CHARS", 100_000)

    tools = Tools()
    tools.write_file("notes.txt", "alpha\nbeta\n")
    assert "alpha" in tools.read_file("notes.txt")


def test_edit_file_requires_a_unique_match(tmp_path, monkeypatch):
    """An ambiguous edit must fail loudly rather than guess."""
    monkeypatch.setattr("jaz_agent.tools.ROOT", tmp_path)
    tools = Tools()
    tools.write_file("dup.txt", "x\nx\n")

    with pytest.raises(ToolError, match="not unique"):
        tools.edit_file("dup.txt", "x", "y")
    assert tools.edit_file("dup.txt", "x", "y", replace_all=True).startswith("replaced 2")


def test_read_missing_file_raises_a_clear_error(tmp_path, monkeypatch):
    """A missing file must say so, not leak an OSError."""
    monkeypatch.setattr("jaz_agent.tools.ROOT", tmp_path)
    with pytest.raises(ToolError, match="no such file"):
        Tools().read_file("nope.txt")


def test_paths_cannot_escape_the_workspace(tmp_path, monkeypatch):
    """A relative path with .. must not reach outside the root."""
    monkeypatch.setattr("jaz_agent.tools.ROOT", tmp_path)
    with pytest.raises(ToolError, match="escapes the workspace root"):
        Tools().read_file("../../etc/passwd")


def test_search_files_finds_a_match(tmp_path, monkeypatch):
    """A regex search over the workspace should locate a known string."""
    monkeypatch.setattr("jaz_agent.tools.ROOT", tmp_path)
    (tmp_path / "code.py").write_text("def needle():\n    pass\n", encoding="utf-8")
    assert "code.py" in search_files("needle", ".")


def test_run_shell_reports_the_exit_code():
    """Shell output must carry the exit status back to the model."""
    out = Tools().run_shell("exit 3")
    assert "3" in out


# --------------------------------------------------------------------------
# bridge
# --------------------------------------------------------------------------


def test_emit_does_not_nest_the_payload():
    """``emit(..., final=True)`` must land as ``payload['final']``.

    Regression test for a real bug: the dataclass field was named ``data``, so
    ``emit(..., data={...})`` was captured by the ``**kwargs`` catch-all and
    nested one level deeper. Every ``payload.get('idle')`` check then read
    ``None`` and the UI never left its busy state.
    """
    from queue import SimpleQueue

    q: SimpleQueue = SimpleQueue()
    bridge = EventBridge(q)
    bridge.emit(Kind.STATUS, "idle", idle=True)
    ev = q.get_nowait()
    assert ev.payload == {"idle": True}
    assert ev.payload.get("idle") is True


def test_queue_drain_forwards_and_survives_a_bad_sink():
    """A raising sink must not abort the drain loop."""
    from queue import SimpleQueue

    q: SimpleQueue = SimpleQueue()
    q.put(Event(Kind.STATUS, "a"))
    q.put(Event(Kind.STATUS, "b"))

    seen: list[str] = []

    def sink(ev: Event) -> None:
        seen.append(ev.text)
        raise RuntimeError("boom")

    drained = QueueDrain(q, sink)()
    assert len(drained) == 2
    assert seen == ["a", "b"]


def test_queue_drain_is_bounded():
    """A burst larger than the limit is truncated, not dropped silently."""
    from queue import SimpleQueue

    q: SimpleQueue = SimpleQueue()
    for i in range(500):
        q.put(Event(Kind.OUTPUT, str(i)))

    got: list[Event] = []
    QueueDrain(q, got.append, limit=50)()
    assert len(got) == 50


# --------------------------------------------------------------------------
# session / agent loop
# --------------------------------------------------------------------------


def test_agent_loop_runs_tools_and_finishes():
    """A full turn: tool call, tool output, then finish()."""
    mock, fn = scripted(
        "listing = tools.list_dir('.')\nprint(len(listing.splitlines()))",
        "return finish('counted the workspace')",
    )
    session = AgentSession()
    session._llm = mock
    session.max_iterations = 6

    report = session.run_sync("count things")

    assert report == "counted the workspace"
    assert len(fn.seen) == 2

    kinds = [e.kind for e in drain(session)]
    assert Kind.CODE in kinds
    assert Kind.RESULT in kinds
    assert kinds[-1] is Kind.STATUS


def test_a_tool_error_is_recoverable():
    """The loop must survive a tool raising, and show the traceback.

    This is the property the whole tool design rests on: jaz feeds the
    traceback back to the model, which then corrects itself. Without it a typo in
    a path would end the run.
    """
    mock, fn = scripted(
        "tools.read_file('definitely-missing.txt')",
        "return finish('recovered')",
    )
    session = AgentSession()
    session._llm = mock
    session.max_iterations = 6

    report = session.run_sync("read a file that is not there")

    assert report == "recovered"
    assert len(fn.seen) == 2
    outputs = [e.text for e in drain(session) if e.kind is Kind.OUTPUT]
    assert any("no such file" in o or "Error" in o for o in outputs)


def test_iteration_limit_is_reported_not_hung():
    """A model that never finishes must stop with an error, not spin forever."""
    mock, fn = scripted("x = 1")  # never returns
    session = AgentSession()
    session._llm = mock
    session.max_iterations = 3

    session.run_sync("loop forever")

    errors = [e for e in drain(session) if e.kind is Kind.ERROR]
    assert errors, "expected an error event"
    assert "IterationLimit" in errors[0].text
    assert len(fn.seen) == 3


def test_history_is_carried_into_the_next_turn():
    """Turn 2's prompt must contain turn 1's content.

    jaz's ``invoke`` is stateless, so without an explicit carry-over the agent
    has no memory of what it just did. The previous turn is passed in as a
    ``prior_turns`` input, which jaz renders into the prompt -- so the check
    that matters is whether the earlier text reaches the model's messages.
    """
    prompts: list[str] = []

    def spy(model, messages, **kwargs):
        prompts.append("\n".join(str(m.get("content")) for m in messages))
        return "return finish('ok')"

    session = AgentSession()
    session._llm = MockLLMClient(fn=spy)
    session.max_iterations = 4

    session.run_sync("ALPHA_MARKER first task")
    session.run_sync("BETA_MARKER second task")

    assert len(prompts) >= 2, "expected at least two model calls"
    assert "ALPHA_MARKER" in prompts[0], "turn 1 lost its own prompt"
    assert "ALPHA_MARKER" in prompts[-1], "turn 2 did not receive turn 1"
    assert "BETA_MARKER" in prompts[-1], "turn 2 lost its own prompt"


def test_make_repl_opens_the_sandbox_for_coding():
    """The REPL must allow attribute access, imports and workspace file IO.

    jaz's default REPL denies all three, which makes a file-editing agent
    impossible. This test pins the configuration so a future change that
    re-tightens it fails here rather than in a live run.
    """
    repl = make_repl(ROOT)
    # jaz normalizes the allow-lists to tuples in the constructor, so compare
    # against a set of the values rather than the exact container type.
    assert set(repl.allowed_attributes) == {"*"}
    assert set(repl.allowed_imports) == {"*"}
    assert repl.exec_timeout and repl.exec_timeout >= 60
    assert any(str(ROOT) in p for p in repl.allowed_read_paths)
    assert any(str(ROOT) in p for p in repl.allowed_write_paths)


def test_llm_config_requires_a_key(monkeypatch):
    """Missing credentials must fail at construction with actionable text."""
    from jaz_agent.llm_config import LLMConfigError, build_llm

    for var in ("OPENROUTER_API_KEY", "OR_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(LLMConfigError, match="openrouter.ai/keys"):
        build_llm()


def test_llm_config_prefixes_the_model(monkeypatch):
    """The model id must carry the openrouter/ prefix or LiteLLM cannot route it."""
    from jaz_agent.llm_config import build_llm

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    llm = build_llm("stealth/space-bunny-alpha")
    assert llm.model == "openrouter/stealth/space-bunny-alpha"

    # A caller who already wrote the prefix should not get it doubled.
    llm2 = build_llm("openrouter/stealth/space-bunny-alpha")
    assert llm2.model == "openrouter/stealth/space-bunny-alpha"


# --------------------------------------------------------------------------
# model switching
# --------------------------------------------------------------------------


def test_backend_routes_are_idempotent():
    """``route`` must not double a prefix the caller already supplied."""
    from jaz_agent.llm_config import resolve_backend

    backend = resolve_backend("openrouter")
    assert backend.route("foo/bar") == "openrouter/foo/bar"
    assert backend.route("openrouter/foo/bar") == "openrouter/foo/bar"
    # bare() is the inverse, so a prefixed id round-trips.
    assert backend.bare("openrouter/foo/bar") == "foo/bar"
    assert backend.bare("foo/bar") == "foo/bar"


def test_unknown_backend_lists_the_known_ones():
    """A typo must be immediately actionable, not a KeyError."""
    from jaz_agent.llm_config import LLMConfigError, resolve_backend

    with pytest.raises(LLMConfigError, match="openrouter"):
        resolve_backend("openruter")


def test_switch_model_commits_and_reports(monkeypatch):
    """A valid switch changes the session state and says so.

    Uses an injected mock so no network is touched; the point under test is the
    state transition, not the probe (which gets its own tests below).
    """
    from jaz_agent.llm_config import BACKENDS

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    session = AgentSession(llm=MockLLMClient(fn=lambda *a, **k: "return 'x'"))
    assert session.backend is BACKENDS["openrouter"]

    note = session.switch_model("openai/gpt-5-mini")

    assert session.model == "openai/gpt-5-mini"
    assert session.backend.name == "openrouter"  # same backend
    assert "gpt-5-mini" in note
    assert "same backend" in note


def test_switch_backend_and_model_together(monkeypatch):
    """Naming a backend moves the session to that provider."""
    from jaz_agent.llm_config import BACKENDS

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    session = AgentSession(llm=MockLLMClient(fn=lambda *a, **k: "return 'x'"))

    session.switch_model("claude-sonnet-4-5", backend="anthropic")

    assert session.backend is BACKENDS["anthropic"]
    assert session.model == "claude-sonnet-4-5"
    assert session.model_name == "claude-sonnet-4-5 via anthropic"


def test_switch_strips_a_prefix_from_the_wrong_backend(monkeypatch):
    """A prefixed id is normalised against the *target* backend.

    ``/model anthropic anthropic/claude-x`` and ``/model anthropic claude-x``
    must mean the same thing -- the UI routes one-word and two-word forms to
    different code paths, so the normalisation has to live in one place.
    """
    from jaz_agent.llm_config import BACKENDS

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    session = AgentSession(llm=MockLLMClient(fn=lambda *a, **k: "return 'x'"))

    session.switch_model("anthropic/claude-sonnet-4-5", backend="anthropic")
    assert session.model == "claude-sonnet-4-5"
    assert BACKENDS["anthropic"].route(session.model) == "anthropic/claude-sonnet-4-5"


def test_a_failed_probe_leaves_the_model_alone(monkeypatch):
    """The whole point of probing first: a bad target changes nothing.

    Regression test for the failure mode this feature exists to prevent -- the
    user switches to a broken model and only finds out after a full agent turn
    has been spent.
    """
    from jaz_agent.llm_config import LLMConfigError
    import jaz_agent.session as session_mod

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    calls: list[int] = []

    def fake_probe(llm):
        calls.append(1)
        return "model not found, or not available on this backend"

    monkeypatch.setattr(session_mod, "probe", fake_probe)

    session = AgentSession()  # no injected llm -> probing is live
    before = (session.backend.name, session.model)

    with pytest.raises(LLMConfigError, match="unusable"):
        session.switch_model("does/not-exist")

    assert (session.backend.name, session.model) == before, "state changed on failure"
    assert calls, "probe was never called"


def test_switch_refuses_while_a_turn_is_running(monkeypatch):
    """Swapping the backend mid-turn would mix two models in one history."""
    from jaz_agent.llm_config import LLMConfigError

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    session = AgentSession()
    session.running = True
    try:
        with pytest.raises(LLMConfigError, match="while a turn is running"):
            session.switch_model("some/other-model")
    finally:
        session.running = False


def test_switch_requires_a_model_name(monkeypatch):
    """An empty target with no backend named is a usage error."""
    from jaz_agent.llm_config import LLMConfigError

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    session = AgentSession()
    with pytest.raises(LLMConfigError, match="no model given"):
        session.switch_model("")


def test_bare_backend_switch_resolves_that_defaults_model(monkeypatch):
    """``switch_model(backend=...)`` with no model must pick that backend's default.

    Regression test with a live failure behind it: this raised "no model given"
    because the default was resolved in the TUI layer only, so the session API
    and the UI disagreed. The rule now lives in one place -- here.
    """
    from jaz_agent.llm_config import BACKENDS, default_model_for

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    session = AgentSession(llm=MockLLMClient(fn=lambda *a, **k: "return 'x'"))

    session.switch_model(backend="anthropic")

    assert session.backend is BACKENDS["anthropic"]
    assert session.model == default_model_for(BACKENDS["anthropic"])


def test_default_model_for_prefers_the_env_override(monkeypatch):
    """JAZ_MODEL must win over the built-in default for OpenRouter."""
    from jaz_agent.llm_config import default_model_for, resolve_backend

    monkeypatch.setenv("JAZ_MODEL", "some/other-model")
    assert default_model_for(resolve_backend("openrouter")) == "some/other-model"


def test_bare_backend_switch_reports_a_missing_key(monkeypatch):
    """Moving to a provider with no credential must say which variable is missing."""
    from jaz_agent.llm_config import LLMConfigError, resolve_backend

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    session = AgentSession()  # no injected llm -> the real build path runs
    before = (session.backend.name, session.model)

    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        session.switch_model(backend="anthropic")

    assert (session.backend.name, session.model) == before, "state changed on failure"


def test_probe_translates_common_failures():
    """The probe's job is to say *why*, not to dump a provider traceback."""
    from jaz_agent.llm_config import _describe

    class AuthenticationError(Exception):
        pass

    assert "authentication" in _describe(AuthenticationError("401 unauthorized"))
    assert "not found" in _describe(Exception("model does not exist"))
    assert "rate limited" in _describe(Exception("429 rate limit exceeded"))
    assert "credits" in _describe(Exception("insufficient credits"))
    assert "network" in _describe(Exception("connection refused"))
    # An unknown error still says something, and stays one line.
    generic = _describe(ValueError("line one\nline two"))
    assert "ValueError" in generic
    assert "\n" not in generic


def test_probe_names_a_model_rejected_as_invalid():
    """OpenRouter's actual wording must map to the friendly message.

    Real payload, captured from a live probe:
        litellm.BadRequestError: OpenrouterException - {"error":{"message":
        "definitely/not-a-real-model-xyz-123 is not a valid model ID","code":400}}
    The earlier wording-matching missed this and leaked litellm's wrapper text
    to the user verbatim.
    """
    from jaz_agent.llm_config import _describe

    raw = (
        'litellm.BadRequestError: OpenrouterException - {"error":{"message":'
        '"foo/bar is not a valid model ID","code":400},"user_id":"user_30ABC"}'
    )
    message = _describe(Exception(raw))
    assert message == "model not found, or not available on this backend"


def test_probe_redacts_account_identifiers():
    """A provider error must not leak the account's user id into the transcript.

    OpenRouter embeds ``user_id`` in every error payload. Unredacted, it would
    land in the TUI and in any --log file the user keeps around.
    """
    from jaz_agent.llm_config import _describe, _redact

    raw = '{"error":{"message":"boom"},"user_id":"user_30LPZNXKOSuDyOqBHcP39AfNLr7"}'
    cleaned = _redact(raw)
    assert "30LPZNXKOSuDyOqBHcP39AfNLr7" not in cleaned
    assert "user_id" in cleaned  # the key stays, the value goes

    # And through the fallback path, where an unrecognised error is passed on.
    message = _describe(Exception(raw))
    assert "30LPZNXKOSuDyOqBHcP39AfNLr7" not in message


def test_redact_leaves_ordinary_text_alone():
    """Redaction must not mangle a message that has nothing sensitive in it."""
    from jaz_agent.llm_config import _redact

    assert _redact("connection reset by peer") == "connection reset by peer"


def test_model_listing_falls_back_when_offline(monkeypatch):
    """Discovery failure must degrade the menu, not break it."""
    import jaz_agent.llm_config as cfg

    monkeypatch.setattr(cfg, "_list_openrouter_models", lambda backend: [])
    models = cfg.list_models(cfg.resolve_backend("openrouter"))
    assert models, "expected the well-known fallback"
    assert cfg.DEFAULT_MODEL in models


def test_available_backends_reports_key_presence(monkeypatch):
    """The /backends menu marks which providers are usable right now."""
    import jaz_agent.llm_config as cfg

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    seen = dict(cfg.available_backends())
    assert seen["openrouter"] is True
    assert seen["anthropic"] is False


# --------------------------------------------------------------------------
# architecture
#
# These are not behaviour tests. They pin the two structural properties the
# README claims and the rest of the suite relies on: only the UI knows about
# Textual, and the internal import graph has no cycles. A refactor that breaks
# either would still pass every other test here, so they are checked explicitly.
# --------------------------------------------------------------------------

_SRC = Path(__file__).resolve().parent.parent / "jaz_agent"


def _imports_of(py: Path) -> tuple[set[str], set[str]]:
    """Return (top-level external modules, internal module names) for one file."""
    tree = ast.parse(py.read_text(encoding="utf-8"))
    external: set[str] = set()
    internal: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("jaz_agent."):
                    internal.add(alias.name.split(".")[1])
                else:
                    external.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.level == 1:
                internal.add(node.module.split(".")[0])
            elif node.level == 0 and not node.module.startswith("jaz_agent"):
                external.add(node.module.split(".")[0])

    return external, internal - {py.stem}


def test_only_the_ui_imports_textual():
    """Textual must stay confined to tui.py.

    This is what lets the agent loop be tested with no terminal, and the bridge
    be tested with no UI.
    """
    offenders = []
    for py in sorted(_SRC.glob("*.py")):
        if py.name == "tui.py":
            continue
        external, _ = _imports_of(py)
        if any(name.startswith("textual") for name in external):
            offenders.append(py.name)
    assert not offenders, f"textual leaked into {offenders}"


def test_internal_import_graph_has_no_cycles():
    """Module dependencies must be acyclic, or the layering claim is false."""
    edges = {py.stem: _imports_of(py)[1] for py in sorted(_SRC.glob("*.py"))}
    edges = {k: v for k, v in edges.items() if v}

    for start in edges:
        seen: set[str] = set()
        stack = list(edges[start])
        while stack:
            node = stack.pop()
            if node == start:
                pytest.fail(f"import cycle through {start}")
            if node not in seen:
                seen.add(node)
                stack.extend(edges.get(node, ()))


def test_only_the_entrypoint_reaches_the_ui():
    """Only __main__ may import tui, and it imports it lazily.

    The entry point has to dispatch to the UI, so an import there is correct.
    What must not happen is the *agent layers* reaching up into it, and what
    must not happen either is a module-level import in __main__: Textual is
    slow to import, so `-p` one-shot mode would pay for it unnecessarily.
    """
    for py in sorted(_SRC.glob("*.py")):
        if py.name in ("tui.py", "__main__.py"):
            continue
        _, internal = _imports_of(py)
        assert "tui" not in internal, f"{py.name} imports tui"

    # The __main__ import of .tui must sit inside a function body.
    main_src = (_SRC / "__main__.py").read_text(encoding="utf-8")
    tree = ast.parse(main_src)
    module_level: set[str] = set()
    for node in tree.body:  # top level only, not nested in def/if/with
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "tui":
            module_level.add("tui")
    assert not module_level, "__main__ imports tui at module level; make it lazy"