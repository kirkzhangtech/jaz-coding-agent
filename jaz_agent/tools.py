"""Filesystem + shell tools exposed to the jaz agent.

In jaz a "tool" is just a function passed as a keyword argument to ``invoke``.
Its signature and docstring are what the model reads, so the docstrings below are
the actual tool descriptions -- keep them one line and imperative.

Every tool returns a plain string. jaz renders the return value into the REPL
namespace and feeds it back to the model on the next turn, so strings keep the
loop simple and readable.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path

# Output is truncated before it ever reaches the model. A runaway build log or a
# minified bundle would otherwise eat the context window in one turn.
MAX_READ_CHARS = 40_000
MAX_BASH_CHARS = 30_000
MAX_LS_ENTRIES = 500
MAX_GREP_MATCHES = 200

# Resolved once at import so a tool call cannot be tricked into escaping the
# workspace by a relative path containing "..".
ROOT = Path(os.environ.get("JAZ_AGENT_ROOT", Path.cwd())).resolve()


class ToolError(Exception):
    """Raised for conditions the model should see and correct, not crash on."""


def _resolve(path_like: str) -> Path:
    """Resolve *path_like* against the workspace root, refusing escapes.

    Absolute paths are allowed (editors and terminals routinely need them) but
    still normalized. Relative paths are anchored to :data:`ROOT` so the agent
    behaves the same no matter what the process cwd drifted to.
    """
    p = Path(path_like).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    if p != ROOT and ROOT not in p.parents:
        raise ToolError(
            f"path escapes the workspace root ({ROOT}): {path_like!r}"
        )
    return p


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    return f"{text[:limit]}\n... [truncated {dropped} chars]"


def _read_text(path: Path) -> str:
    """Read *path* as UTF-8 with replacement, so binary reads cannot raise."""
    return path.read_bytes().decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


def read_file(path: str, start_line: int | None = None, num_lines: int | None = None) -> str:
    """Read a text file. Returns it with 1-based line numbers prefixed.

    Use start_line/num_lines for large files; default reads the whole file.
    """
    p = _resolve(path)
    if not p.exists():
        raise ToolError(f"no such file: {p}")
    if p.is_dir():
        raise ToolError(f"{p} is a directory, not a file -- use ls")
    text = _read_text(p)

    lines = text.splitlines()
    total = len(lines)
    if start_line is None and num_lines is None:
        body = _clip(text, MAX_READ_CHARS)
        return f"{p} ({total} lines)\n{body}"

    first = max(1, start_line or 1)
    last = min(total, first - 1 + (num_lines or total))
    if first > total:
        return f"{p} has only {total} lines; start_line {first} is past the end"
    chunk = lines[first - 1 : last]
    width = len(str(last))
    numbered = "\n".join(f"{first + i:>{width}}\t{ln}" for i, ln in enumerate(chunk))
    return f"{p} lines {first}-{last} of {total}\n{numbered}"


def write_file(path: str, content: str) -> str:
    """Create or overwrite a file with `content`. Creates parent dirs as needed."""
    p = _resolve(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        p.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"could not write {p}: {exc}") from exc
    verb = "overwrote" if existed else "created"
    n_lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
    return f"{verb} {p} ({n_lines} lines, {len(content)} chars)"


def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace `old_string` with `new_string` in a file.

    old_string must appear verbatim and be unique unless replace_all is True.
    Read the file first so old_string matches the real text exactly.
    """
    p = _resolve(path)
    if not p.is_file():
        raise ToolError(f"no such file: {p}")
    text = _read_text(p)

    if old_string not in text:
        raise ToolError(
            f"old_string not found in {p}. It must match the file byte-for-byte, "
            f"including indentation and newlines. Read the file again and copy exactly."
        )
    count = text.count(old_string)
    if count > 1 and not replace_all:
        raise ToolError(
            f"old_string appears {count} times in {p}; it is not unique. "
            f"Add surrounding context to make it unique, or pass replace_all=True."
        )

    updated = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
    try:
        p.write_text(updated, encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"could not write {p}: {exc}") from exc
    n = count if replace_all else 1
    return f"replaced {n} occurrence(s) in {p} ({len(old_string)} -> {len(new_string)} chars)"


def list_dir(path: str = ".", pattern: str = "*") -> str:
    """List directory entries. `pattern` is a glob like '*.py' or 'src/**'."""
    p = _resolve(path)
    if not p.is_dir():
        raise ToolError(f"{p} is not a directory")
    try:
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    except OSError as exc:
        raise ToolError(f"could not list {p}: {exc}") from exc

    shown: list[str] = []
    for entry in entries:
        if not fnmatch.fnmatch(entry.name, pattern):
            continue
        if entry.is_dir():
            shown.append(f"{entry.name}/")
        else:
            try:
                size = entry.stat().st_size
            except OSError:
                size = -1
            shown.append(f"{entry.name} ({size})")
        if len(shown) >= MAX_LS_ENTRIES:
            break

    # This tool is not recursive, so an empty result here is ambiguous: the
    # directory may be empty, or the pattern may simply not match at this
    # level while files exist further down. An earlier phrasing ("... is empty")
    # pushed the model to report zero files when the real answer was "look in
    # the subdirectories", so the sibling names are listed instead of a claim.
    if not shown:
        siblings = [e.name + ("/" if e.is_dir() else "") for e in entries[:20]]
        hint = (
            f"no entries in {p} match {pattern!r}. "
            f"list_dir does not recurse -- {len(entries)} entries exist here: "
            + (", ".join(siblings) if siblings else "(none)")
        )
        return hint

    body = "\n".join(shown)
    if len(entries) > MAX_LS_ENTRIES:
        body += f"\n... [showing {MAX_LS_ENTRIES} of {len(entries)} entries]"
    return f"{p}\n{body}"


def search_files(pattern: str, path: str = ".", glob: str = "*", ignore_case: bool = False) -> str:
    """Regex-search file contents under `path`; `glob` limits which files are read.

    Returns `path:lineno: line` per match. Prefer search_files over shelling out
    to grep -- it is portable and skips huge/binary files.
    """
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        raise ToolError(f"invalid regex {pattern!r}: {exc}") from exc

    root = _resolve(path)
    candidates = [root] if root.is_file() else sorted(root.rglob("*"))
    hits: list[str] = []
    scanned = 0

    for f in candidates:
        if len(hits) >= MAX_GREP_MATCHES:
            hits.append(f"... [stopped at {MAX_GREP_MATCHES} matches]")
            break
        if not f.is_file():
            continue
        rel = f.relative_to(ROOT) if ROOT in f.parents else f
        if not fnmatch.fnmatch(f.name, glob):
            continue
        try:
            if f.stat().st_size > 2_000_000:
                continue
            text = _read_text(f)
        except OSError:
            continue
        scanned += 1
        for n, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{rel}:{n}: {line.strip()[:300]}")
                if len(hits) >= MAX_GREP_MATCHES:
                    break

    if not hits:
        return f"no matches for {pattern!r} under {root} ({scanned} files scanned)"
    return "\n".join(hits)


def run_shell(command: str, timeout: int = 120) -> str:
    """Run a shell command in the workspace and return combined stdout+stderr.

    Use for git, builds, tests, package managers. Prefer the file tools for
    reading and editing -- shell quoting is error-prone. Returns the exit code
    in the last line. Has a hard timeout.
    """
    if not command.strip():
        raise ToolError("empty command")
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"[timeout] command exceeded {timeout}s and was killed:\n{command}"
    except OSError as exc:
        raise ToolError(f"could not run command: {exc}") from exc

    out = (proc.stdout or "") + (proc.stderr or "")
    if not out.strip():
        return f"(no output)\nexit code: {proc.returncode}"
    return _clip(out, MAX_BASH_CHARS) + f"\n[exit code: {proc.returncode}]"


def git_status() -> str:
    """Show the git working-tree status. Use to check what changed before editing."""
    return run_shell("git status --short --branch")


def git_diff(staged: bool = False) -> str:
    """Show the current git diff. Pass staged=True to see staged changes."""
    cmd = "git diff --cached" if staged else "git diff"
    return run_shell(cmd)


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


class Tools:
    """Tool catalog bound to the agent as a single ``tools`` input.

    jaz renders a class as its docstring plus public members, so the agent sees
    ``tools.read_file(...)``, ``tools.run_shell(...)`` etc. Grouping them keeps
    the invoke() call readable and gives one place to add or hide a tool.

    The methods below delegate to the module-level functions of the same name.
    That indirection is not redundant: assigning a plain function as a class
    attribute would make it an *unbound* function, so ``self`` would arrive as
    the first positional argument and every tool would receive the catalog
    instance where it expected a path. Declaring them as real methods is what
    binds ``self`` correctly. The module-level functions stay callable
    directly, which is what the tests and the ``__main__`` smoke check use.
    """

    """Filesystem, shell and git tools for software engineering tasks."""

    def read_file(self, path: str, start_line: int | None = None, num_lines: int | None = None) -> str:
        """Read a text file. Returns it with 1-based line numbers prefixed.

        Use start_line/num_lines for large files; default reads the whole file.
        """
        return read_file(path, start_line, num_lines)

    def write_file(self, path: str, content: str) -> str:
        """Create or overwrite a file with `content`. Creates parent dirs as needed."""
        return write_file(path, content)

    def edit_file(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """Replace `old_string` with `new_string` in a file.

        old_string must appear verbatim and be unique unless replace_all is True.
        Read the file first so old_string matches the real text exactly.
        """
        return edit_file(path, old_string, new_string, replace_all)

    def list_dir(self, path: str = ".", pattern: str = "*") -> str:
        """List one directory level. `pattern` is a glob like '*.py' or 'test_*'.

        Does NOT recurse. If a pattern matches nothing, the reply lists the
        sibling names so you can tell "empty directory" from "wrong level" --
        when you want a recursive search use search_files instead.
        """
        return list_dir(path, pattern)

    def search_files(self, pattern: str, path: str = ".", glob: str = "*", ignore_case: bool = False) -> str:
        """Regex-search file contents under `path`; `glob` limits which files are read.

        Returns `path:lineno: line` per match. Prefer this over shelling out to
        grep -- it is portable and skips huge or binary files.
        """
        return search_files(pattern, path, glob, ignore_case)

    def run_shell(self, command: str, timeout: int = 120) -> str:
        """Run a shell command in the workspace and return combined stdout+stderr.

        Use for git, builds, tests, package managers. Prefer the file tools for
        reading and editing -- shell quoting is error-prone. Returns the exit
        code in the last line. Has a hard timeout.
        """
        return run_shell(command, timeout)

    def git_status(self) -> str:
        """Show the git working-tree status. Use to check what changed before editing."""
        return git_status()

    def git_diff(self, staged: bool = False) -> str:
        """Show the current git diff. Pass staged=True to see staged changes."""
        return git_diff(staged)


def tool_catalog() -> str:
    """Human-readable summary of the tools, for the UI status bar."""
    return ", ".join(
        name
        for name in vars(Tools)
        if not name.startswith("_") and callable(getattr(Tools, name))
    )


if __name__ == "__main__":  # manual smoke check
    print(f"root={ROOT}")
    print(f"python={sys.version.split()[0]}")
    print(f"tools={tool_catalog()}")
    print(list_dir("."))