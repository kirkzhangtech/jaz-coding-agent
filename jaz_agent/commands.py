"""The slash-command table, as data.

One list drives three things -- ``/help`` output, the inline hint that appears
when you type ``/``, and Tab completion. When those were written separately they
drifted immediately: a command added to the dispatcher was invisible to the
completions and undocumented in ``/help``.

Each :class:`Command` carries its own usage string and summary rather than
repeating the name in prose, so the rendered help is assembled from the same
fields the completer matches on. That is the whole point: there is nowhere for
the two to disagree.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Command:
    """One slash command.

    ``usage`` is the argument shape (``""`` when the command takes none) and
    ``summary`` is the one line shown in help and completion. ``target`` is what
    the argument refers to, used to offer model-name completion after
    ``/model`` and ``/switchmodules``.
    """

    name: str
    usage: str
    summary: str
    #: None, "model" or "backend" -- drives what the completer offers.
    target: str | None = None
    aliases: tuple[str, ...] = ()

    @property
    def signature(self) -> str:
        """``/name <usage>`` or ``/name``."""
        return f"/{self.name} {self.usage}".strip()

    def matches(self, prefix: str) -> bool:
        """True if this command answers to *prefix* (case-insensitive)."""
        p = prefix.lower()
        return self.name.startswith(p) or any(a.startswith(p) for a in self.aliases)


#: The table. Order is the order they appear in ``/help`` and in completion.
COMMANDS: tuple[Command, ...] = (
    Command("help", "", "list commands", aliases=("h",)),
    Command("status", "", "current model, workspace and tool list"),
    Command(
        "switchmodules",
        "[model]",
        "browse and switch models on the current backend",
        target="model",
        aliases=("switchmodule",),
    ),
    Command(
        "model",
        "[name|backend [name]]",
        "show or switch model; validated live before committing",
        target="model",
    ),
    Command("backends", "", "list backends and whether each has a key"),
    Command("new", "", "drop conversation history"),
    Command("clear", "", "clear the transcript"),
    Command("cancel", "", "stop the running turn"),
    Command("quit", "", "exit", aliases=("q", "exit")),
)

#: Commands that need no model name after them.
NO_ARGUMENT = frozenset(c.name for c in COMMANDS if not c.usage)

#: Commands whose argument is a model id.
MODEL_ARGUMENT = frozenset(c.name for c in COMMANDS if c.target == "model")


def canonical(name: str) -> str | None:
    """Resolve an alias or exact name to the canonical command name.

    Aliases live on each :class:`Command` rather than in a separate map, so
    adding a command and its shorthand is one edit rather than two that can
    disagree.
    """
    key = name.strip().lower()
    if not key:
        return None
    for command in COMMANDS:
        if key == command.name or key in command.aliases:
            return command.name
    return None


def find(name: str) -> Command | None:
    """Look up a command by name or alias."""
    resolved = canonical(name)
    return next((c for c in COMMANDS if c.name == resolved), None) if resolved else None


def complete(prefix: str) -> list[Command]:
    """Commands whose name (or alias) starts with *prefix*, in table order."""
    p = prefix.lower().lstrip("/")
    if not p:
        return list(COMMANDS)
    return [c for c in COMMANDS if c.matches(p)]


def help_text() -> str:
    """Render ``/help`` from the table.

    Two columns, aligned on the longest signature so the summaries line up.
    """
    rows = [(c.signature, c.summary) for c in COMMANDS]
    width = max(len(sig) for sig, _ in rows)
    lines = [f"  {sig.ljust(width)}  {summary}" for sig, summary in rows]
    body = "\n".join(lines)
    return (
        f"{body}\n\n"
        "A model switch is validated with one live request before it is "
        "committed, so a rejected key or a wrong model name leaves the current "
        "model in place.\n"
        "Type / and press Tab to complete a command."
    )


def hint_text(prefix: str = "") -> str:
    """The one-line hint shown while typing a slash command.

    Renders the candidates that match *prefix*, or a generic pointer when the
    input is just ``/``. Kept to one line so it never pushes the prompt around
    while the user is typing.

    Each candidate keeps its leading slash: the hint mirrors what the user
    should type next, and ``status switchmodules`` reads like prose while
    ``/status  /switchmodules [model]`` reads like the input it replaces.
    """
    matches = complete(prefix)
    if not matches:
        return "no such command — /help lists them all"
    if len(matches) == 1:
        only = matches[0]
        extra = f" {only.usage}" if only.usage else ""
        return f"{only.signature}{extra} — {only.summary}"

    shown = matches[:6]
    parts = [c.signature if c.usage else f"/{c.name}" for c in shown]
    text = "   ".join(parts)
    if len(matches) > len(shown):
        text += f"   (+{len(matches) - len(shown)} more)"
    return text