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
    ``/switchmodules``.
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
        "[model|@backend]",
        "browse and switch models, across backends",
        target="model",
        aliases=("switchmodule",),
    ),
    Command("backends", "", "list backends and whether each has a key"),
    Command("new", "", "drop conversation history"),
    Command("context", "", "what the agent remembers, and what it sends"),
    Command("clear", "", "clear the transcript"),
    Command("cancel", "", "stop the running turn"),
    Command("cost", "", "turns, spend and elapsed time", aliases=("c",)),
    Command("quit", "", "exit", aliases=("q", "exit")),
)

#: Commands that need no model name after them.
NO_ARGUMENT = frozenset(c.name for c in COMMANDS if not c.usage)

#: Commands whose argument is a model id.
MODEL_ARGUMENT = frozenset(c.name for c in COMMANDS if c.target == "model")

#: Removed commands mapped to what replaced them.
#:
#: Kept because deleting a command does not delete the muscle memory or the old
#: transcripts and shell history that still say the old name. Answering
#: ``/model`` with a bare "unknown command" would be technically true and
#: useless; naming the replacement turns a dead end into a redirect.
#:
#: This is not the same mechanism as :func:`canonical`. An alias still exists;
#: a retired name is gone and the user needs to be told where it went.
RETIRED: dict[str, str] = {
    "model": "switchmodules",
    "models": "switchmodules",
}


def canonical(name: str) -> str | None:
    """Resolve an alias or exact name to the canonical command name.

    Aliases live on each :class:`Command` rather than in a separate map, so
    adding a command and its shorthand is one edit rather than two that can
    disagree.

    A leading slash is stripped, matching :func:`complete`. The two disagreed
    for a while -- ``complete("/sw")`` matched and ``canonical("/sw")`` did not
    -- which is the kind of inconsistency that only shows up when a new caller
    passes the text a user actually typed.
    """
    key = name.strip().lower().lstrip("/")
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


@dataclass(frozen=True, slots=True)
class Candidate:
    """One selectable row in the hint.

    A row rather than a string because the list is navigable, and navigation
    needs the command *name* separately from its rendered text: ``name`` is what
    goes into the input box, ``signature`` is what gets displayed.

    The distinction matters. ``Command.signature`` is the display form and
    carries the usage placeholder -- ``/switchmodules [model]`` -- so feeding it
    back into the prompt would submit the literal ``/switchmodules [model]`` and
    be rejected as an unknown command. ``name`` is ``/switchmodules``.

    ``lines`` is the pre-formatted block for this row. Pre-formatting is
    deliberate: the layout (columns, wrapping, indentation) is worked out once
    for the whole list, and re-deriving it per row would both duplicate that
    work and give each row a different notion of the column width, so the block
    boundaries would not line up.
    """

    #: What goes into the input box when this row is chosen: ``/switchmodules``.
    name: str
    #: The display form, with the usage placeholder: ``/switchmodules [model]``.
    signature: str
    #: One-line description shown beside it.
    summary: str
    #: The rendered lines, already indented and column-aligned.
    lines: tuple[str, ...]

    @property
    def height(self) -> int:
        """How many display lines this row occupies.

        Needed by the scrolling viewport: a row whose summary wraps is taller
        than one that does not, so a window measured in rows rather than lines
        would scroll by the wrong amount and drift further off the longer the
        list gets.
        """
        return len(self.lines) or 1


def picker_rows(
    items: list[str],
    *,
    marks: dict[str, str] | None = None,
) -> list[Candidate]:
    """Build navigable rows from a plain list of ids.

    Used for the model browser, which has the same shape of list -- a set of
    identifiers, one of which is current, the user picks one -- but a different
    domain from the command table. Sharing the row type means the navigation,
    the highlight and the block layout are written once rather than twice, and
    the two lists cannot drift apart in behaviour.

    *marks* prefixes a row's id in the display, which is how the browser marks
    the current model (``✓``) without that knowledge leaking into this module.

    No column padding: model ids run from ``aion-labs/aion-2.0`` to
    ``vendor/some-very-long-model-name-v2:batch``, and padding every row to the
    longest would push the useful part off the screen. One id per line is what a
    browser wants.

    Every line starts with two spaces so that the selection cursor, which the
    renderer puts in column 0, lines up with the rest of the list instead of
    shifting the selected row left by one character.
    """
    marks = marks or {}
    return [
        Candidate(
            name=item,
            signature=item,
            summary=marks.get(item, ""),
            lines=(f"  {marks.get(item, ' ')} {item}",),
        )
        for item in items
    ]


def candidates(prefix: str = "", width: int = 0) -> list[Candidate]:
    """The selectable command rows matching *prefix*.

    Empty when there is nothing to navigate: no match, or exactly one match --
    a single row needs no cursor, and offering one would make the arrow keys
    look broken.
    """
    matches = complete(prefix)
    if len(matches) < 2:
        return []

    # Lay the whole list out once, then hand each command its own block. Doing
    # it per-command would re-measure the column width n times, which is both
    # wasteful and -- worse -- gives every row a *different* notion of the
    # column width, so the block boundaries would not line up.
    blocks = _blocks(matches, width)
    return [
        Candidate(f"/{c.name}", c.signature, c.summary, tuple(blocks[i]))
        for i, c in enumerate(matches)
    ]


def _blocks(matches: list[Command], width: int) -> list[list[str]]:
    """Lay out *matches*, returning one list of display lines per command.

    Each command may occupy one line (two-column layout) or several (the
    summary stacked underneath), so the result is a list of blocks rather than
    a flat list of lines.
    """
    width = max(width, 40)
    gap = "   "
    signature_width = max(len(c.signature) for c in matches)
    longest_summary = max(len(c.summary) for c in matches)
    fits_beside = signature_width + len(gap) + longest_summary + 2 <= width

    blocks: list[list[str]] = []
    for command in matches:
        if fits_beside:
            pad = signature_width - len(command.signature)
            blocks.append([f"  {command.signature}{' ' * pad}{gap}{command.summary}"])
        else:
            block = [f"  {command.signature}"]
            block.extend(f"      {line}" for line in _wrap(command.summary, width - 6))
            blocks.append(block)
    return blocks


def hint_text(prefix: str = "", width: int = 0) -> str:
    """The hint shown while typing a slash command.

    A thin wrapper over the layout so that callers which only need to *display*
    the list do not have to know about selection. Everything the keyboard
    navigation needs is on the :class:`Candidate`; everything the terminal needs
    is here.
    """
    matches = complete(prefix)
    if not matches:
        return "no such command — /help lists them all"
    if len(matches) == 1:
        only = matches[0]
        extra = f" {only.usage}" if only.usage else ""
        return f"{only.signature}{extra} — {only.summary}"
    return "\n".join(line for block in _blocks(matches, width) for line in block)


def _wrap(text: str, width: int) -> list[str]:
    """Break *text* into lines of at most *width* characters."""
    lines: list[str] = []
    current: list[str] = []
    for word in text.split():
        candidate = " ".join([*current, word])
        if current and len(candidate) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def hint_text(prefix: str = "", width: int = 0) -> str:
    """The hint shown while typing a slash command.

    A thin wrapper over the same layout :func:`candidates` uses, so that callers
    which only need to *display* the list do not have to know about selection.
    Everything the keyboard navigation needs is on the :class:`Candidate`;
    everything a caller that only prints needs is here. Deriving both from
    ``_blocks`` is the point -- two layouts would drift, and the drift would
    only show up as a highlight on the wrong line.
    """
    matches = complete(prefix)
    if not matches:
        return "no such command — /help lists them all"
    if len(matches) == 1:
        only = matches[0]
        extra = f" {only.usage}" if only.usage else ""
        return f"{only.signature}{extra} — {only.summary}"
    return "\n".join(line for block in _blocks(matches, width) for line in block)