"""Putting text on the system clipboard.

Why this module exists at all: ``App.copy_to_clipboard`` writes an OSC 52 escape
sequence and lets the *terminal* own the clipboard. That is the right mechanism
for a session in another machine -- it works over ssh and inside tmux, where
nothing else can reach the desktop -- and it is the only clipboard a Python
process can write without touching the OS. It is also silent where it is not
implemented: most terminals on Windows ignore OSC 52, so Ctrl+C looked exactly
like a key that had been swallowed.

So the local clipboard is written directly and OSC 52 is kept as well, for the
sessions where it is the only thing that can work. :func:`copy_text` reports
whether the direct write happened, which is what the caller needs to know to
decide between the two.

Nothing here raises: a clipboard that will not take the text is not worth a
traceback over a chat window.
"""

from __future__ import annotations

import sys
import time

__all__ = ["copy_text"]

#: ``OpenClipboard`` is a lock, not a queue: any process may hold it, briefly.
#: Browsers and clipboard managers take it while they copy, so a refusal here
#: usually means "in a moment" rather than "no".
_LOCK_ATTEMPTS = 10
_LOCK_PAUSE = 0.01


def _windows_clipboard(text: str) -> bool:
    """Put *text* on the Windows clipboard. ``False`` if Windows refused.

    ``clip.exe`` would be shorter, but it is spawn-and-encode-to-UTF-16 and it
    takes the clipboard for a moment of its own; the Win32 API is synchronous,
    in-process, and needs no agreement about the console codepage.

    The ctypes types matter. ``SetClipboardData`` returns a HANDLE and
    ``GlobalAlloc`` returns a pointer, and ctypes defaults every return value to
    ``c_int``: on 64-bit Windows that truncates both, which fails at the point
    of the paste rather than at the point of the call. Both are declared.

    ``\\n`` is turned into ``\\r\\n`` because CF_UNICODETEXT is the Windows line
    convention -- a lone ``\\n`` still shows as one long line in Notepad.

    Ownership of the block transfers to the clipboard on success, so it must not
    be freed here; the failure paths, which do not transfer it, free it.
    """
    import ctypes
    from ctypes import wintypes

    CF_UNICODETEXT = 13
    GHND = 0x0042  # GMEM_MOVEABLE | GMEM_ZEROINIT

    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32

    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]

    buffer = ctypes.create_unicode_buffer(
        text.replace("\r\n", "\n").replace("\n", "\r\n")
    )
    size = ctypes.sizeof(buffer)

    for _ in range(_LOCK_ATTEMPTS):
        if user32.OpenClipboard(None):
            break
        time.sleep(_LOCK_PAUSE)
    else:
        return False

    block = None
    try:
        user32.EmptyClipboard()
        block = kernel32.GlobalAlloc(GHND, size)
        if not block:
            return False
        pointer = kernel32.GlobalLock(block)
        if not pointer:
            return False
        ctypes.memmove(pointer, buffer, size)
        kernel32.GlobalUnlock(block)
        if not user32.SetClipboardData(CF_UNICODETEXT, block):
            return False
        block = None  # the clipboard owns it now
        return True
    finally:
        user32.CloseClipboard()
        if block:
            kernel32.GlobalFree(block)


def copy_text(text: str) -> bool:
    """Put *text* on this machine's clipboard, if this platform can.

    Returns whether it happened. ``False`` is not an error: it means the caller
    has nothing better than the terminal's own mechanism, which is the state
    every non-Windows session is in.
    """
    if not text or sys.platform != "win32":
        return False
    try:
        return _windows_clipboard(text)
    except Exception:
        # A deliberate catch-all: this runs on the UI thread, where an exception
        # out of a Win32 call would take the app down rather than become a copy
        # that did not happen.
        return False
