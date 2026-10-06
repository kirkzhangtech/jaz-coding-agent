"""The clipboard is one function, so what is tested is the platform it writes.

The Windows path is the one worth a test: it cannot be exercised on any other
machine, and its failure mode is silent -- ctypes defaults every return value to
``c_int``, so a truncated handle does not raise, it produces a paste of the wrong
thing or of nothing.

The round trip below reads the clipboard back the way a paste would, and puts
back what was there.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
import sys

import pytest

from jaz_agent.clipboard import copy_text

#: Read the clipboard and print it as base64 of its UTF-16 code units.
#:
#: Raw code units on purpose. Anything textual would be re-encoded on the way
#: out -- PowerShell's console pipeline rewrites line endings, and Python's text
#: mode does it again on the way in -- and then "did the ``\\n`` become
#: ``\\r\\n``?" would pass without the conversion ever happening.
READ_BACK = (
    "[Console]::OutputEncoding=[Text.Encoding]::ASCII; "
    "$t = Get-Clipboard -Raw -ErrorAction SilentlyContinue; "
    "if ($null -eq $t) { 'EMPTY' } else { "
    "[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($t)) }"
)


def _powershell(script: str) -> str:
    """Run *script* and return its stdout, or skip when there is no PowerShell."""
    exe = shutil.which("powershell") or shutil.which("pwsh")
    if exe is None:
        pytest.skip("no PowerShell available to read the clipboard back")
    done = subprocess.run(
        [exe, "-NoProfile", "-Command", script],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def _read_clipboard() -> str:
    """Exactly what is on the clipboard, or ``""`` when it holds no text."""
    raw = _powershell(READ_BACK).strip()
    if not raw or raw == "EMPTY":
        return ""
    return base64.b64decode(raw).decode("utf-16-le")


def _restore(before: str) -> None:
    """Put back what was on the clipboard, if there was anything to put back.

    The contents are read as code units rather than as text, so what is restored
    is what was there; the only case that cannot be repaired is a clipboard
    holding something that is not text at all, which comes back empty.
    """
    if not before:
        print("clipboard left as-is: it held no text before the test")
    elif not copy_text(before):
        print("clipboard left as-is: the previous text could not be restored")


def test_empty_text_is_not_a_copy():
    """An empty copy is not worth pretending about: the caller keeps OSC 52."""
    assert copy_text("") is False


@pytest.mark.skipif(sys.platform != "win32", reason="the native path is Windows-only")
def test_the_windows_clipboard_takes_multi_line_text():
    """Round trip through the real clipboard, restoring what was there."""
    before = _read_clipboard()
    try:
        assert copy_text("jaz-agent clipboard check\nsecond line") is True
        # CRLF, because that is the form CF_UNICODETEXT is defined in: a lone
        # \n renders as one long line in Notepad.
        assert _read_clipboard() == "jaz-agent clipboard check\r\nsecond line"
    finally:
        _restore(before)


@pytest.mark.skipif(sys.platform != "win32", reason="the native path is Windows-only")
def test_non_ascii_text_survives():
    """The conversion to UTF-16 must not be lossy: this UI prints ✗ and ✓."""
    before = _read_clipboard()
    try:
        assert copy_text("✗ could not reach the provider — 12 turns") is True
        assert _read_clipboard() == "✗ could not reach the provider — 12 turns"
    finally:
        _restore(before)
