"""PyInstaller entry point for the frozen exe.

Kept as a real module rather than a `--onefile` command-line invocation so the
same file works for both builds and so the spec can import the metadata
(version, description) from one place instead of duplicating it.

`sys.frozen` matters for one reason: when frozen there is no `__file__`-relative
source tree, and jaz's REPL sandbox resolves allowed paths against the
workspace, which the user passes with `-w`. Nothing else needs to know it is
frozen, which is the point -- if more code had to check, the freeze would be
leaking into the product.
"""

from __future__ import annotations

import multiprocessing
import sys

from jaz_agent.__main__ import main

if __name__ == "__main__":
    # Required before anything spawns a process; harmless otherwise. Without it
    # a frozen build re-runs the whole app in child processes instead of erroring,
    # which turns one crash into an infinite spawn loop.
    multiprocessing.freeze_support()
    sys.exit(main())