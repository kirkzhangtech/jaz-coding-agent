# PyInstaller spec for the frozen jaz-agent executable.
#
# Run with:  .\.venv\Scripts\python.exe -m PyInstaller jaz_agent.spec
#
# Kept as a spec rather than a command line because the interesting part of this
# build is the list of things a static analyser cannot see. Writing it as a
# `--onefile --hidden-import=...` invocation would be shorter and would rot:
# nobody can diff it against the reason each entry exists.
#
# The one non-obvious requirement is litellm. It resolves providers and their
# pricing tables at runtime, through `importlib.resources` and dynamic import,
# so PyInstaller's graph analysis finds none of it. A build with no hidden
# imports produces an exe that starts, prints a traceback about a missing
# `model_prices.json`, and dies -- which is exactly what the first probe did.

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# litellm ships JSON/CSV tables it reads relative to its own package: provider
# lists, model prices, context windows. These are data, not code, so no amount
# of `--hidden-import` would collect them.
#
# Unfiltered on purpose. `collect_data_files("litellm")` returns 933 files, and
# most of them are litellm's bundled web proxy -- a compiled Next.js admin
# panel, PNG logos, JS chunks -- which this agent never starts. Filtering them
# out was worth ~40 MB in the onedir build, but "all dependencies" means all of
# them: a file this size is only worth having if it works when moved somewhere
# unexpected, and a data file discovered to be missing is exactly the failure
# that only shows up on someone else's machine.
litellm_datas = collect_data_files("litellm")

# The provider adapters themselves. litellm imports these lazily by name, one
# per integration, so a build that only sees the OpenRouter path statically will
# fail the moment a user switches backend with `/backends`. All of them are kept
# for the same reason as the data files above: `/backends` offers four providers
# and the user may pick any of them at runtime, with no way to test all four
# combinations in a frozen build.
litellm_imports = collect_submodules("litellm")

# jaz is small but reaches for `importlib.resources` in its own sandbox
# setup; collecting its data keeps the REPL's secure-path resolution working
# once there is no source tree to fall back to.
jaz_datas = collect_data_files("jaz")

# Textual loads its built-in CSS and SVG glyphs from package data at startup;
# without these the app renders but every box comes out blank. This returns
# (source, destination) pairs, so it belongs in `datas` -- not in
# `hiddenimports`, which takes module names and fails on a tuple.
textual_datas = collect_data_files("textual")

block_cipher = None

a = Analysis(
    ["jaz_agent/__freeze__.py"],
    pathex=[],
    binaries=[],
    datas=litellm_datas + jaz_datas + textual_datas,
    hiddenimports=litellm_imports
    # The default provider is OpenRouter, so its adapter must survive even when
    # litellm's own module scan misses it.
    + ["litellm.llms.openrouter.chat"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Nothing is excluded from this build. An earlier onedir build dropped
    # tkinter and litellm's web proxy to save ~40 MB, which was the right call
    # for a directory that has to be shipped as-is. This build exists to be
    # *complete* -- one file that works with no Python and no neighbours -- so
    # anything PyInstaller can carry is carried. If litellm grows a UI path that
    # imports tkinter at module scope, excluding it here is what would turn a
    # working build into a `ModuleNotFoundError` on someone else's machine.
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ONE FILE. Everything is embedded in the exe, so it can be copied anywhere and
# run on a machine with no Python installed.
#
# The two changes from a standard onedir build are `exclude_binaries=False` --
# which makes EXE absorb the binary blobs instead of handing them to COLLECT --
# and the removal of COLLECT entirely.
#
# Cost, accepted deliberately: every launch extracts the whole payload to
# %TEMP% and cleans up afterwards, so startup takes seconds rather than
# milliseconds. The trade is one self-contained file that survives being moved,
# emailed or dropped onto a USB stick, against a directory that is fast but only
# works next to its own `_internal`.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    exclude_binaries=False,
    name="jaz-agent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX compresses the payload inside the exe, which is most of what makes a
    # onefile this size bearable. It costs build time and is occasionally
    # flaky on AV engines, but a 100 MB exe that starts instantly beats a
    # 60 MB one that does not.
    upx=True,
    # console=True, not console=False: this is a terminal application. A
    # windowed build on Windows detaches from the terminal, which breaks the
    # TUI outright -- nowhere to draw, no way for the user to type.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)