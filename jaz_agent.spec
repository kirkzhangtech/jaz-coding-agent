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
# The `includes` filter is not an optimisation, it is a correctness fix.
# Unfiltered, `collect_data_files("litellm")` returns 933 files, and the great
# majority of them are litellm's bundled web proxy -- a compiled Next.js admin
# panel, PNG logos, JS chunks. This agent never starts a proxy, so all of it is
# dead weight that bloats the build and, worse, buries the handful of JSON
# tables that pricing actually depends on.
litellm_datas = collect_data_files("litellm", includes=["*.json", "*.csv"])

# The provider adapters themselves. litellm imports these lazily by name, one
# per integration, so a build that only sees the OpenRouter path statically will
# fail the moment a user switches backend with `/backends`.
#
# `proxy` is excluded deliberately: it is litellm's web server, pulls in FastAPI
# and friends, and this agent never runs it.
litellm_imports = [
    name
    for name in collect_submodules("litellm")
    if ".proxy" not in name
]

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
    # Standard library modules litellm reaches for through `importlib` probes
    # rather than direct imports.
    excludes=[
        # tkinter is pulled in by litellm's model-download UI paths and adds
        # several MB for a TUI that never uses it.
        "tkinter",
        "unittest",
        "pydoc_data",
        "test",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="jaz-agent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # No console=False: this is a terminal application. A windowed build on
    # Windows detaches from the terminal, which breaks the TUI outright --
    # there would be nowhere to draw and no way for the user to type.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="jaz-agent",
)