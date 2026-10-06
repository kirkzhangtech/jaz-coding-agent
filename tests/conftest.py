"""Shared test fixtures.

The suite's README claims it needs no API key, but several tests set one to
exercise the keyed path -- and none of them cleared the *others*. That was
harmless while every feature looked at a single backend, and stops being
harmless the moment one iterates the backends: ``/switchmodules`` now lists
what every keyed backend offers, so a developer with ``DEEPSEEK_API_KEY``
exported would see a different list, a different count, and different rows than
CI does. A test whose result depends on the machine is not a test.

Clearing every backend's credential before each test makes the environment the
test's rather than the machine's. Tests that want a key still call
``monkeypatch.setenv``, which runs after this fixture and therefore wins.

``JAZ_BACKEND`` and ``JAZ_MODEL`` are cleared for the same reason: they decide
what a session starts on, so an exported one would silently re-point the
default-backend tests.
"""

from __future__ import annotations

import pytest

from jaz_agent.llm_config import BACKENDS

#: Not credentials, but they choose the starting backend and model -- which is
#: the same kind of machine-dependent input and just as able to change a result.
_CONFIG_VARS = ("JAZ_BACKEND", "JAZ_MODEL", "JAZ_API_BASE")


@pytest.fixture(autouse=True)
def _no_ambient_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a clean, keyless environment to opt into."""
    for backend in BACKENDS.values():
        for var in backend.key_vars:
            monkeypatch.delenv(var, raising=False)
    for var in _CONFIG_VARS:
        monkeypatch.delenv(var, raising=False)
