"""Backends and models the agent can switch between at runtime.

Two ideas do the work here.

**A backend is a small record, not a class.** Each :class:`Backend` knows how to
build a litellm route and how to discover the models it offers. Adding a provider
means adding one entry to ``BACKENDS``, not writing a subclass -- which matters
because jaz already owns the pluggable-component machinery and we want to add to
it, not re-implement it.

**A switch is proven before it is committed.** :func:`probe` sends one minimal
live request; the caller only commits the new choice when it succeeds. A failed
switch therefore leaves the session on a model that works, which is the same
contract ``jaz.console.switch_model`` gives. Discovering a model is broken after
a full agent turn has burned is far worse than a one-line error.

Following jaz's own settings-file policy, **no API key is ever written down** --
only the chosen backend and model ids.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from jaz.llm import LiteLLM

#: Default OpenRouter model. Space Bunny Alpha, the model this project targets.
DEFAULT_MODEL = "stealth/space-bunny-alpha"

#: OpenRouter's public API root; litellm appends ``/chat/completions``.
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

#: DeepSeek's OpenAI-compatible root. litellm's own deepseek default points at
#: ``/beta``; ``/v1`` is the documented, stable equivalent and takes the same
#: ``/chat/completions`` path, so pinning it here keeps the choice visible.
DEEPSEEK_BASE = "https://api.deepseek.com/v1"

#: Where to get a key, shown when one is missing -- per backend, because the
#: console differs and "set an env var" alone does not tell you where to look.
KEY_HINTS: dict[str, str] = {
    "openrouter": "OpenRouter keys come from https://openrouter.ai/keys",
    "deepseek": "DeepSeek keys come from https://platform.deepseek.com/api_keys",
}

SITE_URL = "https://github.com/jaz-lang/jaz"
APP_TITLE = "jaz-coding-agent"


class LLMConfigError(RuntimeError):
    """Raised for a backend or model that cannot be used."""


@dataclass(frozen=True, slots=True)
class Backend:
    """One way of reaching a model, plus how to list what it offers.

    ``prefix`` turns a bare model id into the provider-prefixed form litellm
    needs. ``key_vars`` are the environment variables that may hold this
    backend's credential, most specific first.
    """

    name: str
    prefix: str
    key_vars: tuple[str, ...]
    base: str
    headers: dict[str, str] | None = None
    #: Models offered when discovery is unavailable, so the menu still works
    #: offline. Kept short -- this is a fallback, not a catalogue.
    well_known: tuple[str, ...] = field(default=())

    def route(self, model: str) -> str:
        """Return *model* in litellm's provider-prefixed form.

        A model that already carries the prefix is returned unchanged, so
        ``/switchmodules openrouter/foo`` and ``/switchmodules foo`` mean the same thing.
        """
        model = model.strip()
        if not model:
            raise LLMConfigError("empty model id")
        if model.startswith(self.prefix + "/"):
            return model
        return f"{self.prefix}/{model}"

    def bare(self, model: str) -> str:
        """Inverse of :meth:`route`: strip this backend's prefix if present.

        Session state stores the bare id so ``model_name`` reads cleanly, and
        ``route`` re-adds the prefix at build time. Doing it this way means a
        user typing either ``foo/bar`` or ``openrouter/foo/bar`` ends up with
        the same stored value.
        """
        model = str(model).strip()
        marker = self.prefix + "/"
        return model[len(marker) :] if model.startswith(marker) else model

    def has_key(self) -> bool:
        """True if a credential for this backend is present in the environment."""
        return any(os.environ.get(var, "").strip() for var in self.key_vars)

    def find_key(self) -> str:
        """Return the first credential this backend accepts.

        Raises :class:`LLMConfigError` naming the variables to set, rather than
        failing later with a 401 that costs a full agent turn.
        """
        for var in self.key_vars:
            value = os.environ.get(var, "").strip()
            if value:
                return value
        wanted = " or ".join(self.key_vars)
        hint = KEY_HINTS.get(self.name, "")
        raise LLMConfigError(
            f"no API key for backend {self.name!r}.\nSet one of: {wanted}"
            + (f"\n{hint}" if hint else "")
        )

    def build(self, model: str, **request_defaults: Any) -> LiteLLM:
        """Construct a litellm backend for *model*.

        The key is read here and handed to the backend; it is never persisted.
        """
        key = self.find_key()
        defaults: dict[str, Any] = {
            "timeout": float(os.environ.get("JAZ_HTTP_TIMEOUT", "180")),
            "max_retries": int(os.environ.get("JAZ_MAX_RETRIES", "3")),
        }
        if self.headers:
            defaults["extra_headers"] = dict(self.headers)
        defaults.update(request_defaults)

        return LiteLLM(
            model=self.route(model),
            api_key=key,
            api_base=os.environ.get("JAZ_API_BASE", self.base),
            **defaults,
        )


#: The backends on offer. OpenRouter is first because it is the default; the
#: rest exist so ``/switchmodules`` is useful without reconfiguring anything.
BACKENDS: dict[str, Backend] = {
    b.name: b
    for b in (
        Backend(
            name="openrouter",
            prefix="openrouter",
            key_vars=("OPENROUTER_API_KEY", "OR_API_KEY"),
            base=OPENROUTER_BASE,
            headers={"HTTP-Referer": SITE_URL, "X-Title": APP_TITLE},
            well_known=(
                "stealth/space-bunny-alpha",
                "anthropic/claude-sonnet-4.5",
                "openai/gpt-5-mini",
                "google/gemini-2.5-flash",
            ),
        ),
        Backend(
            name="anthropic",
            prefix="anthropic",
            key_vars=("ANTHROPIC_API_KEY",),
            base="https://api.anthropic.com/v1",
            well_known=("claude-sonnet-4-5", "claude-opus-4-1", "claude-haiku-4-5"),
        ),
        Backend(
            name="openai",
            prefix="openai",
            key_vars=("OPENAI_API_KEY",),
            base="https://api.openai.com/v1",
            well_known=("gpt-5-mini", "gpt-5", "gpt-4.1-mini"),
        ),
        Backend(
            name="google",
            prefix="gemini",
            key_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            base="https://generativelanguage.googleapis.com/v1beta",
            well_known=("gemini-2.5-flash", "gemini-2.5-pro"),
        ),
        Backend(
            name="deepseek",
            prefix="deepseek",
            key_vars=("DEEPSEEK_API_KEY",),
            base=DEEPSEEK_BASE,
            # Only the ids DeepSeek's own ``/models`` advertises. The older
            # ``deepseek-chat`` / ``deepseek-reasoner`` / ``deepseek-coder``
            # names still resolve as aliases, but the API rejects the
            # *undocumented* spellings outright --
            #   "The supported API model names are deepseek-flash,
            #    deepseek-v4-pro, but you passed deepseek-v4.1-flash."
            # -- so offering aliases would put rows in the browser that cannot
            # be checked against anything. Order matters: the first is the
            # default a bare ``-b deepseek`` lands on.
            well_known=("deepseek-v4-pro", "deepseek-flash"),
        ),
    )
}

#: What a fresh session starts on, unless the environment says otherwise.
DEFAULT_BACKEND = "openrouter"


def resolve_backend(name: str | None) -> Backend:
    """Look up a backend by name.

    Raises :class:`LLMConfigError` listing the known names, so a typo is
    immediately actionable.
    """
    if not name:
        name = os.environ.get("JAZ_BACKEND") or DEFAULT_BACKEND
    key = str(name).strip().lower()
    if key not in BACKENDS:
        raise LLMConfigError(
            f"unknown backend {name!r}. Known: {', '.join(sorted(BACKENDS))}."
        )
    return BACKENDS[key]


def available_backends() -> list[tuple[str, bool]]:
    """``(name, has_key)`` for every backend, for ``/backends``."""
    return [(name, backend.has_key()) for name, backend in BACKENDS.items()]


#: Marks an explicit backend in a switch argument: ``@deepseek deepseek-v4-pro``.
#:
#: A marker is needed rather than inferring the provider from the model id,
#: because a routed id is genuinely ambiguous. ``deepseek/deepseek-v4-pro`` is a
#: real OpenRouter id *and* what DeepSeek's own ``deepseek-v4-pro`` routes to, so
#: a rule like "the prefix names the backend" would silently move existing
#: ``/switchmodules openai/gpt-5-mini`` -- today a switch *within* OpenRouter --
#: onto the OpenAI backend and its key. ``@`` is the marker because no provider
#: uses one in a model id.
BACKEND_MARK = "@"


@dataclass(frozen=True, slots=True)
class SwitchRequest:
    """What a ``/switchmodules`` argument asks for, decoded.

    Two shapes, and they are told apart by :attr:`is_switch`:

    * **a switch** -- at least one of ``backend`` / ``model`` is set. A bare
      ``backend`` with no ``model`` means "that backend's default".
    * **a browse** -- neither is set, and ``filter`` narrows the list.
    """

    #: ``None`` means "stay on the backend we are already using".
    backend: Backend | None = None
    #: ``None`` means "whatever that backend starts on". Only meaningful
    #: alongside a ``backend``.
    model: str | None = None
    #: The text to narrow the browser with. Only set for a browse.
    filter: str = ""

    @property
    def is_switch(self) -> bool:
        """True when this names a target rather than narrowing a list."""
        return self.backend is not None or self.model is not None


def parse_switch_request(argument: str, *, current: Backend) -> SwitchRequest:
    """Decode a ``/switchmodules`` argument.

    Three forms, in the order they are tried:

    ``@<backend> [model]``
        Explicit provider. ``@deepseek`` alone means that backend's default
        model, so the form is symmetric with the bare name below.
    ``<backend>``
        Shorthand for ``@<backend>``. Safe to read as a backend because model
        ids carry a vendor prefix and a backend name never contains ``/`` --
        and it is checked against the known names rather than against that
        shape, so a typo falls through to the filter path instead of raising.
    anything else
        A model id or a filter, decided later by catalogue membership. This is
        what keeps ``/switchmodules openai/gpt-5-mini`` meaning what it always
        did: a switch *within* the current backend.

    ``current`` is taken so the last case is not guessed at here; only the
    caller knows the catalogue.
    """
    text = argument.strip()
    if not text:
        return SwitchRequest()

    if text.startswith(BACKEND_MARK):
        name, _, model = text[len(BACKEND_MARK) :].strip().partition(" ")
        # Not caught: an unknown name must raise, and the raise lists the known
        # backends. Falling through to a filter here would report "no model
        # matches '@deapseak'" and hide the typo inside a search.
        return SwitchRequest(backend=resolve_backend(name), model=model.strip() or None)

    try:
        return SwitchRequest(backend=resolve_backend(text))
    except LLMConfigError:
        return SwitchRequest(model=text)


def list_models(backend: Backend) -> list[str]:
    """Best-effort model list for *backend*.

    Falls back to ``backend.well_known`` when discovery fails -- being offline
    should degrade the menu, not break it. A failure here is never fatal: the
    user can always type a model id by hand.

    The dispatch names each lister rather than holding a ``name -> function``
    map. A map would capture the function object at import time, so
    ``monkeypatch.setattr(module, "_list_deepseek_models", ...)`` would be
    silently ignored and every test would reach the real network -- the same
    binding trap that once made a TUI test depend on the live catalogue.
    """
    found: list[str] = []
    if backend.name == "openrouter":
        found = _list_openrouter_models(backend)
    elif backend.name == "deepseek":
        found = _list_deepseek_models(backend)
    return found or list(backend.well_known)


def _fetch_model_ids(url: str, headers: dict[str, str] | None = None) -> list[str]:
    """GET an OpenAI-shaped ``/models`` payload and return its sorted ids.

    Returns ``[]`` on any failure, for the reason ``list_models`` gives: the
    menu degrades, it does not break. ``headers`` carries the credential where
    the endpoint is keyed, which is every provider here except OpenRouter.
    """
    try:
        import httpx

        response = httpx.get(url, headers=headers, timeout=10.0)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []

    ids = {
        entry["id"]
        for entry in data
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    return sorted(ids)


def _list_openrouter_models(backend: Backend) -> list[str]:
    """Query OpenRouter's public catalogue. Returns ``[]`` on any failure."""
    return _fetch_model_ids(f"{backend.base}/models")


def _list_deepseek_models(backend: Backend) -> list[str]:
    """Query DeepSeek's catalogue. Returns ``[]`` on any failure.

    Unlike OpenRouter's, this endpoint is keyed -- so with no credential there
    is nothing to ask, and the caller falls back to ``well_known``.
    """
    try:
        key = backend.find_key()
    except LLMConfigError:
        return []
    return _fetch_model_ids(f"{backend.base}/models", {"Authorization": f"Bearer {key}"})


def probe(llm: LiteLLM) -> str | None:
    """Send one minimal request to prove *llm* works.

    Returns ``None`` on success, or a human-readable reason on failure. One
    short completion is the whole test: the point is to catch a rejected key, a
    nonexistent model and a bad route *before* the user commits to them, not to
    benchmark anything.
    """
    try:
        llm.complete(
            model=llm.model,
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            max_tokens=8,
        )
        return None
    except Exception as exc:
        return _describe(exc)


def _describe(exc: Exception) -> str:
    """Turn an exception into one actionable line.

    Three jobs, in order of importance:

    * Name the common cause -- a rejected key, a bad model, a rate limit, no
      credit, no network. Those are the ones a user can act on.
    * **Redact anything account-identifying.** OpenRouter embeds a ``user_id``
      in its error payloads; that must not reach the transcript or a log file.
    * Stay one line, so a multi-line provider error cannot flood the UI.

    The wording is matched loosely on purpose: every provider phrases these
    differently, and litellm wraps them so the original text arrives as a
    prefix (``litellm.BadRequestError: OpenrouterException - {...}``).
    """
    name = type(exc).__name__
    raw = " ".join(str(exc).split())
    lowered = raw.lower()
    bare = name.lower()

    if "auth" in bare or "unauthorized" in lowered or "invalid api key" in lowered:
        return "authentication failed — the API key was rejected"
    # Covers "is not a valid model ID", "model not found", "does not exist".
    if (
        "not a valid model" in lowered
        or "not found" in lowered
        or "does not exist" in lowered
        or "unknown model" in lowered
    ):
        return "model not found, or not available on this backend"
    if "rate limit" in lowered or "429" in lowered:
        return "rate limited — try again shortly"
    if "credit" in lowered and ("insufficient" in lowered or "exceeded" in lowered):
        return "insufficient credits on this account"
    if "connect" in bare or "timeout" in bare or "connection" in lowered:
        return "could not reach the provider — check the network"

    return f"{name}: {_redact(raw)[:200]}" if raw else name


#: Keys whose values must never appear in output, whatever the provider says.
_SECRET_KEYS = ("user_id", "userid", "user", "api_key", "apikey", "key", "token", "email")


def _redact(text: str) -> str:
    """Strip account identifiers out of a provider error string.

    OpenRouter's JSON errors carry ``user_id``; Anthropic and OpenAI have their
    own equivalents. Rather than enumerate providers, blank the value of any
    JSON-looking key that smells like an identifier. This is best-effort by
    design: an unrecognised leak is possible, which is why the fallback path
    truncates too.
    """
    import re

    pattern = re.compile(
        r'("(?:%s)"\s*:\s*)"[^"]*"' % "|".join(_SECRET_KEYS),
        re.IGNORECASE,
    )
    return pattern.sub(r'\1"<redacted>"', text)


# --- wrappers preserving the module's original surface --------------------


def build_llm(
    model: str | None = None, backend: str | None = None, **request_defaults: Any
) -> LiteLLM:
    """Build a litellm backend for *model*, defaulting to the configured backend."""
    resolved = resolve_backend(backend)
    bare = model or os.environ.get("JAZ_MODEL") or default_model_for(resolved)
    return resolved.build(bare, **request_defaults)


def default_model_for(backend: Backend) -> str:
    """The model a fresh session starts on for *backend*.

    Named without a leading underscore because the session resolves its starting
    model from here, and a private name imported across modules is a smell.
    """
    if backend.name == "openrouter":
        return os.environ.get("JAZ_MODEL") or DEFAULT_MODEL
    return backend.well_known[0] if backend.well_known else DEFAULT_MODEL


def describe_model(backend: str | None = None, model: str | None = None) -> str:
    """One-line ``model via backend`` description for the UI header."""
    resolved = resolve_backend(backend)
    bare = model or os.environ.get("JAZ_MODEL") or default_model_for(resolved)
    return f"{bare} via {resolved.name}"