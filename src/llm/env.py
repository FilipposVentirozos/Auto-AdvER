"""Minimal, dependency-free ``.env`` loader.

The public Auto-AdvER release reads every credential from the process
environment.  Credentials live in a gitignored ``.env`` file at the repository
root; ``.env.example`` is committed with empty placeholders.

Recognised keys (see ``.env.example``)::

    OPENAI_API_KEY=      # OpenAI API key for the `openai` backend
    HF_TOKEN=            # Hugging Face token, for gated checkpoints
    VERTEX_PROJECT=      # Google Cloud project, for the (unimplemented) Gemini path
    VERTEX_LOCATION=     # Google Cloud region, ditto

Nothing here reaches out to a network and nothing is printed: the loader only
copies ``KEY=VALUE`` lines into :data:`os.environ`.  Variables that are already
set in the environment always win, so ``OPENAI_API_KEY=... python ...`` and CI
secrets both behave as expected.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

__all__ = ["load_env", "find_env_file", "require"]

_REPO_ROOT_CANDIDATE = Path(__file__).resolve().parents[2]


def find_env_file() -> Optional[Path]:
    """Return the first plausible ``.env`` file, or ``None``.

    Search order: the current working directory, then the repository root
    inferred from this file's location (``src/llm/env.py`` -> two levels up).
    """
    for candidate in (Path.cwd() / ".env", _REPO_ROOT_CANDIDATE / ".env"):
        if candidate.is_file():
            return candidate
    return None


def _strip_value(value: str) -> str:
    """Normalise the right-hand side of a ``KEY=VALUE`` line.

    Surrounding single or double quotes are removed.  For an *unquoted* value, a
    trailing ``#`` comment is stripped -- ``.env.example`` documents each key
    that way, so without this a half-filled copy would set ``HF_TOKEN`` to the
    comment text.  Quote the value if it genuinely contains a ``#``.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    hash_at = value.find("#")
    if hash_at != -1:
        value = value[:hash_at]
    return value.strip()


def load_env(path: Optional[str] = None, override: bool = False) -> Dict[str, str]:
    """Load ``KEY=VALUE`` pairs from *path* into :data:`os.environ`.

    Blank lines and lines whose first non-space character is ``#`` are ignored,
    as is a leading ``export``.  Single or double quotes around a value are
    stripped.  Values are *not* expanded or interpolated.

    Unless *override* is true, a key that is already present in the environment
    is left untouched.  Returns the mapping that was read from the file
    (regardless of whether it was applied), so callers can report what a file
    defines without ever printing the values.

    A missing file is not an error: the environment may already carry the
    credentials.
    """
    env_path = Path(path) if path is not None else find_env_file()
    parsed: Dict[str, str] = {}
    if env_path is None or not Path(env_path).is_file():
        return parsed

    with open(env_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key:
                continue
            parsed[key] = _strip_value(value)

    for key, value in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


def require(key: str, hint: str = "") -> str:
    """Return ``os.environ[key]`` or raise a helpful :class:`RuntimeError`.

    The value itself is never included in the message.
    """
    load_env()
    value = os.environ.get(key, "")
    if not value:
        message = (
            f"{key} is not set. Put it in a .env file at the repository root "
            f"(copy .env.example) or export it in your shell."
        )
        if hint:
            message += " " + hint
        raise RuntimeError(message)
    return value
