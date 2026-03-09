from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_if_present(dotenv_path: Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from `.env` into the process environment.

    Existing environment variables win. This is intentionally tiny so the
    distillation tooling works outside the Nix shell too.
    """
    if dotenv_path is None:
        dotenv_path = Path(__file__).resolve().parent.parent / ".env"

    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

    if "VASTAI_API_KEY" in os.environ and "VAST_API_KEY" not in os.environ:
        os.environ["VAST_API_KEY"] = os.environ["VASTAI_API_KEY"]
