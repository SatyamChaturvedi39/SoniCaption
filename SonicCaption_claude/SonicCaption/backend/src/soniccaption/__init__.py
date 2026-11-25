"""
SonicCaption - Intelligent Audio Understanding & Captioning System.
"""

import os
from pathlib import Path

__version__ = "1.0.0"


def _load_dotenv_once() -> None:
    """Populate os.environ from project .env at package import without overriding explicit env vars."""
    root = Path(__file__).resolve().parents[3]
    for env_path in (root / ".env", root / "backend" / ".env"):
        if not env_path.is_file():
            continue
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_dotenv_once()
