"""FlyWire / CAVE authentication from environment (no secrets logged)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _strip_unreadable_path_entries() -> None:
    raw = os.environ.get("PATH", "")
    if not raw:
        return
    safe: list[str] = []
    for entry in raw.split(os.pathsep):
        if not entry:
            continue
        try:
            _ = Path(entry).exists()
            safe.append(entry)
        except (PermissionError, OSError):
            continue
    os.environ["PATH"] = os.pathsep.join(safe)


def load_dotenv_if_present(repo_root: Optional[Path] = None) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = repo_root or Path.cwd()
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


def apply_flywire_token_from_env(overwrite: bool = True) -> bool:
    """
    If FLYWIRE_TOKEN or CAVE_TOKEN is set, register it with fafbseg/caveclient.

    Returns True if a token was applied, False if none was found.
    """
    token = (os.environ.get("FLYWIRE_TOKEN") or os.environ.get("CAVE_TOKEN") or "").strip()
    if not token:
        return False

    _strip_unreadable_path_entries()
    from fafbseg import flywire

    flywire.set_chunkedgraph_secret(token, overwrite=overwrite)
    logger.info("Registered FlyWire/CAVE token from environment (FLYWIRE_TOKEN or CAVE_TOKEN).")
    return True
