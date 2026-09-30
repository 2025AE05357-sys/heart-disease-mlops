"""YAML configuration loading shared by every pipeline stage."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file into a plain dictionary."""
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)
