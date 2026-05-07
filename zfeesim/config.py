"""Scenario configuration loading from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_scenario(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)
