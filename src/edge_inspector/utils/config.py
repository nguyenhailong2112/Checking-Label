from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    data: dict[str, Any]

    def get(self, key_path: str, default: Any = None) -> Any:
        current: Any = self.data
        for key in key_path.split("."):
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return AppConfig(data=data)