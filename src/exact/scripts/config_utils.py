from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_toml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("rb") as file:
        return tomllib.load(file)
