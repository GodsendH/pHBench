from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {source}")
    config = copy.deepcopy(config)
    config["_config_path"] = str(source)
    config["_root"] = str(_project_root(source))
    return config


def _project_root(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    value: Any = config
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def path(config: dict[str, Any], key: str, default: str | None = None) -> Path:
    value = get(config, key, default)
    if value is None:
        raise KeyError(f"missing path configuration: {key}")
    result = Path(str(value)).expanduser()
    if not result.is_absolute():
        result = Path(config["_root"]) / result
    return result.resolve()


def config_hash(config: dict[str, Any]) -> str:
    payload = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_resolved(config: dict[str, Any], destination: str | Path) -> None:
    payload = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
