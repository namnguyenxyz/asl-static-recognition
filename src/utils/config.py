from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path = "configs/baseline.yaml") -> dict[str, Any]:
    """Load YAML config and resolve project-relative paths lazily."""
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_output_dirs(config: dict[str, Any]) -> None:
    for key in ("metadata", "splits", "metrics", "figures", "logs"):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["model"]).parent.mkdir(parents=True, exist_ok=True)
