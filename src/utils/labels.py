import json
from pathlib import Path


def save_classes(classes: list[str], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")


def load_classes(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
