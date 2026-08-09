from __future__ import annotations
import hashlib
from pathlib import Path
import cv2
import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_dataset(raw_dir: str | Path, classes: list[str], output_dir: str | Path) -> pd.DataFrame:
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    records: list[dict] = []
    for label in classes:
        class_dir = raw_dir / label
        if not class_dir.exists():
            records.append({"label": label, "path": "", "status": "missing_class_directory", "sha256": ""})
            continue
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = cv2.imread(str(path))
            records.append({
                "label": label, "path": str(path),
                "status": "ok" if image is not None else "unreadable",
                "sha256": image_hash(path) if image is not None else "",
            })
    frame = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "dataset_audit.csv", index=False)
    summary = frame[frame.status == "ok"].groupby("label").size().rename("images").reindex(classes, fill_value=0)
    summary.to_csv(output_dir / "class_distribution.csv")
    duplicates = frame[(frame.status == "ok") & frame.sha256.duplicated(False)]
    duplicates.to_csv(output_dir / "duplicate_images.csv", index=False)
    return frame
