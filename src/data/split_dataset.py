from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def create_splits(segmentation_csv: str | Path, output_dir: str | Path, seed: int = 42,
                  train_ratio: float = 0.8, val_ratio: float = 0.1) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(segmentation_csv)
    frame = frame[frame.status == "ok"].copy()
    train, holdout = train_test_split(frame, train_size=train_ratio, stratify=frame.label, random_state=seed)
    val_fraction = val_ratio / (1 - train_ratio)
    val, test = train_test_split(holdout, train_size=val_fraction, stratify=holdout.label, random_state=seed)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "val": val, "test": test}
    all_paths = set()
    for name, split in splits.items():
        assert not (set(split.processed_path) & all_paths), "Split leakage detected"
        all_paths.update(split.processed_path)
        split[["processed_path", "label"]].to_csv(output_dir / f"{name}.csv", index=False)
    return splits
