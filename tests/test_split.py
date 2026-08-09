import pandas as pd
from src.data.split_dataset import create_splits


def test_splits_are_disjoint_and_stratified(tmp_path):
    rows = []
    for label in ("0", "A"):
        for index in range(20):
            rows.append({"label": label, "processed_path": f"/{label}/{index}.jpg", "status": "ok"})
    source = tmp_path / "segmentation.csv"; pd.DataFrame(rows).to_csv(source, index=False)
    splits = create_splits(source, tmp_path / "splits", seed=42)
    assert [len(splits[key]) for key in ("train", "val", "test")] == [32, 4, 4]
    assert not (set(splits["train"].processed_path) & set(splits["test"].processed_path))
