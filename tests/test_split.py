import pandas as pd
from src.data.split_dataset import create_splits


def test_splits_are_disjoint_and_stratified_after_deduplication(tmp_path):
    rows = []
    for label in ("0", "A"):
        for index in range(20):
            source = f"/{label}/{index}.jpg"
            rows.append({"label": label, "source_path": source, "processed_path": f"/processed{source}", "status": "ok"})
    source = tmp_path / "segmentation.csv"; pd.DataFrame(rows).to_csv(source, index=False)
    audit = tmp_path / "audit.csv"
    pd.DataFrame([
        {"label": row["label"], "path": row["source_path"], "status": "ok", "sha256": f"{row['label']}-{index:03d}"}
        for index, row in enumerate(rows)
    ]).to_csv(audit, index=False)
    splits = create_splits(source, tmp_path / "splits", audit, seed=42)
    assert [len(splits[key]) for key in ("train", "val", "test")] == [32, 4, 4]
    assert not (set(splits["train"].processed_path) & set(splits["test"].processed_path))


def test_exact_duplicate_hashes_are_removed_before_split(tmp_path):
    rows, audit_rows = [], []
    for label in ("0", "A"):
        for index in range(20):
            source = f"/{label}/{index}.jpg"
            rows.append({"label": label, "source_path": source, "processed_path": f"/processed{source}", "status": "ok"})
            digest = f"{label}-shared" if index < 2 else f"{label}-{index:03d}"
            audit_rows.append({"label": label, "path": source, "status": "ok", "sha256": digest})
    segmentation = tmp_path / "segmentation.csv"; pd.DataFrame(rows).to_csv(segmentation, index=False)
    audit = tmp_path / "audit.csv"; pd.DataFrame(audit_rows).to_csv(audit, index=False)
    splits = create_splits(segmentation, tmp_path / "splits", audit, seed=42)
    assert sum(len(split) for split in splits.values()) == 38
    manifest = pd.read_csv(tmp_path / "splits" / "deduplication_manifest.csv")
    assert manifest.is_canonical.sum() == 38
    assert manifest.duplicate_group_size.max() == 2
