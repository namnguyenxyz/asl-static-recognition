from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def _deduplicate_exact_images(segmentation: pd.DataFrame, audit_csv: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one deterministic representative of every raw-image SHA-256 group."""
    audit = pd.read_csv(audit_csv)
    required = {"path", "label", "status", "sha256"}
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(f"Audit manifest is missing columns: {sorted(missing)}")
    audit = audit[audit.status == "ok"][["path", "label", "sha256"]].copy()
    if audit.path.duplicated().any():
        raise ValueError("Audit manifest contains duplicate source paths")
    if audit.sha256.eq("").any() or audit.sha256.isna().any():
        raise ValueError("Readable audit records must include SHA-256 hashes")

    merged = segmentation.merge(
        audit, how="left", left_on="source_path", right_on="path", suffixes=("", "_audit"), validate="one_to_one"
    )
    if merged.sha256.isna().any():
        raise ValueError("Some segmentation records do not match the audit manifest")
    if (merged.label != merged.label_audit).any():
        raise ValueError("Label mismatch between segmentation and audit manifests")
    if (merged.groupby("sha256").label.nunique() > 1).any():
        raise ValueError("The same raw image hash occurs with more than one label")

    merged = merged.sort_values(["sha256", "source_path"], kind="stable").reset_index(drop=True)
    merged["duplicate_group_size"] = merged.groupby("sha256").sha256.transform("size")
    merged["canonical_source_path"] = merged.groupby("sha256").source_path.transform("first")
    merged["is_canonical"] = merged.source_path.eq(merged.canonical_source_path)
    manifest = merged[[
        "label", "source_path", "processed_path", "sha256", "duplicate_group_size", "canonical_source_path", "is_canonical"
    ]].copy()
    return merged[merged.is_canonical].copy(), manifest


def create_splits(segmentation_csv: str | Path, output_dir: str | Path, audit_csv: str | Path,
                  seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1) -> dict[str, pd.DataFrame]:
    """Create reproducible stratified splits after exact raw-image deduplication.

    ``audit_csv`` is mandatory so a split cannot silently place byte-identical
    images in separate train/validation/test partitions.
    """
    segmentation = pd.read_csv(segmentation_csv)
    segmentation = segmentation[segmentation.status == "ok"].copy()
    frame, dedup_manifest = _deduplicate_exact_images(segmentation, audit_csv)
    if frame.sha256.duplicated().any():
        raise AssertionError("Deduplication failed to make SHA-256 values unique")

    train, holdout = train_test_split(frame, train_size=train_ratio, stratify=frame.label, random_state=seed)
    val_fraction = val_ratio / (1 - train_ratio)
    val, test = train_test_split(holdout, train_size=val_fraction, stratify=holdout.label, random_state=seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dedup_manifest.to_csv(output_dir / "deduplication_manifest.csv", index=False)
    splits = {"train": train, "val": val, "test": test}
    all_paths, all_hashes = set(), set()
    for name, split in splits.items():
        assert not (set(split.processed_path) & all_paths), "Path leakage detected"
        assert not (set(split.sha256) & all_hashes), "Hash leakage detected"
        all_paths.update(split.processed_path)
        all_hashes.update(split.sha256)
        split[["processed_path", "label"]].to_csv(output_dir / f"{name}.csv", index=False)
    summary = {
        "seed": seed,
        "policy": "one deterministic representative per exact raw-image SHA-256",
        "segmentation_ok_before_deduplication": int(len(segmentation)),
        "unique_images_after_deduplication": int(len(frame)),
        "removed_exact_duplicates": int(len(segmentation) - len(frame)),
        "split_counts": {name: int(len(split)) for name, split in splits.items()},
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return splits
