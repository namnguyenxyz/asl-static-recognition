# Reusable audit metadata cache

Audit-only metadata for the ASL image dataset has been published to Hugging Face.

- Dataset repository: `hnam25/asl-hand-gesture-images`
- Audit artifact directory: `metadata/colab-audit-2026-08-10/`
- Source dataset revision audited: `bad9dd9297697ec2da901562b203a5a14ebb93d1`
- Raw archive: `ASL_HG_36000/ASL_Raw_Images.zip`
- Raw archive SHA-256: `594cfa0158044085ed61351315c187a6f3a3f9087795b4f4cdba68ecd04f12b1`
- Publish commit: `8f36ac00ece6dfce94410a980a839d93a912d366`

## Artifacts

- `audit.csv` and `audit.parquet`: one row per source image, including label, portable relative path, status and SHA-256.
- `duplicates.csv` and `duplicates.parquet`: every image in an exact duplicate hash group.
- `class_distribution.csv`: readable-image count by class.
- `cache_manifest.json`: schema, source revision, raw archive fingerprint and aggregate counts.

The audit found 36,000 readable images, no unreadable files, and 994 images in 435 exact duplicate hash groups. Do not split individual images at random before handling/grouping those duplicates.

## Reuse without re-auditing

Download the raw archive at the pinned source revision and download the metadata directory. Before using `audit.csv`, calculate the SHA-256 of the local raw archive and compare it with `raw_archive_sha256` in `cache_manifest.json`. Only reuse the cache if they match.

```bash
hf download hnam25/asl-hand-gesture-images \
  ASL_HG_36000/ASL_Raw_Images.zip \
  --repo-type dataset \
  --revision bad9dd9297697ec2da901562b203a5a14ebb93d1 \
  --local-dir data/raw

hf download hnam25/asl-hand-gesture-images \
  metadata/colab-audit-2026-08-10 \
  --repo-type dataset \
  --local-dir cache-download/audit

sha256sum data/raw/ASL_HG_36000/ASL_Raw_Images.zip
```

`notebooks/ASL_End_to_End_Colab.ipynb` reuses `data/metadata/dataset_audit.csv` when it already exists, but it does not automatically download or fingerprint-check this published cache. Perform the manifest check above, then copy/use the validated `audit.csv` deliberately. `notebooks/01_audit_only_publish.ipynb` is the reproducible producer for a new cache version when the raw archive changes.
