#!/usr/bin/env python3
"""Publish Colab-built ASL audit and MediaPipe cache artifacts from local disk."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "hnam25/asl-hand-gesture-images"
DEFAULT_PREFIX = "derived/mediapipe-hand-landmarker-v1"
REQUIRED_FILES = (
    "audit.parquet",
    "duplicates.parquet",
    "segmentation_manifest.parquet",
    "cache_manifest.json",
    "processed_hand_crops.zip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Local directory downloaded from /content/asl-cache-build/data/metadata.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required in the local environment; it is never read on Colab.")

    cache_dir = args.cache_dir.expanduser().resolve()
    missing = [name for name in REQUIRED_FILES if not (cache_dir / name).is_file()]
    if missing:
        raise SystemExit(f"Missing cache artifact(s) in {cache_dir}: {', '.join(missing)}")

    api = HfApi(token=token)
    for name in REQUIRED_FILES:
        path = cache_dir / name
        target = f"{args.prefix.strip('/')}/{name}"
        print(f"Uploading {path.name} ({path.stat().st_size:,} bytes) -> {target}", flush=True)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=target,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Publish reusable ASL cache: {name}",
        )

    print(f"Published cache to https://huggingface.co/datasets/{args.repo_id}/tree/main/{args.prefix}")


if __name__ == "__main__":
    main()
