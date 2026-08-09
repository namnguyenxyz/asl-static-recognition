"""Download the team's public Hugging Face dataset repository into data/raw."""
from __future__ import annotations
import argparse
from pathlib import Path
import zipfile
from huggingface_hub import snapshot_download

DEFAULT_REPO = "hnam25/asl-hand-gesture-images"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ASL-HG images from Hugging Face")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--revision", default=None, help="Commit SHA/tag for a reproducible dataset snapshot")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id, repo_type="dataset", local_dir=output, revision=args.revision,
        allow_patterns="**/ASL_Raw_Images.zip",
    )
    archives = list(output.rglob("ASL_Raw_Images.zip"))
    if not archives:
        raise RuntimeError(f"{args.repo_id} does not contain ASL_Raw_Images.zip")
    with zipfile.ZipFile(archives[0]) as archive:
        for item in archive.infolist():
            target = (output / item.filename).resolve()
            if output.resolve() not in target.parents and target != output.resolve():
                raise RuntimeError(f"Unsafe ZIP member: {item.filename}")
        archive.extractall(output)
    image_files = [path for path in output.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    if not image_files:
        raise RuntimeError(
            f"{args.repo_id} downloaded but contains no supported image files. "
            "Push the class directories/images (or update this script for the repository's archive layout)."
        )
    print(f"Downloaded {len(image_files)} images to {output}")


if __name__ == "__main__":
    main()
