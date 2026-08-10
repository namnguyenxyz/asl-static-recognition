"""Run the reusable baseline stages after the equivalent notebooks are validated."""
from pathlib import Path
import argparse
from src.data.validate_dataset import audit_dataset
from src.data.segment_hands import HandCropper, segment_manifest
from src.data.split_dataset import create_splits
from src.models.train import train_baseline
from src.evaluation.evaluate import evaluate_model
from src.utils.config import ensure_output_dirs, load_config
from src.utils.labels import save_classes


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--stage", choices=["audit", "segment", "split", "train", "evaluate", "all"], default="all")
    args = parser.parse_args(); cfg = load_config(args.config); ensure_output_dirs(cfg)
    save_classes(cfg["classes"], Path(cfg["paths"]["metadata"]) / "class_names.json")
    if args.stage in ("audit", "all"):
        audit_dataset(cfg["paths"]["raw"], cfg["classes"], cfg["paths"]["metadata"])
    if args.stage in ("segment", "all"):
        import pandas as pd
        cropper = HandCropper(cfg["mediapipe"]["model_path"], cfg["mediapipe"]["padding_ratio"], cfg["mediapipe"]["min_hand_detection_confidence"], cfg["mediapipe"]["min_hand_presence_confidence"], cfg["mediapipe"]["min_tracking_confidence"])
        try: segment_manifest(pd.read_csv(Path(cfg["paths"]["metadata"]) / "dataset_audit.csv"), cfg["paths"]["processed"], cropper)
        finally: cropper.close()
    if args.stage in ("split", "all"):
        create_splits(Path(cfg["paths"]["metadata"]) / "segmentation_failures.csv", cfg["paths"]["splits"], Path(cfg["paths"]["metadata"]) / "dataset_audit.csv", cfg["seed"], cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
    if args.stage in ("train", "all"): train_baseline(cfg)
    if args.stage in ("evaluate", "all"): print(evaluate_model(cfg))


if __name__ == "__main__": main()
