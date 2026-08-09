from __future__ import annotations
from pathlib import Path
import cv2
import mediapipe as mp
import pandas as pd


class HandCropper:
    """MediaPipe Hand Landmarker cropper. Requires the official .task model asset."""
    def __init__(self, model_path: str, padding_ratio: float = 0.18, detection_confidence: float = 0.5,
                 presence_confidence: float = 0.5, tracking_confidence: float = 0.5):
        model = Path(model_path)
        if not model.exists():
            raise FileNotFoundError(
                f"Missing {model}. Download the MediaPipe Hand Landmarker task model and set mediapipe.model_path."
            )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.padding_ratio = padding_ratio

    def crop(self, bgr_image):
        if bgr_image is None:
            return None, "unreadable"
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            return None, "no_hand_detected"
        height, width = bgr_image.shape[:2]
        points = result.hand_landmarks[0]
        xs, ys = [p.x * width for p in points], [p.y * height for p in points]
        side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * self.padding_ratio)
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        x1, y1 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
        x2, y2 = min(width, int(cx + side / 2)), min(height, int(cy + side / 2))
        if x2 <= x1 or y2 <= y1:
            return None, "invalid_bbox"
        crop = bgr_image[y1:y2, x1:x2]
        return crop, "ok"

    def close(self):
        self.detector.close()


def segment_manifest(manifest: pd.DataFrame, processed_dir: str | Path, cropper: HandCropper) -> pd.DataFrame:
    processed_dir = Path(processed_dir)
    outcomes = []
    for row in manifest[manifest.status == "ok"].itertuples():
        source = Path(row.path)
        image = cv2.imread(str(source))
        crop, status = cropper.crop(image)
        destination = processed_dir / row.label / source.name
        if status == "ok":
            destination.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination), crop)
        outcomes.append({"label": row.label, "source_path": str(source), "processed_path": str(destination), "status": status})
    result = pd.DataFrame(outcomes)
    processed_dir.parent.joinpath("metadata").mkdir(parents=True, exist_ok=True)
    result.to_csv(processed_dir.parent / "metadata" / "segmentation_failures.csv", index=False)
    return result
