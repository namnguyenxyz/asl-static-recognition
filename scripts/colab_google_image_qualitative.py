"""Qualitative 36-class check on one Google-discovered educational image per class.

The inputs are transiently downloaded from Pics4Learning URLs surfaced by Google
Image Search.  This is a demonstration table, not an accuracy benchmark.
"""
from pathlib import Path
import csv
import json
import subprocess
import sys
import urllib.request


OUT = Path('/content/google_image_qualitative')
SOURCE = 'https://images2.pics4learning.com/catalog/{label}/{label}.jpg'
TASK_URL = ('https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
            'hand_landmarker/float16/1/hand_landmarker.task')
MODEL_REPO = 'hnam25/asl-hg-mediapipe-roi-p010-final'
LABELS = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def main() -> None:
    root = Path('/content/asl-static-recognition')
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/namnguyenxyz/asl-static-recognition.git', str(root)], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(root / 'requirements-colab-transfer.txt')], check=True)
    import cv2
    import mediapipe as mp
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from safetensors.torch import load_file
    from torchvision import transforms
    import timm
    from timm.data import resolve_model_data_config

    OUT.mkdir(parents=True, exist_ok=True)
    images = OUT / 'images'; images.mkdir(exist_ok=True)
    for label in LABELS:
        url = SOURCE.format(label=label.lower())
        try:
            urllib.request.urlretrieve(url, images / f'{label}.jpg')
        except Exception as error:
            raise RuntimeError(f'Could not download source image for {label}: {url}') from error
    task = OUT / 'hand_landmarker.task'; urllib.request.urlretrieve(TASK_URL, task)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(task)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=.5, min_hand_presence_confidence=.5, min_tracking_confidence=.5)
    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
    model_path = Path(hf_hub_download(repo_id=MODEL_REPO, filename='models/model.safetensors', local_dir=str(OUT / 'checkpoint')))
    model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=False, num_classes=1000)
    model.reset_classifier(num_classes=36); model.load_state_dict(load_file(model_path), strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda': raise RuntimeError('Expected Colab GPU.')
    model.to(device).eval(); config = resolve_model_data_config(model)
    transform = transforms.Compose([transforms.Resize(tuple(config['input_size'][1:])), transforms.ToTensor(), transforms.Normalize(mean=config['mean'], std=config['std'])])
    def predict(image):
        tensor = transform(Image.fromarray(image).convert('RGB')).unsqueeze(0).to(device)
        with torch.inference_mode(): probabilities = torch.softmax(model(tensor), dim=1)[0]
        index = int(probabilities.argmax().item())
        return LABELS[index], float(probabilities[index].item())
    rows = []
    try:
        for label in LABELS:
            path = images / f'{label}.jpg'; image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            raw_label, raw_conf = predict(image)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
            crop, detected, status = image, False, 'no_hand_detected_fallback_raw'
            if result.hand_landmarks:
                height, width = image.shape[:2]; points = result.hand_landmarks[0]
                xs, ys = [point.x * width for point in points], [point.y * height for point in points]
                side = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.2
                cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                x1, y1 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
                x2, y2 = min(width, int(cx + side / 2)), min(height, int(cy + side / 2))
                if x2 > x1 and y2 > y1:
                    crop, detected, status = image[y1:y2, x1:x2], True, 'roi_p010'
            roi_label, roi_conf = predict(crop)
            rows.append({'expected_label': label, 'source_url': SOURCE.format(label=label.lower()),
                         'raw_prediction': raw_label, 'raw_confidence': raw_conf,
                         'mediapipe_roi_detected': detected, 'roi_status': status,
                         'roi_prediction': roi_label, 'roi_confidence': roi_conf})
    finally:
        detector.close()
    with (OUT / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = {'type': 'qualitative_google_image_demo_not_a_benchmark', 'source': 'Pics4Learning images located via Google Image Search', 'classes': len(rows), 'model_repo': MODEL_REPO, 'preprocess_variants': ['raw_image', 'mediapipe_roi_padding_0.10_with_raw_fallback'], 'raw_exact_matches': sum(row['expected_label'] == row['raw_prediction'] for row in rows), 'roi_exact_matches': sum(row['expected_label'] == row['roi_prediction'] for row in rows), 'roi_detection_count': sum(row['mediapipe_roi_detected'] for row in rows)}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
