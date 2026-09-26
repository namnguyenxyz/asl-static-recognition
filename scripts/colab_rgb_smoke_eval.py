"""Small, reproducible external RGB smoke test for the locked ROI candidate."""
from pathlib import Path
import hashlib
import json
import random
import subprocess
import sys
import urllib.request


OUT = Path('/content/asl_rgb_smoke_eval')
DATASET = 'Marxulia/asl_sign_languages_alphabets_v03'
MODEL_REPO = 'hnam25/asl-hg-mediapipe-roi-p010-final'
TASK_URL = ('https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
            'hand_landmarker/float16/1/hand_landmarker.task')
LETTERS = [chr(code) for code in range(ord('a'), ord('z') + 1) if chr(code) not in {'j', 'z'}]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''): value.update(block)
    return value.hexdigest()


def main() -> None:
    root = Path('/content/asl-static-recognition')
    if not root.exists():
        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/namnguyenxyz/asl-static-recognition.git', str(root)], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(root / 'requirements-colab-transfer.txt'), 'datasets>=3.0,<4'], check=True)
    import cv2
    import mediapipe as mp
    import numpy as np
    import pandas as pd
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from safetensors.torch import load_file
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    import timm
    from timm.data import resolve_model_data_config

    OUT.mkdir(parents=True, exist_ok=True)
    data = load_dataset(DATASET, split='train')
    label_names = data.features['label'].names
    normalized = {name.lower(): index for index, name in enumerate(label_names)}
    if set(LETTERS) - set(normalized): raise RuntimeError(f'Unexpected source labels: {label_names}')
    rng = random.Random(42)
    indices_by_label = {}
    for index, label_id in enumerate(data['label']):
        indices_by_label.setdefault(label_id, []).append(index)
    selected = []
    for letter in LETTERS:
        indices = indices_by_label.get(normalized[letter], [])
        if len(indices) < 25: raise RuntimeError(f'Not enough images for {letter}: {len(indices)}')
        selected.extend((letter, index) for index in rng.sample(indices, 25))
    rng.shuffle(selected)
    task = OUT / 'hand_landmarker.task'
    if not task.exists(): urllib.request.urlretrieve(TASK_URL, task)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(task)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=.5, min_hand_presence_confidence=.5, min_tracking_confidence=.5)
    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
    crops = OUT / 'crops'; crops.mkdir(exist_ok=True)
    records = []
    try:
        for number, (letter, source_index) in enumerate(selected):
            image = np.asarray(data[source_index]['image'].convert('RGB'))
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
            crop, detected, status = None, False, 'no_hand_detected'
            if result.hand_landmarks:
                height, width = image.shape[:2]; points = result.hand_landmarks[0]
                xs, ys = [point.x * width for point in points], [point.y * height for point in points]
                side = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.2
                cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                x1, y1 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
                x2, y2 = min(width, int(cx + side / 2)), min(height, int(cy + side / 2))
                if x2 > x1 and y2 > y1: crop, detected, status = image[y1:y2, x1:x2], True, 'ok'
            if crop is None: crop, status = image, status + '_fallback_raw'
            crop_path = crops / f'{number:04d}.png'
            if not cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)): raise RuntimeError('crop write failed')
            records.append({'source_index': source_index, 'truth': letter, 'crop_path': str(crop_path), 'roi_detected': detected, 'status': status})
    finally:
        detector.close()
    frame = pd.DataFrame(records)
    checkpoint = Path(hf_hub_download(repo_id=MODEL_REPO, filename='models/model.safetensors', local_dir=str(OUT / 'checkpoint')))
    model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=False, num_classes=1000)
    model.reset_classifier(num_classes=36); model.load_state_dict(load_file(checkpoint), strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda': raise RuntimeError('Expected Colab GPU.')
    model.to(device).eval(); config = resolve_model_data_config(model)
    transform = transforms.Compose([transforms.Resize(tuple(config['input_size'][1:])), transforms.ToTensor(), transforms.Normalize(mean=config['mean'], std=config['std'])])
    class Samples(Dataset):
        def __len__(self): return len(frame)
        def __getitem__(self, index):
            with Image.open(frame.iloc[index].crop_path) as image: return transform(image.convert('RGB')), LETTERS.index(frame.iloc[index].truth)
    allowed = torch.tensor([10 + ord(letter) - ord('a') for letter in LETTERS], device=device)
    truth, pred = [], []
    with torch.inference_mode():
        for images, labels in DataLoader(Samples(), batch_size=128, num_workers=2, pin_memory=True):
            output = allowed[model(images.to(device)).index_select(1, allowed).argmax(1)] - 10
            truth.extend(labels.tolist()); pred.extend(output.cpu().tolist())
    truth_letters = [LETTERS[item] for item in truth]
    prediction_letters = [chr(ord('a') + item) for item in pred]
    frame['prediction'] = prediction_letters
    frame.to_csv(OUT / 'predictions.csv', index=False)
    pd.DataFrame(classification_report(truth_letters, prediction_letters, labels=LETTERS, target_names=[item.upper() for item in LETTERS], output_dict=True, zero_division=0)).transpose().to_csv(OUT / 'classification_report.csv')
    summary = {'evaluation_type': 'external_rgb_smoke_test_no_training_or_adaptation', 'dataset': DATASET, 'dataset_file': 'data/train-00000-of-00001.parquet', 'sampling': 'seed 42; 25 images per shared static letter; 600 images total', 'model_repo': MODEL_REPO, 'model_sha256': digest(checkpoint), 'preprocess': 'MediaPipe ROI p=0.10, raw-image fallback; model-native resize/normalization', 'roi_detection_coverage': float(frame.roi_detected.mean()), 'accuracy': accuracy_score(truth_letters, prediction_letters), 'macro_f1': f1_score(truth_letters, prediction_letters, labels=LETTERS, average='macro', zero_division=0), 'device': str(device)}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
