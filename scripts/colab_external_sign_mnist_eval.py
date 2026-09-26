"""Evaluate the locked MediaPipe-ROI candidate on external Sign Language MNIST.

This is deliberately evaluation-only: it downloads the published test CSV, never
uses its training CSV, and writes a compact, reproducible evidence bundle.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys


ROOT = Path('/content/asl-static-recognition')
OUT = Path('/content/external_sign_mnist')
CSV_URL = ('https://raw.githubusercontent.com/gurpreet0610/sign_language_CNN/'
           'master/sign-language-mnist/sign_mnist_test.csv')
MODEL_REPO = 'hnam25/asl-hg-mediapipe-roi-p010-final'
TIMM_MODEL = 'mobilenetv4_conv_small.e2400_r224_in1k'
# Sign Language MNIST omits dynamic J (9) and Z (25). Its numeric labels retain
# alphabet positions, so they map directly to the project's A--Z positions +10.
ASL_MNIST_LABELS = [index for index in range(26) if index not in (9, 25)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/namnguyenxyz/asl-static-recognition.git', str(ROOT)], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                    str(ROOT / 'requirements-colab-transfer.txt')], check=True)

    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from safetensors.torch import load_file
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    import timm
    from timm.data import resolve_model_data_config

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / 'sign_mnist_test.csv'
    subprocess.run(['curl', '-L', '--fail', '--retry', '3', '--retry-delay', '3',
                    '-o', str(csv_path), CSV_URL], check=True)
    frame = pd.read_csv(csv_path)
    if list(frame.columns[:2]) != ['label', 'pixel1'] or frame.shape != (7172, 785):
        raise RuntimeError(f'Unexpected Sign Language MNIST test CSV shape/schema: {frame.shape}')
    if set(frame.label.unique()) != set(ASL_MNIST_LABELS):
        raise RuntimeError('Unexpected external label set.')

    checkpoint = Path(hf_hub_download(repo_id=MODEL_REPO, filename='models/model.safetensors',
                                      local_dir=str(OUT / 'checkpoint')))
    model = timm.create_model(TIMM_MODEL, pretrained=False, num_classes=1000)
    model.reset_classifier(num_classes=36)
    model.load_state_dict(load_file(checkpoint), strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('Expected a Colab GPU for this evaluation.')
    model.to(device).eval()
    config = resolve_model_data_config(model)
    transform = transforms.Compose([
        transforms.Resize(tuple(config['input_size'][1:])), transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std']),
    ])

    class TestDataset(Dataset):
        def __len__(self): return len(frame)
        def __getitem__(self, index):
            row = frame.iloc[index]
            pixels = row.iloc[1:].to_numpy(dtype=np.uint8).reshape(28, 28)
            # The source is grayscale; replicate it into RGB, then use exactly the
            # model's ImageNet resize/normalization. No retraining or adaptation.
            image = Image.fromarray(pixels, mode='L').convert('RGB')
            return transform(image), int(row.label)

    loader = DataLoader(TestDataset(), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    allowed_model_indices = torch.tensor([10 + item for item in ASL_MNIST_LABELS], device=device)
    truth, predicted = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            # Evaluate in the shared 24-letter label space. Digits/J/Z are outside
            # this external benchmark rather than plausible target classes.
            outputs = allowed_model_indices[logits.index_select(1, allowed_model_indices).argmax(1)] - 10
            truth.extend(labels.tolist())
            predicted.extend(outputs.cpu().tolist())

    names = [chr(ord('A') + item) for item in ASL_MNIST_LABELS]
    report = classification_report(truth, predicted, labels=ASL_MNIST_LABELS,
                                   target_names=names, output_dict=True, zero_division=0)
    summary = {
        'evaluation_type': 'external_test_only_no_training_or_adaptation',
        'model_repo': MODEL_REPO,
        'model_file': 'models/model.safetensors',
        'model_sha256': sha256(checkpoint),
        'dataset_name': 'Sign Language MNIST test split',
        'dataset_source_url': CSV_URL,
        'dataset_sha256': sha256(csv_path),
        'samples': len(truth),
        'source_image_format': '28x28 grayscale CSV',
        'inference_transform': 'grayscale->RGB replication; model-native resize and normalization',
        'class_protocol': '24 shared static letters A-I,K-Y; logits restricted to those 24 labels',
        'excluded_dynamic_letters': ['J', 'Z'],
        'accuracy': accuracy_score(truth, predicted),
        'macro_f1': f1_score(truth, predicted, labels=ASL_MNIST_LABELS, average='macro', zero_division=0),
        'device': str(device),
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    pd.DataFrame(report).transpose().to_csv(OUT / 'classification_report.csv')
    pd.DataFrame({'truth_external_label': truth, 'prediction_external_label': predicted}).to_csv(
        OUT / 'predictions.csv', index=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
