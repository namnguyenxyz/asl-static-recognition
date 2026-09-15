#!/usr/bin/env python3
"""Run the final candidate: MediaPipe ROI crops + MobileNetV4 full fine-tuning.

The script intentionally evaluates only once on the pinned P9 split. Select all
training choices using P2 validation before reading test metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=Path('outputs/mp_mnv4_003'))
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--padding-ratio', type=float, default=.18)
    parser.add_argument('--force-recrop', action='store_true')
    parser.add_argument('--hf-repo-id', default='hnam25/asl-hg-mediapipe-roi-transfer')
    # Colab executes source through an IPython kernel and appends `-f kernel.json`.
    # Ignore that kernel-only argument while retaining normal CLI options.
    return parser.parse_known_args()[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def extract_safely(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise RuntimeError(f'Unsafe ZIP member: {member.filename}')
        archive.extractall(destination)


def class_root(root: Path, labels: list[str]) -> Path:
    candidates = [path.parent for path in root.rglob('0') if path.is_dir() and all((path.parent / label).is_dir() for label in labels)]
    if not candidates:
        raise RuntimeError('Could not locate direct class directories after extraction.')
    return min(candidates, key=lambda path: len(path.parts))


def main() -> None:
    args = arguments()
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import torch
        from PIL import Image
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        from matplotlib import pyplot as plt
        from safetensors.torch import load_file, save_file
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        import timm
        from timm.data import resolve_model_data_config
    except ImportError as error:
        raise SystemExit(f'Missing dependency: {error}. Run: python -m pip install -r requirements.txt') from error

    experiment_id = 'mp-mnv4-003-raw-mediapipe-roi-participant-disjoint-full-finetune'
    dataset_repo = 'hnam25/asl-hand-gesture-images'
    dataset_revision = '8f36ac00ece6dfce94410a980a839d93a912d366'
    raw_archive_name = 'ASL_HG_36000/ASL_Raw_Images.zip'
    cnn_repo, cnn_revision = 'hnam25/asl-hg-cnn-baseline', '064551d5634ef34d3e6a9132ab3d4a374b2f5633'
    timm_repo, timm_revision = 'timm/mobilenetv4_conv_small.e2400_r224_in1k', '331fb803779522b685cf942e15f914fb6741c1eb'
    model_name = 'mobilenetv4_conv_small.e2400_r224_in1k'
    task_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    labels = [str(index) for index in range(10)] + [chr(index) for index in range(ord('A'), ord('Z') + 1)]
    seed, lr, weight_decay = 42, 1e-4, 1e-4

    root = args.work_dir.resolve()
    hf_root, raw_dir, crops = root / 'hf', root / 'raw', root / 'mediapipe_crops'
    outputs = root / 'outputs'
    for path in (hf_root, raw_dir, crops, outputs / 'models', outputs / 'metrics', outputs / 'figures', outputs / 'logs', outputs / 'metadata'):
        path.mkdir(parents=True, exist_ok=True)

    token = os.environ.get('HF_TOKEN')
    if not token:
        raise SystemExit('HF_TOKEN is required: this run persists checkpoints to Hugging Face after every improvement.')
    api = HfApi(token=token)
    api.create_repo(repo_id=args.hf_repo_id, repo_type='model', private=False, exist_ok=True)

    def persist(path: Path, path_in_repo: str, message: str) -> None:
        for attempt in range(1, 4):
            try:
                api.upload_file(path_or_fileobj=str(path), path_in_repo=path_in_repo, repo_id=args.hf_repo_id, repo_type='model', commit_message=message)
                return
            except Exception as error:
                if attempt == 3:
                    raise RuntimeError(f'Could not persist {path.name} after 3 attempts') from error
                time.sleep(attempt * 3)

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise SystemExit('A CUDA GPU is required for this 35,441-image fine-tuning run. Use Colab GPU or a CUDA host.')
    print({'experiment_id': experiment_id, 'device': str(device), 'torch': torch.__version__, 'timm': timm.__version__}, flush=True)

    snapshot_download(repo_id=dataset_repo, repo_type='dataset', revision=dataset_revision, local_dir=hf_root, allow_patterns=[raw_archive_name])
    snapshot_download(repo_id=cnn_repo, revision=cnn_revision, local_dir=hf_root / 'cnn_artifact', allow_patterns=['metadata/deduplication_manifest.csv', 'metadata/split_manifest.json', 'metadata/experiment_config.json'])
    archive = hf_root / raw_archive_name
    cnn_root = hf_root / 'cnn_artifact'
    cnn_config = json.loads((cnn_root / 'metadata' / 'experiment_config.json').read_text())
    expected_raw_sha = '594cfa0158044085ed61351315c187a6f3a3f9087795b4f4cdba68ecd04f12b1'
    if sha256(archive) != expected_raw_sha:
        raise RuntimeError('Raw archive checksum differs from the pinned dataset.')
    if not any(raw_dir.iterdir()):
        extract_safely(archive, raw_dir)
    source_root = class_root(raw_dir, labels)

    dedup = pd.read_csv(cnn_root / 'metadata' / 'deduplication_manifest.csv')
    dedup['is_canonical'] = dedup['is_canonical'].astype(bool)
    canonical = dedup[dedup.is_canonical].copy()
    raw_paths = {path.relative_to(source_root).as_posix(): path for label in labels for path in (source_root / label).rglob('*') if path.is_file()}
    canonical['source_path'] = canonical['relative_path'].map(raw_paths)
    if canonical.source_path.isna().any() or len(canonical) != 35441:
        raise RuntimeError('Raw archive does not map exactly to the canonical manifest.')
    split_manifest = json.loads((cnn_root / 'metadata' / 'split_manifest.json').read_text())
    participants = split_manifest['participants']
    canonical['split'] = 'train'
    canonical.loc[canonical.participant.eq(participants['validation']), 'split'] = 'validation'
    canonical.loc[canonical.participant.eq(participants['test']), 'split'] = 'test'
    if set(canonical[canonical.split == 'train'].sha256) & set(canonical[canonical.split == 'test'].sha256):
        raise RuntimeError('Hash leakage detected.')

    previous_manifest_path = outputs / 'metadata' / 'mediapipe_crop_manifest.csv'
    previous_records = {}
    if previous_manifest_path.exists() and not args.force_recrop:
        previous = pd.read_csv(previous_manifest_path)
        required = {'relative_path', 'status', 'fallback_raw'}
        if required <= set(previous.columns):
            previous_records = previous.set_index('relative_path')[['status', 'fallback_raw']].to_dict('index')

    task_path = root / 'hand_landmarker.task'
    if not task_path.exists():
        urllib.request.urlretrieve(task_url, task_path)
    task_sha = sha256(task_path)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(task_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=.5,
        min_hand_presence_confidence=.5,
        min_tracking_confidence=.5,
    )
    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
    records = []
    try:
        for number, row in enumerate(canonical.itertuples(index=False), start=1):
            destination = crops / row.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            cached = previous_records.get(row.relative_path)
            status, used_fallback = 'uncropped', False
            # Reuse a crop only if its prior detection outcome is available;
            # otherwise recrop so coverage statistics cannot be fabricated.
            if destination.exists() and cached is not None and not args.force_recrop:
                status = str(cached['status'])
                value = cached['fallback_raw']
                used_fallback = value if isinstance(value, bool) else str(value).strip().lower() == 'true'
            else:
                image = cv2.imread(str(row.source_path))
                if image is None:
                    raise RuntimeError(f'Unreadable canonical source: {row.source_path}')
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                crop = None
                if result.hand_landmarks:
                    height, width = image.shape[:2]
                    points = result.hand_landmarks[0]
                    xs, ys = [point.x * width for point in points], [point.y * height for point in points]
                    side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * args.padding_ratio)
                    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                    x1, y1 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
                    x2, y2 = min(width, int(cx + side / 2)), min(height, int(cy + side / 2))
                    if x2 > x1 and y2 > y1:
                        crop, status = image[y1:y2, x1:x2], 'ok'
                    else:
                        status, used_fallback = 'invalid_bbox_fallback_raw', True
                else:
                    status, used_fallback = 'no_hand_detected_fallback_raw', True
                if crop is None:
                    crop = image
                if not cv2.imwrite(str(destination), crop):
                    raise RuntimeError(f'Could not write crop: {destination}')
            records.append({'relative_path': row.relative_path, 'crop_path': str(destination), 'label': row.label, 'participant': row.participant, 'split': row.split, 'sha256': row.sha256, 'status': status, 'fallback_raw': used_fallback})
            if number % 500 == 0 or number == len(canonical):
                print(f'[crop] {number}/{len(canonical)}', flush=True)
    finally:
        detector.close()
    cropped = pd.DataFrame(records)
    crop_manifest_path = outputs / 'metadata' / 'mediapipe_crop_manifest.csv'
    cropped.to_csv(crop_manifest_path, index=False)
    coverage = cropped.assign(detected=~cropped.fallback_raw).groupby('split').agg(images=('label', 'size'), detected=('detected', 'sum'), fallback_raw=('fallback_raw', 'sum')).reset_index()
    coverage_path = outputs / 'metrics' / 'detection_coverage.csv'
    coverage.to_csv(coverage_path, index=False)
    persist(crop_manifest_path, 'metadata/mediapipe_crop_manifest.csv', 'Persist MediaPipe crop manifest before training')
    persist(coverage_path, 'metrics/detection_coverage.csv', 'Persist MediaPipe detection coverage before training')
    for split in ('train', 'validation', 'test'):
        cropped[cropped.split.eq(split)][['crop_path', 'label', 'participant', 'sha256', 'status', 'fallback_raw']].to_csv(outputs / 'metadata' / f'{split}.csv', index=False)

    weight_path = Path(hf_hub_download(repo_id=timm_repo, filename='model.safetensors', revision=timm_revision))
    weight_sha = sha256(weight_path)
    model = timm.create_model(model_name, pretrained=False, num_classes=1000)
    loaded = model.load_state_dict(load_file(weight_path), strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError('Pinned ImageNet weights did not load exactly.')
    model.reset_classifier(num_classes=len(labels))
    model = model.to(device)
    config = resolve_model_data_config(model)
    transform = transforms.Compose([transforms.Resize(tuple(config['input_size'][1:])), transforms.ToTensor(), transforms.Normalize(mean=config['mean'], std=config['std'])])
    label_index = {label: index for index, label in enumerate(labels)}

    class ASLDataset(Dataset):
        def __init__(self, frame): self.frame = frame.reset_index(drop=True)
        def __len__(self): return len(self.frame)
        def __getitem__(self, index):
            row = self.frame.iloc[index]
            with Image.open(row.crop_path) as image:
                return transform(image.convert('RGB')), label_index[row.label]

    generator = torch.Generator().manual_seed(seed)
    loaders = {split: DataLoader(ASLDataset(cropped[cropped.split.eq(split)]), batch_size=args.batch_size, shuffle=split == 'train', num_workers=args.num_workers, pin_memory=True, generator=generator) for split in ('train', 'validation', 'test')}
    criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=.2)
    checkpoint = outputs / 'models' / 'mp_mnv4_003_raw_mediapipe_roi_full_finetune.safetensors'
    history, best_loss, stale = [], float('inf'), 0
    for epoch in range(1, args.epochs + 1):
        started = time.time(); model.train(); loss_sum = correct = total = 0
        for images, targets in loaders['train']:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(); logits = model(images); loss = criterion(logits, targets); loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(targets); correct += (logits.argmax(1) == targets).sum().item(); total += len(targets)
        model.eval(); validation_loss = validation_correct = validation_total = 0
        with torch.no_grad():
            for images, targets in loaders['validation']:
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                logits = model(images); validation_loss += criterion(logits, targets).item() * len(targets)
                validation_correct += (logits.argmax(1) == targets).sum().item(); validation_total += len(targets)
        row = {'epoch': epoch, 'train_loss': loss_sum / total, 'train_accuracy': correct / total, 'val_loss': validation_loss / validation_total, 'val_accuracy': validation_correct / validation_total, 'learning_rate': optimizer.param_groups[0]['lr'], 'elapsed_seconds': time.time() - started}
        history.append(row)
        history_path = outputs / 'logs' / 'training_history.csv'
        pd.DataFrame(history).to_csv(history_path, index=False)
        scheduler.step(row['val_loss'])
        persist(history_path, 'logs/training_history.csv', f'Persist training history through epoch {epoch}')
        print(row, flush=True)
        if row['val_loss'] < best_loss:
            best_loss, stale = row['val_loss'], 0
            save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(checkpoint))
            persist(checkpoint, 'models/mp_mnv4_003_raw_mediapipe_roi_full_finetune.safetensors', f'Best validation-loss checkpoint at epoch {epoch}')
        else:
            stale += 1
            if stale >= 5: break

    model.load_state_dict(load_file(checkpoint)); model.eval(); truth, predicted = [], []
    with torch.no_grad():
        for images, targets in loaders['test']:
            predicted.extend(model(images.to(device, non_blocking=True)).argmax(1).cpu().tolist()); truth.extend(targets.tolist())
    report = classification_report(truth, predicted, labels=range(36), target_names=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=range(36))
    pd.DataFrame(report).T.to_csv(outputs / 'metrics' / 'classification_report.csv')
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(outputs / 'metrics' / 'confusion_matrix.csv')
    oi, zi = label_index['O'], label_index['0']
    summary = {'experiment_id': experiment_id, 'test_accuracy': float(accuracy_score(truth, predicted)), 'macro_precision': report['macro avg']['precision'], 'macro_recall': report['macro avg']['recall'], 'macro_f1': report['macro avg']['f1-score'], 'O_recall': report['O']['recall'], '0_recall': report['0']['recall'], 'O_to_0': int(matrix[oi, zi]), '0_to_O': int(matrix[zi, oi]), 'best_validation_accuracy': max(row['val_accuracy'] for row in history), 'epochs_ran': len(history)}
    summary_path = outputs / 'metrics' / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    report_path, matrix_path = outputs / 'metrics' / 'classification_report.csv', outputs / 'metrics' / 'confusion_matrix.csv'
    plt.figure(figsize=(16, 13)); sns.heatmap(matrix, cmap='Blues', xticklabels=labels, yticklabels=labels); plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout(); figure_path = outputs / 'figures' / 'confusion_matrix.png'; plt.savefig(figure_path, dpi=180); plt.close()
    for local_path, remote_path in ((summary_path, 'metrics/summary.json'), (report_path, 'metrics/classification_report.csv'), (matrix_path, 'metrics/confusion_matrix.csv'), (figure_path, 'figures/confusion_matrix.png')):
        persist(local_path, remote_path, 'Persist final evaluation artifact')
    experiment_config = {'experiment_id': experiment_id, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'dataset_repo': dataset_repo, 'dataset_revision': dataset_revision, 'raw_archive': raw_archive_name, 'raw_archive_sha256': expected_raw_sha, 'cnn_artifact_repo': cnn_repo, 'cnn_artifact_revision': cnn_revision, 'split_manifest': split_manifest, 'mediapipe': {'task_url': task_url, 'task_sha256': task_sha, 'num_hands': 1, 'padding_ratio': args.padding_ratio, 'fallback_policy': 'use the unmodified raw image when no hand or valid ROI is detected'}, 'model': {'name': model_name, 'timm_repo': timm_repo, 'timm_revision': timm_revision, 'weights_sha256': weight_sha}, 'training': {'seed': seed, 'epochs_max': args.epochs, 'batch_size': args.batch_size, 'learning_rate': lr, 'weight_decay': weight_decay, 'all_layers_trainable': True}, 'environment': {'torch': torch.__version__, 'timm': timm.__version__, 'mediapipe': mp.__version__}}
    config_path = outputs / 'metadata' / 'experiment_config.json'
    environment_path = outputs / 'metadata' / 'environment.txt'
    config_path.write_text(json.dumps(experiment_config, indent=2))
    environment_path.write_text('\n'.join([f'torch={torch.__version__}', f'timm={timm.__version__}', f'mediapipe={mp.__version__}']) + '\n')
    for local_path, remote_path in ((config_path, 'metadata/experiment_config.json'), (environment_path, 'metadata/environment.txt')):
        persist(local_path, remote_path, 'Persist final run provenance')
    print(json.dumps({'summary': summary, 'detection_coverage': coverage.to_dict(orient='records'), 'artifact_repo': args.hf_repo_id}, indent=2))


if __name__ == '__main__':
    main()
