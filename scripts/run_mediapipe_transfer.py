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
import sys
import time
import tarfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=Path('outputs/mp_mnv4_003'))
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--input-mode', choices=('mediapipe_roi', 'raw_image'), default='mediapipe_roi')
    parser.add_argument('--padding-ratio', type=float, default=.18)
    parser.add_argument('--fallback-policy', choices=('raw_image', 'center_crop'), default='raw_image')
    parser.add_argument('--selection-only', action='store_true', help='Evaluate P2 only; never read the locked P9 test split.')
    parser.add_argument('--no-upload', action='store_true', help='Keep artifacts in --work-dir instead of publishing them to Hugging Face.')
    parser.add_argument('--resume-state', type=Path, help='Resume from an atomic training_state.pt saved after a completed epoch.')
    parser.add_argument('--stop-after-crop', action='store_true', help='Build and archive the crop cache, then exit before model training.')
    parser.add_argument('--crop-cache-archive', type=Path, help='Write a portable crop-cache archive after cropping.')
    parser.add_argument('--restore-crop-archive', type=Path, help='Restore a crop-cache archive into --work-dir before processing.')
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


def extract_tar_safely(archive_path: Path, destination: Path) -> None:
    """Extract a locally created crop archive without permitting path traversal."""
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise RuntimeError(f'Unsafe TAR member: {member.name}')
        archive.extractall(destination, filter='data')


def create_crop_archive(archive_path: Path, root: Path, crops: Path, metadata: Path, coverage_path: Path) -> None:
    """Atomically package reusable crops plus manifests needed to validate them."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix('.partial')
    with tarfile.open(temporary, 'w:gz') as archive:
        archive.add(crops, arcname=str(crops.relative_to(root)))
        for path in (metadata / 'mediapipe_crop_manifest.csv', metadata / 'train.csv', metadata / 'validation.csv', metadata / 'test.csv', coverage_path):
            archive.add(path, arcname=str(path.relative_to(root)))
    temporary.replace(archive_path)


def class_root(root: Path, labels: list[str]) -> Path:
    candidates = [path.parent for path in root.rglob('0') if path.is_dir() and all((path.parent / label).is_dir() for label in labels)]
    if not candidates:
        raise RuntimeError('Could not locate direct class directories after extraction.')
    return min(candidates, key=lambda path: len(path.parts))


def main() -> None:
    args = arguments()
    if args.padding_ratio < 0:
        raise SystemExit('--padding-ratio must be non-negative.')
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

    variant = f'{args.input_mode}-padding-{args.padding_ratio:g}-fallback-{args.fallback_policy}'
    experiment_id = f'mp-mnv4-003-{variant}-participant-disjoint-full-finetune'
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
    crop_cache_name = variant.replace('.', '_')
    hf_root, raw_dir, crops = root / 'hf', root / 'raw', root / 'crop_cache' / crop_cache_name
    outputs = root / 'outputs'
    for path in (hf_root, raw_dir, crops, outputs / 'models', outputs / 'metrics', outputs / 'figures', outputs / 'logs', outputs / 'metadata'):
        path.mkdir(parents=True, exist_ok=True)
    if args.restore_crop_archive:
        if not args.restore_crop_archive.is_file():
            raise SystemExit(f'Crop cache archive not found: {args.restore_crop_archive}')
        extract_tar_safely(args.restore_crop_archive, root)
        print({'restored_crop_archive': str(args.restore_crop_archive)}, flush=True)

    token = os.environ.get('HF_TOKEN')
    if not token and not args.no_upload:
        raise SystemExit('HF_TOKEN is required: this run persists checkpoints to Hugging Face after every improvement.')
    api = HfApi(token=token) if token else None
    if api is not None:
        api.create_repo(repo_id=args.hf_repo_id, repo_type='model', private=False, exist_ok=True)

    def persist(path: Path, path_in_repo: str, message: str) -> None:
        if api is None:
            return
        for attempt in range(1, 4):
            try:
                api.upload_file(path_or_fileobj=str(path), path_in_repo=path_in_repo, repo_id=args.hf_repo_id, repo_type='model', commit_message=message)
                return
            except Exception as error:
                if attempt == 3:
                    raise RuntimeError(f'Could not persist {path.name} after 3 attempts') from error
                time.sleep(attempt * 3)

    def download_snapshot(**kwargs) -> None:
        """Retry transient Hugging Face rate limits without changing revisions."""
        for attempt in range(1, 6):
            try:
                snapshot_download(**kwargs)
                return
            except Exception as error:
                if attempt == 5:
                    raise
                delay = attempt * 20
                print(f'[download] attempt {attempt}/5 failed ({type(error).__name__}); retrying in {delay}s', flush=True)
                time.sleep(delay)

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise SystemExit('A CUDA GPU is required for this 35,441-image fine-tuning run. Use Colab GPU or a CUDA host.')
    print({'experiment_id': experiment_id, 'device': str(device), 'torch': torch.__version__, 'timm': timm.__version__}, flush=True)

    download_snapshot(repo_id=dataset_repo, repo_type='dataset', revision=dataset_revision, local_dir=hf_root, allow_patterns=[raw_archive_name])
    download_snapshot(repo_id=cnn_repo, revision=cnn_revision, local_dir=hf_root / 'cnn_artifact', allow_patterns=['metadata/deduplication_manifest.csv', 'metadata/split_manifest.json', 'metadata/experiment_config.json'])
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

    preprocess_signature = json.dumps({
        'input_mode': args.input_mode,
        'padding_ratio': args.padding_ratio,
        'fallback_policy': args.fallback_policy,
    }, sort_keys=True)
    previous_manifest_path = outputs / 'metadata' / 'mediapipe_crop_manifest.csv'
    previous_records = {}
    if previous_manifest_path.exists() and not args.force_recrop:
        previous = pd.read_csv(previous_manifest_path)
        required = {'relative_path', 'status', 'fallback_raw', 'roi_detected', 'preprocess_signature'}
        if required <= set(previous.columns):
            previous = previous[previous.preprocess_signature.eq(preprocess_signature)]
            previous_records = previous.set_index('relative_path')[['status', 'fallback_raw', 'roi_detected']].to_dict('index')

    detector, task_sha = None, None
    if args.input_mode == 'mediapipe_roi':
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
            status, used_fallback, roi_detected = 'uncropped', False, False
            # Reuse a crop only if its prior detection outcome is available;
            # otherwise recrop so coverage statistics cannot be fabricated.
            if destination.exists() and cached is not None and not args.force_recrop:
                status = str(cached['status'])
                value = cached['fallback_raw']
                used_fallback = value if isinstance(value, bool) else str(value).strip().lower() == 'true'
                value = cached['roi_detected']
                roi_detected = value if isinstance(value, bool) else str(value).strip().lower() == 'true'
            else:
                image = cv2.imread(str(row.source_path))
                if image is None:
                    raise RuntimeError(f'Unreadable canonical source: {row.source_path}')
                crop = image if args.input_mode == 'raw_image' else None
                if args.input_mode == 'raw_image':
                    status = 'raw_image'
                else:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                    if result.hand_landmarks:
                        height, width = image.shape[:2]
                        points = result.hand_landmarks[0]
                        xs, ys = [point.x * width for point in points], [point.y * height for point in points]
                        side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * args.padding_ratio)
                        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                        x1, y1 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
                        x2, y2 = min(width, int(cx + side / 2)), min(height, int(cy + side / 2))
                        if x2 > x1 and y2 > y1:
                            crop, status, roi_detected = image[y1:y2, x1:x2], 'ok', True
                        else:
                            status = 'invalid_bbox'
                    else:
                        status = 'no_hand_detected'
                    if crop is None:
                        used_fallback = True
                        if args.fallback_policy == 'raw_image':
                            crop, status = image, f'{status}_fallback_raw'
                        else:
                            height, width = image.shape[:2]
                            side, x1, y1 = min(height, width), (width - min(height, width)) // 2, (height - min(height, width)) // 2
                            crop, status = image[y1:y1 + side, x1:x1 + side], f'{status}_fallback_center_crop'
                if not cv2.imwrite(str(destination), crop):
                    raise RuntimeError(f'Could not write crop: {destination}')
            records.append({'relative_path': row.relative_path, 'crop_path': str(destination), 'label': row.label, 'participant': row.participant, 'split': row.split, 'sha256': row.sha256, 'status': status, 'fallback_raw': used_fallback, 'roi_detected': roi_detected, 'input_mode': args.input_mode, 'fallback_policy': args.fallback_policy, 'preprocess_signature': preprocess_signature})
            if number % 500 == 0 or number == len(canonical):
                print(f'[crop] {number}/{len(canonical)}', flush=True)
    finally:
        if detector is not None:
            detector.close()
    cropped = pd.DataFrame(records)
    crop_manifest_path = outputs / 'metadata' / 'mediapipe_crop_manifest.csv'
    cropped.to_csv(crop_manifest_path, index=False)
    coverage = cropped.groupby('split').agg(images=('label', 'size'), roi_detected=('roi_detected', 'sum'), fallback_images=('fallback_raw', 'sum')).reset_index()
    coverage['roi_attempted'] = args.input_mode == 'mediapipe_roi'
    coverage['roi_detection_coverage'] = coverage['roi_detected'] / coverage['images'] if args.input_mode == 'mediapipe_roi' else None
    coverage_path = outputs / 'metrics' / 'detection_coverage.csv'
    coverage.to_csv(coverage_path, index=False)
    persist(crop_manifest_path, 'metadata/mediapipe_crop_manifest.csv', 'Persist MediaPipe crop manifest before training')
    persist(coverage_path, 'metrics/detection_coverage.csv', 'Persist MediaPipe detection coverage before training')
    for split in ('train', 'validation', 'test'):
        cropped[cropped.split.eq(split)][['relative_path', 'crop_path', 'label', 'participant', 'sha256', 'status', 'fallback_raw', 'roi_detected', 'input_mode', 'fallback_policy']].to_csv(outputs / 'metadata' / f'{split}.csv', index=False)
    crop_archive_path = args.crop_cache_archive or outputs / 'metadata' / f'crop_cache_{crop_cache_name}.tar.gz'
    if args.stop_after_crop:
        create_crop_archive(crop_archive_path, root, crops, outputs / 'metadata', coverage_path)
        print(json.dumps({'crop_cache_archive': str(crop_archive_path), 'sha256': sha256(crop_archive_path), 'images': len(cropped)}, indent=2))
        return

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
    training_state_path = outputs / 'models' / 'training_state.pt'
    history, best_loss, stale, start_epoch, best_model_state = [], float('inf'), 0, 1, None
    if args.resume_state:
        if not args.resume_state.is_file():
            raise SystemExit(f'Resume state not found: {args.resume_state}')
        state = torch.load(args.resume_state, map_location='cpu', weights_only=False)
        if state.get('experiment_id') != experiment_id or state.get('preprocess_signature') != preprocess_signature:
            raise RuntimeError('Resume state belongs to a different experiment or preprocessing variant.')
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        history = state['history']
        best_loss, stale, start_epoch = state['best_loss'], state['stale'], state['next_epoch']
        best_model_state = state['best_model_state']
        print({'resume_state': str(args.resume_state), 'next_epoch': start_epoch, 'completed_epochs': len(history)}, flush=True)

    def save_training_state(next_epoch: int) -> None:
        """Atomically persist all state needed to continue after a runtime loss."""
        payload = {
            'format_version': 1,
            'experiment_id': experiment_id,
            'preprocess_signature': preprocess_signature,
            'next_epoch': next_epoch,
            'model_state': {key: value.detach().cpu() for key, value in model.state_dict().items()},
            'best_model_state': best_model_state,
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'history': history,
            'best_loss': best_loss,
            'stale': stale,
        }
        temporary = training_state_path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(training_state_path)

    for epoch in range(start_epoch, args.epochs + 1):
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
            best_model_state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
            save_file(best_model_state, str(checkpoint))
            persist(checkpoint, 'models/mp_mnv4_003_raw_mediapipe_roi_full_finetune.safetensors', f'Best validation-loss checkpoint at epoch {epoch}')
        else:
            stale += 1
        save_training_state(epoch + 1)
        persist(training_state_path, 'models/training_state.pt', f'Persist resumable training state through epoch {epoch}')
        if stale >= 5:
            break

    if best_model_state is None:
        raise RuntimeError('No completed epoch produced a checkpoint.')
    model.load_state_dict(best_model_state); model.eval()

    def collect_predictions(split: str) -> tuple[list[int], list[int], list[float], list[float]]:
        truth, predicted, confidences, margins = [], [], [], []
        with torch.no_grad():
            for images, targets in loaders[split]:
                probabilities = torch.softmax(model(images.to(device, non_blocking=True)), dim=1).cpu()
                top_values, top_indices = probabilities.topk(k=2, dim=1)
                predicted.extend(top_indices[:, 0].tolist())
                confidences.extend(top_values[:, 0].tolist())
                margins.extend((top_values[:, 0] - top_values[:, 1]).tolist())
                truth.extend(targets.tolist())
        return truth, predicted, confidences, margins

    validation_truth, validation_predicted, _, _ = collect_predictions('validation')
    validation_report = classification_report(validation_truth, validation_predicted, labels=range(36), target_names=labels, output_dict=True, zero_division=0)
    validation_summary = {
        'experiment_id': experiment_id,
        'selection_split': 'validation',
        'selection_only': args.selection_only,
        'accuracy': float(accuracy_score(validation_truth, validation_predicted)),
        'macro_f1': validation_report['macro avg']['f1-score'],
        'macro_precision': validation_report['macro avg']['precision'],
        'macro_recall': validation_report['macro avg']['recall'],
        'epochs_ran': len(history),
    }
    validation_summary_path = outputs / 'metrics' / 'validation_summary.json'
    validation_summary_path.write_text(json.dumps(validation_summary, indent=2))
    persist(validation_summary_path, 'metrics/validation_summary.json', 'Persist validation-only model-selection metrics')
    if args.selection_only:
        print(json.dumps({'validation_summary': validation_summary, 'artifact_repo': None if args.no_upload else args.hf_repo_id}, indent=2))
        return

    truth, predicted, confidences, margins = collect_predictions('test')
    report = classification_report(truth, predicted, labels=range(36), target_names=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=range(36))
    pd.DataFrame(report).T.to_csv(outputs / 'metrics' / 'classification_report.csv')
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(outputs / 'metrics' / 'confusion_matrix.csv')
    test_rows = cropped[cropped.split.eq('test')].reset_index(drop=True)
    predictions_frame = test_rows[['relative_path', 'crop_path', 'label', 'participant', 'status', 'fallback_raw', 'roi_detected']].rename(columns={'label': 'true_label'})
    predictions_frame['predicted_label'] = [labels[index] for index in predicted]
    predictions_frame['confidence'] = confidences
    predictions_frame['margin'] = margins
    predictions_frame['correct'] = predictions_frame.true_label.eq(predictions_frame.predicted_label)
    predictions_path = outputs / 'metrics' / 'predictions.csv'
    predictions_frame.to_csv(predictions_path, index=False)
    from src.evaluation.error_analysis import summarize_errors
    error_examples, error_pairs, error_by_class = summarize_errors(predictions_frame)
    error_examples_path, error_pairs_path, error_by_class_path = outputs / 'metrics' / 'error_examples.csv', outputs / 'metrics' / 'error_pairs.csv', outputs / 'metrics' / 'error_by_class.csv'
    error_examples.to_csv(error_examples_path, index=False)
    error_pairs.to_csv(error_pairs_path, index=False)
    error_by_class.to_csv(error_by_class_path, index=False)
    oi, zi = label_index['O'], label_index['0']
    summary = {'experiment_id': experiment_id, 'test_accuracy': float(accuracy_score(truth, predicted)), 'macro_precision': report['macro avg']['precision'], 'macro_recall': report['macro avg']['recall'], 'macro_f1': report['macro avg']['f1-score'], 'O_recall': report['O']['recall'], '0_recall': report['0']['recall'], 'O_to_0': int(matrix[oi, zi]), '0_to_O': int(matrix[zi, oi]), 'best_validation_accuracy': max(row['val_accuracy'] for row in history), 'epochs_ran': len(history)}
    summary_path = outputs / 'metrics' / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    report_path, matrix_path = outputs / 'metrics' / 'classification_report.csv', outputs / 'metrics' / 'confusion_matrix.csv'
    plt.figure(figsize=(16, 13)); sns.heatmap(matrix, cmap='Blues', xticklabels=labels, yticklabels=labels); plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout(); figure_path = outputs / 'figures' / 'confusion_matrix.png'; plt.savefig(figure_path, dpi=180); plt.close()
    for local_path, remote_path in ((summary_path, 'metrics/summary.json'), (validation_summary_path, 'metrics/validation_summary.json'), (report_path, 'metrics/classification_report.csv'), (matrix_path, 'metrics/confusion_matrix.csv'), (predictions_path, 'metrics/predictions.csv'), (error_examples_path, 'metrics/error_examples.csv'), (error_pairs_path, 'metrics/error_pairs.csv'), (error_by_class_path, 'metrics/error_by_class.csv'), (figure_path, 'figures/confusion_matrix.png')):
        persist(local_path, remote_path, 'Persist final evaluation artifact')
    experiment_config = {'experiment_id': experiment_id, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'dataset_repo': dataset_repo, 'dataset_revision': dataset_revision, 'raw_archive': raw_archive_name, 'raw_archive_sha256': expected_raw_sha, 'cnn_artifact_repo': cnn_repo, 'cnn_artifact_revision': cnn_revision, 'split_manifest': split_manifest, 'preprocessing': {'input_mode': args.input_mode, 'preprocess_signature': json.loads(preprocess_signature), 'mediapipe': {'task_url': task_url if args.input_mode == 'mediapipe_roi' else None, 'task_sha256': task_sha, 'num_hands': 1 if args.input_mode == 'mediapipe_roi' else 0, 'padding_ratio': args.padding_ratio if args.input_mode == 'mediapipe_roi' else None, 'fallback_policy': args.fallback_policy if args.input_mode == 'mediapipe_roi' else None}}, 'model': {'name': model_name, 'timm_repo': timm_repo, 'timm_revision': timm_revision, 'weights_sha256': weight_sha}, 'training': {'seed': seed, 'epochs_max': args.epochs, 'batch_size': args.batch_size, 'learning_rate': lr, 'weight_decay': weight_decay, 'all_layers_trainable': True}, 'environment': {'torch': torch.__version__, 'timm': timm.__version__, 'mediapipe': mp.__version__}}
    config_path = outputs / 'metadata' / 'experiment_config.json'
    environment_path = outputs / 'metadata' / 'environment.txt'
    config_path.write_text(json.dumps(experiment_config, indent=2))
    environment_path.write_text('\n'.join([f'torch={torch.__version__}', f'timm={timm.__version__}', f'mediapipe={mp.__version__}']) + '\n')
    for local_path, remote_path in ((config_path, 'metadata/experiment_config.json'), (environment_path, 'metadata/environment.txt')):
        persist(local_path, remote_path, 'Persist final run provenance')
    print(json.dumps({'summary': summary, 'detection_coverage': coverage.to_dict(orient='records'), 'artifact_repo': None if args.no_upload else args.hf_repo_id}, indent=2))


if __name__ == '__main__':
    main()
