"""Self-contained, one-time P9 evaluation of the validation-selected control."""
from pathlib import Path
import subprocess
import sys

root = Path('/content/asl-static-recognition')
subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/namnguyenxyz/asl-static-recognition.git', str(root)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(root / 'requirements-colab-transfer.txt')], check=True)
subprocess.run([
    sys.executable, '-c',
    "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='hnam25/asl-hg-mp-mnv4-raw-control', filename='checkpoints/training_state_epoch_11.pt', local_dir='/content', local_dir_use_symlinks=False)",
], check=True)
subprocess.run([
    sys.executable, str(root / 'scripts' / 'run_mediapipe_transfer.py'),
    '--work-dir', '/content/mp_mnv4_004_raw_control', '--input-mode', 'raw_image',
    '--epochs', '15', '--no-upload',
    '--resume-state', '/content/checkpoints/training_state_epoch_11.pt',
], check=True)
