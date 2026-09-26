"""Evaluate the validation-selected raw-image control on P9 exactly once."""
from pathlib import Path
import subprocess
import sys

work_dir = Path('/content/mp_mnv4_004_raw_control')
subprocess.run([
    sys.executable, '/content/run_mediapipe_transfer.py',
    '--work-dir', str(work_dir), '--input-mode', 'raw_image', '--epochs', '15',
    '--no-upload', '--resume-state', '/content/training_state_epoch_11.pt',
], check=True)
