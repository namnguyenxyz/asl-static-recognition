"""Resume the raw-image ablation control on a Colab runtime."""
from pathlib import Path
import subprocess
import sys

work_dir = Path('/content/mp_mnv4_004_raw_control')
work_dir.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        sys.executable,
        '/content/run_mediapipe_transfer.py',
        '--work-dir', str(work_dir),
        '--input-mode', 'raw_image',
        '--epochs', '15',
        '--selection-only',
        '--no-upload',
        '--resume-state', '/content/training_state_epoch_6.pt',
    ],
    check=True,
)
