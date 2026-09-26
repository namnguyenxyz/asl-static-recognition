"""Start the raw-image control in a background Colab process."""
from pathlib import Path
import subprocess
import sys

root = Path('/content')
work_dir = Path('/content/mp_mnv4_004_raw_control')
log_path = work_dir / 'train.log'
work_dir.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable, str(root / 'run_mediapipe_transfer.py'),
    '--work-dir', str(work_dir), '--input-mode', 'raw_image',
    '--epochs', '15', '--selection-only', '--no-upload',
]
with log_path.open('ab', buffering=0) as log:
    process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print({'pid': process.pid, 'log': str(log_path), 'command': command})
