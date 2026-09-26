from pathlib import Path
import sys

for path in (Path('/content/asl-static-recognition'), Path('/content/training_state_epoch_6.pt'), Path('/content/mp_mnv4_004_raw_control')):
    print(path, path.exists())
print(sys.version)
