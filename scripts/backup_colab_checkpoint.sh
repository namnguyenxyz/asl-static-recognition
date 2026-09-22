#!/usr/bin/env bash
# Download completed-epoch artifacts from a live Colab session until interrupted.
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 SESSION REMOTE_WORK_DIR LOCAL_BACKUP_DIR [INTERVAL_SECONDS]" >&2
  exit 2
fi

session="$1"
remote_work_dir="$2"
local_backup_dir="$3"
interval_seconds="${4:-120}"
mkdir -p "$local_backup_dir"

while true; do
  temporary="$local_backup_dir/training_state.pt.partial"
  if colab download -s "$session" "$remote_work_dir/outputs/models/training_state.pt" "$temporary"; then
    mv "$temporary" "$local_backup_dir/training_state.pt"
    echo "[$(date -u +%FT%TZ)] backed up completed-epoch state"
  fi
  colab download -s "$session" "$remote_work_dir/outputs/logs/training_history.csv" "$local_backup_dir/training_history.csv" || true
  colab download -s "$session" "$remote_work_dir/outputs/metadata/mediapipe_crop_manifest.csv" "$local_backup_dir/mediapipe_crop_manifest.csv" || true
  sleep "$interval_seconds"
done
