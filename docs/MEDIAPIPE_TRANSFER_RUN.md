# Run `mp-mnv4-003`: MediaPipe ROI + MobileNetV4

Đây là runner cho ứng viên phương pháp cuối: raw image canonical được crop ROI bởi MediaPipe Hand Landmarker, sau đó MobileNetV4 Conv-S ImageNet được fine-tune toàn bộ. Nó dùng revision dữ liệu, canonicalization và participant split giống `mnv4-002`, nên hai run có thể so sánh trực tiếp.

## Điều kiện

- Colab hoặc máy CUDA có đủ dung lượng trống cho raw archive, crop cache và checkpoint.
- Python environment cài dependencies trong `requirements.txt`.
- Biến môi trường `HF_TOKEN` có quyền write vào model repository `hnam25/asl-hg-mediapipe-roi-transfer` (hoặc repo truyền qua `--hf-repo-id`) nếu cần publish artifact. Token không được ghi vào notebook, log hay Git. Với run local/Colab không publish, dùng `--no-upload` và tải artifact về thủ công.
- Không thay đổi hyperparameter hay đọc test P9 để lựa chọn phương án; chọn checkpoint bằng validation P2.

## Chạy trên Colab

```bash
!git clone <repository-url> asl-static-recognition
%cd asl-static-recognition
!python -m pip install -r requirements-colab-transfer.txt
!export HF_TOKEN='...token có quyền write...'
!python scripts/run_mediapipe_transfer.py --work-dir /content/mp_mnv4_003 --hf-repo-id hnam25/asl-hg-mediapipe-roi-transfer --input-mode mediapipe_roi --padding-ratio 0.18 --fallback-policy raw_image
```

`requirements-colab-transfer.txt` là environment tối thiểu cho runner PyTorch/timm và tránh TensorFlow, vì TensorFlow theo range chính của project có thể chưa có wheel cho Python Colab hiện tại.

Để chạy lại crop cache từ đầu, thêm `--force-recrop`. Không dùng cờ này khi tiếp tục training nếu crop manifest đã được xác minh.

## Phục hồi khi Colab mất session

Chia run dài thành crop, backup crop và các chặng train ngắn. Không chạy toàn bộ pipeline trong một cell.

### 1. Crop một lần, đóng gói và tải về local

```bash
python scripts/run_mediapipe_transfer.py \
  --work-dir /content/ablation_roi_p018 \
  --input-mode mediapipe_roi --padding-ratio 0.18 --fallback-policy raw_image \
  --stop-after-crop --no-upload
```

Lệnh tạo `outputs/metadata/crop_cache_mediapipe_roi-padding-0_18-fallback-raw_image.tar.gz`. Tải file này về local ngay bằng `colab download`. Ở session mới, upload archive rồi dùng `--restore-crop-archive /content/crop_cache.tar.gz`; runner xác minh manifest/signature trước khi tái sử dụng crop.

### 2. Train theo chặng và backup state local

Runner ghi nguyên tử `outputs/models/training_state.pt` sau **mỗi epoch hoàn chỉnh**. File này chứa model, optimizer, scheduler, lịch sử, best checkpoint và epoch kế tiếp; không dùng checkpoint giữa epoch. Chạy chặng đầu, ví dụ 3 epoch:

```bash
python scripts/run_mediapipe_transfer.py \
  --work-dir /content/ablation_roi_p018 \
  --input-mode mediapipe_roi --padding-ratio 0.18 --fallback-policy raw_image \
  --epochs 3 --selection-only --no-upload
```

Tải `training_state.pt` về local sau mỗi chặng, hoặc mở terminal local thứ hai để backup định kỳ:

```bash
bash scripts/backup_colab_checkpoint.sh asl-ablation-roi /content/ablation_roi_p018 backups/roi-p018
```

Chạy chặng sau với max epoch cao hơn; ví dụ từ epoch 3 đến epoch 6:

```bash
python scripts/run_mediapipe_transfer.py \
  --work-dir /content/ablation_roi_p018 \
  --input-mode mediapipe_roi --padding-ratio 0.18 --fallback-policy raw_image \
  --epochs 6 --selection-only --no-upload \
  --resume-state /content/training_state.pt
```

Nếu runtime mất, tạo session mới, upload crop archive và `training_state.pt`, rồi resume:

```bash
python scripts/run_mediapipe_transfer.py \
  --work-dir /content/ablation_roi_p018 \
  --input-mode mediapipe_roi --padding-ratio 0.18 --fallback-policy raw_image \
  --epochs 6 --selection-only --no-upload \
  --restore-crop-archive /content/crop_cache.tar.gz \
  --resume-state /content/training_state.pt
```

Chỉ resume state/crop archive có cùng variant, dataset revision và protocol. Với cách này, failure khi train chỉ mất tối đa một chặng ngắn, còn crop không phải làm lại.

## Output cần lưu/publish

Runner tự persist crop manifest và detection coverage trước khi train; sau mỗi epoch nó upload training history, và upload checkpoint mỗi khi validation loss cải thiện. Sau khi đánh giá test, runner upload toàn bộ metrics/provenance. Thư mục `outputs/mp_mnv4_003/outputs/` cũng phải được giữ nguyên để đối chiếu với artifact gồm:

- `models/mp_mnv4_003_raw_mediapipe_roi_full_finetune.safetensors`;
- `metrics/summary.json`, `classification_report.csv`, `confusion_matrix.csv`, `detection_coverage.csv`, `predictions.csv`, `error_examples.csv`, `error_pairs.csv`, `error_by_class.csv`;
- `metadata/mediapipe_crop_manifest.csv`, split/dedup manifest, `experiment_config.json`, `environment.txt`;
- `logs/training_history.csv` và `figures/confusion_matrix.png`.

Sau khi run hoàn tất, tải ngược checkpoint từ Hugging Face và đối chiếu SHA-256 trước khi cập nhật bảng kết quả. Chỉ khi artifact đầy đủ và metrics test đã xác minh mới chọn `mp-mnv4-003` làm mô hình cuối. Nếu nó không vượt `mnv4-002`, giữ `mnv4-002` là mô hình được chọn và báo cáo `mp-mnv4-003` như ablation âm.

## Chạy ablation và phân tích lỗi

Không ghi đè artifact đã công bố. Ví dụ, chạy raw-image control trong một thư mục và repository riêng:

```bash
!python scripts/run_mediapipe_transfer.py --work-dir /content/mp_mnv4_003_raw --hf-repo-id hnam25/asl-hg-mp-mnv4-raw-control --input-mode raw_image --selection-only --no-upload
```

Thêm `--selection-only` cho mọi run dùng chọn padding/fallback: runner chỉ xuất `validation_summary.json` trên P2 và không đọc P9. Sau khi khóa một variant, chạy lại đúng variant đó **không** có cờ này để đánh giá P9 một lần và đọc các bảng lỗi tạo sẵn trong `outputs/metrics/`. Có thể tạo lại các bảng này từ artifact đã tải về bằng:

```bash
python scripts/analyze_errors.py outputs/metrics/predictions.csv
```
