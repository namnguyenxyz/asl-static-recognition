# Run `mp-mnv4-003`: MediaPipe ROI + MobileNetV4

Đây là runner cho ứng viên phương pháp cuối: raw image canonical được crop ROI bởi MediaPipe Hand Landmarker, sau đó MobileNetV4 Conv-S ImageNet được fine-tune toàn bộ. Nó dùng revision dữ liệu, canonicalization và participant split giống `mnv4-002`, nên hai run có thể so sánh trực tiếp.

## Điều kiện

- Colab hoặc máy CUDA có đủ dung lượng trống cho raw archive, crop cache và checkpoint.
- Python environment cài dependencies trong `requirements.txt`.
- Biến môi trường `HF_TOKEN` có quyền write vào model repository `hnam25/asl-hg-mediapipe-roi-transfer` (hoặc repo truyền qua `--hf-repo-id`). Token không được ghi vào notebook, log hay Git.
- Không thay đổi hyperparameter hay đọc test P9 để lựa chọn phương án; chọn checkpoint bằng validation P2.

## Chạy trên Colab

```bash
!git clone <repository-url> asl-static-recognition
%cd asl-static-recognition
!python -m pip install -r requirements.txt
!export HF_TOKEN='...token có quyền write...'
!python scripts/run_mediapipe_transfer.py --work-dir /content/mp_mnv4_003 --hf-repo-id hnam25/asl-hg-mediapipe-roi-transfer
```

Để chạy lại crop cache từ đầu, thêm `--force-recrop`. Không dùng cờ này khi tiếp tục training nếu crop manifest đã được xác minh.

## Output cần lưu/publish

Runner tự persist crop manifest và detection coverage trước khi train; sau mỗi epoch nó upload training history, và upload checkpoint mỗi khi validation loss cải thiện. Sau khi đánh giá test, runner upload toàn bộ metrics/provenance. Thư mục `outputs/mp_mnv4_003/outputs/` cũng phải được giữ nguyên để đối chiếu với artifact gồm:

- `models/mp_mnv4_003_raw_mediapipe_roi_full_finetune.safetensors`;
- `metrics/summary.json`, `classification_report.csv`, `confusion_matrix.csv`, `detection_coverage.csv`;
- `metadata/mediapipe_crop_manifest.csv`, split/dedup manifest, `experiment_config.json`, `environment.txt`;
- `logs/training_history.csv` và `figures/confusion_matrix.png`.

Sau khi run hoàn tất, tải ngược checkpoint từ Hugging Face và đối chiếu SHA-256 trước khi cập nhật bảng kết quả. Chỉ khi artifact đầy đủ và metrics test đã xác minh mới chọn `mp-mnv4-003` làm mô hình cuối. Nếu nó không vượt `mnv4-002`, giữ `mnv4-002` là mô hình được chọn và báo cáo `mp-mnv4-003` như ablation âm.
