# ASL Project Handoff — 2026-08-09

## Trạng thái hiện tại

- Không còn Colab session đang chạy; session `asl-debug` đã được dừng để giải phóng T4.
- Notebook chính: `notebooks/ASL_End_to_End_Colab.ipynb`.
- Dataset nguồn: `hnam25/asl-hand-gesture-images` trên Hugging Face.
- Cách chạy bắt buộc: Google Colab CLI (`colab`).

## Đã hoàn thành

### Dataset và audit

- Xác nhận archive `ASL_Raw_Images.zip` giải nén thành **36.000 ảnh**.
- Có đủ 36 lớp `0-9`, `A-Z`; mỗi lớp có **1.000 ảnh**.
- Audit thành công: 36.000 ảnh đọc được, không có ảnh unreadable.
- Đã tải artifact audit từ Colab về local:
  - `data/metadata/dataset_audit.csv` — 36.000 dòng.
  - `data/metadata/duplicate_images.csv` — 994 dòng duplicate.
  - `data/metadata/segmentation_failures.csv` — 36.000 dòng.

### Hand segmentation

- MediaPipe Hand Landmarker đã chạy toàn bộ dữ liệu qua CLI.
- Kết quả: **35.971** ảnh crop thành công, **29** ảnh `no_hand_detected` (0,08%).
- Train/validation/test split đã tạo trên Colab:
  - Train: 28.776 ảnh.
  - Validation: 3.597 ảnh.
  - Test: 3.598 ảnh.

### Training

- TensorFlow nhận diện T4 GPU thành công.
- MobileNetV2 baseline bắt đầu train; checkpoint `baseline_mobilenetv2.keras` đã được sinh trên Colab.
- Chưa tải được metrics/checkpoint về local và chưa có kết quả accuracy, precision/recall hoặc confusion matrix cuối cùng.

## Các lỗi đã phát hiện và khắc phục

1. **Google Colab CLI 0.6.0 không tương thích dependency `jupyter-kernel-client` mới.**
   - CLI gọi `KernelClient`, nhưng dependency hiện export `ColabKernelClient`.
   - Đã vá local CLI runtime để dùng `ColabKernelClient(proxy_token=...)`.

2. **CLI không stream trạng thái notebook dài đáng tin cậy.**
   - Session có thể báo `BUSY` trong khi terminal không hiện tiến trình cell.
   - Đã thêm logger text `outputs/logs/cli_debug.log` trong notebook.
   - Log ghi marker: environment, download, audit, segmentation mỗi 1.000 ảnh, split, training và evaluation.

3. **Rich output gây khó theo dõi với CLI.**
   - Thay `display(DataFrame)` và `plt.show()` bằng `print(..., flush=True)` và file output.
   - Confusion matrix được lưu file PNG, không render trực tiếp trong CLI.

4. **Không cài lại TensorFlow trong notebook.**
   - Colab đã có TensorFlow GPU; cài lại gây conflict với `tensorflow-text`.
   - Notebook hiện chỉ cài MediaPipe, Hugging Face Hub và thư viện hỗ trợ.

5. **Audit cache.**
   - Notebook tái sử dụng `data/metadata/dataset_audit.csv` nếu đã upload sẵn vào runtime, tránh hash lại 36.000 ảnh.

## Chạy tiếp bằng CLI

### 1. Tạo session và tạo thư mục remote

```bash
colab new -s asl-training --gpu T4

printf "from pathlib import Path; Path('/content/asl-static-recognition/data/metadata').mkdir(parents=True, exist_ok=True)" \
  | colab exec -s asl-training
```

### 2. Upload audit cache local

```bash
colab upload -s asl-training data/metadata/dataset_audit.csv \
  /content/asl-static-recognition/data/metadata/dataset_audit.csv

colab upload -s asl-training data/metadata/duplicate_images.csv \
  /content/asl-static-recognition/data/metadata/duplicate_images.csv

colab upload -s asl-training data/metadata/segmentation_failures.csv \
  /content/asl-static-recognition/data/metadata/segmentation_failures.csv
```

### 3. Chạy notebook và theo dõi log

```bash
colab exec -s asl-training -f notebooks/ASL_End_to_End_Colab.ipynb --timeout 7200

colab download -s asl-training \
  /content/asl-static-recognition/outputs/logs/cli_debug.log /tmp/asl-cli-debug.log

tail -n 30 /tmp/asl-cli-debug.log
```

### 4. Khi hoàn tất, tải artifact và dừng session

```bash
colab download -s asl-training \
  /content/asl-static-recognition/outputs/models/baseline_mobilenetv2.keras \
  outputs/models/baseline_mobilenetv2.keras

colab download -s asl-training \
  /content/asl-static-recognition/outputs/metrics/o_zero_analysis.json \
  outputs/metrics/o_zero_analysis.json

colab download -s asl-training \
  /content/asl-static-recognition/outputs/metrics/confusion_matrix.csv \
  outputs/metrics/confusion_matrix.csv

colab stop -s asl-training
```

## Việc cần làm tiếp theo

1. Chạy lại notebook với audit cache upload sẵn.
2. Tải checkpoint, training history, classification report, confusion matrix và `o_zero_analysis.json` về local.
3. Đánh giá accuracy tổng, precision/recall/F1 từng lớp và lỗi nhầm chéo `O → 0`, `0 → O`.
4. Nếu recall `O` hoặc `0` dưới 95%, thực hiện fine-tuning/augmentation theo notebook thí nghiệm.
5. Để bỏ qua **cả segmentation** ở các lần chạy sau, cần lưu/upload thêm toàn bộ `data/processed/`; ba file CSV metadata chỉ giúp bỏ qua audit, không thay thế ảnh crop.
