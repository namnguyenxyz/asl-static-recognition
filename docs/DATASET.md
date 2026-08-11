# ASL-HG dataset

## Nguồn download chuẩn

- Hugging Face: [hnam25/asl-hand-gesture-images](https://huggingface.co/datasets/hnam25/asl-hand-gesture-images).
- API: `https://huggingface.co/api/datasets/hnam25/asl-hand-gesture-images`.
- Tải bằng `python scripts/download_dataset.py`; có thể khóa commit với `--revision <sha>` để tái lập thí nghiệm.
- Mục tiêu dữ liệu: 36.000 ảnh RGB, 36 lớp tĩnh gồm `A-Z` và `0-9`.

Repo là public, không gated và hiện chứa `ASL_HG_36000/ASL_Raw_Images.zip` cùng archive processed. Notebook/script chỉ tải archive raw để tránh tải dư; sau khi giải nén chúng tự tìm thư mục chứa đủ 36 folder lớp.

## Derived cache

Các cache do pipeline tạo được publish trong cùng dataset repo. Cache crop lịch sử nằm dưới `derived/mediapipe-hand-landmarker-v1/`; cache landmark hai tay tái dùng cho `mp-svm-001` nằm dưới `derived/mediapipe-two-hand-landmarks-v1/`.

- `audit.parquet`, `duplicates.parquet`, `segmentation_manifest.parquet`;
- `cache_manifest.json` có SHA-256 của raw archive và cấu hình MediaPipe;
- `processed_hand_crops.zip` chứa toàn bộ crop đầu vào classifier;
- cache two-hand mới chứa `landmark_manifest.parquet`, `landmark_status.csv` và `detection_summary.json`; nó lưu 130D landmark feature cùng trạng thái detect để không phải chạy MediaPipe lại.

`01_audit_only_publish.ipynb` tạo/publish metadata audit cho revision mới. Các notebook tái lập `08`--`11` tải archive và artifact bằng revision khóa, kiểm tra checksum trước khi tái dựng canonical split.

## Quy ước local

Giải nén sao cho mỗi nhãn là một thư mục trực tiếp dưới `data/raw/`, ví dụ:

```text
data/raw/0/*.jpg
data/raw/O/*.jpg
data/raw/Z/*.jpg
```

Không thay đổi dữ liệu gốc. Metadata audit được tạo bởi `01_audit_only_publish.ipynb`; các run tái lập dùng canonical representative theo SHA-256 để ngăn leakage do ảnh trùng. Archive ảnh đã xử lý của dataset được dùng làm đầu vào cho baseline ảnh hiện hành.

Nếu dataset là bản sao hoặc bản biến đổi từ một nguồn học thuật khác, bổ sung citation/license của nguồn đó trong báo cáo trước khi nộp.
