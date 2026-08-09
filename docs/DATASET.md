# ASL-HG dataset

## Nguồn download chuẩn

- Hugging Face: [hnam25/asl-hand-gesture-images](https://huggingface.co/datasets/hnam25/asl-hand-gesture-images).
- API: `https://huggingface.co/api/datasets/hnam25/asl-hand-gesture-images`.
- Tải bằng `python scripts/download_dataset.py`; có thể khóa commit với `--revision <sha>` để tái lập thí nghiệm.
- Mục tiêu dữ liệu: 36.000 ảnh RGB, 36 lớp tĩnh gồm `A-Z` và `0-9`.

Repo là public, không gated và hiện chứa `ASL_HG_36000/ASL_Raw_Images.zip` cùng archive processed. Notebook/script chỉ tải archive raw để tránh tải dư; sau khi giải nén chúng tự tìm thư mục chứa đủ 36 folder lớp.

## Derived cache

Cache do pipeline tạo được publish dưới `derived/mediapipe-hand-landmarker-v1/` trong cùng dataset repo:

- `audit.parquet`, `duplicates.parquet`, `segmentation_manifest.parquet`;
- `cache_manifest.json` có SHA-256 của raw archive và cấu hình MediaPipe;
- `processed_hand_crops.zip` chứa toàn bộ crop đầu vào classifier.

`01_audit_cache_publish.ipynb` tạo/publish cache. `02_train_from_published_cache.ipynb` xác minh raw archive SHA-256 trước khi restore cache; nếu không khớp, nó dừng và yêu cầu rebuild cache để tránh reuse dữ liệu sai. `ASL_End_to_End_Colab.ipynb` vẫn là pipeline rebuild đầy đủ.

## Quy ước local

Giải nén sao cho mỗi nhãn là một thư mục trực tiếp dưới `data/raw/`, ví dụ:

```text
data/raw/0/*.jpg
data/raw/O/*.jpg
data/raw/Z/*.jpg
```

Không thay đổi dữ liệu gốc. `01_dataset_audit.ipynb` tạo `data/metadata/dataset_audit.csv`, thống kê số ảnh/lớp và phát hiện file không đọc được hoặc ảnh có SHA-256 trùng. Sau đó `02_hand_segmentation.ipynb` tạo dữ liệu crop riêng trong `data/processed/`.

Nếu dataset là bản sao hoặc bản biến đổi từ một nguồn học thuật khác, bổ sung citation/license của nguồn đó trong báo cáo trước khi nộp.
