# ASL-HG dataset

## Nguồn download chuẩn

- Hugging Face: [hnam25/asl-hand-gesture-images](https://huggingface.co/datasets/hnam25/asl-hand-gesture-images).
- API: `https://huggingface.co/api/datasets/hnam25/asl-hand-gesture-images`.
- Tải bằng `python scripts/download_dataset.py`; có thể khóa commit với `--revision <sha>` để tái lập thí nghiệm.
- Mục tiêu dữ liệu: 36.000 ảnh RGB, 36 lớp tĩnh gồm `A-Z` và `0-9`.

Repo là public, không gated và hiện chứa `ASL_HG_36000/ASL_Raw_Images.zip` cùng archive processed. Notebook/script chỉ tải archive raw để tránh tải dư; sau khi giải nén chúng tự tìm thư mục chứa đủ 36 folder lớp.

## Quy ước local

Giải nén sao cho mỗi nhãn là một thư mục trực tiếp dưới `data/raw/`, ví dụ:

```text
data/raw/0/*.jpg
data/raw/O/*.jpg
data/raw/Z/*.jpg
```

Không thay đổi dữ liệu gốc. `01_dataset_audit.ipynb` tạo `data/metadata/dataset_audit.csv`, thống kê số ảnh/lớp và phát hiện file không đọc được hoặc ảnh có SHA-256 trùng. Sau đó `02_hand_segmentation.ipynb` tạo dữ liệu crop riêng trong `data/processed/`.

Nếu dataset là bản sao hoặc bản biến đổi từ một nguồn học thuật khác, bổ sung citation/license của nguồn đó trong báo cáo trước khi nộp.
