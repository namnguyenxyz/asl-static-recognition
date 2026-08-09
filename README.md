# Static ASL Hand Gesture Recognition

Nhận dạng 36 cử chỉ ASL tĩnh (A–Z, 0–9) bằng MediaPipe Hand Landmarker và TensorFlow/Keras transfer learning. Notebook hoàn chỉnh [ASL_End_to_End_Colab.ipynb](notebooks/ASL_End_to_End_Colab.ipynb) là entry point chính; `src/` là phiên bản module hóa cho demo local và kiểm thử.

## Chuẩn bị

1. Tải dataset từ Hugging Face repo công khai của nhóm: `python scripts/download_dataset.py`. Script chỉ tải `ASL_Raw_Images.zip`, giải nén và tự tìm cấu trúc lớp `0`–`9`, `A`–`Z`.
2. Tải MediaPipe Hand Landmarker task model và lưu tại `assets/hand_landmarker.task` (hoặc sửa `configs/baseline.yaml`).
3. Cài môi trường: `python -m pip install -r requirements.txt`.

## Chạy baseline

**Cách chạy chính:** mở `notebooks/ASL_End_to_End_Colab.ipynb` trong Colab và chọn Run all. Notebook tự cài dependencies, tải `hnam25/asl-hand-gesture-images`, tải MediaPipe task model, train và xuất metrics/artifacts.

Để tạo cache tái sử dụng cho audit và MediaPipe crops, chạy một lần `notebooks/01_audit_cache_publish.ipynb` trên Colab. Notebook này **không có Hugging Face write token**: nó chỉ tạo Parquet metadata, manifest fingerprint và archive `processed_hand_crops.zip`. Dùng `colab download` tải năm artifact về local, rồi chạy `HF_TOKEN=... python scripts/upload_hf_cache.py --cache-dir cache-download` tại local để publish lên Hugging Face. Sau đó dùng `notebooks/02_train_from_published_cache.ipynb` để train/evaluate mà không audit hoặc chạy MediaPipe lại.

Các notebook nhỏ và CLI bên dưới chỉ dùng khi cần debug từng bước:

Mở notebook theo thứ tự `00` đến `05`, hoặc chạy một stage bằng:

```bash
python run_baseline.py --stage audit
python run_baseline.py --stage segment
python run_baseline.py --stage split
python run_baseline.py --stage train
python run_baseline.py --stage evaluate
python -m src.inference.webcam_demo
```

Kết quả được lưu trong `outputs/`; ảnh crop, metadata và split manifests được lưu dưới `data/`. Models tự áp dụng preprocessing ImageNet phù hợp với backbone; input dataset giữ giá trị RGB `0..255`. Không commit `data/raw`, `data/processed` hay model artifacts.

Khi repo có commit dữ liệu ổn định, ghi SHA vào experiment log và tải tái lập bằng `python scripts/download_dataset.py --revision <commit-sha>`.

## Colab

Quy trình dùng session GPU và cách dừng session an toàn nằm trong [docs/GOOGLE_COLAB_CLI.md](docs/GOOGLE_COLAB_CLI.md).

## Cấu trúc

- `notebooks/`: điều phối các thí nghiệm có thể trình bày trong báo cáo.
- `src/`: triển khai pipeline có thể kiểm thử/tái sử dụng.
- `configs/`: tham số tái lập thí nghiệm.
- `docs/`: nguồn dữ liệu, vận hành Colab và nhật ký thực nghiệm.
