# Google Colab CLI runbook

Project dùng lệnh `colab` của gói `google-colab-cli` (đã kiểm tra local: v0.6.0). CLI tạo runtime Colab, thực thi file `.py`/`.ipynb`, quản lý file remote và dừng runtime từ terminal.

## Xác thực

Lần đầu, dùng OAuth hoặc ADC theo cấu hình của máy. Kiểm tra trạng thái an toàn bằng:

```bash
colab sessions
```

Nếu dùng ADC, cần đăng nhập lại với scope Colab trước khi tạo session. Không dùng `colab auth` để sửa lỗi xác thực CLI: lệnh này chỉ cấp credentials *bên trong VM* cho dịch vụ GCP.

## Session huấn luyện bền vững

```bash
colab new -s asl-training --gpu T4
colab status -s asl-training
colab install -s asl-training -r requirements.txt
```

Khả năng cấp T4/L4 phụ thuộc quota. Nếu T4 không khả dụng, thử `L4`; CPU chỉ dùng smoke test. Luôn đặt tên session để không thao tác nhầm runtime.

## Đồng bộ và chạy notebook

Thư mục mặc định remote là `/content`. Upload source/data theo từng file hoặc archive; giải nén/project setup bằng code chạy qua `colab exec`. Chạy notebook:

```bash
colab exec -s asl-training -f notebooks/ASL_End_to_End_Colab.ipynb --timeout 3600
colab download -s asl-training /content/asl-static-recognition/outputs/models/baseline_mobilenetv2.keras outputs/models/baseline_mobilenetv2.keras
colab download -s asl-training /content/asl-static-recognition/outputs/metrics/o_zero_analysis.json outputs/metrics/o_zero_analysis.json
```

Mỗi notebook phải đọc YAML, không dựa vào biến từ notebook trước. Lưu experiment ID, config, metrics, figures và checkpoint vào `outputs/`.

## Theo dõi và phục hồi

```bash
colab status -s asl-training
colab log -s asl-training -n 30
colab restart-kernel -s asl-training
```

Nếu session biến mất, chạy `colab sessions`, tạo lại session rồi sync source/data cần thiết. Không dùng `colab drivemount` trong automation không có TTY; việc mount Drive yêu cầu thao tác người dùng.

## Kết thúc bắt buộc

GPU có thể tiêu compute units khi còn sống. Sau khi tải artifacts và xuất log, luôn chạy:

```bash
colab stop -s asl-training
```

`colab run --gpu T4 script.py` là lựa chọn job một lần; nó tự giải phóng runtime nếu không dùng `--keep`.
## Cache build without a remote Hugging Face token

`notebooks/01_audit_cache_publish.ipynb` only builds the cache on Colab. Do not put `HF_TOKEN` in a notebook, Colab command, or remote VM.

```bash
colab exec -s asl-training -f notebooks/01_audit_cache_publish.ipynb
mkdir -p cache-download
colab download -s asl-training /content/asl-cache-build/data/metadata/audit.parquet cache-download/audit.parquet
colab download -s asl-training /content/asl-cache-build/data/metadata/duplicates.parquet cache-download/duplicates.parquet
colab download -s asl-training /content/asl-cache-build/data/metadata/segmentation_manifest.parquet cache-download/segmentation_manifest.parquet
colab download -s asl-training /content/asl-cache-build/data/metadata/cache_manifest.json cache-download/cache_manifest.json
colab download -s asl-training /content/asl-cache-build/data/metadata/processed_hand_crops.zip cache-download/processed_hand_crops.zip

# Runs locally, after setting a write-capable token in the local environment.
HF_TOKEN=hf_... python scripts/upload_hf_cache.py --cache-dir cache-download
```

The local `cache-download/` directory is ignored by Git. After a successful upload, remove the downloaded cache if disk space is needed and stop the GPU session.
