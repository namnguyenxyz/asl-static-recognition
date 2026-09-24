# Static ASL Hand Gesture Recognition

Đồ án phân loại 36 ký hiệu ASL tĩnh (`0–9`, `A–Z`) theo hướng tái lập được. Repository lưu mã nguồn, notebook và tài liệu; dữ liệu lớn, checkpoint và metrics đầy đủ được version hóa trên Hugging Face.

## Kết quả hiện tại

Mọi baseline dưới đây dùng cùng protocol: archive dữ liệu đã khóa phiên bản, dedup theo SHA-256, split participant-disjoint cố định (train: P1, P3–P8, P10; validation: P2; test: P9) và seed 42.

| ID | Phương pháp | Val acc. | Test acc. | Macro-F1 | Artifact |
|---|---|---:|---:|---:|---|
| `cnn-001` | CNN from scratch | 72.10% | 83.25% | 80.10% | [Hugging Face](https://huggingface.co/hnam25/asl-hg-cnn-baseline) |
| `mnv4-001` | MobileNetV4 Conv-S ImageNet, frozen | 66.42% | 79.88% | 77.29% | [Hugging Face](https://huggingface.co/hnam25/asl-hg-mobilenetv4-baseline) |
| `mp-svm-001` | MediaPipe two-hand landmarks + RBF SVM | 95.01% | 90.53% | 88.03% | [Hugging Face](https://huggingface.co/hnam25/asl-hg-mediapipe-svm-baseline) |
| `mnv4-002` | MobileNetV4 Conv-S ImageNet, full fine-tuning trên archive processed | 92.11% | 94.51% | 93.38% | [Hugging Face](https://huggingface.co/hnam25/asl-hg-mobilenetv4-finetune-baseline) |
| `mp-mnv4-003-p010` | **Raw image → MediaPipe ROI padding 0.10 → MobileNetV4 Conv-S full fine-tuning** | **92.77%** | **96.38%** | **95.64%** | [Hugging Face](https://huggingface.co/hnam25/asl-hg-mediapipe-roi-p010-final) |

`mp-mnv4-003-p010` là mô hình được chọn. Nó crop ROI từ raw image canonical bằng MediaPipe Hand Landmarker (padding 0,10), dùng raw image fallback khi không detect được tay, rồi fine-tune toàn bộ MobileNetV4. Trên test P9, detection coverage là 100% (3.589/3.589); train có 29/28.365 fallback raw. Kết quả chỉ áp dụng cho protocol participant-disjoint đã khóa. Báo cáo chi tiết nằm trong [báo cáo tái lập](docs/THESIS_REPRODUCIBILITY_REPORT.md).

## Tái lập nhanh

1. Cài môi trường:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Dùng các notebook tái lập trên Google Colab:

   - `notebooks/01_audit_only_publish.ipynb`: audit SHA-256 và publish metadata cache.
   - `notebooks/08_cnn_baseline_reproducible.ipynb`: `cnn-001`.
   - `notebooks/09_mobilenetv4_timm_baseline_reproducible.ipynb`: `mnv4-001`.
   - `notebooks/10_mediapipe_svm_baseline_reproducible.ipynb`: `mp-svm-001`.
   - `notebooks/11_mobilenetv4_timm_finetune_reproducible.ipynb`: `mnv4-002`, đối chứng archive processed.
   - `notebooks/12_mediapipe_crop_mobilenetv4_finetune_reproducible.ipynb`: thí nghiệm crop cache MediaPipe.
   - `notebooks/13_progressive_transfer_mobilenetv4_reproducible.ipynb`: thí nghiệm progressive transfer.
   - `scripts/run_mediapipe_transfer.py`: `mp-mnv4-003`, mô hình được chọn; crop ROI từ raw image bằng MediaPipe rồi fine-tune MobileNetV4; xem [runbook](docs/MEDIAPIPE_TRANSFER_RUN.md).

Các notebook là mã tái lập; chúng không chứa output thực thi được commit. Số liệu chính thức phải đọc từ artifact bất biến và experiment log. Hướng dẫn vận hành session nằm ở [Google Colab CLI runbook](docs/GOOGLE_COLAB_CLI.md).

3. Dùng artifacts đã publish thay vì train lại. Mỗi model repository chứa checkpoint, metrics, split/config manifest, environment và notebook tương ứng. Landmark cache tái dùng của MediaPipe nằm tại [dataset repository](https://huggingface.co/datasets/hnam25/asl-hand-gesture-images/tree/main/derived/mediapipe-two-hand-landmarks-v1).

## Tính tái lập

- Dataset: [`hnam25/asl-hand-gesture-images`](https://huggingface.co/datasets/hnam25/asl-hand-gesture-images), revision `8f36ac00ece6dfce94410a980a839d93a912d366`.
- Raw archive SHA-256: `594cfa0158044085ed61351315c187a6f3a3f9087795b4f4cdba68ecd04f12b1`.
- Sau dedup: 35,441 ảnh canonical; loại 559 exact duplicates.
- Không commit raw data, cache download hoặc checkpoints vào Git. Các file này được kiểm tra lại bằng SHA-256 từ Hugging Face.

Xem [DATASET.md](docs/DATASET.md), [AUDIT_CACHE.md](docs/AUDIT_CACHE.md), [EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) và [BASELINE_STRATEGY.md](docs/BASELINE_STRATEGY.md) để biết chi tiết.

## Cấu trúc

- `notebooks/`: các thí nghiệm tái lập được trên Colab.
- `src/`: pipeline module hóa và inference local.
- `tests/`: smoke tests cho split/model.
- `docs/`: protocol, audit, experiment log và báo cáo thesis-ready.
