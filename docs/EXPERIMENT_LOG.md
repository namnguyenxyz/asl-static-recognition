# Nhật ký thí nghiệm

Tất cả test score được đọc sau khi model selection hoàn tất trên validation P2. Test P9 không dùng để chọn hyperparameter.

| ID | Model/config | Val acc. | Test acc. | Macro precision | Macro recall | Macro-F1 | Coverage test | Quyết định |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `cnn-001` | CNN from scratch | 72.10% | 83.25% | 79.57% | 83.31% | 80.10% | 3,589/3,589 | Mốc CNN học từ đầu. Recall lớp `0` = 0%. |
| `mnv4-001` | MobileNetV4 Conv-S ImageNet, frozen backbone | 66.42% | 79.88% | 77.65% | 79.94% | 77.29% | 3,589/3,589 | Transfer frozen không vượt CNN tổng thể. |
| `mp-svm-001` | MediaPipe 2-hand landmarks (130D) + StandardScaler + RBF SVM (`C=10`) | 95.01% | 90.53% | 87.04% | 90.52% | 88.03% | 3,589/3,589 | 29/35,441 landmark failures, đều ở train; test coverage 100%. |
| `mnv4-002` | MobileNetV4 Conv-S ImageNet, full fine-tuning trên archive processed | 92.11% | **94.51%** | 94.05% | 94.53% | **93.38%** | 3,589/3,589 | Mô hình được chọn; artifact bất biến đã công bố. |
| `mp-mnv4-002`* | Progressive transfer trên archive processed: head 3 epoch rồi full fine-tune | 82.33% | 85.37% | — | — | 83.77% | — | Exploratory; số liệu từ tổng hợp thực nghiệm, chưa có artifact bất biến được liệt kê. |
| `mp-mnv4-003` | Raw image → MediaPipe ROI (padding 0,18; raw fallback) → MobileNetV4 full fine-tuning | **95.18%** | **96.66%** | **95.37%** | **96.67%** | **95.75%** | 3,589/3,589 | Mô hình được chọn; artifact bất biến đã công bố. |

## Protocol chung

- Dataset revision: `8f36ac00ece6dfce94410a980a839d93a912d366`.
- Raw archive SHA-256: `594cfa0158044085ed61351315c187a6f3a3f9087795b4f4cdba68ecd04f12b1`.
- Exact duplicate policy: một canonical representative cho mỗi raw-image SHA-256; 36,000 → 35,441 ảnh.
- Participant split: train P1/P3–P8/P10, validation P2, test P9; seed 42.
- `mp-mnv4-003` có recall 100% ở cả `O` và `0`, không có nhầm `O→0` hoặc `0→O`; `cnn-001` có recall `0` = 0%.
- MediaPipe ROI coverage của `mp-mnv4-003`: test 3.589/3.589 và validation 3.487/3.487 detect; train có 29/28.365 fallback raw.

Dấu `*` chỉ run exploratory, không dùng làm bằng chứng chính. Metrics chi tiết, confusion matrix, environment và config bất biến của các baseline chính nằm trong model artifacts trên Hugging Face; xem [báo cáo tái lập](THESIS_REPRODUCIBILITY_REPORT.md).

## Addendum ROI padding 0,10

`mp-mnv4-003-p010-raw`: chọn theo P2 (accuracy 92,77%, Macro-F1 91,64%); P9 chạy một lần cho **96,38% accuracy** và **95,64% Macro-F1**. Artifact: [asl-hg-mediapipe-roi-p010-final](https://huggingface.co/hnam25/asl-hg-mediapipe-roi-p010-final), commit `e2faeb2768d2b081a9f7c9acc410401f7b758f1b`.

## Ablation raw-image control và external evaluation

Raw-image control dùng cùng raw archive, canonical participant split, seed 42, MobileNetV4 Conv-S và full fine-tuning như ứng viên ROI; chỉ thay preprocessing thành raw image. P2 selection đạt 84,74% accuracy; P9 đánh giá một lần đạt **90,22% accuracy** và **88,81% Macro-F1**. So với ROI p=0,10 trên P9, MediaPipe ROI có đóng góp quan sát được **+6,16 điểm accuracy** và **+6,75 điểm Macro-F1**. Artifact: [asl-hg-mp-mnv4-raw-control](https://huggingface.co/hnam25/asl-hg-mp-mnv4-raw-control).

| Kiểm tra ngoài tập (zero-shot, không fine-tune) | Phạm vi | Accuracy | Macro-F1 | Diễn giải |
|---|---:|---:|---:|---|
| Sign Language MNIST | 7.172 ảnh, 24 chữ tĩnh chung A--I/K--Y | 14,15% | 12,42% | Ảnh 28×28 grayscale đã căn chỉnh; khác miền dữ liệu ASL-HG. |
| RGB smoke | 600 ảnh, 25 ảnh/lớp tĩnh, seed 42 | 14,33% | 14,64% | Nguồn RGB khác; ROI detect 79,83%. Không phải official held-out benchmark. |

Hai phép đo ngoài tập là bằng chứng giới hạn generalization, không dùng để chọn checkpoint/hyperparameter. Artifact: [MNIST](https://huggingface.co/hnam25/asl-hg-external-sign-mnist-eval), [RGB smoke](https://huggingface.co/hnam25/asl-hg-external-rgb-smoke-600).

Demo định tính 36 ảnh ASL giáo dục tìm qua Google Image Search: raw 19/36 đúng, MediaPipe ROI 25/36 đúng, detection 34/36. Vì một ảnh/lớp có nền sạch và không phải sampling benchmark, nó chỉ minh họa tác động ROI; không được diễn giải thành accuracy tổng quát. [Artifact demo](https://huggingface.co/hnam25/asl-hg-google-image-qualitative-36).
