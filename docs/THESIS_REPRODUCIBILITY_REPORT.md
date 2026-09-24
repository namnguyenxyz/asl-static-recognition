# Báo cáo tái lập và tổng kết thực nghiệm ASL tĩnh

## 1. Mục tiêu và phạm vi

Đồ án phân loại 36 ký hiệu ASL tĩnh (`0–9`, `A–Z`) từ ảnh. Mục tiêu của repository là tạo chuỗi bằng chứng tái lập: audit dữ liệu, khóa split, chạy baseline, publish artifact và kiểm tra lại checksum.

## 2. Dữ liệu, audit và split

Nguồn chuẩn là [hnam25/asl-hand-gesture-images](https://huggingface.co/datasets/hnam25/asl-hand-gesture-images). Các run dùng revision `8f36ac00ece6dfce94410a980a839d93a912d366`; raw archive `ASL_Raw_Images.zip` có SHA-256 `594cfa0158044085ed61351315c187a6f3a3f9087795b4f4cdba68ecd04f12b1`.

Audit đọc được 36,000 ảnh, không có file hỏng và phát hiện 994 dòng thuộc 435 nhóm exact duplicate. Không random split từng ảnh. Sau canonicalization theo raw-image SHA-256, còn 35,441 ảnh và loại 559 duplicate crop. Split participant-disjoint cố định là:

| Split | Participants | Số ảnh canonical |
|---|---|---:|
| Train | P1, P3, P4, P5, P6, P7, P8, P10 | 28,365 |
| Validation | P2 | 3,487 |
| Test | P9 | 3,589 |

Điều này ngăn cùng exact-image SHA xuất hiện ở nhiều split. Audit cache, manifest và hướng tái sử dụng nằm ở [AUDIT_CACHE.md](AUDIT_CACHE.md).

## 3. Protocol đánh giá

Seed là 42. Validation P2 chọn checkpoint/hyperparameter; test P9 chỉ chạy sau khi selection hoàn tất. Mọi artifact phải chứa config, environment, split/dedup manifest, metrics, confusion matrix và checkpoint. Artifact upload xong được tải ngược rồi so SHA-256 với file local.

Bởi validation/test chỉ tương ứng một participant mỗi bên, chênh lệch validation-test có thể phản ánh khác biệt người ký hiệu hơn là tổng quát hóa thật sự. Đây là hạn chế chính: score không thay thế cho leave-one-participant-out hoặc external test set.

## 4. Kết quả baseline

| ID | Phương pháp | Val acc. | Test acc. | Macro-F1 | Nhận xét |
|---|---|---:|---:|---:|---|
| `cnn-001` | CNN from scratch | 72.10% | 83.25% | 80.10% | Mốc pixel model đơn giản; không nhận đúng lớp `0`. |
| `mnv4-001` | MobileNetV4 Conv-S ImageNet, frozen | 66.42% | 79.88% | 77.29% | Freeze toàn backbone chưa phù hợp với domain. |
| `mp-svm-001` | MediaPipe 2-hand landmarks + RBF SVM | 95.01% | 90.53% | 88.03% | Pose-only mạnh; 100% landmark coverage ở validation/test. |
| `mnv4-002` | MobileNetV4 Conv-S ImageNet, full fine-tune | 92.11% | 94.51% | 93.38% | Đối chứng archive processed. |
| `mp-mnv4-003` | Raw image → MediaPipe ROI (padding 0,18; raw fallback) → MobileNetV4 full fine-tune | **95.18%** | **96.66%** | **95.75%** | Mô hình được chọn. |

`mp-mnv4-003` vượt `mnv4-002` 2,15 điểm phần trăm test accuracy và 2,37 điểm Macro-F1. MediaPipe detect 100% ở validation/test; train có 29 fallback raw. Các kết luận này chỉ áp dụng cho split đã khóa.

## 5. Provenance artifact

| Run | Hugging Face model/artifact | Commit bất biến |
|---|---|---|
| `cnn-001` | [asl-hg-cnn-baseline](https://huggingface.co/hnam25/asl-hg-cnn-baseline) | `064551d5634ef34d3e6a9132ab3d4a374b2f5633` |
| `mnv4-001` | [asl-hg-mobilenetv4-baseline](https://huggingface.co/hnam25/asl-hg-mobilenetv4-baseline) | `8b78dcf957e05d68e00fe252f939023011a29a92` |
| `mp-svm-001` | [asl-hg-mediapipe-svm-baseline](https://huggingface.co/hnam25/asl-hg-mediapipe-svm-baseline) | `9cfccdf2e0fb39f7a517472ad3bb4f2988001518` |
| `mnv4-002` | [asl-hg-mobilenetv4-finetune-baseline](https://huggingface.co/hnam25/asl-hg-mobilenetv4-finetune-baseline) | `0c9311dddce54fd8f7e8e6c688626cbbd8ba0ab0` |
| `mp-mnv4-003` | [asl-hg-mediapipe-roi-transfer](https://huggingface.co/hnam25/asl-hg-mediapipe-roi-transfer) | `fcb5d06567f8cdacce7ebfe70c63fa4f396c6007` |

Landmark cache của `mp-svm-001` nằm tại `derived/mediapipe-two-hand-landmarks-v1/` trong dataset repository, commit `815e149b53ebbc5be9022fbfddb5b126a8938abf`. `mnv4-002` checkpoint đã được re-download và xác minh SHA-256 `a3a30a27290340a52e5cac408bc7dbbaa592cc6ebef3654113e3c004126b0101`.

## 6. Cách tái lập

Mở notebook theo run cần tái lập trên Colab. Notebook tải dataset/artifact bằng revision cụ thể, xác minh archive checksum, tái dựng canonical set và split, sau đó lưu outputs. `notebooks/11_mobilenetv4_timm_finetune_reproducible.ipynb` log batch 1/mỗi 50 batch/batch cuối, elapsed và ETA; ghi `training_progress.csv` sau mỗi epoch để không phụ thuộc console stream.

Không cần train lại để kiểm tra số liệu: download model repository ở bảng trên tại đúng commit, đọc `metrics/summary.json`, `metadata/experiment_config.json` và checksum checkpoint. Quy trình Colab CLI đầy đủ ở [GOOGLE_COLAB_CLI.md](GOOGLE_COLAB_CLI.md).

## 7. Hướng nghiên cứu tiếp theo

Phương pháp được chọn là raw image--MediaPipe ROI--MobileNetV4 full fine-tuning (`mp-mnv4-003`). ROI một tay dùng padding 0,18; fallback raw image được khóa trước khi xem test. So sánh với `mnv4-002` là ablation archive processed so với raw-image ROI. Leave-one-participant-out/external evaluation và one-hand/two-hand vẫn là các hướng tiếp theo.

Để đưa vào luận văn, nên bổ sung leave-one-participant-out hoặc external evaluation, latency đo trên cùng hardware, và phân tích lỗi theo từng lớp. Không tuyên bố hiệu năng tổng quát ngoài protocol P9 hiện tại.

## Addendum ROI padding 0,10

Ablation P2 chọn ROI padding 0,10 + raw fallback (accuracy 92,77%, Macro-F1 91,64%). Đánh giá P9 duy nhất đạt **96,38% accuracy** và **95,64% Macro-F1**; artifact immutable: [asl-hg-mediapipe-roi-p010-final](https://huggingface.co/hnam25/asl-hg-mediapipe-roi-p010-final), commit `e2faeb2768d2b081a9f7c9acc410401f7b758f1b`.

## Final selected ROI 0.10 artifact

The P2-selected ROI padding 0.10 raw-fallback run was evaluated on P9 once: 96.38% accuracy and 95.64% Macro-F1. Artifact: [asl-hg-mediapipe-roi-p010-final](https://huggingface.co/hnam25/asl-hg-mediapipe-roi-p010-final), commit `e2faeb2768d2b081a9f7c9acc410401f7b758f1b`.
