# Tổng hợp thực nghiệm và phương pháp được chọn

## Mục tiêu

Đồ án phân loại 36 ký hiệu ASL tĩnh (`0–9`, `A–Z`). Các thí nghiệm dùng dữ liệu đã audit, loại exact duplicate theo SHA-256 và participant-disjoint split cố định: train P1/P3–P8/P10, validation P2, test P9. Seed là 42.

Ảnh đầu vào cho các image baseline được lấy từ archive ảnh đã xử lý đã publish của ASL-HG (`ASL_Processed_Images.zip`). Các run `mnv4-001` và `mnv4-002` không chạy lại MediaPipe trong lúc huấn luyện. Vì archive này là đầu vào có sẵn của dataset, kết quả hiện tại không cô lập được lợi ích của một pipeline crop MediaPipe do nhóm tạo.

## Các phương pháp đã thử

| ID | Phương pháp | Val accuracy | Test accuracy | Macro-F1 | Kết luận |
|---|---|---:|---:|---:|---|
| `cnn-001` | CNN from scratch | 72.10% | 83.25% | 80.10% | Baseline pixel cơ bản; recall lớp `0` bằng 0%. |
| `mnv4-001` | MobileNetV4 pretrained ImageNet, đóng băng backbone, chỉ train classifier | 66.42% | 79.88% | 77.29% | Transfer learning frozen không phù hợp. |
| `mp-svm-001` | MediaPipe two-hand landmarks + StandardScaler + RBF SVM | 95.01% | 90.53% | 88.03% | Baseline landmark cạnh tranh. |
| `mnv4-002` | MobileNetV4 pretrained ImageNet, full fine-tuning ngay từ đầu trên archive processed | 92.11% | **94.51%** | **93.38%** | Mô hình được chọn. |
| `mp-mnv4-002` | Progressive transfer trên archive processed: train classifier head 3 epoch, sau đó full fine-tune | 82.33% | 85.37% | 83.77% | Thấp hơn mô hình được chọn; không được chọn. |
| `mp-mnv4-003` | Raw image → MediaPipe ROI (padding 0,18; raw fallback) → MobileNetV4 full fine-tuning | **95.18%** | **96.66%** | **95.75%** | Mô hình được chọn; hơn `mnv4-002` 2,15 điểm test accuracy. |

Ở `mp-mnv4-003`, recall của cả `O` và `0` là 100%, không có lỗi nhầm `O → 0` hoặc `0 → O`; detection coverage ở validation/test là 100%. Thí nghiệm progressive transfer có recall `O` chỉ 54% và không được dùng làm bằng chứng chính.

## Phương pháp cuối cùng

Phương pháp cuối cùng là **Transfer Learning dạng full fine-tuning**:

1. Dùng raw image canonical của ASL-HG đã khóa revision.
2. Dùng MediaPipe Hand Landmarker một tay để trích xuất ROI vuông với padding 0,18; nếu không detect được tay/ROI không hợp lệ thì dùng raw image làm fallback và ghi manifest.
3. Dùng MobileNetV4 Conv-S đã tiền huấn luyện ImageNet, thay classifier đầu ra thành 36 lớp.
4. Fine-tune toàn bộ backbone và classifier ngay từ đầu trên tập train đã khóa.
5. Chọn checkpoint theo validation P2; chỉ báo cáo test P9 sau khi cấu hình đã khóa.

Phương pháp đạt **96,66% test accuracy** và **95,75% macro-F1**. Checkpoint, crop manifest, coverage, metrics và environment được publish tại commit `fcb5d06567f8cdacce7ebfe70c63fa4f396c6007` của [artifact `mp-mnv4-003`](https://huggingface.co/hnam25/asl-hg-mediapipe-roi-transfer). Đây là lựa chọn đáp ứng yêu cầu MediaPipe kết hợp transfer learning và tốt hơn các baseline đã có trên protocol hiện tại.

## Hạn chế

Kết quả áp dụng cho split participant-disjoint hiện tại; validation và test chỉ tương ứng một participant mỗi bên. Không nên khẳng định khả năng tổng quát hóa ngoài protocol này nếu chưa có leave-one-participant-out hoặc external test set.
## Addendum: ablation ROI padding 0,10

Theo lựa chọn trên validation P2, cấu hình ROI padding 0,10 với raw-image fallback đạt 92,77% validation accuracy và 91,64% Macro-F1. Đánh giá P9 được chạy đúng một lần sau khi khóa cấu hình: **96,38% test accuracy**, **95,64% Macro-F1**, macro precision 95,42% và macro recall 96,39%. Recall `O` và `0` đều 100%, không có lỗi `O→0` hoặc `0→O`.

Checkpoint, P9 summary, predictions và error pairs được publish tại [asl-hg-mediapipe-roi-p010-final](https://huggingface.co/hnam25/asl-hg-mediapipe-roi-p010-final), commit `e2faeb2768d2b081a9f7c9acc410401f7b758f1b`. Kết quả P9 này không được dùng để chọn padding/fallback; so sánh với run ROI padding 0,18 trước đó phải nêu rõ khác biệt lựa chọn theo P2.

## Ablation MediaPipe và giới hạn generalization

Để cô lập MediaPipe ROI, raw-image control giữ nguyên raw archive, canonical split, seed 42, MobileNetV4 Conv-S và full fine-tuning; chỉ thay ROI bằng ảnh raw. Control đạt 90,22% test accuracy và 88,81% Macro-F1 trên P9. So với ROI p=0,10 (96,38% / 95,64%), MediaPipe ROI đóng góp quan sát được **+6,16 điểm accuracy** và **+6,75 điểm Macro-F1** trong protocol P9. Đây là kết luận ablation chính; không suy rộng tự động ra mọi camera/dataset.

Hai kiểm tra zero-shot ngoài tập, không fine-tune, cho kết quả thấp: Sign Language MNIST 7.172 ảnh đạt 14,15% accuracy / 12,42% Macro-F1; RGB smoke test 600 ảnh đạt 14,33% / 14,64%, với ROI coverage 79,83%. Do đó mô hình mạnh trong miền ASL-HG nhưng chưa tổng quát hóa tốt sang nguồn ảnh khác. Demo 36 ảnh giáo dục tìm qua Google Image Search có raw 19/36 đúng và ROI 25/36 đúng (34/36 detect), nhưng chỉ là ví dụ định tính, không phải benchmark.
