# Tổng hợp thực nghiệm và phương pháp được chọn

## Mục tiêu

Đồ án phân loại 36 ký hiệu ASL tĩnh (`0–9`, `A–Z`). Các thí nghiệm dùng dữ liệu đã audit, loại exact duplicate theo SHA-256 và participant-disjoint split cố định: train P1/P3–P8/P10, validation P2, test P9. Seed là 42.

Ảnh đầu vào cho các image baseline được lấy từ archive crop bàn tay MediaPipe đã publish (`ASL_Processed_Images.zip`). Vì vậy, không chạy lại MediaPipe ở mỗi thí nghiệm.

## Các phương pháp đã thử

| ID | Phương pháp | Val accuracy | Test accuracy | Macro-F1 | Kết luận |
|---|---|---:|---:|---:|---|
| `cnn-001` | CNN from scratch | 72.10% | 83.25% | 80.10% | Baseline pixel cơ bản; recall lớp `0` bằng 0%. |
| `mnv4-001` | MobileNetV4 pretrained ImageNet, đóng băng backbone, chỉ train classifier | 66.42% | 79.88% | 77.29% | Transfer learning frozen không phù hợp. |
| `mp-svm-001` | MediaPipe two-hand landmarks + StandardScaler + RBF SVM | 95.01% | 90.53% | 88.03% | Baseline landmark cạnh tranh. |
| `mp-mnv4-001` / `mnv4-002` | MediaPipe crop cache + MobileNetV4 pretrained ImageNet, full fine-tuning ngay từ đầu | 92.11% | **94.51%** | **93.38%** | Tốt nhất. |
| `mp-mnv4-002` | Progressive transfer: train classifier head 3 epoch, sau đó full fine-tune | 82.33% | 85.37% | 83.77% | Thấp hơn phương pháp tốt nhất 9.14 điểm test accuracy. |

Ở `mp-mnv4-001` / `mnv4-002`, recall của cả `O` và `0` là 100%, không có lỗi nhầm `O → 0` hoặc `0 → O`. Thí nghiệm progressive transfer có recall `O` chỉ 54%, vì vậy không được chọn.

## Phương pháp cuối cùng

Phương pháp cuối cùng là **Transfer Learning dạng full fine-tuning**:

1. Dùng MobileNetV4 Conv-S đã tiền huấn luyện trên ImageNet.
2. Dùng ảnh bàn tay đã crop bởi MediaPipe từ cache đã publish.
3. Thay classifier đầu ra thành 36 lớp ASL.
4. Fine-tune toàn bộ backbone và classifier ngay từ đầu trên tập train đã khóa.
5. Chọn checkpoint theo validation P2; chỉ báo cáo test P9 sau khi cấu hình đã khóa.

Phương pháp đạt **94.51% test accuracy** và **93.38% macro-F1**. Đây là lựa chọn đáp ứng yêu cầu dùng transfer learning và tốt hơn tất cả baseline đã thử trên protocol hiện tại.

## Hạn chế

Kết quả áp dụng cho split participant-disjoint hiện tại; validation và test chỉ tương ứng một participant mỗi bên. Không nên khẳng định khả năng tổng quát hóa ngoài protocol này nếu chưa có leave-one-participant-out hoặc external test set.
