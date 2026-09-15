# Chiến lược baseline và phương pháp đề xuất

## Trạng thái đã hoàn tất

Bốn baseline đã hoàn tất theo cùng protocol và không được train lại/ghi đè. Kết quả chuẩn nằm trong [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

1. `cnn-001`: CNN from scratch — chứng minh mốc học từ đầu.
2. `mnv4-001`: MobileNetV4 ImageNet frozen — cô lập hiệu ứng feature extractor frozen.
3. `mp-svm-001`: MediaPipe two-hand landmarks + RBF SVM — mốc hand-pose không dùng pixels.
4. `mnv4-002`: MobileNetV4 ImageNet full fine-tuning — mốc image transfer mạnh nhất hiện tại.

## Phương pháp đề xuất tiếp theo

`mp-mnv4-003`: **raw image → MediaPipe Hand Landmarker ROI (một tay) → MobileNetV4 Conv-S full fine-tuning**. Runner là `scripts/run_mediapipe_transfer.py`.

Mỗi canonical raw image được crop bằng MediaPipe với padding cố định 0,18. Nếu không phát hiện được tay hoặc ROI không hợp lệ, runner dùng nguyên ảnh raw làm fallback và ghi lại trạng thái theo split. Mục tiêu là kiểm tra MediaPipe có mang lại cải thiện thực sự so với `mnv4-002` hay không, dùng đúng canonical set và participant split; tuyệt đối không đổi test P9.

## Ablation tối thiểu

- Archive processed hiện hành (`mnv4-002`) so với raw-image MediaPipe ROI (`mp-mnv4-003`).
- One-hand ROI hiện hành so với two-hand/union crop nếu mở rộng sau này.
- Landmark-only (`mp-svm-001`) so với image-only (`mnv4-002`) so với phương pháp đề xuất.
- Detection coverage và fallback: nếu không detect được tay, dùng raw image theo policy cố định và report số lượng fallback theo từng split.

## Tiêu chí báo cáo

Report test accuracy, macro precision/recall/F1, per-class recall/F1, confusion matrix, detection coverage, latency và số parameters. Hyperparameter chỉ chọn theo validation P2. Kết luận improvement chỉ hợp lệ nếu so với `mnv4-002` trong cùng protocol; phải nêu hạn chế chỉ có một participant test.
