# Chiến lược baseline và phương pháp đề xuất

## Trạng thái đã hoàn tất

Bốn baseline đã hoàn tất theo cùng protocol và không được train lại/ghi đè. Kết quả chuẩn nằm trong [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

1. `cnn-001`: CNN from scratch — chứng minh mốc học từ đầu.
2. `mnv4-001`: MobileNetV4 ImageNet frozen — cô lập hiệu ứng feature extractor frozen.
3. `mp-svm-001`: MediaPipe two-hand landmarks + RBF SVM — mốc hand-pose không dùng pixels.
4. `mnv4-002`: MobileNetV4 ImageNet full fine-tuning — mốc image transfer mạnh nhất hiện tại.

## Phương pháp đề xuất tiếp theo

`mp-mnv4-001`: **MediaPipe Hand Landmarker → two-hand-guided crop/normalization → MobileNetV4 Conv-S full fine-tuning**.

Mục tiêu không chỉ là tối đa accuracy mà còn kiểm tra MediaPipe có mang lại cải thiện thực sự so với `mnv4-002` hay không. Dùng landmark cache versioned hiện có, cùng exact canonical set và participant split; tuyệt đối không đổi test P9.

## Ablation tối thiểu

- Image gốc (`mnv4-002`) so với MediaPipe-guided crop.
- One-hand crop so với two-hand crop/union crop.
- Landmark-only (`mp-svm-001`) so với image-only (`mnv4-002`) so với phương pháp đề xuất.
- Detection coverage và fallback: nếu không detect được tay, dùng image gốc theo một policy cố định, report số lượng fallback theo từng split.

## Tiêu chí báo cáo

Report test accuracy, macro precision/recall/F1, per-class recall/F1, confusion matrix, detection coverage, latency và số parameters. Hyperparameter chỉ chọn theo validation P2. Kết luận improvement chỉ hợp lệ nếu so với `mnv4-002` trong cùng protocol; phải nêu hạn chế chỉ có một participant test.
