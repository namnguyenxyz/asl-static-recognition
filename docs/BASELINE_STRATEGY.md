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

## Ablation đã được chuẩn hóa để chạy

Mỗi variant phải giữ nguyên revision dữ liệu, canonical split, seed, model, epoch budget và quy tắc chọn checkpoint theo validation P2. Đặt một `--work-dir` và một `--hf-repo-id` riêng cho từng variant; chạy mỗi variant với `--selection-only` để xuất `validation_summary.json` và tuyệt đối không đọc test P9 để chọn padding hay fallback policy.

| Câu hỏi | Variant chạy bằng `scripts/run_mediapipe_transfer.py` | Quyết định theo P2 |
|---|---|---|
| ROI có ích hơn ảnh raw không? | `--input-mode raw_image` so với `--input-mode mediapipe_roi --padding-ratio 0.18` | Macro-F1 và accuracy trong `validation_summary.json` |
| ROI nhạy với vùng đệm không? | ROI với `--padding-ratio 0.10`, `0.18`, `0.26` | Macro-F1 validation; chỉ một variant được đánh giá P9 cuối |
| Fallback có ảnh hưởng không? | ROI với `--fallback-policy raw_image` so với `center_crop` | Macro-F1 validation và số fallback |

Runner cô lập cache crop bằng input mode, padding và fallback policy; không được tái sử dụng crop của variant khác. Artifact phải lưu detection coverage, fallback count, `predictions.csv`, `error_examples.csv`, `error_pairs.csv` và `error_by_class.csv`.

## Phân tích lỗi bắt buộc

Sau mỗi đánh giá cuối, dùng `error_pairs.csv` để chọn các cặp nhầm lẫn nhiều nhất và dùng `error_examples.csv` để kiểm tra tối đa 5 ảnh cho mỗi cặp. Gán nguyên nhân quan sát được vào một trong các nhóm: ROI/detection, crop quá chặt hoặc quá rộng, góc/che khuất bàn tay, mờ/chất lượng ảnh, nền/ánh sáng, hoặc hình dạng lớp tương tự. Chỉ kết luận nguyên nhân khi ảnh tương ứng được kiểm tra; nếu không rõ, ghi là "không xác định".

## Tiêu chí báo cáo

Report test accuracy, macro precision/recall/F1, per-class recall/F1, confusion matrix, detection coverage và số parameters. Hyperparameter chỉ chọn theo validation P2. Kết luận improvement chỉ hợp lệ nếu so với `mnv4-002` trong cùng protocol; phải nêu hạn chế chỉ có một participant test.
