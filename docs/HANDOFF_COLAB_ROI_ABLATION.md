# Handoff: ROI ablation trên Google Colab

Tài liệu này ghi lại trạng thái thí nghiệm hiện tại để một AI khác có thể tiếp quản mà không phải suy đoán hoặc chạy lại từ đầu.

## Mục tiêu

Hoàn tất ablation cho pipeline nhận dạng ASL tĩnh 36 lớp với split **participant-disjoint**. Biến thể đang chạy là P2 (selection/validation), dùng crop ROI từ MediaPipe trước khi fine-tune MobileNetV4. Chỉ sau khi chốt cấu hình từ P2 mới được chạy P9/test một lần.

## Những thay đổi đã có trong working tree

- Đã bỏ webcam/realtime demo chưa hoàn thiện để tránh gây hiểu nhầm phạm vi đồ án.
- Đã bổ sung phân tích lỗi: `src/evaluation/error_analysis.py` và `scripts/analyze_errors.py`.
- Runner chính: `scripts/run_mediapipe_transfer.py`.
  - Hỗ trợ `--input-mode mediapipe_roi`, crop cache, archive/restore cache.
  - Hỗ trợ resume qua `--resume-state`; state được ghi atomically sau từng epoch.
  - Hỗ trợ `--selection-only` để không vô tình đánh giá P9/test.
- Hướng dẫn chạy: `docs/MEDIAPIPE_TRANSFER_RUN.md`.
- Dependencies cho Colab: `requirements-colab-transfer.txt`.
- Các thay đổi này chưa được commit; giữ nguyên thay đổi người dùng trong working tree.

## Baseline và kết quả hiện tại

| Cấu hình | Split | Epoch | Accuracy | Macro-F1 | Ghi chú |
| --- | --- | ---: | ---: | ---: | --- |
| Raw image control | P2 validation | 15 | 84.34% | 81.77% | Control trước ROI |
| MediaPipe ROI, padding 0.18, fallback raw image | P2 validation | 3 | 83.25% | 80.85% | Stage 1 |
| MediaPipe ROI, padding 0.18, fallback raw image | P2 validation | 6 | 83.85% | 81.71% | Stage 2 hiện tại |

Trong Stage 2, validation accuracy tốt nhất xuất hiện ở epoch 5: **85.12%**. Checkpoint resume vẫn có `best_model_state`; không thay thế nó bằng checkpoint epoch cuối khi đánh giá kết quả tốt nhất.

## Artefact đã được bảo toàn local

Các file này nằm trong `.gitignore`, không commit hoặc đẩy public:

| File | Mục đích | SHA-256 |
| --- | --- | --- |
| `local-artifacts/crop_cache_mediapipe_roi-padding-0_18-fallback-raw_image.tar.gz` | Cache crop ROI đã dựng | `52894d6b7f1360445e8db26bf3e937a16253cb66928d4f5e65e694aaeca48b23` |
| `local-artifacts/roi-p018/training_state_epoch_3.pt` | Resume sau Stage 1 | `a5dea2d5643d4ca5417b5127a1591fbbd923123173cb0f56f785349aaed163ff` |
| `local-artifacts/roi-p018/training_state_epoch_6.pt` | Resume sau Stage 2 | `8c030850bc863b1bd7947516819feaf8f08b8b1c7cecfcdf449593e477592629` |
| `local-artifacts/roi-p018/validation_summary_epoch_6.json` | Kết quả Stage 2 | `b9222e949954c69311105303273d7979c7177c8b5e70f644e2507aedafcd100c` |

Cache crop cũng đã được upload vào private Hugging Face repo `hnam25/asl-hg-mp-mnv4-ablation-checkpoints`, path `crop/crop_cache_mediapipe_roi-padding-0_18-fallback-raw_image.tar.gz`. Không in hoặc commit token HF; token chỉ được lấy từ môi trường khi cần tải repo private.

## Trạng thái Colab

- Session gần nhất: `asl-train-hf-stage-1`.
- T4 đã chạy Stage 2 hoàn tất; không còn process train tại thời điểm handoff.
- Working directory trên Colab: `/content/asl-static-recognition`.
- Work directory trên Colab: `/content/ablation_roi_p018`.
- Tài liệu này không giả định session còn tồn tại. Luôn kiểm tra bằng `colab sessions` trước khi dùng lại.

## Next step bắt buộc: chạy Stage 3 (epoch 7--9)

Mục tiêu trước mắt: continue **cùng chính xác cấu hình P2 ROI** đến epoch 9. Chưa đổi padding, fallback, backbone, seed, split hoặc chuyển sang P9/test.

1. Kiểm tra session và trạng thái process:

   ```bash
   colab sessions
   printf '%s\n' \
     'ps -eo pid,etime,cmd | grep "run_mediapipe_transfer.py" | grep -v grep || true' \
     'tail -n 30 /content/ablation_roi_p018/train_stage_2.log 2>/dev/null || true' \
     | colab console -s asl-train-hf-stage-1
   ```

2. Nếu session cũ còn sống, xác nhận có file state ở `/content/ablation_roi_p018/outputs/models/training_state.pt`. Nếu session mất, tạo T4 session mới, clone/cài project, upload hoặc tải lại crop archive, rồi khôi phục state local `training_state_epoch_6.pt` vào đúng đường dẫn work directory.

3. Khởi động Stage 3 trong background để CLI không ngắt job dài:

   ```bash
   printf '%s\n' \
     'cd /content/asl-static-recognition' \
     'nohup python scripts/run_mediapipe_transfer.py --work-dir /content/ablation_roi_p018 --input-mode mediapipe_roi --padding-ratio 0.18 --fallback-policy raw_image --epochs 9 --selection-only --no-upload --resume-state /content/ablation_roi_p018/outputs/models/training_state.pt > /content/ablation_roi_p018/train_stage_3.log 2>&1 & echo "TRAIN_STAGE_3_PID=$!"' \
     | colab console -s asl-train-hf-stage-1
   ```

4. Theo dõi bằng log, không khởi động duplicate process:

   ```bash
   printf '%s\n' \
     'ps -eo pid,etime,cmd | grep "run_mediapipe_transfer.py" | grep -v grep || true' \
     'tail -n 40 /content/ablation_roi_p018/train_stage_3.log' \
     | colab console -s asl-train-hf-stage-1
   ```

5. Khi log có JSON `validation_summary` với `epochs_ran: 9` và process đã kết thúc, download ngay hai artefact sau về local:

   ```bash
   mkdir -p local-artifacts/roi-p018
   colab download -s asl-train-hf-stage-1 /content/ablation_roi_p018/outputs/models/training_state.pt local-artifacts/roi-p018/training_state_epoch_9.pt
   colab download -s asl-train-hf-stage-1 /content/ablation_roi_p018/outputs/metrics/validation_summary.json local-artifacts/roi-p018/validation_summary_epoch_9.json
   sha256sum local-artifacts/roi-p018/training_state_epoch_9.pt local-artifacts/roi-p018/validation_summary_epoch_9.json
   ```

6. Lặp lại cùng mẫu cho epoch 12, rồi epoch 15 (`--epochs 12`, `--epochs 15`). Sao lưu state sau mỗi stage.

## Khi session Colab mất

Không chạy crop MediaPipe lại nếu không cần. Khôi phục theo thứ tự này:

1. Dùng local checkpoint mới nhất (`training_state_epoch_6.pt`, sau này là epoch 9/12/15) và upload vào `/content/ablation_roi_p018/outputs/models/training_state.pt`.
2. Khôi phục cache crop từ local archive qua `--restore-crop-archive /content/crop_cache.tar.gz`, hoặc tải archive từ private HF nếu người vận hành có quyền đọc.
3. Chạy lại runner với đúng `--resume-state` và `--epochs` target của stage tiếp theo.
4. Xác nhận log hiện `next_epoch` đúng (ví dụ Stage 3 phải là `next_epoch: 7`) trước khi để job chạy.

Không dùng Hugging Face token trong command line, log hoặc file repo. Nếu cần token private repo, truyền qua environment tạm thời và xóa file tạm ngay sau khi dùng.

## Điều kiện trước khi chuyển P9/test

- Hoàn tất P2 theo lịch epoch đến 15 và chọn cấu hình chỉ dựa trên P2/validation.
- Ghi lại rõ metric dùng để chọn (khuyến nghị Macro-F1) và checkpoint tốt nhất.
- Chỉ chạy P9/test đúng một lần cho cấu hình đã chốt; sau đó chạy `scripts/analyze_errors.py` trên predictions để tạo error pairs, error examples và error-by-class.

