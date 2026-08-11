# Kế hoạch khởi động viết luận văn LaTeX

## Mục tiêu hiện tại

Khởi tạo khung luận văn và viết trước các phần đã ổn định từ đề tài/pipeline. Các quy chuẩn trình bày, template chính thức, giới hạn số trang và đề mục bắt buộc sẽ được áp dụng sau khi nhận được.

## Cấu trúc LaTeX đề xuất

```text
thesis/
├── main.tex                 # File biên dịch chính
├── chapters/
│   ├── 01_introduction.tex
│   ├── 02_background.tex
│   ├── 03_methodology.tex
│   ├── 04_experiments.tex
│   ├── 05_conclusion.tex
│   └── appendix.tex
├── figures/                 # Sơ đồ pipeline, biểu đồ, confusion matrix
├── tables/                  # Bảng dữ liệu và kết quả thực nghiệm
├── references.bib
└── assets/                  # Logo/trang bìa nếu template yêu cầu
```

## Kế hoạch viết đơn giản

| Bước | Phần | Trạng thái đầu vào | Việc cần làm | Đầu ra |
|---|---|---|---|---|
| 1 | Khởi tạo LaTeX | Chờ template trường | Tạo `main.tex`, cấu trúc chương, bibliography và lệnh build; thay template khi được cung cấp | Khung biên dịch PDF được |
| 2 | Mở đầu | Có thể viết ngay | Bối cảnh ASL, vấn đề nhận dạng cử chỉ tĩnh, mục tiêu 36 lớp và lỗi `O`/`0`, phạm vi, đóng góp, bố cục luận văn | Chương mở đầu bản nháp |
| 3 | Cơ sở lý thuyết/liên quan | Có thể bắt đầu | ASL-HG, CNN, MobileNetV4, MediaPipe Hand Landmarker và các chỉ số đánh giá | Chương nền tảng bản nháp + tài liệu tham khảo |
| 4 | Phương pháp đề xuất | Có thể viết ngay từ source | Mô tả dataset, audit, canonicalization, participant split, MobileNetV4 và quy trình tái lập bằng Colab CLI | Chương phương pháp bản nháp + sơ đồ pipeline |
| 5 | Thiết kế thực nghiệm | Viết ngay | Nêu participant-disjoint split, seed/config, metric, tiêu chí chọn model và protocol phân tích `O`/`0` | Khung chương thực nghiệm |
| 6 | Kết quả và thảo luận | Có dữ liệu baseline | Chèn metrics, confusion matrix và phân tích `O`/`0`; để TODO cho latency/FPS và MediaPipe-guided crop | Chương kết quả |
| 7 | Kết luận | Viết khung ngay, hoàn thiện sau | Tổng kết đóng góp, hạn chế, hướng phát triển (fine-tuning, TFLite, realtime) | Chương kết luận |
| 8 | Rà soát cuối | Chờ quy chuẩn trường | Đồng bộ thuật ngữ, trích dẫn, bảng/hình, phụ lục, kiểm tra PDF và tái lập kết quả | Bản nộp hoàn chỉnh |

## Việc có thể bắt đầu ngay

- [ ] Khởi tạo thư mục `thesis/` khi nhận template hoặc quy định trường.
- [ ] Viết Chương 1: Giới thiệu.
- [ ] Thu thập và ghi BibTeX cho dataset ASL-HG, MediaPipe, MobileNetV4 và các tài liệu nền tảng.
- [ ] Viết mô tả pipeline hiện có dựa trên `docs/DATASET.md`, `docs/AUDIT_CACHE.md` và source code.
- [ ] Vẽ sơ đồ pipeline: archive khóa revision → canonicalization → participant split → MobileNetV4 → evaluation.
- [ ] Tạo khung bảng kết quả, chưa điền số liệu chưa được xác nhận.

## Nội dung phải chờ kết quả hoặc đầu vào bổ sung

- Template LaTeX, quy định font/lề/trang bìa và đề mục bắt buộc của trường.
- Kết quả baseline/fine-tuning cuối cùng, biểu đồ training và confusion matrix.
- Benchmark FPS/latency realtime.
- Tiêu chí chính thức để khẳng định đã giảm nhầm lẫn giữa `O` và `0`.
- Danh mục tài liệu tham khảo do giảng viên yêu cầu.

## Nguyên tắc cập nhật

- Không bịa số liệu; dùng `TODO` hoặc bảng trống cho phần chưa chạy.
- Mỗi hình/bảng phải truy vết được về notebook, experiment ID hoặc file trong `outputs/`.
- Khi có template, chỉ cần ánh xạ các chương trên vào các đề mục bắt buộc của template, không viết lại nội dung đã có.
