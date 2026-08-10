# Phân tích repository và kế hoạch baseline cho đồ án ASL tĩnh

> Phạm vi: repository tại thời điểm phân tích. Đây là đồ án **phân loại ảnh tĩnh 36 lớp ASL** (`0–9`, `A–Z`) chạy trên Google Colab qua `google-colab-cli`. Webcam chỉ là demo local phụ trợ, không phải luồng tạo số liệu.

## 1. Kết luận ngắn

- Luồng phù hợp để chạy thí nghiệm là **Colab CLI → notebook → artifacts trong `outputs/`**. Không nên dùng webcam hoặc các lệnh stage local để tạo số liệu báo cáo.
- Baseline hiện được hiện thực đầy đủ trong hai notebook chạy Colab là **MobileNetV2 pretrained ImageNet, frozen backbone, head 36 lớp**. Chưa có accuracy/metric cuối được lưu trong repository, vì vậy chưa được phép ghi nhận đây là baseline có số liệu.
- Có cache crop tay đã được thiết kế để tái sử dụng. Sau khi cache đã publish trên Hugging Face, `notebooks/02_train_from_published_cache.ipynb` là đường chạy nhanh, đúng đắn nhất cho nhiều baseline.
- Trước khi chốt bất kỳ baseline nào, cần sửa protocol split để không tách các ảnh trùng byte-for-byte sang các tập khác nhau. Metadata hiện có 994 ảnh thuộc 435 nhóm trùng; split theo ảnh hiện tại có nguy cơ data leakage.
- Cần xây dựng CNN-from-scratch làm baseline thấp hơn, sau đó MobileNetV2 frozen và fine-tuned làm baseline mạnh. Cùng một tập test khóa, một seed và cùng crop cache là điều kiện để chứng minh improvement công bằng.

## 2. Repository thực sự làm gì

### Dữ liệu và bài toán

- Nguồn dữ liệu: Hugging Face dataset `hnam25/asl-hand-gesture-images`.
- Archive raw: `ASL_HG_36000/ASL_Raw_Images.zip`.
- Nhãn: 36 lớp, theo thứ tự `0–9`, `A–Z`.
- Mục tiêu trong tài liệu: 36.000 ảnh RGB, 1.000 ảnh/lớp.
- Pipeline dùng MediaPipe Hand Landmarker để lấy bounding box bàn tay, thêm padding 18% mỗi phía (hệ số cạnh `1.36`), rồi lưu crop để classifier học trên vùng tay.

### Hai luồng notebook

| Luồng | Notebook | Khi dùng | Vai trò |
|---|---|---|---|
| Full rebuild | `notebooks/ASL_End_to_End_Colab.ipynb` | Lần đầu hoặc khi raw data/tiền xử lý thay đổi | Tải raw, audit, crop MediaPipe, split, train MobileNetV2, evaluate. |
| Reuse cache | `notebooks/01_audit_cache_publish.ipynb` rồi `notebooks/02_train_from_published_cache.ipynb` | Chạy lặp nhiều baseline | Build/publish cache một lần; notebook train chỉ xác minh SHA raw, tải crop và train/evaluate. |

Các notebook nhỏ `00`–`06` gọi code trong `src/` để debug/tách stage. Chúng không phải là lựa chọn nên ưu tiên để thu số liệu trên Colab CLI, vì notebook self-contained đã có logic train/evaluate riêng.

### Webcam không nằm trong pipeline thí nghiệm

`src/inference/webcam_demo.py` và `notebooks/07_realtime_validation.ipynb` chỉ tải model đã train để demo/kiểm tra inference local. Colab là môi trường headless nên không phù hợp webcam; chính notebook end-to-end cũng ghi rõ điều này. Không đưa FPS/webcam vào phần benchmark chính trừ khi đồ án có một mục riêng về triển khai realtime.

## 3. Những gì đã có và mức độ sẵn sàng

### Kết quả dữ liệu có bằng chứng local

Từ `data/metadata/`:

| Kiểm tra | Kết quả |
|---|---:|
| Ảnh audit/readable | 36.000 |
| Ảnh unreadable | 0 |
| Crop MediaPipe thành công | 35.971 |
| Không phát hiện bàn tay | 29 (0,08%) |
| Lỗi `no_hand_detected` theo lớp | `0`: 9, `H`: 17, `R`: 3 |
| Nhóm hash trùng byte-for-byte | 435 |
| Ảnh thuộc nhóm trùng | 994 |
| Nhóm trùng chéo nhãn | 0 |

Handoff ghi split dự kiến sau crop là train 28.776, validation 3.597, test 3.598; các số này phù hợp với 35.971 ảnh và tỉ lệ 80/10/10.

### Baseline MobileNetV2 hiện có

Cả `ASL_End_to_End_Colab.ipynb` và `02_train_from_published_cache.ipynb` dùng cùng setup:

- Input crop RGB `224×224`.
- `MobileNetV2(include_top=False, weights='imagenet')`.
- Backbone frozen.
- `mobilenet_v2.preprocess_input` trong model.
- `GlobalAveragePooling2D → Dropout(0.2) → Dense(36, softmax)`.
- Adam `1e-3`, categorical cross entropy, batch size 32, tối đa 20 epochs.
- ModelCheckpoint theo `val_accuracy`, EarlyStopping theo `val_loss` với patience 4, ReduceLROnPlateau theo `val_loss` với patience 2/factor 0.2.

Artifacts mong đợi sau run:

- `outputs/models/baseline_mobilenetv2.keras`
- `outputs/logs/baseline_history.csv` (full end-to-end) hoặc `outputs/logs/training_history.csv` (cache notebook)
- `outputs/metrics/classification_report.csv`
- `outputs/metrics/confusion_matrix.csv`
- `outputs/metrics/o_zero_analysis.json`
- `outputs/figures/confusion_matrix.png`

Repository hiện không có các artifact kết quả này. Handoff chỉ xác nhận checkpoint đã từng được sinh trên Colab, chưa tải metrics/checkpoint về local. Vì vậy bảng `docs/EXPERIMENT_LOG.md` vẫn trống và không có baseline score có thể trích dẫn.

## 4. Rủi ro phương pháp cần xử lý trước khi so sánh

### P0 — Split có khả năng leakage từ ảnh trùng

Audit đã phát hiện 994 ảnh trong 435 nhóm có SHA-256 giống hệt nhau. Cả notebook full và cache đều gọi `train_test_split(..., stratify=label, random_state=42)` trên từng ảnh, không group theo hash. Các bản sao cùng một lớp có thể xuất hiện giữa train/validation/test, làm accuracy test lạc quan.

Cách xử lý tối thiểu trước khi tạo baseline chính thức:

1. Join manifest segmentation với `audit` theo `source_path/path` để có `sha256`.
2. Chỉ giữ một ảnh đại diện cho mỗi `sha256` **trước** stratified split (vì toàn bộ duplicate group hiện cùng nhãn).
3. Lưu `dedup_policy`, số ảnh còn lại và danh sách hash loại bỏ trong artifact run.
4. Tạo split một lần với seed 42, lưu `train.csv`, `val.csv`, `test.csv`, sau đó **không tạo lại** cho mọi baseline/improvement.

Nếu muốn giữ toàn bộ bản sao để training, cần group toàn bộ bản sao vào một split và kiểm tra một nhóm hash không cắt qua split. Khi đó cần một group-stratified splitter; không được random image-level như hiện tại.

### P1 — Tái lập dữ liệu chưa được khóa revision

`REVISION = None` ở notebook full và script download. Dataset có thể thay đổi theo thời gian. Trước run chính thức, lấy SHA commit của Hugging Face dataset và gán vào `REVISION`/`--revision`; ghi SHA này trong experiment log. Cache notebook đã làm tốt hơn: nó đối chiếu SHA-256 raw archive trước khi dùng crop cache.

### P1 — Hai implementation dễ drift

- Notebook full/cache có code train riêng.
- `src/models/train.py`/`src/evaluation/evaluate.py` có implementation khác dùng YAML.
- `configs/baseline.yaml` cũng là MobileNetV2 frozen nhưng đường output và log name không hoàn toàn đồng nhất với notebook cache.
- `src/models/mobilenetv2.py` là implementation trùng lặp và hiện không được gọi bởi train factory.

Chọn **một** implementation chuẩn cho benchmark. Khuyến nghị dùng `02_train_from_published_cache.ipynb` và đưa mọi architecture/config baseline vào notebook hoặc một module import chung; không sửa song song hai nơi rồi giả định chúng đồng nhất.

### P2 — Cần phân biệt score chọn model và score báo cáo

Validation dùng cho early stopping, learning rate, augmentation và chọn checkpoint. Test chỉ chạy sau khi config được khóa. Không thử nhiều config rồi chọn score test cao nhất. Bảng log hiện có ý đúng nhưng chưa ghi đủ: cần thêm dataset revision, cache SHA, dedup policy, input size, epoch best, số parameters và đường dẫn artifact.

## 5. Baseline nên chạy để có căn cứ improvement

Mục tiêu không phải chỉ một model cao điểm, mà là một chuỗi so sánh có kiểm soát.

| ID gợi ý | Model | Mục đích | Quy tắc công bằng |
|---|---|---|---|
| `data-001` | Không train | Khóa revision, crop cache, deduplicate/group split | Đây là nền dữ liệu dùng chung; không có test score. |
| `cnn-001` | CNN từ đầu | Baseline đơn giản, đo đóng góp của pretrained transfer learning | Cùng crop, split, 36 lớp, optimizer/early-stop hợp lý. |
| `mnet-001` | MobileNetV2 frozen | Baseline mạnh hiện đã có trong repo | Giữ chính xác setup hiện có. |
| `mnet-002` | MobileNetV2 fine-tune | Baseline transfer-learning cải tiến chuẩn | Khởi tạo từ pretrained/frozen checkpoint; chỉ tune `fine_tune_at`, LR nhỏ. |
| `proposed-001` | Phương pháp của đồ án | So sánh cuối | Không đổi test split hay dataset protocol. |

### CNN-from-scratch nên là baseline nào

Một CNN nhỏ, không pretrained, có thể dùng input `128×128` để rẻ hơn nhưng cần ghi rõ khác biệt với MobileNetV2 `224×224`. Nếu muốn ablation thuần model nhất, dùng luôn `224×224` cho cả hai; nếu GPU/time hạn chế, chấp nhận `128×128` và báo cáo nó là trade-off tính toán.

Cấu hình khởi đầu hợp lý:

- `Rescaling(1/255)` trong model.
- Ba block `Conv(32/64/128) → BatchNorm → ReLU → Conv → BatchNorm → ReLU → MaxPool`, dropout nhỏ sau mỗi block.
- Global average pooling → Dense 256 → Dropout 0.3 → Dense 36 softmax.
- Adam `1e-3`, batch 32, tối đa 30 epochs, EarlyStopping/ReduceLROnPlateau giống baseline MobileNetV2.
- Không horizontal flip mặc định: flip có thể làm thay đổi handedness/hình dạng ký hiệu. Chỉ thêm augmentation (rotation nhẹ, translation/zoom nhẹ, brightness) nếu được đánh giá bằng validation và ghi riêng thành experiment khác.

CNN nên được cài thành một architecture có thể gọi từ cùng notebook/module benchmark; không nên chỉ để code rời hoặc thay thế MobileNetV2 mặc định, vì cần giữ lại cả hai để so sánh.

### Thước đo bắt buộc

- Accuracy test tổng.
- Macro precision, macro recall, macro F1 (class-balanced hơn accuracy).
- Per-class precision/recall/F1 từ classification report.
- Confusion matrix.
- `O_recall`, `0_recall`, số `O → 0` và `0 → O` — đây là tiêu chí đặc thù đã có trong repo.
- Best validation accuracy/loss, epoch của best checkpoint và thời gian train.
- Nếu báo cáo tính thực thi: số parameters và latency inference trên cùng một môi trường, tách biệt với benchmark accuracy.

## 6. Quy trình Colab CLI đề xuất

### Một lần: khóa dữ liệu và chuẩn bị cache

1. Chọn commit SHA của Hugging Face dataset.
2. Chạy `01_audit_cache_publish.ipynb` với SHA đó để tạo crop cache và manifest.
3. Tải 5 artifact cache về local, publish bằng `scripts/upload_hf_cache.py` với `HF_TOKEN` chỉ đặt ở máy local.
4. Bổ sung dedup/group-split vào pipeline training trước run benchmark; publish một version cache/split mới nếu crop/split schema thay đổi.

### Mỗi baseline

```bash
colab new -s asl-baseline --gpu T4
colab install -s asl-baseline -r requirements.txt
colab exec -s asl-baseline -f notebooks/02_train_from_published_cache.ipynb --timeout 7200
```

Sau mỗi run, download toàn bộ artifacts (model, history, metrics, figure, config/experiment log), ghi một dòng trong `docs/EXPERIMENT_LOG.md`, rồi dừng runtime:

```bash
colab stop -s asl-baseline
```

Tên session/output phải mang experiment ID, ví dụ `cnn-001`, `mnet-001`, `mnet-002`, để tránh checkpoint và logs ghi đè lên nhau. Không chạy baseline khác trước khi tải xong artifact của run hiện tại.

## 7. Thứ tự triển khai nên làm tiếp

1. Chốt revision dataset, deduplicate/group split và lưu manifest cố định.
2. Hoàn thiện `cnn-001` trong luồng notebook cache, chạy và tải đủ metrics.
3. Chạy lại `mnet-001` trên **cùng split đã dedup**; score MobileNet cũ (nếu có) không so sánh trực tiếp được nếu split thay đổi.
4. Chạy `mnet-002` fine-tuning với LR thấp hơn (repo gợi ý `fine_tune_at=120`, `1e-4`).
5. Chỉ sau khi chọn phương pháp dựa trên validation, chạy test một lần cho `proposed-001` và lập bảng so sánh.

## 8. Tình trạng working tree khi phân tích

Có các thay đổi local chưa commit: `src/data/dataloader.py`, `src/models/cnn.py`, `tests/test_cnn.py`. CNN này hiện chưa được nối vào `src/models/transfer.py`, YAML hoặc các notebook benchmark, nên **chưa phải baseline runnable** và không được ghi vào experiment log. Cần tích hợp có chủ đích theo protocol ở trên hoặc bỏ các thay đổi dở dang trước run chính thức.
