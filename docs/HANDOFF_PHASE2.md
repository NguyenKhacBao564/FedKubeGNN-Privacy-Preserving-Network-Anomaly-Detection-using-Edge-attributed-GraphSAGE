# Tài liệu Bàn giao — Giai đoạn 2: Federated Learning

> **Dành cho người phụ trách GĐ2** (Nguyễn Chí Hiếu hoặc thành viên khác).
> Tóm tắt ngắn gọn những gì GĐ1 đã làm và những gì cần biết để bắt đầu GĐ2.

---

## 1. Artifact GĐ2 sẽ tái dùng

| Artifact | Đường dẫn | Ghi chú |
|---|---|---|
| **E-GraphSAGE** (model tốt nhất) | `src/model.py` | Đã test, class `EGraphSAGE`, `build_model()` factory |
| **Pipeline tiền xử lý** | `src/preprocess.py` | Hàm `transform()` — tái sử dụng cho mọi client FL |
| **Shared preprocessor** | `src/multi_scenario.py` | `class_to_idx` (8 lớp), `feature_dim=55`, shared scaler |
| **Xử lý mất cân bằng** | `src/imbalance.py` | `compute_class_weights()`, `undersample_majority()` |
| **Đồ thị** | `src/graph_build.py` | `build_graph()` — DataFrame → PyG Data |
| **Config** | `config.yaml` | Danh sách scenario, đường dẫn, hyperparameter |

**QUAN TRỌNG**: `src/preprocess.py` đã được viết dạng module tái sử dụng (không phải notebook một lần). GĐ3 có thể dùng lại để xử lý luồng sự kiện Falco real-time.

---

## 2. Điểm mấu chốt để so sánh công bằng với GĐ1

### Protocol "pooled" = "mô hình tập trung"

| Metric | Giá trị | Ý nghĩa |
|---|---|---|
| Pooled macro-F1 | **0.8773** | E-GraphSAGE, class_weight, pooled graph |
| Pooled accuracy | 0.9840 | Không có ý nghĩa nhiều do mất cân bằng |
| Pooled weighted-F1 | 0.9842 | Phản ánh đúng hơn do tính support-weighted |

**FL sẽ so kết quả với con số pooled này.** Nếu FL macro-F1 trên mỗi client thấp hơn 0.8773, cần kiểm tra:
- Có đang dùng đúng shared preprocessor (cùng 55 features, cùng scaler/encoder)?
- Liệu communication rounds có đủ để model hội tụ?
- Có vấn đề non-IID quá nghiêm trọng không?

### Protocol LOSO = "tổng quát hóa sang mạng mới"

| Metric | Giá trị |
|---|---|
| LOSO mean macro-F1 | 0.2334 |

LOSO thấp vì **mất lớp private khi held-out** (34-1: lớp DDoS mất; 36-1: 3 lớp Okiru mất). Đây là vấn đề **data distribution**, không phải lỗi model. FL có thể khắc phục được nếu các client chia sẻ kiến thức qua communication rounds.

---

## 3. Gợi ý chia client cho FL

```
Mỗi client = 1–2 scenario (non-IID tự nhiên)
```

Tại sao:
- IoT-23 đã có sẵn phân bố lệch giữa các scenario — mỗi scenario là một "domain" malware khác nhau.
-FL trên dữ liệu non-IID chính là bài toán thực tế.

**Ví dụ**:
- Client 1: scenario 34-1 (Mirai)
- Client 2: scenario 1-1 (Hide-and-Seek) + 3-1 (Muhstik)
- Client 3: scenario 9-1 (Linux.Hajime)
- Client 4: scenario 36-1 (Okiru) + 39-1 (IRCBot)

---

## 4. Cách sử dụng shared preprocessor

**BẮT BUỘC** mọi client dùng cùng `class_to_idx` và cùng scaler/encoder đã fit ở GĐ1. Nếu không:

- `feature_dim` sẽ lệch giữa các client → **không aggregate được** (FedAvg yêu cầu cùng architecture).
- `num_classes` sẽ lệch → logits vector khác chiều.

Cách dùng:
```python
from src.preprocess import transform
from src.multi_scenario import SharedPreprocessor

# Tải shared preprocessor đã lưu ở GĐ1
import pickle
with open("artifacts/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("artifacts/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Áp dụng cho dữ liệu client mới
df_client = transform(df_raw, encoder=encoder, scaler=scaler)
```

---

## 5. Cảnh báo kế thừa cho FL

### 5.1 Cross-client edges

Khi chia đồ thị thành 2 client, **cạnh nối 2 IP thuộc 2 client khác nhau bị mất**. Ví dụ:
- IP_A thuộc Client 1, IP_B thuộc Client 2
- Flow A→B thuộc Client 1, flow B→A thuộc Client 2
- Khi chia, mỗi client chỉ thấy 1 chiều → cấu trúc đồ thị bị thay đổi.

Đây là **bài toán mở** trong Graph FL. Cần cân nhắc:
- Chia đồ thị theo IP (node partitioning) thay vì chia theo edge.
- Hoặc gom tất cả flow liên quan đến 1 IP vào cùng 1 client (IP-based partitioning).

### 5.2 Lớp private và cực hiếm

| Lớp | Số mẫu toàn dataset | Ghi chú |
|---|---|---|
| Okiru-Attack | **3** | Chỉ ở 36-1 — gần như không học được |
| C&C | 1,650 | Phân bố lệch giữa 36-1 và 39-1 |
| DDoS | 2,879 | Chỉ ở 34-1 (Mirai) — lớp private |

Nếu client giữ scenario 34-1 thì client đó có lớp DDoS; các client khác **không có** → khi aggregate, model chung sẽ mất kiến thức DDoS nếu 34-1 không contribute đủ.

### 5.3 Imbalance handling

GĐ1 đã chứng minh `class_weight` hiệu quả nhất (macro-F1 = 0.8773 pooled). GĐ2 nên **dùng class_weight ở mọi client** để xử lý mất cân bằng cục bộ.

---

## 6. Lệnh chạy lại nếu cần

```bash
# Download data (nếu cần)
bash scripts/download_all.sh --apply

# Chạy lại full experiments
bash scripts/run_full_gpu.sh

# Chỉ chạy pooled (mô hình tập trung — dùng làm baseline)
python -m src.run_experiments \
  --config config.yaml \
  --auto-resume \
  --protocols pooled \
  --epochs 150 \
  --cap-per-class 20000

# Chỉ chạy LOSO (tổng quát hóa)
python -m src.run_experiments \
  --config config.yaml \
  --auto-resume \
  --protocols loso \
  --epochs 150 \
  --cap-per-class 20000
```

---

## 7. Kết quả GĐ1 tóm tắt

```
Model tốt nhất:  E-GraphSAGE
Imbalance mode:  class_weight
Pooled macro-F1: 0.8773  (nền so sánh FL)
LOSO macro-F1:   0.2334  (tổng quát hóa - thấp do private classes)
```

**Câu hỏi quan trọng cho GĐ2:**
1. FL có thể đạt được macro-F1 bao nhiêu trên mỗi client so với pooled baseline?
2. Communication rounds bao nhiêu là đủ để model hội tụ?
3. Non-IID severity: phân bố nhãn lệch giữa các client ảnh hưởng bao nhiêu đến convergence speed?

---

*Tài liệu này được tạo dựa trên kết quả GĐ1: `artifacts/experiments2/results_summary.csv`*
*Chi tiết kỹ thuật: xem `docs/PHASE1_REPORT.md`*
