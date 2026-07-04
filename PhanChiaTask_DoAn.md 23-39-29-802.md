# PHÂN CHIA TASK & ĐÁNH GIÁ ĐỘ PHỨC TẠP

**Đề tài:** Phát hiện hành vi độc hại trong Kubernetes bằng FL + GNN trên bộ dữ liệu IoT-23
**Nhóm:** M06 — Nguyễn Khắc Bảo & Nguyễn Chí Hiếu

Thang đánh giá độ phức tạp: ★☆☆☆☆ Rất dễ · ★★☆☆☆ Dễ · ★★★☆☆ Trung bình · ★★★★☆ Khó · ★★★★★ Rất khó / mang tính nghiên cứu

---

## GIAI ĐOẠN 1 — Baseline GNN tập trung (Sản phẩm 1)

| # | Task | Phụ trách | Độ phức tạp | Vì sao |
|---|---|---|---|---|
| 1.1 | Khảo sát 23 scenario của IoT-23, chọn 5–8 scenario đại diện | Bảo | ★★☆☆☆ | Chỉ cần đọc README từng scenario và cân nhắc, không cần code nhiều |
| 1.2 | Setup môi trường (Kaggle Notebook, cài PyTorch + PyTorch Geometric) | Bảo | ★☆☆☆☆ | Việc cấu hình quen thuộc, tài liệu sẵn nhiều |
| 1.3 | Khám phá dữ liệu (EDA): phân bố nhãn, kiểu dữ liệu, giá trị thiếu | Bảo | ★★☆☆☆ | Thao tác pandas/Polars cơ bản |
| 1.4 | Tách cột gộp `tunnel_parents label detailed-label` thành 3 cột | Bảo | ★★☆☆☆ | Cần viết hàm parse cẩn thận nhưng logic đơn giản |
| 1.5 | Làm sạch dữ liệu: xử lý ký tự `-` (missing), ép kiểu numeric | Bảo | ★★☆☆☆ | Thao tác tiền xử lý chuẩn |
| 1.6 | Encode categorical (proto, service, conn_state, history) + chuẩn hóa numeric | Bảo | ★★★☆☆ | Cần chọn đúng phương pháp encode cho từng loại cột, ảnh hưởng trực tiếp đến chất lượng model |
| 1.7 | Xử lý mất cân bằng lớp cực đoan (class weighting / downsampling) | Bảo | ★★★★☆ | IoT-23 có lớp chênh lệch hàng triệu lần; xử lý sai sẽ khiến model chỉ đoán lớp đa số |
| 1.8 | Dựng đồ thị từ flow data (node = IP, cạnh = flow, edge_index tensor) | Bảo | ★★★☆☆ | Cần thiết kế đúng cấu trúc dữ liệu PyG (Data object), dễ sai ở bước map ID → index |
| 1.9 | Cài đặt kiến trúc **E-GraphSAGE** (custom MessagePassing layer có nhúng đặc trưng cạnh) | Bảo (có hỗ trợ từ Hiếu) | ★★★★★ | PyTorch Geometric **không có sẵn** E-GraphSAGE — phải tự viết lớp `MessagePassing`, dễ sai ở bước ghép (concat) đặc trưng node + cạnh trong hàm `message()` |
| 1.10 | Huấn luyện baseline + tinh chỉnh hyperparameter (learning rate, số lớp, dropout) | Bảo | ★★★☆☆ | Cần nhiều vòng thử nghiệm nhưng không có rủi ro kỹ thuật lớn |
| 1.11 | Cài đặt model đối chứng (GCN, GraphSAGE gốc) để so sánh | Bảo | ★★☆☆☆ | PyG có sẵn layer dựng nhanh (`GCNConv`, `SAGEConv`) |
| 1.12 | Đánh giá: Accuracy, Precision/Recall/F1 theo từng lớp, confusion matrix | Bảo | ★★☆☆☆ | scikit-learn hỗ trợ sẵn hầu hết metric |

**Điểm nghẽn của giai đoạn 1:** Task 1.9 (cài E-GraphSAGE) và 1.7 (xử lý mất cân bằng) là hai task quyết định chất lượng toàn bộ baseline — nên dành nhiều thời gian nhất ở đây, đừng vội chuyển sang Giai đoạn 2 khi baseline chưa ổn định.

---

## GIAI ĐOẠN 2 — Federated Learning (Sản phẩm 2)

| # | Task | Phụ trách | Độ phức tạp | Vì sao |
|---|---|---|---|---|
| 2.1 | Tìm hiểu framework Flower (kiến trúc client–server, vòng đời 1 round) | Hiếu | ★★☆☆☆ | Tài liệu chính thức rõ ràng, nhiều ví dụ mẫu |
| 2.2 | Thiết kế cách chia dữ liệu cho client (mỗi client = 1–2 scenario → non-IID tự nhiên) | Hiếu | ★★★☆☆ | Cần cân nhắc để non-IID có ý nghĩa thực tế chứ không chỉ chia ngẫu nhiên |
| 2.3 | Viết Flower Client (bọc model GNN đã có ở Giai đoạn 1: `get_parameters`, `fit`, `evaluate`) | Hiếu | ★★★★☆ | Phải map đúng tham số PyTorch ↔ định dạng Flower, dễ lỗi lúc serialize/deserialize state_dict |
| 2.4 | Cấu hình chiến lược tổng hợp phía server (FedAvg) | Hiếu | ★★★☆☆ | Flower có sẵn `FedAvg`, chủ yếu là cấu hình đúng tham số |
| 2.5 | Chạy mô phỏng nhiều client trên 1 máy (Flower Simulation) | Hiếu | ★★★☆☆ | Cần quản lý tài nguyên (CPU/RAM) khi giả lập nhiều client cùng lúc |
| 2.6 | Đo & so sánh độ chính xác FL vs baseline tập trung | Hiếu | ★★☆☆☆ | Chỉ là ghi nhận và vẽ biểu đồ so sánh |
| 2.7 | Đo chi phí truyền thông (dung lượng tham số mỗi round) | Hiếu | ★★★☆☆ | Cần tự viết code đo kích thước tensor gửi đi, không có sẵn công cụ đo trực tiếp trong Flower |
| 2.8 | Cài đặt FedProx để xử lý non-IID tốt hơn | Hiếu | ★★★★☆ | Phải sửa hàm loss để thêm proximal term, cần hiểu rõ toán học đằng sau |
| 2.9 | **Xử lý cross-client edges** (khi đồ thị bị chia, các cạnh nối giữa 2 client bị mất) | Cả 2 cùng làm | ★★★★★ | Đây là **bài toán nghiên cứu mở**, chưa có giải pháp chuẩn — độ khó cao nhất toàn đồ án; nên giới hạn phạm vi (ví dụ chấp nhận mất cạnh liên-client ở bản đầu, ghi nhận như hướng phát triển) |
| 2.10 | Phân tích kết quả IID vs non-IID | Hiếu | ★★★☆☆ | Cần chạy nhiều cấu hình và tổng hợp so sánh có hệ thống |

**Điểm nghẽn của giai đoạn 2:** Task 2.9 là rủi ro lớn nhất của **toàn bộ đồ án**. Khuyến nghị: đặt kỳ vọng thực tế — xử lý đơn giản (mỗi client tự xây subgraph độc lập, chấp nhận mất cạnh liên-client) để đảm bảo tiến độ, chỉ đào sâu nếu còn dư thời gian. Task 2.3 cũng dễ phát sinh lỗi kỹ thuật vụn vặt (shape mismatch, sai key trong state_dict) — nên test kỹ với model nhỏ trước khi chạy full.

---

## GIAI ĐOẠN 3 — Triển khai trên 2 cụm Kubernetes (Sản phẩm 3)

| # | Task | Phụ trách | Độ phức tạp | Vì sao |
|---|---|---|---|---|
| 3.1 | Đóng gói model đã huấn luyện thành Docker image (inference service) | Hiếu | ★★★☆☆ | Cần viết REST/gRPC service wrapper quanh model PyTorch |
| 3.2 | Dựng 2 cụm K8s cục bộ để test (kind/minikube) trước khi lên cloud | Hiếu | ★★★☆☆ | Cần hiểu cấu hình multi-cluster cơ bản, nhưng công cụ hỗ trợ tốt |
| 3.3 | Thuê & cấu hình 2 cụm K8s thật trên cloud (DigitalOcean/GKE) | Hiếu | ★★★☆☆ | Chủ yếu là thao tác theo tài liệu nhà cung cấp, rủi ro chính là quản lý chi phí |
| 3.4 | Expose Flower server qua LoadBalancer/Ingress + cấu hình TLS | Hiếu | ★★★★☆ | Networking đa cụm, cấp chứng chỉ TLS, dễ vướng lỗi cấu hình firewall/DNS |
| 3.5 | Deploy client pods ở cụm Edge, kết nối tới server qua endpoint công khai | Hiếu | ★★★☆☆ | Sau khi bước 3.4 xong thì bước này khá thẳng | 
| 3.6 | Tích hợp Falco để thu sự kiện syscalls/audit logs theo thời gian thực | Hiếu | ★★★★☆ | Cần học cú pháp rule của Falco, cấu hình DaemonSet đúng quyền truy cập kernel |
| 3.7 | Xây pipeline: sự kiện Falco → chuyển đổi định dạng → đưa vào model suy luận | Cả 2 cùng làm | ★★★★★ | Đây là phần **tích hợp khó nhất** của giai đoạn 3 — output của Falco không có sẵn định dạng khớp với input mà GNN cần, phải tự thiết kế tầng chuyển đổi |
| 3.8 | Mô phỏng tấn công (kube-hunter, Atomic Red Team, script tự viết) | Hiếu | ★★★☆☆ | Công cụ có sẵn, nhưng cần tùy biến kịch bản cho đúng bối cảnh |
| 3.9 | Đo lường: tỷ lệ phát hiện đúng, báo động giả, độ trễ phát hiện | Cả 2 cùng làm | ★★★☆☆ | Cần thiết kế thực nghiệm và log timestamp chính xác |
| 3.10 | (Tùy chọn) Dashboard giám sát bằng Prometheus + Grafana | Hiếu | ★★★☆☆ | Không bắt buộc, chỉ làm nếu còn thời gian dư |
| 3.11 | Quản lý chi phí cloud (lịch bật/tắt cụm) | Hiếu | ★☆☆☆☆ | Thao tác vận hành đơn giản nhưng cần kỷ luật để tránh phát sinh phí |

**Điểm nghẽn của giai đoạn 3:** Task 3.7 là điểm rủi ro lớn nhất — nhiều đồ án tương tự dừng ở mức "train xong model" mà chưa demo được suy luận thời gian thực. Nên làm mẫu nhỏ (1 loại sự kiện Falco → 1 lần suy luận) sớm để xác nhận tính khả thi, tránh dồn hết vào cuối kỳ. Task 3.4 dễ tốn thời gian debug hơn dự kiến do đặc thù networking đa cụm — nên bắt đầu thử ở cụm cục bộ (3.2) trước khi lên cloud thật.

---

## Công việc chạy song song (toàn bộ kỳ)

| Task | Phụ trách | Độ phức tạp | Ghi chú |
|---|---|---|---|
| Viết chương cơ sở lý thuyết | Cả 2 | ★★☆☆☆ | Không khó về kỹ thuật, nhưng cần làm đều đặn, tránh dồn cuối kỳ |
| Viết báo cáo tổng hợp | Cả 2 | ★★★☆☆ | Nên viết theo từng giai đoạn ngay sau khi hoàn thành, không đợi gộp cuối |
| Chuẩn bị slide + kịch bản demo | Cả 2 | ★★☆☆☆ | Làm ở 1–2 tuần cuối |

---

## Tổng hợp: 4 task rủi ro cao nhất toàn đồ án

Đây là những task nên được ưu tiên thời gian, thử nghiệm sớm, và có phương án dự phòng nếu không kịp:

1. **Task 1.9** — Cài đặt E-GraphSAGE (★★★★★): không có sẵn trong thư viện, phải tự viết. *Dự phòng: nếu quá khó, dùng SAGEConv của PyG kèm kỹ thuật nối đặc trưng cạnh vào đặc trưng node trước khi đưa vào layer, chấp nhận đây là bản "gần đúng" của E-GraphSAGE.*
2. **Task 2.9** — Cross-client edges trong Federated GNN (★★★★★): bài toán nghiên cứu mở. *Dự phòng: mỗi client tự xây subgraph độc lập, không cố ghép cạnh liên-client; nêu rõ đây là giới hạn và hướng phát triển trong báo cáo.*
3. **Task 3.7** — Pipeline Falco → model suy luận real-time (★★★★★): không có công cụ dựng sẵn, phải tự thiết kế tầng chuyển đổi dữ liệu. *Dự phòng: làm demo ở quy mô nhỏ (1–2 loại sự kiện) thay vì cố bao phủ mọi loại hành vi.*
4. **Task 3.4** — Networking đa cụm + TLS (★★★★☆): nhiều điểm dễ lỗi về cấu hình. *Dự phòng: thử nghiệm kỹ trên cụm cục bộ (kind/minikube) trước, chỉ lên cloud khi pipeline đã chạy ổn định.*

**Nguyên tắc chung khi phân bổ thời gian:** dồn nhiều thời gian nhất cho các task ★★★★☆ trở lên, và luôn có "phương án B" đơn giản hơn để không bị kẹt tiến độ nếu phiên bản đầy đủ không kịp hoàn thành.
