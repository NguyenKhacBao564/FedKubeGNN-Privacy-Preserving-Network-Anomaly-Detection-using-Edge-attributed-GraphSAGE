# IoT-23 FL+GNN — Giai đoạn 1: Baseline GNN tập trung

Đồ án TTTN nhóm M06 (Nguyễn Khắc Bảo & Nguyễn Chí Hiếu):
*Phát hiện hành vi độc hại trong Kubernetes bằng Federated Learning + GNN
trên bộ dữ liệu IoT-23.*

Repo này phục vụ **Giai đoạn 1** do **Nguyễn Khắc Bảo** phụ trách: tiền xử lý
IoT-23 → dựng đồ thị hành vi → huấn luyện **E-GraphSAGE** tập trung trên 1 máy,
làm **mốc hiệu năng cơ sở (baseline)** để Giai đoạn 2 (Federated Learning)
so sánh.

## Cấu trúc repo

```
.
├── CLAUDE.md               # bộ nhớ dự án (đọc kỹ trước khi code)
├── README.md               # file này
├── requirements.txt
├── .gitignore
├── config.yaml             # cấu hình trung tâm: scenario, đường dẫn, hyperparameter
├── data/                   # conn.log.labeled tải về (gitignore)
├── artifacts/              # file trung gian đã xử lý (gitignore)
├── checkpoints/            # model đã train (gitignore)
├── scripts/
│   └── download_data.sh    # wget conn.log.labeled 6 scenario
└── src/
    ├── data_io.py          # đọc conn.log.labeled → DataFrame, tách cột 21
    ├── preprocess.py       # làm sạch, encode, scale (hàm tái sử dụng)
    ├── imbalance.py        # tính class weights / undersample
    ├── graph_build.py      # DataFrame → PyG Data, lưu .pt
    ├── model.py            # E-GraphSAGE + baselines GCN/GraphSAGE
    ├── train.py            # vòng train device-agnostic, checkpoint
    └── evaluate.py         # macro-F1, per-class F1, confusion matrix
```

## Setup môi trường

### A. Local (MacBook M2 Pro — Apple Silicon, không CUDA)

Mục đích: viết code và test trên mẫu nhỏ. **KHÔNG train thật ở local.**

```bash
# 1. Tạo venv và kích hoạt
python3 -m venv .venv
source .venv/bin/activate

# 2. Cài torch bản CPU-only (PyG trên MPS hay lỗi)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Cài torch-geometric (bản CPU)
pip install torch-geometric

# 4. Cài các dependency còn lại
pip install -r requirements.txt

# 5. (Tùy chọn) tải dataset bằng script
bash scripts/download_data.sh
```

Trong code luôn dùng `device = "cuda" if torch.cuda.is_available() else "cpu"`
và `.to(device)` — không hardcode `.cuda()`. Trên Mac M2 sẽ tự rơi về CPU.

### B. Trên vast.ai (Linux + CUDA — train thật)

```bash
# 1. Chọn image có CUDA (vd: pytorch 2.x + cu121), SSH vào instance.
# 2. Pull code về
git clone <repo-url> iot23-fl-gnn && cd iot23-fl-gnn

# 3. Tạo venv
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 4. Cài torch ĐÚNG phiên bản CUDA của image.
#    Ví dụ image có CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 5. Cài torch-geometric + các optional sparse package
pip install torch-geometric
pip install torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# 6. Cài phần còn lại
pip install -r requirements.txt

# 7. Tải data & chạy pipeline
bash scripts/download_data.sh
python src/preprocess.py --config config.yaml
python src/graph_build.py --config config.yaml
python src/train.py --config config.yaml
python src/evaluate.py --config config.yaml --checkpoint checkpoints/best.pt
```

> Tra cứu wheel chính xác cho `torch_scatter` / `torch_sparse` tại
> <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>.

## Tài liệu tham chiếu

- Quyết định thiết kế, quy tắc tiền xử lý, mô hình: xem `CLAUDE.md`.
- Dataset: <https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/>
- PyTorch Geometric: <https://pytorch-geometric.readthedocs.io/>