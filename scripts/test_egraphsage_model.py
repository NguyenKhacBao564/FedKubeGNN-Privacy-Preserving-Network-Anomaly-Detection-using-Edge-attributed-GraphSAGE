"""
test_egraphsage_model.py — End-to-end test cho EGraphSAGE trên đồ thị thật.

Mục đích (Task 1.10 + 1.12 smoke):
    1. Chạy nguyên pipeline:  load conn.log.labeled (CTU-IoT-34-1)
                              -> clean_flows -> fit_preprocessor -> transform
                              -> build_graph -> Data.
    2. Build EGraphSAGE qua build_model('egraphsage', data, cfg), đọc
       edge_dim / num_classes / node_in_dim **động từ data** (không hardcode).
    3. forward() 1 lần: kiểm tra shape, không NaN, argmax hợp lệ trong
       [0, num_classes-1].
    4. CE-loss + backward(): chứng minh gradient chảy về tới layer đầu.

Chạy:
    /Users/nguyen_bao/Projects/AIproject/FedKube-IDS/.venv/bin/python \\
        scripts/test_egraphsage_model.py

Lưu ý:
    Chạy ở CPU (Mac M2 Pro — không CUDA, theo CLAUDE.md mục 2).
"""

import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ----- đường dẫn -----
REPO_ROOT = "/Users/nguyen_bao/Projects/AIproject/FedKube-IDS"
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

LOG_PATH = "/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/data/CTU-IoT-Malware-Capture-34-1/conn.log.labeled"
CFG_PATH = "/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/config.yaml"


def main():
    t_total = time.perf_counter()

    # ---- 0) Reproducibility ----
    torch.manual_seed(42)
    np.random.seed(42)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("test_egraphsage_model")

    if not os.path.isfile(LOG_PATH):
        raise FileNotFoundError(f"Không tìm thấy file log: {LOG_PATH}")
    if not os.path.isfile(CFG_PATH):
        raise FileNotFoundError(f"Không tìm thấy config: {CFG_PATH}")

    # ---- 1) Pipeline dữ liệu (load -> clean -> transform -> build_graph) ----
    from sklearn.model_selection import train_test_split
    from src.data_io import load_scenario
    from src.preprocess import clean_flows, fit_preprocessor, transform
    from src.imbalance import compute_class_weights
    from src.graph_build import build_graph, graph_stats

    logger.info("Pipeline: load -> clean -> fit_preprocessor -> transform ...")
    t0 = time.perf_counter()
    df_clean = clean_flows(load_scenario(LOG_PATH))
    pre = fit_preprocessor(df_clean)
    df_feat = transform(df_clean, pre)
    logger.info(
        "  shape sau transform: %s  (%.1fs)", df_feat.shape, time.perf_counter() - t0,
    )

    # 80/20 stratified — đồ thị chỉ dựng trên TRAIN (đúng phép train/test).
    df_train, _ = train_test_split(
        df_feat,
        test_size=0.2,
        stratify=df_feat["detailed-label"],
        random_state=42,
    )
    df_train = df_train.reset_index(drop=True)
    logger.info("  TRAIN rows: %s", df_train.shape)

    _, class_to_idx, _ = compute_class_weights(
        df_train["detailed-label"].tolist(), scheme="balanced",
    )
    logger.info("  class_to_idx: %s", class_to_idx)

    logger.info("Build đồ thị PyG trên TRAIN ...")
    t0 = time.perf_counter()
    data = build_graph(
        df_train,
        class_to_idx=class_to_idx,
        feature_columns=pre.feature_columns,
    )
    logger.info("  build_graph xong (%.2fs)", time.perf_counter() - t0)

    stats = graph_stats(data)
    print()  # tách dòng cho dễ đọc

    # ===========================================================
    # [Sanity] Đọc chiều ĐỘNG từ data — không hardcode
    # ===========================================================
    edge_dim = int(data.feature_dim)
    num_classes = int(data.num_classes)
    node_in_dim = int(data.x.shape[1])
    num_edges = int(data.edge_index.shape[1])
    num_nodes = int(data.num_nodes)

    print(f"[SANITY] edge_dim     = {edge_dim}  (đọc từ data.feature_dim)")
    print(f"[SANITY] num_classes  = {num_classes}  (đọc từ data.num_classes)")
    print(f"[SANITY] node_in_dim  = {node_in_dim}  (đọc từ data.x.shape[1])")
    print(f"[SANITY] num_edges    = {num_edges:,}")
    print(f"[SANITY] num_nodes    = {num_nodes:,}")
    assert edge_dim > 0, "edge_dim phải > 0"
    assert num_classes >= 2, "num_classes phải >= 2 (Benign + ít nhất 1 lớp độc hại)"
    assert node_in_dim >= 1
    print()

    # ---- 2) Build model qua factory ----
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    logger.info("config.yaml['model']: %s", cfg['model'])

    from src.model import build_model

    model = build_model("egraphsage", data, cfg)
    model.train()  # dropout active (đúng regime train)
    logger.info(
        "Model khởi tạo: layers=%d, hidden=%d, dropout=%s",
        model.num_layers, model.hidden_dim, model.dropout_p,
    )

    # ---- 3) Param count ----
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PARAM] Tổng số tham số học được: {n_params:,}")
    print(f"[PARAM]  - theo layer:")
    for name, p in model.named_parameters():
        print(f"           {name:<40s}  {tuple(p.shape)}  = {p.numel():>10,}  tham số")
    print()

    # ============================================================
    # [4] Forward pass
    # ============================================================
    t0 = time.perf_counter()
    logits = model(data)
    dt_fwd = time.perf_counter() - t0

    print(f"[4] logits.shape = {tuple(logits.shape)}  "
          f"(kỳ vọng: ({num_edges}, {num_classes}))")
    assert logits.shape == (num_edges, num_classes), (
        f"shape sai: {tuple(logits.shape)} != ({num_edges}, {num_classes})"
    )

    has_nan = torch.isnan(logits).any().item() or torch.isinf(logits).any().item()
    print(f"    NaN/Inf trong logits? {has_nan}")
    assert not has_nan, "logits chứa NaN/Inf."
    print(f"    Forward pass trên {num_edges:,} cạnh xong trong {dt_fwd*1000:.1f} ms")
    print("    PASS")
    print()

    # ============================================================
    # [5] Range argmax — dự đoán còn ngẫu nhiên, chỉ cần nằm trong [0, K)
    # ============================================================
    preds = logits.argmax(dim=-1)            # [E]
    unique_preds = sorted(torch.unique(preds).tolist())
    print(f"[5] Unique argmax classes (model chưa train): {unique_preds}")
    assert all(0 <= p < num_classes for p in unique_preds), (
        f"argmax ra nhãn ngoài [0, {num_classes}): {unique_preds}"
    )
    # Không bắt buộc đủ cả num_classes lớp — model chưa train, phân phối
    # có thể nghiêng. Chỉ cần range hợp lệ.
    assert len(unique_preds) >= 1
    print(f"    Có {len(unique_preds)}/{num_classes} lớp xuất hiện trong dự đoán.")
    print("    PASS (range hợp lệ)")
    print()

    # ============================================================
    # [6] Backward — gradient phải chảy về layer đầu
    # ============================================================
    # Đảm bảo model ở chế độ train (dropout active) trước khi loss/backward.
    model.train()

    edge_label = data.edge_label
    assert edge_label.dtype == torch.long, (
        f"edge_label phải long; got {edge_label.dtype}"
    )

    loss = F.cross_entropy(logits, edge_label)
    print(f"[6] loss (untrained model) = {loss.item():.4f}")

    # Zero grad sạch trước (đề phòng nếu có state cũ).
    model.zero_grad()
    loss.backward()

    # Kiểm tra gradient ở layer đầu: lin_msg.weight
    first_layer = model.layers[0]
    assert hasattr(first_layer, "lin_msg"), (
        "EGraphSAGELayer lớp đầu phải có self.lin_msg"
    )
    g_msg = first_layer.lin_msg.weight.grad
    print(f"    first_layer.lin_msg.weight.grad.shape = {tuple(g_msg.shape)}")
    print(f"    first_layer.lin_msg.weight.grad (l2-norm) = "
          f"{g_msg.norm().item():.6f}")
    assert g_msg is not None, "lin_msg.weight.grad = None — gradient không chảy về."
    assert torch.isfinite(g_msg).all(), "gradient có NaN/Inf."
    assert g_msg.abs().sum().item() > 0.0, (
        "lin_msg.weight.grad toàn 0 — đồ thị tính toán bị đứt ở đâu đó "
        "(head không nhận được gradient từ layer đầu)."
    )

    # Bonus: kiểm tra luôn head + lin_upd layer đầu.
    head_lin0 = model.head[0]            # nn.Linear đầu của head MLP
    head_lin0_g = head_lin0.weight.grad
    print(f"    head.0.weight.grad        (l2-norm) = "
          f"{head_lin0_g.norm().item():.6f}")
    upd_lin_g = first_layer.lin_upd.weight.grad
    print(f"    first_layer.lin_upd.weight.grad (l2-norm) = "
          f"{upd_lin_g.norm().item():.6f}")
    assert head_lin0_g is not None and head_lin0_g.abs().sum().item() > 0.0
    assert upd_lin_g is not None and upd_lin_g.abs().sum().item() > 0.0

    print("    PASS — gradient thông suốt từ head → lin_upd → lin_msg ở layer đầu.")
    print()

    # ============================================================
    # Tổng kết
    # ============================================================
    print("=" * 70)
    print("ALL CHECKS PASSED.")
    print("=" * 70)
    print(f"Tổng thời gian: {time.perf_counter() - t_total:.2f}s")
    print()
    print("EGraphSAGE (Task 1.10) đã sẵn sàng nối vào train.py.")
    print(
        "Bước tiếp theo: viết vòng train device-agnostic + đánh giá "
        "macro-F1/per-class F1/confusion matrix (Task 1.11–1.12)."
    )


if __name__ == "__main__":
    main()
