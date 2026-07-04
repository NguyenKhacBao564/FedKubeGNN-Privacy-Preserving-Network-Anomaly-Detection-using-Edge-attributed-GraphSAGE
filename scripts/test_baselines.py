"""
test_baselines.py — So sánh 5 model edge-classification trên đồ thị thật 34-1.

Mục đích (Task 1.11 + mở rộng GAT)
    Chứng minh việc dùng đặc trưng cạnh (E-GraphSAGE) tốt hơn các cách
    không tận dụng đầy đủ đặc trưng cạnh. So sánh 5 model trên CÙNG
    đồ thị, CÙNG head, CÙNG seed:

        1. E-GraphSAGE          — ghép edge feature vào message().
        2. GCNBaseline          — GCNConv, KHÔNG dùng edge feature.
        3. GraphSAGEBaseline    — SAGEConv, KHÔNG dùng edge feature.
        4. SAGEEdgeConcatBaseline — "nhồi" edge feature vào node input.
        5. GATBaseline          — GATv2Conv với edge_dim; edge feature
                                   điều chỉnh attention score.

    Với mỗi model:
        * Forward 1 lần, kiểm tra shape [E, 4] và không NaN/Inf.
        * Tổng số param học được — để giải thích trong báo cáo tại sao
          E-GraphSAGE có nhiều param hơn (lin_msg có thêm edge_dim).
        * Chạy thử backward (CE loss) — xác nhận gradient chảy về
          thông suốt.

Chạy:
    /Users/nguyen_bao/Projects/AIproject/FedKube-IDS/.venv/bin/python \\
        scripts/test_baselines.py

Lưu ý:
    Chạy ở CPU (Mac M2 Pro, không CUDA, theo CLAUDE.md mục 2).
"""

import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = "/Users/nguyen_bao/Projects/AIproject/FedKube-IDS"
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

LOG_PATH = (
    "/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/"
    "data/CTU-IoT-Malware-Capture-34-1/conn.log.labeled"
)
CFG_PATH = "/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/config.yaml"

# 5 model cần so sánh (phải đúng tên `build_model` chấp nhận).
MODEL_TYPES = ["egraphsage", "gat", "gcn", "graphsage", "sage_edge_concat"]


def _count_params(model: torch.nn.Module) -> int:
    """Số tham số học được (yêu cầu gradient)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _layer_breakdown(model: torch.nn.Module) -> str:
    """In thông tin từng layer — gọn cho report."""
    lines = []
    for name, p in model.named_parameters():
        lines.append(f"      {name:<40s}  {tuple(p.shape)}  {p.numel():>8,}")
    return "\n".join(lines)


def main():
    t_total = time.perf_counter()

    # ---- 0) Reproducibility ----
    torch.manual_seed(42)
    np.random.seed(42)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("test_baselines")

    if not os.path.isfile(LOG_PATH):
        raise FileNotFoundError(f"Không tìm thấy log: {LOG_PATH}")
    if not os.path.isfile(CFG_PATH):
        raise FileNotFoundError(f"Không tìm thấy config: {CFG_PATH}")

    # ---- 1) Pipeline dữ liệu (CHỈ chạy MỘT LẦN cho cả 5 model) ----
    from sklearn.model_selection import train_test_split
    from src.data_io import load_scenario
    from src.preprocess import clean_flows, fit_preprocessor, transform
    from src.imbalance import compute_class_weights
    from src.graph_build import build_graph
    from src.model import build_model

    logger.info("Pipeline: load -> clean -> fit_preprocessor -> transform ...")
    t0 = time.perf_counter()
    df_clean = clean_flows(load_scenario(LOG_PATH))
    pre = fit_preprocessor(df_clean)
    df_feat = transform(df_clean, pre)
    logger.info(
        "  shape sau transform: %s  (%.2fs)",
        df_feat.shape, time.perf_counter() - t0,
    )

    df_train, _ = train_test_split(
        df_feat,
        test_size=0.2,
        stratify=df_feat["detailed-label"],
        random_state=42,
    )
    df_train = df_train.reset_index(drop=True)

    _, class_to_idx, _ = compute_class_weights(
        df_train["detailed-label"].tolist(), scheme="balanced",
    )
    logger.info("  class_to_idx: %s", class_to_idx)

    data = build_graph(
        df_train,
        class_to_idx=class_to_idx,
        feature_columns=pre.feature_columns,
    )
    E = int(data.edge_index.shape[1])
    K = int(data.num_classes)
    F_dim = int(data.feature_dim)
    N = int(data.num_nodes)
    logger.info(
        "  Graph: N=%d, E=%d, F=%d, K=%d",
        N, E, F_dim, K,
    )

    # ---- 2) Load config ----
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    logger.info("config.yaml['model']: %s", cfg['model'])

    # ---- 3) Chạy từng model: forward + shape/NaN/param + backward ----
    # Kết quả gom lại để in bảng so sánh cuối.
    results = []

    for model_type in MODEL_TYPES:
        print()
        print("=" * 70)
        print(f" MODEL: {model_type}")
        print("=" * 70)

        torch.manual_seed(42)  # CÙNG seed cho mọi model → khởi tạo tương đương.
        model = build_model(model_type, data, cfg)
        model.train()

        # ---- Forward ----
        t0 = time.perf_counter()
        logits = model(data)
        dt_fwd = (time.perf_counter() - t0) * 1000

        # ---- Shape / NaN ----
        has_nan = (
            torch.isnan(logits).any().item()
            or torch.isinf(logits).any().item()
        )
        ok_shape = (tuple(logits.shape) == (E, K))
        print(f"  logits.shape = {tuple(logits.shape)}   "
              f"(kỳ vọng ({E}, {K}))  -> {'OK' if ok_shape else 'SAI'}")
        print(f"  NaN/Inf?      {has_nan}")
        print(f"  Forward time  {dt_fwd:.1f} ms (CPU)")
        assert ok_shape, f"{model_type}: shape sai"
        assert not has_nan, f"{model_type}: có NaN/Inf"

        # ---- Param count ----
        n_params = _count_params(model)
        print(f"  Tổng param học được: {n_params:,}")
        print(f"  Layer breakdown:")
        print(_layer_breakdown(model))

        # ---- Backward sanity ----
        loss = F.cross_entropy(logits, data.edge_label)
        model.zero_grad()
        loss.backward()
        # Kiểm tra ít nhất 1 tham số có grad ≠ 0.
        any_nonzero = any(
            (p.grad is not None) and (p.grad.abs().sum().item() > 0.0)
            for p in model.parameters()
        )
        n_with_grad = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum().item() > 0.0
        )
        print(f"  loss (untrained) = {loss.item():.4f}")
        print(f"  Layers có gradient ≠ 0: "
              f"{n_with_grad}/{sum(1 for _ in model.parameters())}")
        assert any_nonzero, (
            f"{model_type}: không có tham số nào có gradient ≠ 0 — "
            f"đồ thị tính toán bị đứt."
        )

        results.append({
            "model_type": model_type,
            "n_params": n_params,
            "loss": loss.item(),
            "fwd_ms": dt_fwd,
        })

    # ---- 4) Bảng so sánh ----
    print()
    print("=" * 70)
    print(" BẢNG SO SÁNH 5 MODEL (cùng đồ thị 34-1, cùng seed 42, cùng head)")
    print("=" * 70)
    # Trục cột:
    header = f"  {'Model':<22s}  {'#Params':>10s}  {'Loss':>10s}  {'Fwd (ms)':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(
            f"  {r['model_type']:<22s}  "
            f"{r['n_params']:>10,d}  "
            f"{r['loss']:>10.4f}  "
            f"{r['fwd_ms']:>10.1f}"
        )

    # So sánh tương đối — E-GraphSAGE lớn hơn baseline vì lin_msg có thêm edge_dim.
    egs = next(r for r in results if r["model_type"] == "egraphsage")
    gcn = next(r for r in results if r["model_type"] == "gcn")
    sage = next(r for r in results if r["model_type"] == "graphsage")
    sec = next(r for r in results if r["model_type"] == "sage_edge_concat")
    gat = next(r for r in results if r["model_type"] == "gat")

    print()
    print("  Ghi chú cho báo cáo:")
    print(
        f"    • E-GraphSAGE có {egs['n_params']:,} param "
        f"({egs['n_params']/gcn['n_params']:.1f}x GCN, "
        f"{egs['n_params']/sage['n_params']:.1f}x GraphSAGE, "
        f"{egs['n_params']/sec['n_params']:.1f}x SAGE+EdgeConcat, "
        f"{egs['n_params']/gat['n_params']:.1f}x GAT)."
    )
    print(
        "      Phần dư đến từ lin_msg(in_dim + edge_dim) và "
        "lin_upd(in_dim + out_dim) trong mỗi EGraphSAGELayer."
    )
    print(
        f"    • GAT có {gat['n_params']:,} param — nhiều nhất trong các "
        f"baseline vì ``heads`` lần ma trận trọng số + thêm ``lin_edge(edge_dim → "
        f"heads*out_channels)``."
    )
    print(
        "    • Loss chưa có ý nghĩa so sánh (model chưa train) — chỉ dùng để"
    )
    print(
        "      xác nhận gradient chảy về. So sánh thật sẽ ở evaluate.py (macro-F1)."
    )

    # ---- Tổng kết ----
    print()
    print("=" * 70)
    print("ALL CHECKS PASSED — cả 5 model build + forward + backward OK.")
    print(f"Tổng thời gian: {time.perf_counter() - t_total:.2f}s")
    print()
    print("Bước tiếp theo: train từng model với cùng seed/hyperparam, đánh giá")
    print("macro-F1 / per-class F1 / confusion matrix trên test set (evaluate.py).")


if __name__ == "__main__":
    main()
