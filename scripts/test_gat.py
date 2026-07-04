"""
test_gat.py — End-to-end test cho GATBaseline (mới) trên đồ thị thật 34-1.

Mục đích
--------
Hai chứng minh quan trọng nhất:

    1. GATBaseline build + forward + backward KHÔNG lỗi trên đồ thị thật
       (49 node, 23 145 cạnh MP, edge_dim=45, num_classes=4).

    2. **EDGE SENSITIVITY** (ĐIỂM MẤU CHỐT) — chứng minh GAT THỰC SỰ
       dùng ``edge_attr`` trong lan truyền (qua cơ chế attention).

       Cách kiểm: ``model.eval()`` (tắt dropout cho deterministic), chạy
       forward 2 lần — lần 1 với ``edge_attr`` gốc, lần 2 với
       ``edge_attr * 100``. Nếu output GIỐNG HỆT nhau → GAT không dùng
       edge_dim (sai mục đích baseline). Nếu KHÁC → GAT đang tham chiếu
       edge feature trong attention.

       Ngưỡng 1e-3: gấp ~7 bậc so với floating-point noise (~ 1e-7), đủ
       phân biệt "dùng" vs "không dùng".

    3. Backward: gradient CHẢY VỀ layer GAT đầu (chứng minh model học
       được, không bị đứt đồ thị tính toán).

    4. Param count (tham khảo) — để so sánh với 4 model khác.

Chạy:
    /Users/nguyen_bao/Projects/AIproject/FedKube-IDS/.venv/bin/python \\
        scripts/test_gat.py

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

REPO_ROOT = "/Users/nguyen_bao/Projects/AIproject/FedKube-IDS"
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

LOG_PATH = (
    REPO_ROOT + "/data/CTU-IoT-Malware-Capture-34-1/conn.log.labeled"
)
CFG_PATH = REPO_ROOT + "/config.yaml"

# Ngưỡng nhỏ nhất cho max|Δ| giữa 2 forward. Nếu GAT KHÔNG dùng
# edge_attr, output bit-identical → max|Δ| = 0 → fail. Nếu dùng, output
# khác rõ rệt (thường >> 0.1 sau khi nhân ×100).
EDGE_SENSITIVITY_THRESHOLD = 1e-3


def main() -> None:
    t_total = time.perf_counter()

    # ---- 0) Repro ----
    torch.manual_seed(42)
    np.random.seed(42)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("test_gat")

    if not os.path.isfile(LOG_PATH):
        raise FileNotFoundError(f"Thiếu file: {LOG_PATH}")
    if not os.path.isfile(CFG_PATH):
        raise FileNotFoundError(f"Thiếu config: {CFG_PATH}")

    # ---- 1) Pipeline dữ liệu ----
    from sklearn.model_selection import train_test_split
    from src.data_io import load_scenario
    from src.preprocess import clean_flows, fit_preprocessor, transform
    from src.imbalance import compute_class_weights
    from src.graph_build import build_graph
    from src.model import build_model

    logger.info("Pipeline: load → clean → fit_preprocessor → transform ...")
    df_clean = clean_flows(load_scenario(LOG_PATH))
    pre = fit_preprocessor(df_clean)
    df_feat = transform(df_clean, pre)
    logger.info("  df_feat.shape = %s", df_feat.shape)

    df_train, _ = train_test_split(
        df_feat, test_size=0.2,
        stratify=df_feat["detailed-label"], random_state=42,
    )
    df_train = df_train.reset_index(drop=True)

    _, class_to_idx, _ = compute_class_weights(
        df_train["detailed-label"].tolist(), scheme="balanced",
    )
    logger.info("  class_to_idx: %s", class_to_idx)

    data = build_graph(
        df_train, class_to_idx=class_to_idx,
        feature_columns=pre.feature_columns,
    )
    E = int(data.edge_index.shape[1])
    K = int(data.num_classes)
    F_dim = int(data.feature_dim)
    logger.info("  Graph: N=%d, E=%d, F=%d, K=%d",
                int(data.num_nodes), E, F_dim, K)

    # ---- 2) Build GAT qua factory ----
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    model = build_model("gat", data, cfg)
    model = model  # đã ở CPU; không .to(device) ở đây

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print()
    print("=" * 70)
    print(f" GATBaseline — forward + edge-sensitivity + backward")
    print("=" * 70)
    print(f"  hidden_dim          = {model.hidden_dim}")
    print(f"  num_layers          = {model.num_layers}")
    print(f"  dropout             = {model.dropout_p}")
    print(f"  heads               = {model.heads}")
    print(f"  edge_dim            = {model.edge_dim}")
    print(f"  num_classes         = {model.num_classes}")
    print(f"  node_in_dim         = {model.node_in_dim}")
    print(f"  Tổng tham số học được: {n_params:,}")

    # Layer breakdown (để thấy lin_edge của GATv2Conv).
    print("  Layer breakdown:")
    for name, p in model.named_parameters():
        print(f"    {name:<55s}  {tuple(p.shape)}  {p.numel():>8,}")
    print()

    # ============================================================
    # [3] Forward — shape + không NaN
    # ============================================================
    model.eval()
    with torch.no_grad():
        logits = model(data)

    print(f"[3] logits.shape = {tuple(logits.shape)}  "
          f"(kỳ vọng ({E}, {K}))")
    assert logits.shape == (E, K), (
        f"shape sai: {tuple(logits.shape)} != ({E}, {K})"
    )
    has_nan = (
        torch.isnan(logits).any().item()
        or torch.isinf(logits).any().item()
    )
    print(f"    NaN/Inf trong logits? {has_nan}")
    assert not has_nan, "logits chứa NaN/Inf."
    print("    PASS — forward shape OK, không NaN/Inf.\n")

    # ============================================================
    # [4] EDGE SENSITIVITY — ĐIỂM MẤU CHỐT
    # ============================================================
    print("=" * 70)
    print(" [4] EDGE SENSITIVITY (chứng minh GAT dùng edge_attr)")
    print("=" * 70)

    # Lưu edge_attr gốc để khôi phục sau.
    edge_attr_orig = data.edge_attr.clone()

    # --- Forward #1: edge_attr gốc ---
    model.eval()
    with torch.no_grad():
        logits_orig = model(data)

    # --- Perturb: scale edge_attr × 100 ---
    # Dùng in-place * trên tensor leaf để tránh autograd graph cũ (ở
    # eval mode thì không cần autograd, nhưng vẫn cẩn thận).
    data.edge_attr = data.edge_attr * 100.0

    # --- Forward #2: edge_attr đã nhân 100 ---
    with torch.no_grad():
        logits_perturbed = model(data)

    # Khôi phục edge_attr gốc — KHÔNG ảnh hưởng các bước sau.
    data.edge_attr = edge_attr_orig

    delta = (logits_orig - logits_perturbed).abs().max().item()
    print(f"  max|Δ| (original vs scale×100) = {delta:.6f}")
    print(f"  ngưỡng đạt                    = {EDGE_SENSITIVITY_THRESHOLD}")
    if delta < EDGE_SENSITIVITY_THRESHOLD:
        print(f"  ✗ FAIL — GAT KHÔNG nhạy với edge_attr.")
        print("    Có thể 'edge_dim' không được GATv2Conv sử dụng,")
        print("    hoặc forward không truyền edge_attr vào GATv2Conv.")
        raise AssertionError(
            f"GAT edge-sensitivity fail: max|Δ|={delta:.6f} "
            f"< threshold {EDGE_SENSITIVITY_THRESHOLD}."
        )
    else:
        print(f"  ✓ PASS — GAT CÓ dùng edge_attr trong attention.\n")

    # ============================================================
    # [5] Backward — gradient chảy về layer đầu
    # ============================================================
    print("=" * 70)
    print(" [5] Backward (gradient chảy về layer GAT đầu)")
    print("=" * 70)

    model.train()
    edge_label = data.edge_label
    assert edge_label.dtype == torch.long

    logits_train = model(data)
    loss = F.cross_entropy(logits_train, edge_label)
    print(f"  loss (untrained) = {loss.item():.4f}")

    model.zero_grad()
    loss.backward()

    # Layer GAT đầu — kiểm tra gradient ≠ 0 ở lin_src (Linear W cho src).
    first_gat = model.layers[0]
    print(f"  Layer GAT đầu: {type(first_gat).__name__}")

    grad_info = []
    has_nonzero_grad = False
    for name, p in first_gat.named_parameters():
        if p.grad is None:
            grad_info.append((name, "None", 0.0))
            continue
        norm = float(p.grad.norm().item())
        is_finite = bool(torch.isfinite(p.grad).all().item())
        sum_abs = float(p.grad.abs().sum().item())
        if sum_abs > 0.0:
            has_nonzero_grad = True
        grad_info.append((name, "finite" if is_finite else "NaN/Inf",
                          norm))

    for name, status, norm in grad_info:
        print(f"    {name:<45s}  grad_status={status:<8s}  "
              f"l2_norm={norm:.6f}")

    assert has_nonzero_grad, (
        "Không có gradient ≠ 0 ở layer GAT đầu — đồ thị tính toán bị đứt."
    )
    print("  ✓ PASS — gradient thông suốt từ head về lin_src layer đầu.\n")

    # ============================================================
    # TỔNG KẾT
    # ============================================================
    print("=" * 70)
    print(" ALL GAT CHECKS PASSED")
    print("=" * 70)
    print(f"  hidden_dim          = {model.hidden_dim}")
    print(f"  num_layers          = {model.num_layers}")
    print(f"  heads               = {model.heads}")
    print(f"  Tổng tham số        = {n_params:,}")
    print(f"  edge-sensitivity Δ  = {delta:.6f}  "
          f"(>{EDGE_SENSITIVITY_THRESHOLD})")
    print(f"  Tổng thời gian      = {time.perf_counter() - t_total:.2f}s")
    print()
    print("Bước tiếp theo:")
    print("  • test_baselines.py đã được cập nhật để in bảng so sánh 5 model.")
    print("  • evaluate.py đã bao gồm 'gat' trong default list và argparse.")
    print("  • Chạy lại test_baselines.py để đối chiếu param của 5 model.")


if __name__ == "__main__":
    main()
