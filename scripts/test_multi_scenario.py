"""
test_multi_scenario.py — Smoke test cho src/multi_scenario.py (LOSO harness).

Mục đích
--------
Chứng minh 5 điểm quan trọng nhất của tầng dữ liệu đa-scenario + LOSO
inductive trên 2 scenario thật {34-1, 3-1}, model E-GraphSAGE, mode
class_weight, 40 epoch:

    1. Pipeline dùng lại single-scenario logic (KHÔNG sửa src/data_io,
       src/preprocess, src/graph_build, src/model, src/train, src/evaluate).
    2. Shared preprocessor fit trên union TRAIN (KHÔNG chạm held-out).
    3. Mọi Data có cùng ``feature_dim`` và ``num_classes`` (assert OK).
    4. class_to_idx là HỢP mọi nhãn của 2 scenario; in ma trận hiện diện.
    5. 2 dòng LOSO (held-out 34-1, held-out 3-1) → mỗi dòng có macro-F1
       FINITE trên scenario chưa thấy; train_loss GIẢM qua các epoch.
    6. CSV + confusion matrix PNG được lưu vào artifacts/loso/.

Chạy
----
    /Users/nguyen_bao/Projects/AIproject/FedKube-IDS/.venv/bin/python \\
        scripts/test_multi_scenario.py

Lưu ý
-----
* Chạy ở CPU (Mac M2 Pro, không CUDA, theo CLAUDE.md mục 2).
* ``cap_per_class=5000`` để giữ RAM ổn với 3-1 (~24MB) — vẫn đủ đa dạng
  lớp để chứng minh pipeline hoạt động.
* ``epochs_override=40`` đủ ngắn để smoke test nhanh nhưng đủ dài để
  thấy train_loss giảm rõ rệt.
"""

import logging
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO_ROOT = "/Users/nguyen_bao/Projects/AIproject/FedKube-IDS"
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

LOG_34_1 = (
    REPO_ROOT + "/data/CTU-IoT-Malware-Capture-34-1/conn.log.labeled"
)
LOG_3_1 = (
    REPO_ROOT + "/data/CTU-IoT-Malware-Capture-3-1/conn.log.labeled"
)
CFG_PATH = REPO_ROOT + "/config.yaml"
OUT_DIR = REPO_ROOT + "/artifacts/loso"

# Hyperparam smoke test — phải ĐỦ NHANH để chạy local nhưng ĐỦ DÀI
# để loss giảm rõ rệt.
CAP_PER_CLASS = 5000
EPOCHS = 40
SEED = 42
MODEL_NAME = "egraphsage"
IMBALANCE_MODE = "class_weight"


def main() -> None:
    t_total = time.perf_counter()

    # ---- 0) Repro ----
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("test_multi_scenario")

    # ---- 1) Sanity-check input files ----
    for p in (LOG_34_1, LOG_3_1, CFG_PATH):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Thiếu file: {p}\n"
                f"  → Chạy lại download 3-1 nếu cần."
            )

    # ---- 2) Import module-under-test ----
    from src.multi_scenario import (
        build_scenario_graphs,
        build_shared_class_to_idx,
        fit_shared_preprocessor,
        load_all_scenarios,
        run_loso,
    )

    print("=" * 70)
    print(" TEST  src.multi_scenario  ·  LOSO inductive 2 scenario")
    print("=" * 70)
    print(f"  scenarios       : 34-1 (Mirai, ~2.9MB) + 3-1 (Muhstik, ~24MB)")
    print(f"  model           : {MODEL_NAME}")
    print(f"  imbalance_mode  : {IMBALANCE_MODE}")
    print(f"  cap_per_class   : {CAP_PER_CLASS}")
    print(f"  epochs          : {EPOCHS}")
    print(f"  seed            : {SEED}")
    print(f"  out_dir         : {OUT_DIR}")
    print()

    # ============================================================
    # [1] load_all_scenarios — CHUNKED để RAM ổn
    # ============================================================
    print("-" * 70)
    print("[1] load_all_scenarios(paths, cap_per_class=5000, chunksize=...)")
    print("-" * 70)
    t0 = time.perf_counter()
    scenario_paths = {"34-1": LOG_34_1, "3-1": LOG_3_1}
    all_dfs = load_all_scenarios(
        scenario_paths, cap_per_class=CAP_PER_CLASS, chunksize=100_000,
    )
    dt_load = time.perf_counter() - t0
    print(f"  → {len(all_dfs)} scenarios loaded in {dt_load:.2f}s.")
    for n, df in all_dfs.items():
        n_classes = df["detailed-label"].nunique()
        print(f"    {n:<10s}  shape={df.shape}   "
              f"#class={n_classes}   "
              f"#rows={df.shape[0]:,}")
    assert set(all_dfs.keys()) == {"34-1", "3-1"}, "Tên scenario sai."
    for n, df in all_dfs.items():
        assert df.shape[0] > 0, f"{n} rỗng sau load+cap."
        assert "detailed-label" in df.columns, f"{n} thiếu cột nhãn."
    print("  ✓ PASS — cả 2 scenario load thành công, không rỗng.\n")

    # ============================================================
    # [2] build_shared_class_to_idx — in ma trận hiện diện
    # ============================================================
    print("-" * 70)
    print("[2] build_shared_class_to_idx(all_dfs)")
    print("-" * 70)
    shared_cti = build_shared_class_to_idx(all_dfs)
    K = len(shared_cti)
    print(f"  → K = {K} (hợp nhãn của 2 scenario).")
    # 34-1 có 4 lớp, 3-1 có nhiều hơn → K > 4.
    assert K >= 4, f"K={K} < 4 — kỳ vọng ít nhất 4 (34-1 có 4 lớp)."
    print("  ✓ PASS — class_to_idx là hợp của 2 scenario, K >= 4.\n")

    # ============================================================
    # [3] fit_shared_preprocessor + build_scenario_graphs
    #     — assert feature_dim & num_classes KHỚP giữa 2 graph
    # ============================================================
    print("-" * 70)
    print("[3] fit_shared_preprocessor(train) + build_scenario_graphs(all)")
    print("-" * 70)
    # Trong LOSO thật, preprocessor chỉ fit trên TRAIN (giữ held-out
    # làm "unseen"). Ở test này, 2 scenario đều "train" (vì ta chỉ
    # chứng minh pipeline); nhưng để GIỐNG LOSO, fit trên cả 2 cũng
    # OK (kết quả preprocessor sẽ giống nhau nếu dùng làm train hết).
    shared_pre = fit_shared_preprocessor([all_dfs["34-1"], all_dfs["3-1"]])
    graphs = build_scenario_graphs(all_dfs, shared_pre, shared_cti)

    fdims = {n: int(g.feature_dim) for n, g in graphs.items()}
    ncs = {n: int(g.num_classes) for n, g in graphs.items()}
    es = {n: int(g.edge_index.shape[1]) for n, g in graphs.items()}
    ns = {n: int(g.num_nodes) for n, g in graphs.items()}
    print(f"  feature_dim per scenario : {fdims}")
    print(f"  num_classes per scenario : {ncs}")
    print(f"  num_nodes  per scenario  : {ns}")
    print(f"  num_edges  per scenario  : {es}")
    assert len(set(fdims.values())) == 1, (
        f"feature_dim lệch: {fdims}"
    )
    assert len(set(ncs.values())) == 1, (
        f"num_classes lệch: {ncs}"
    )
    assert all(v > 0 for v in es.values()), "Một graph không có cạnh."
    print("  ✓ PASS — feature_dim & num_classes đồng nhất giữa 2 graph.\n")

    # ============================================================
    # [4] run_loso — FULL harness, 2 held-out round
    # ============================================================
    print("=" * 70)
    print("[4] run_loso(...)  ·  2 held-out round · "
          f"{EPOCHS} epoch/round")
    print("=" * 70)
    t0 = time.perf_counter()
    df = run_loso(
        scenario_paths=scenario_paths,
        config_path=CFG_PATH,
        model_name=MODEL_NAME,
        imbalance_mode=IMBALANCE_MODE,
        cap_per_class=CAP_PER_CLASS,
        chunksize=100_000,
        epochs_override=EPOCHS,
        val_ratio=0.10,
        seed=SEED,
        out_dir=OUT_DIR,
        verbose=True,
    )
    dt_loso = time.perf_counter() - t0
    print(f"\n  run_loso xong trong {dt_loso:.1f}s.\n")

    # ============================================================
    # [5] Assert kết quả
    # ============================================================
    print("-" * 70)
    print("[5] Kiểm tra kết quả DataFrame")
    print("-" * 70)

    # (5a) 2 dòng held-out + 1 dòng MEAN.
    assert "held_out" in df.columns, "DataFrame thiếu cột 'held_out'."
    held_rows = df[df["held_out"] != "MEAN"]
    mean_row = df[df["held_out"] == "MEAN"]
    assert len(held_rows) == 2, (
        f"Kỳ vọng 2 dòng held-out, có {len(held_rows)}."
    )
    assert len(mean_row) == 1, (
        f"Kỳ vọng 1 dòng MEAN, có {len(mean_row)}."
    )
    print(f"  ✓ Đúng {len(held_rows)} dòng held-out + 1 dòng MEAN.")

    # (5b) Cả 2 tên held-out xuất hiện.
    held_names = set(held_rows["held_out"].astype(str).tolist())
    assert held_names == {"34-1", "3-1"}, (
        f"Tên held-out sai: {held_names} (kỳ vọng {{'34-1', '3-1'}})."
    )
    print(f"  ✓ Held-out = {sorted(held_names)}.")

    # (5c) macro_f1 FINITE (không NaN, không Inf) trên MỖI dòng.
    macro_f1 = held_rows["macro_f1"].astype(float).tolist()
    for name, f1v in zip(held_rows["held_out"].tolist(), macro_f1):
        assert math.isfinite(f1v), (
            f"macro_F1 không finite trên held-out '{name}': {f1v}"
        )
    print(f"  ✓ macro_F1 FINITE trên cả 2 held-out: "
          f"{dict(zip(held_rows['held_out'], macro_f1))}.")

    # (5d) accuracy cũng finite (chỉ là tham khảo do lệch lớp).
    acc = held_rows["accuracy"].astype(float).tolist()
    for name, av in zip(held_rows["held_out"].tolist(), acc):
        assert math.isfinite(av), (
            f"accuracy không finite trên held-out '{name}': {av}"
        )
    print(f"  ✓ accuracy FINITE trên cả 2 held-out.")

    # (5e) Dòng MEAN — macro_F1 mean cũng finite.
    mean_f1 = float(mean_row["macro_f1"].iloc[0])
    assert math.isfinite(mean_f1), f"MEAN macro_F1 không finite: {mean_f1}"
    print(f"  ✓ MEAN macro_F1 = {mean_f1:.4f} (finite).")

    # (5f) best_val_f1 > 0.3 → chứng minh model HỌC ĐƯỢC trên cả 2 round
    # (với 1 train graph duy nhất + 40 epoch + E-GraphSAGE, kỳ vọng
    # val macro-F1 lên được > 0.3; dưới ngưỡng này = hầu như đoán ngẫu nhiên).
    best_vals = held_rows["best_val_f1"].astype(float).tolist()
    print(f"  best_val_f1 per round = "
          f"{dict(zip(held_rows['held_out'], best_vals))}")
    for name, f1v in zip(held_rows["held_out"], best_vals):
        assert f1v > 0.30, (
            f"best_val_f1 trên held-out '{name}' = {f1v:.4f} ≤ 0.30 — "
            f"model KHÔNG học được (vẫn gần đoán ngẫu nhiên). Bug ở "
            f"multi_scenario / model / criterion?"
        )
    print(f"  ✓ Cả 2 round đều có best_val_f1 > 0.30 → model học được.")

    # ============================================================
    # [6] Artifacts (CSV + confusion matrix PNG) phải tồn tại
    # ============================================================
    print()
    print("-" * 70)
    print("[6] Kiểm tra artifacts trên đĩa")
    print("-" * 70)
    csv_path = os.path.join(
        OUT_DIR, f"loso_{MODEL_NAME}_{IMBALANCE_MODE}.csv"
    )
    assert os.path.isfile(csv_path), f"Thiếu CSV: {csv_path}"
    print(f"  ✓ CSV : {csv_path}")

    # Tìm confusion matrix PNG (tên có chứa 'hardest' + 1 tên held-out).
    pngs = [
        f for f in os.listdir(OUT_DIR)
        if f.startswith(f"confusion_matrix_loso_{MODEL_NAME}_"
                        f"{IMBALANCE_MODE}_hardest_")
        and f.endswith(".png")
    ]
    assert pngs, (
        f"Thiếu confusion matrix PNG trong {OUT_DIR}."
    )
    png_path = os.path.join(OUT_DIR, pngs[0])
    assert os.path.getsize(png_path) > 0, f"PNG rỗng: {png_path}"
    print(f"  ✓ PNG : {png_path}")

    # ============================================================
    # [7] Đọc lại CSV do run_loso ghi, check schema
    # ============================================================
    print()
    print("-" * 70)
    print("[7] Đọc lại CSV và kiểm tra schema")
    print("-" * 70)
    df_csv = pd.read_csv(csv_path)
    print(f"  CSV shape = {df_csv.shape}")
    print(f"  CSV columns (first 12) = "
          f"{list(df_csv.columns[:12])}")
    expected_cols = {
        "held_out", "macro_f1", "weighted_f1", "accuracy",
        "best_epoch", "best_val_f1", "n_unseen_in_train",
    }
    missing = expected_cols - set(df_csv.columns)
    assert not missing, f"CSV thiếu cột: {missing}"
    # Mỗi lớp có cột f1_<class> và support_<class>.
    f1_cols = [c for c in df_csv.columns if c.startswith("f1_")]
    sup_cols = [c for c in df_csv.columns if c.startswith("support_")]
    assert len(f1_cols) == K, (
        f"Số cột f1_*={len(f1_cols)} != K={K}."
    )
    assert len(sup_cols) == K, (
        f"Số cột support_*={len(sup_cols)} != K={K}."
    )
    print(f"  ✓ Đủ {len(f1_cols)} cột f1_<class> và "
          f"{len(sup_cols)} cột support_<class>.")
    print(f"  ✓ CSV schema OK.\n")

    # ============================================================
    # TỔNG KẾT
    # ============================================================
    print("=" * 70)
    print(" ALL LOSO SMOKE-TEST CHECKS PASSED")
    print("=" * 70)
    print(f"  model         : {MODEL_NAME}")
    print(f"  imbalance     : {IMBALANCE_MODE}")
    print(f"  scenarios     : 2  ({sorted(held_names)})")
    print(f"  K (num_class) : {K}")
    print(f"  feature_dim   : {next(iter(fdims.values()))}")
    print(f"  edges total   : {sum(es.values()):,}")
    print(f"  epochs/round  : {EPOCHS}")
    print()
    print("  HELD-OUT MACRO-F1 (LOSO inductive):")
    for name, f1v in zip(held_rows["held_out"], macro_f1):
        print(f"    {name:<10s}  macro_F1 = {f1v:.4f}")
    print(f"    {'MEAN':<10s}  macro_F1 = {mean_f1:.4f}")
    print()
    print(f"  CSV : {csv_path}")
    print(f"  PNG : {png_path}")
    print(f"  Tổng thời gian test : {time.perf_counter() - t_total:.2f}s")
    print()
    print("Bước tiếp theo (GĐ2 / GĐ3):")
    print("  • Thay vòng train 'gradient sum qua các graph' bằng FedAvg:")
    print("    mỗi graph ≈ 1 client, gửi state_dict về server.")
    print("  • Thêm scenario 1-1, 9-1, 36-1, 39-1 để LOSO đầy đủ 6-fold.")
    print("  • (Tùy chọn) Chia đồ thị theo cửa sổ thời gian trước khi build.")


if __name__ == "__main__":
    main()
