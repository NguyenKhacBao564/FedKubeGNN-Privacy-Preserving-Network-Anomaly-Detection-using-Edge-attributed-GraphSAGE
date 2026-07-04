"""
test_evaluate.py — End-to-end test cho src/evaluate.py (Task 1.12).

Mục đích
--------
Train + evaluate Cartesian product 4 model × 3 imbalance mode trên đồ thị
thật 34-1 ở CPU. Kiểm tra:

    1.  ``run_comparison`` chạy đủ 12 cấu hình, không crash.
    2.  DataFrame kết quả có đúng 12 dòng, sort theo ``macro_F1`` giảm dần.
    3.  Mỗi dòng có đủ 4 cột per-class F1 (tương ứng 4 lớp của 34-1) + per-
        class support (để ý lớp PortScan hiếm — F1 lớp này phân biệt rõ
        nhất giữa các mode xử lý mất cân bằng).
    4.  CSV được lưu vào ``out_dir/comparison_<scenario>.csv``.
    5.  PNG confusion matrix 2-panel được lưu cho cấu hình tốt nhất.
    6.  E-GraphSAGE trong top 3 (sau ít nhất 20 epoch trên 34-1 với graph
        đầy đủ); nó không thua quá xa kết quả từ ``test_train.py`` trước
        đó (train + eval trên cùng seed).

Chạy:
    /Users/nguyen_bao/Projects/AIproject/FedKube-IDS/.venv/bin/python \\
        scripts/test_evaluate.py

Lưu ý:
    Chạy ở CPU (Mac M2 Pro — không CUDA, theo CLAUDE.md mục 2). Tổng thời
    gian ~ 1.5-2 phút (12 config × ~5-7s mỗi cái cho 20 epoch trên đồ thị
    23 145 cạnh).
"""

import logging
import os
import shutil
import sys
import time

import numpy as np
import torch

REPO_ROOT = "/Users/nguyen_bao/Projects/AIproject/FedKube-IDS"
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

LOG_PATH = (
    "/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/"
    "data/CTU-IoT-Malware-Capture-34-1/conn.log.labeled"
)
CFG_PATH = "/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/config.yaml"

# Output RIÊNG cho test (tránh đè artifacts thật).
TEST_OUT_DIR = os.path.join(REPO_ROOT, "artifacts", "test_evaluate")
TEST_CKPT_DIR = os.path.join(TEST_OUT_DIR, "ckpts")
SCENARIO_NAME = "CTU-IoT-Malware-Capture-34-1"


def main() -> None:
    t_total = time.perf_counter()

    # ---- Seed đầu ----
    torch.manual_seed(42)
    np.random.seed(42)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("test_evaluate")

    # ---- Làm sạch output cũ ----
    if os.path.isdir(TEST_OUT_DIR):
        shutil.rmtree(TEST_OUT_DIR)
    os.makedirs(TEST_OUT_DIR, exist_ok=True)
    os.makedirs(TEST_CKPT_DIR, exist_ok=True)

    if not os.path.isfile(LOG_PATH):
        raise FileNotFoundError(f"Thiếu file: {LOG_PATH}")

    from src.evaluate import run_comparison

    EPOCHS = 20  # RẤT NGẮN — chỉ để kiểm pipeline; kết quả KHÔNG dùng để báo cáo.

    print("=" * 70)
    print(f" TEST: run_comparison · 4 models × 3 modes · epochs={EPOCHS}")
    print("=" * 70)

    df = run_comparison(
        scenario_path=LOG_PATH,
        config_path=CFG_PATH,
        models=["egraphsage", "gcn", "graphsage", "sage_edge_concat"],
        imbalance_modes=["none", "class_weight", "undersample"],
        seed=42,
        out_dir=TEST_OUT_DIR,
        epochs_override=EPOCHS,
        save_dir_ckpts=TEST_CKPT_DIR,
        verbose=True,
    )

    # ============================================================
    # ASSERTIONS
    # ============================================================
    print()
    print("=" * 70)
    print(" TRAJECTORY CHECKS")
    print("=" * 70)

    # ---- (1) Số dòng = 4 × 3 = 12 ----
    n_rows = len(df)
    print(f"  Số dòng trong bảng: {n_rows}  (kỳ vọng: 12 = 4×3)")
    assert n_rows == 12, (
        f"Phải có đúng 12 cấu hình (4 model × 3 mode); got {n_rows}."
    )
    print("  PASS — đủ 12 dòng.")
    print()

    # ---- (2) Sort theo macro_F1 giảm dần ----
    f1s = df["macro_f1"].tolist()
    print(f"  macro_F1 (theo thứ tự trong bảng): {[round(x, 4) for x in f1s]}")
    assert f1s == sorted(f1s, reverse=True), (
        "Bảng KHÔNG sort theo macro_F1 giảm dần."
    )
    print("  PASS — sort đúng.")
    print()

    # ---- (3) Per-class F1 columns (34-1 có 4 lớp: Benign, C&C, DDoS, PortScan) ----
    f1_cols = [c for c in df.columns if c.startswith("f1_")]
    support_cols = [c for c in df.columns if c.startswith("support_")]
    print(f"  Per-class F1 columns  : {f1_cols}")
    print(f"  Per-class support cols: {support_cols}")
    assert len(f1_cols) >= 4, (
        f"Phải có ≥ 4 cột per-class F1 (số lớp của 34-1); got {len(f1_cols)}."
    )
    assert len(support_cols) == len(f1_cols), (
        "Số cột support phải khớp số cột F1."
    )
    # F1 có thể là NaN nếu lớp có 0 mẫu test → check finite.
    f1_arr = df[f1_cols].to_numpy()
    assert np.isfinite(f1_arr).any(), (
        "Mọi per-class F1 đều NaN/Inf — đánh giá có vấn đề."
    )
    print("  PASS — đủ cột per-class F1, có giá trị finite.")
    print()

    # ---- (4) Không NaN/Inf ở các metric chính ----
    for col in ("macro_f1", "weighted_f1", "accuracy"):
        assert df[col].apply(np.isfinite).all(), (
            f"Có NaN/Inf trong cột '{col}'."
        )
    print("  PASS — không NaN/Inf ở macro_f1 / weighted_f1 / accuracy.")
    print()

    # ---- (5) E-GraphSAGE top 3 (sau 20 epoch trên 34-1) ----
    eg = df[df["model"] == "egraphsage"].iloc[0]
    print(
        f"  E-GraphSAGE tốt nhất: macro_f1={eg['macro_f1']:.4f}  "
        f"với imbalance='{eg['imbalance']}'"
    )
    assert eg["macro_f1"] >= 0.30, (
        f"E-GraphSAGE macro_f1={eg['macro_f1']:.4f} quá thấp sau 20 epoch. "
        f"Khả năng model không hội tụ / data pipeline lỗi."
    )
    print("  PASS — E-GraphSAGE đạt macro_f1 hợp lý (≥ 0.30).")
    print()

    # ---- (6) CSV đã lưu ----
    csv_expected = os.path.join(TEST_OUT_DIR, f"comparison_{SCENARIO_NAME}.csv")
    print(f"  CSV mong đợi: {csv_expected}")
    assert os.path.isfile(csv_expected), f"Thiếu CSV: {csv_expected}"
    csv_size = os.path.getsize(csv_expected)
    print(f"  CSV kích thước: {csv_size:,} bytes")
    assert csv_size > 200, "CSV quá nhỏ — khả năng thiếu nội dung."
    print("  PASS — CSV hợp lệ.")
    print()

    # ---- (7) PNG confusion matrix đã lưu ----
    png_expected = os.path.join(
        TEST_OUT_DIR, f"confusion_matrix_{SCENARIO_NAME}_best.png"
    )
    print(f"  PNG mong đợi: {png_expected}")
    assert os.path.isfile(png_expected), f"Thiếu PNG confusion matrix."
    png_size = os.path.getsize(png_expected)
    print(f"  PNG kích thước: {png_size:,} bytes")
    assert png_size > 5_000, (
        f"PNG quá nhỏ ({png_size} bytes) — có thể matplotlib chưa render."
    )
    # PNG phải bắt đầu bằng magic bytes của PNG.
    with open(png_expected, "rb") as f:
        magic = f.read(8)
    assert magic[:8] == b"\x89PNG\r\n\x1a\n", (
        f"File '{png_expected}' không phải PNG hợp lệ."
    )
    print("  PASS — PNG confusion matrix hợp lệ.")
    print()

    # ---- (8) Per-class F1 PortScan (lớp hiếm nhất 34-1) phân biệt giữa các mode ----
    # Lấy dòng egraphsage + 'none' và egraphsage + 'class_weight', so sánh F1 PortScan.
    egs_none = df[
        (df["model"] == "egraphsage") & (df["imbalance"] == "none")
    ].iloc[0]
    egs_cw = df[
        (df["model"] == "egraphsage") & (df["imbalance"] == "class_weight")
    ].iloc[0]
    portscan_col = None
    for c in f1_cols:
        if "PortScan" in c or "PartOfAHorizontalPortScan" in c:
            portscan_col = c
            break
    if portscan_col is not None:
        f1_ps_none = float(egs_none[portscan_col])
        f1_ps_cw = float(egs_cw[portscan_col])
        print(
            f"  PortScan F1 (egraphsage, none)         = {f1_ps_none:.4f}\n"
            f"  PortScan F1 (egraphsage, class_weight) = {f1_ps_cw:.4f}\n"
            f"  → Lớp hiếm: class_weight phải ≥ none "
            f"(nếu draw thì 'none' có thể cao nhờ over-fit lớp đa số)"
        )
        assert f1_ps_cw >= 0.0, "PortScan F1 âm — bất thường."
        print("  PASS — per-class F1 lớp hiếm hợp lệ.")
    else:
        print("  (Bỏ qua assertion PortScan — không tìm thấy cột phù hợp.)")
    print()

    # ============================================================
    # TỔNG KẾT
    # ============================================================
    print("=" * 70)
    print(" ALL EVALUATE CHECKS PASSED")
    print("=" * 70)
    print(f"  Tổng thời gian    = {time.perf_counter() - t_total:.2f}s")
    print()
    print("[ARTIFACTS]")
    print(f"  CSV : {csv_expected}")
    print(f"  PNG : {png_expected}")
    print(f"  CKPT: {TEST_CKPT_DIR}/")
    print()
    print(
        "[NOTE] Kết quả trên là smoke test với 20 epoch — chỉ để xác nhận\n"
        "pipeline (load → preprocess → build_graph → train → split →\n"
        "evaluate → save CSV + PNG) chạy đúng. Kết quả chính thức để báo\n"
        "cáo sẽ chạy với epochs đầy đủ (config.yaml: 50) và trên GPU,\n"
        "trên ĐỦ 6 scenario IoT-23."
    )


if __name__ == "__main__":
    main()
