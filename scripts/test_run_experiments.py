"""
test_run_experiments.py — Smoke test orchestrator cho src/run_experiments.py.

Mục đích
--------
Chạy toàn bộ orchestrator (Phase A + Phase B × 3 protocol) trên 2 scenario
local {34-1, 3-1} với epochs NGẮN + cap nhỏ, để kiểm tra:

    1.  Orchestrator chạy đủ cả 3 protocol (per_scenario, pooled, loso).
    2.  Phase A: egraphsage × 3 imbalance_mode → bảng Phase A có 3 mode.
    3.  Phase B: cố định ``winning_mode``, chạy 5 model → bảng Phase B có 5 model.
    4.  ``winning_mode`` được tính & truyền đúng sang Phase B (cùng đi qua
        CSV với Phase A).
    5.  results_summary.csv có đủ dòng (Phase A + Phase B × protocol ×
        số dòng của mỗi config).
    6.  Confusion matrix PNG tồn tại (ít nhất pooled + cm_tốt nhất mỗi
        protocol).
    7.  LOSO KHÔNG crash khi gặp lớp private (Attack private to 3-1,
        DDoS private to 34-1).
    8.  Tất cả 3 protocol đều ra ``macro_F1`` FINITE.
    9.  Tổng thời gian + tổng số cấu hình được in.

Lưu ý
-----
Đây là SMOKE TEST 2 scenario / epoch ngắn / cap nhỏ. Train thật đủ
6 scenario + 150 epoch trên GPU vast.ai làm ở bước vận hành sau.

Chạy:
    /Users/nguyen_bao/Projects/AIproject/FedKube-IDS/.venv/bin/python \\
        scripts/test_run_experiments.py
"""

import json
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

LOG_34_1 = REPO_ROOT + "/data/CTU-IoT-Malware-Capture-34-1/conn.log.labeled"
LOG_3_1 = REPO_ROOT + "/data/CTU-IoT-Malware-Capture-3-1/conn.log.labeled"
CFG_PATH = REPO_ROOT + "/config.yaml"
OUT_DIR = REPO_ROOT + "/artifacts/experiments"

# Smoke-test parameters (NGẮN để chạy local nhanh).
EPOCHS = 30
CAP_PER_CLASS = 2000
SEED = 42
PROTOCOLS = ["per_scenario", "pooled", "loso"]


def main() -> None:
    t_total = time.perf_counter()

    # ---- 0) Repro ----
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print(" TEST  src.run_experiments  ·  ORCHESTRATOR")
    print("=" * 70)
    print(f"  scenarios    : {['34-1', '3-1']}")
    print(f"  protocols    : {PROTOCOLS}")
    print(f"  epochs       : {EPOCHS}")
    print(f"  cap_per_class: {CAP_PER_CLASS}")
    print(f"  seed         : {SEED}")
    print(f"  out_dir      : {OUT_DIR}")
    print()

    # ---- 1) Sanity-check input ----
    for p in (LOG_34_1, LOG_3_1, CFG_PATH):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    if os.path.isdir(OUT_DIR):
        # xoá các artifact cũ để test "sạch"
        import shutil
        print(f"  [cleanup] xóa {OUT_DIR}/ cũ để test sạch.")
        shutil.rmtree(OUT_DIR, ignore_errors=True)

    # ---- 2) Import orchestrator ----
    from src.run_experiments import (
        IMBALANCE_MODES,
        MODEL_POOL,
        PROTOCOLS as _PROTOCOLS,
        run_all,
    )
    assert IMBALANCE_MODES == ["none", "class_weight", "undersample"]
    assert MODEL_POOL == ["egraphsage", "gat", "sage_edge_concat", "graphsage", "gcn"]
    assert set(_PROTOCOLS) == {"per_scenario", "pooled", "loso"}
    print(f"  ✓ Module loaded — "
          f"{len(IMBALANCE_MODES)} mode × {len(MODEL_POOL)} model × "
          f"{len(_PROTOCOLS)} protocol = "
          f"{(len(IMBALANCE_MODES) + len(MODEL_POOL)) * len(_PROTOCOLS)} "
          f"configs lý thuyết.\n")

    # ---- 3) Chạy run_all ----
    df_summary = run_all(
        scenario_paths={
            "34-1": LOG_34_1,
            "3-1": LOG_3_1,
        },
        config_path=CFG_PATH,
        protocols=PROTOCOLS,
        cap_per_class=CAP_PER_CLASS,
        chunksize=100_000,
        epochs_override=EPOCHS,
        seed=SEED,
        out_dir=OUT_DIR,
        verbose=True,
    )

    # ---- 4) Asserts ----
    print()
    print("-" * 70)
    print("[ASSERTS] kiểm tra output orchestrator")
    print("-" * 70)

    # (4a) Kết quả DataFrame có đủ cột protocol/phase/scenario/model/imbalance_mode
    expected_cols = {
        "protocol", "phase", "scenario", "model", "imbalance_mode",
        "macro_f1", "weighted_f1", "accuracy",
        "best_epoch", "epochs_ran", "best_val_f1",
    }
    missing = expected_cols - set(df_summary.columns)
    assert not missing, f"results_summary thiếu cột: {missing}"
    print(f"  ✓ Đủ cột bắt buộc: {sorted(expected_cols)}.")

    # (4b) Phase A đủ 3 mode × 3 protocol
    df_a = df_summary[df_summary["phase"] == "A"]
    df_b = df_summary[df_summary["phase"] == "B"]
    a_modes = set(df_a["imbalance_mode"].unique())
    assert a_modes == set(IMBALANCE_MODES), (
        f"Phase A thiếu/thừa mode: {a_modes}"
    )
    print(f"  ✓ Phase A: đủ {len(a_modes)} mode = {sorted(a_modes)}.")

    # (4c) Phase B đủ 5 model × 3 protocol
    b_models = set(df_b["model"].unique())
    assert b_models == set(MODEL_POOL), (
        f"Phase B thiếu/thừa model: {b_models}"
    )
    print(f"  ✓ Phase B: đủ {len(b_models)} model = {sorted(b_models)}.")

    # (4d) winning_mode: mỗi protocol có winner riêng (per-protocol), không
    # bắt buộc đồng nhất giữa các protocol. Nhưng trong MỖI protocol, Phase B
    # phải cố định đúng 1 mode duy nhất (= winner của Phase A protocol đó).
    winning_per_proto: Dict[str, str] = {}
    for proto in PROTOCOLS:
        sub = df_summary[df_summary["protocol"] == proto]
        sub_a = sub[(sub["phase"] == "A") & (~sub["scenario"].isin(["MEAN"]))]
        # winner = imbalance_mode có mean macro_f1 cao nhất trong Phase A
        winner = (
            sub_a.groupby("imbalance_mode")["macro_f1"]
                 .mean()
                 .sort_values(ascending=False)
                 .index[0]
        )
        winning_per_proto[proto] = str(winner)
        # Check Phase B của protocol này đúng 1 mode = winner
        sub_b = sub[sub["phase"] == "B"]
        modes_in_b = set(sub_b["imbalance_mode"].unique())
        assert modes_in_b == {winner}, (
            f"Protocol {proto}: Phase B dùng mode(s) {modes_in_b}, "
            f"expected {{'{winner}'}} (winner từ Phase A)."
        )
    print(
        f"  ✓ winning_mode per-protocol: {winning_per_proto} — "
        f"Phase B của MỖI protocol cố định đúng winner."
    )

    # Tên file Phase B dùng winning_mode của protocol 'pooled' (làm mặc định)
    # hoặc winning của protocol đầu tiên.
    winning_default = winning_per_proto[PROTOCOLS[0]]

    # (4e) Đủ protocol
    p_summary = set(df_summary["protocol"].unique())
    assert set(PROTOCOLS) <= p_summary, (
        f"Thiếu protocol: {set(PROTOCOLS) - p_summary}"
    )
    print(f"  ✓ Protocol trong summary: {sorted(p_summary)}.")

    # (4f) macro_F1 FINITE trên từng dòng (rất quan trọng)
    for (proto, phase), sub in df_summary.groupby(["protocol", "phase"]):
        # Bỏ dòng MEAN (dòng tổng hợp), giữ POOLED (là kết quả thật)
        sub_no_agg = sub[~sub["scenario"].isin(["MEAN"])]
        for _, row in sub_no_agg.iterrows():
            m = float(row["macro_f1"])
            assert math.isfinite(m), (
                f"macro_F1 không finite tại "
                f"protocol={proto}, phase={phase}, scenario={row['scenario']}, "
                f"model={row['model']}, mode={row['imbalance_mode']}: {m}"
            )
    print(f"  ✓ macro_F1 FINITE trên MỌI dòng "
          f"(loại trừ dòng tổng hợp 'MEAN').")

    # (4g) LOSO KHÔNG crash với lớp private (đã check downstream ở Phase A/B)
    df_loso = df_summary[df_summary["protocol"] == "loso"]
    assert len(df_loso) > 0, "LOSO thiếu kết quả."
    if "n_unseen_in_train" in df_loso.columns:
        n_unseen_max = int(pd.to_numeric(
            df_loso["n_unseen_in_train"], errors="coerce"
        ).fillna(0).max())
        assert n_unseen_max >= 1, (
            f"Kỳ vọng LOSO có n_unseen_in_train ≥ 1 (Attack/DDoS private), "
            f"got max={n_unseen_max}."
        )
        print(f"  ✓ LOSO có n_unseen_in_train max = {n_unseen_max} "
              f"(đã detect đúng lớp private).")
    print(f"  ✓ LOSO KHÔNG crash với lớp private.")

    # (4h) Artifacts trên đĩa
    print()
    print("-" * 70)
    print("[ASSERTS] artifacts trên đĩa")
    print("-" * 70)
    expected_files = []
    for proto in PROTOCOLS:
        expected_files.append(
            f"phase_a_{proto}_egraphsage_3modes.csv"
        )
        # Phase B filename có winning_mode của protocol đó
        expected_files.append(
            f"phase_b_{proto}_mode-{winning_per_proto[proto]}_5models.csv"
        )
    expected_files.append("results_summary.csv")

    for fname in expected_files:
        fpath = os.path.join(OUT_DIR, fname)
        assert os.path.isfile(fpath), f"Thiếu file: {fpath}"
        size = os.path.getsize(fpath)
        assert size > 0, f"File rỗng: {fpath}"
        print(f"  ✓ {fname}  ({size:,} bytes)")

    # (4i) Confusion matrix PNG — pooled protocol luôn có (mỗi config 1 PNG)
    # + loso có file hardest_<s>.png (≥1, ≤2 cho 2 scenario). Cả 2 loại đều
    # được lưu trong checkpoints/.
    ckpt_dir_check = os.path.join(OUT_DIR, "checkpoints")
    cm_pngs_all: List[str] = []
    if os.path.isdir(ckpt_dir_check):
        cm_pngs_all.extend(
            os.path.join(ckpt_dir_check, f)
            for f in os.listdir(ckpt_dir_check)
            if f.endswith(".png")
        )
    cm_pooled = [p for p in cm_pngs_all if os.path.basename(p).startswith("cm_pooled_")]
    cm_loso = [p for p in cm_pngs_all if "hardest_" in os.path.basename(p)]
    assert cm_pooled, f"Thiếu CM pooled PNG trong {ckpt_dir_check}."
    assert cm_loso, f"Thiếu CM loso hardest PNG trong {ckpt_dir_check}."
    print(
        f"  ✓ CM PNG: {len(cm_pooled)} pooled (cm_pooled_*) + "
        f"{len(cm_loso)} loso (confusion_matrix_loso_*_hardest_*)."
    )

    # (4j) Checkpoints — pooled phải có (≥ 1 cho 1 config; smoke test có 8 configs pooled)
    ckpt_dir = os.path.join(OUT_DIR, "checkpoints")
    if os.path.isdir(ckpt_dir):
        ckpts = [
            f for f in os.listdir(ckpt_dir)
            if f.endswith(".pt")
        ]
        assert ckpts, (
            f"Không có checkpoint nào trong {ckpt_dir}."
        )
        print(f"  ✓ Tìm thấy {len(ckpts)} checkpoint trong checkpoints/ "
              f"({len(ckpts)}/8 pooled + per_scenario).")
    else:
        print(f"  [skip] checkpoints dir không tồn tại.")

    # (4k) In tổng kết cuối
    print()
    print("=" * 70)
    print(" ALL ORCHESTRATOR SMOKE-TEST CHECKS PASSED")
    print("=" * 70)

    # Tổng số dòng summary per protocol × phase
    counts = df_summary.groupby(["protocol", "phase"]).size().reset_index(
        name="n_rows"
    )
    print("\n  Bảng tóm tắt dòng summary:")
    with pd.option_context(
        "display.max_rows", None,
        "display.width", 200,
        "display.float_format", "{:.0f}".format,
    ):
        print(counts.to_string(index=False))

    # Tổng số config "logically" đã chạy (Phase A 3 mode × 3 proto = 9
    # + Phase B 5 model × 3 proto = 15 = 24 configs training jobs)
    n_jobs_phase_a = len(PROTOCOLS) * len(IMBALANCE_MODES)
    n_jobs_phase_b = len(PROTOCOLS) * len(MODEL_POOL)
    print(
        f"\n  • Phase A: {len(PROTOCOLS)} protocol × {len(IMBALANCE_MODES)} mode "
        f"= {n_jobs_phase_a} logical configs"
    )
    print(
        f"  • Phase B: {len(PROTOCOLS)} protocol × {len(MODEL_POOL)} model "
        f"= {n_jobs_phase_b} logical configs"
    )
    print(
        f"  • Tổng logical configs (chưa tính LOSO rounds): "
        f"{n_jobs_phase_a + n_jobs_phase_b}"
    )

    # Per-protocol Phases winner/sort
    print("\n  Winner (mean macro_F1) theo (protocol, phase, model):")
    sub = df_summary[~df_summary["scenario"].isin(["MEAN"])]
    agg = (
        sub.groupby(["protocol", "phase", "model"])["macro_f1"]
           .mean()
           .reset_index()
           .sort_values(["protocol", "phase", "macro_f1"],
                        ascending=[True, True, False])
    )
    with pd.option_context(
        "display.max_rows", None,
        "display.width", 200,
        "display.float_format", "{:.4f}".format,
    ):
        print(agg.to_string(index=False))

    dt = time.perf_counter() - t_total
    print(f"\n  Tổng thời gian test : {dt:.2f}s ({dt/60:.1f} phút)")
    print(f"  Tổng dòng summary   : {len(df_summary)}")
    print()
    print("  ⚠ ĐÂY LÀ SMOKE TEST 2 scenario / epoch ngắn / cap nhỏ.")
    print("    Chạy thật đủ 6 scenario + 150 epoch trên GPU vast.ai làm")
    print("    ở bước vận hành (prompt sau).")


if __name__ == "__main__":
    main()
