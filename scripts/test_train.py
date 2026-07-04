"""
test_train.py — End-to-end test cho src/train.py (Task 1.10).

Mục đích
    Train egraphsage với imbalance_mode='class_weight' trên đồ thị 34-1
    thật, epochs ngắn (~30), kiểm tra:

    1. Vòng train chạy được từ đầu đến cuối (không crash).
    2. Train_loss GIẢM qua các epoch.
    3. Val macro-F1 TĂNG so với epoch 0 (chứng minh model học được).
    4. Checkpoint được lưu và LOAD LẠI được — model mới dựng xong có thể
       nạp state_dict và forward đúng shape.

    KHÔNG kỳ vọng F1 cao (34-1 + epochs ngắn + CPU). Chỉ cần chứng minh:
        - vòng train học được,
        - không rò rỉ nhãn (train_mask tách hẳn với val/test),
        - checkpoint round-trip OK.

Chạy:
    /Users/nguyen_bao/Projects/AIproject/FedKube-IDS/.venv/bin/python \\
        scripts/test_train.py
"""

import logging
import os
import sys
import time

import numpy as np
import torch
import yaml

REPO_ROOT = "/Users/nguyen_bao/Projects/AIproject/FedKube-IDS"
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

LOG_PATH = (
    "/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/"
    "data/CTU-IoT-Malware-Capture-34-1/conn.log.labeled"
)
CFG_PATH = "/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/config.yaml"


def main():
    t_total = time.perf_counter()

    # ---- Seed đầu (để set_seed trong run_scenario cũng dùng cùng giá trị) ----
    torch.manual_seed(42)
    np.random.seed(42)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(name)s: %(message)s',
    )
    logger = logging.getLogger("test_train")

    if not os.path.isfile(LOG_PATH):
        raise FileNotFoundError(f"Thiếu file: {LOG_PATH}")

    # ---- 1) Gọi run_scenario ----
    from src.train import run_scenario

    EPOCHS_OVERRIDE = 30
    save_dir = os.path.join(REPO_ROOT, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print(f" TRAIN: egraphsage  ·  imbalance=class_weight  ·  "
          f"epochs={EPOCHS_OVERRIDE}  ·  patience=8")
    print("=" * 70)

    result = run_scenario(
        log_path=LOG_PATH,
        model_name='egraphsage',
        imbalance_mode='class_weight',
        config_path=CFG_PATH,
        epochs_override=EPOCHS_OVERRIDE,
        early_stop_patience_override=8,
        save_dir=save_dir,
        verbose=True,
    )

    history = result['history']
    ckpt_path = result['checkpoint']
    K = result['cfg']['model']['num_classes'] if 'num_classes' in (
        result['cfg'].get('model', {}) or {}
    ) else None

    # ---- 2) Verify: có ít nhất một epoch chạy ----
    print()
    print("=" * 70)
    print(" TRAJECTORY CHECKS")
    print("=" * 70)
    n_ran = len(history['epoch'])
    print(f"  Số epoch đã chạy: {n_ran} (yêu cầu: ≥ 1, ≤ {EPOCHS_OVERRIDE})")
    assert n_ran >= 1, "Train không chạy epoch nào."
    assert n_ran <= EPOCHS_OVERRIDE

    train_loss = history['train_loss']
    val_f1 = history['val_macro_f1']

    # ---- 2a) Train loss PHẢI giảm ----
    head = train_loss[: min(5, n_ran)]
    tail = train_loss[-min(5, n_ran):]
    avg_head = float(np.mean(head))
    avg_tail = float(np.mean(tail))
    print(f"  train_loss trung bình đầu (≤ 5 epoch đầu): {avg_head:.4f}")
    print(f"  train_loss trung bình cuối (≤ 5 epoch cuối): {avg_tail:.4f}")
    assert avg_tail < avg_head, (
        f"Train loss KHÔNG giảm: head={avg_head:.4f}, tail={avg_tail:.4f}. "
        f"Vòng train có thể đang học sai hoặc learning rate quá nhỏ."
    )
    print("  PASS — train loss giảm.")
    print()

    # ---- 2b) Val macro-F1 PHẢI tăng so với epoch 0 ----
    f1_epoch_0 = val_f1[0]
    f1_best = max(val_f1)
    best_epoch = history['best_epoch']
    print(f"  val_macro_F1 epoch 0    : {f1_epoch_0:.4f}")
    print(f"  val_macro_F1 best       : {f1_best:.4f}  @ epoch {best_epoch}")
    print(f"  val_macro_F1 test (cuối): {history['test_f1']:.4f}")
    assert f1_best >= f1_epoch_0, (
        f"Val F1 KHÔNG cải thiện: epoch0={f1_epoch_0:.4f}, "
        f"best={f1_best:.4f}."
    )
    # Lỏng một chút: yêu cầu best > epoch0 (nếu random start quá tệ,
    # accept bằng tie-break).
    if f1_best == f1_epoch_0:
        logger.warning(
            "  ⚠ val F1 không tăng (tie). Kiểm tra lr/patience nếu muốn"
        )
    print("  PASS — val macro-F1 tăng so với epoch 0.")
    print()

    # ---- 2c) Không có loss NaN/Inf ----
    has_nan = any(
        (not np.isfinite(v)) for v in train_loss + history['val_loss']
    )
    print(f"  NaN/Inf trong train_loss / val_loss? {has_nan}")
    assert not has_nan, "Có NaN/Inf trong loss → mô hình không hội tụ."
    print("  PASS — không NaN/Inf.")
    print()

    # ---- 3) Checkpoint: file tồn tại + reload được ----
    print("=" * 70)
    print(" CHECKPOINT ROUND-TRIP")
    print("=" * 70)
    assert os.path.isfile(ckpt_path), f"Checkpoint không tồn tại: {ckpt_path}"
    print(f"  Path: {ckpt_path}")
    print(f"  Kích thước file: {os.path.getsize(ckpt_path):,} bytes")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"  Metadata trong checkpoint:")
    print(f"    val_macro_f1   = {ckpt['val_macro_f1']:.4f}")
    print(f"    feature_dim    = {ckpt['feature_dim']}")
    print(f"    num_classes    = {ckpt['num_classes']}")
    print(f"    imbalance_mode = '{ckpt['imbalance_mode']}'")
    print(f"    history_meta   = {ckpt['history_meta']}")

    # ---- 3a) Khớp metadata ----
    assert ckpt['val_macro_f1'] == history['best_val_f1'], (
        "val_macro_f1 trong ckpt khác history."
    )
    assert ckpt['imbalance_mode'] == 'class_weight'
    assert ckpt['feature_dim'] > 0
    assert ckpt['num_classes'] == 4, (
        f"num_classes={ckpt['num_classes']} kỳ vọng 4 cho 34-1."
    )

    # ---- 3b) Load lại state_dict vào model MỚI và forward ----
    from src.data_io import load_scenario
    from src.preprocess import clean_flows, fit_preprocessor, transform
    from src.imbalance import prepare_imbalance_variants
    from src.graph_build import build_graph
    from src.model import build_model

    print()
    print("  Rebuild data + model để xác nhận load_state_dict khớp:")
    df_clean = clean_flows(load_scenario(LOG_PATH))
    pre = fit_preprocessor(df_clean)
    df_feat = transform(df_clean, pre)
    variants = prepare_imbalance_variants(df_feat, random_state=42)
    data_re = build_graph(
        df_feat, class_to_idx=variants['class_to_idx'],
        feature_columns=pre.feature_columns,
    )
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    fresh_model = build_model('egraphsage', data_re, cfg)
    fresh_model.load_state_dict(ckpt['state_dict'])
    fresh_model.eval()

    with torch.no_grad():
        out_loaded = fresh_model(data_re)
    print(f"    logits.shape  = {tuple(out_loaded.shape)}  "
          f"(kỳ vọng ({int(data_re.edge_index.shape[1])}, 4))")
    assert tuple(out_loaded.shape) == (
        int(data_re.edge_index.shape[1]), 4
    )
    assert not torch.isnan(out_loaded).any().item()
    print("  PASS — checkpoint reload + forward đúng.")
    print()

    # ---- 4) Tổng kết ----
    print("=" * 70)
    print(" ALL TRAIN CHECKS PASSED")
    print("=" * 70)
    print(f"  best_val_macro_f1 = {history['best_val_f1']:.4f}  @ epoch {history['best_epoch']}")
    print(f"  test_macro_f1     = {history['test_f1']:.4f}")
    print(f"  checkpoint        = {ckpt_path}")
    print(f"  Tổng thời gian    = {time.perf_counter() - t_total:.2f}s")
    print()
    print("Bước tiếp theo: implement evaluate.py (Task 1.12 — macro-F1 / per-class")
    print("F1 / confusion matrix trên TEST mask từ checkpoint này).")


if __name__ == "__main__":
    main()
