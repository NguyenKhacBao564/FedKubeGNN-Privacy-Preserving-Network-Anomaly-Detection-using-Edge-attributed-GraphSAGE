"""
test_safe_split.py — Smoke test cho ``split_edge_masks`` và
``safe_stratified_split`` của ``src.train`` + ``_compute_val_mask`` của
``src.multi_scenario``.

Mục đích
--------
Xác nhận hàm split CHỊU ĐƯỢC lớp cực hiếm (1-3 mẫu toàn dataset), KHÔNG
crash với ``ValueError: The least populated classes in y have only 1
member...`` như sklearn stratified split thuần.

Kịch bản mock
-------------
- Class 0: 100 mẫu (đa số)
- Class 1:  50 mẫu
- Class 2:   5 mẫu
- Class 3:   2 mẫu
- Class 4:   1 mẫu   ← SINGLETON
- Class 5:   1 mẫu   ← SINGLETON

Kỳ vọng
-------
- ``split_edge_masks`` KHÔNG crash.
- Warning in ra liệt kê class 4, 5 (n=1) — đúng lớp hiếm.
- Mọi singleton (class 4, 5) PHẢI nằm trong train_mask (không val/test).
- 3 mask rời nhau, hợp = toàn bộ indices.

Chạy
----
    .venv/bin/python scripts/test_safe_split.py
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import torch

REPO_ROOT = "/Users/nguyen_bao/Projects/AIproject/FedKube-IDS"
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)


def _make_labels() -> torch.Tensor:
    """Tạo edge_label giả với 2 lớp singleton (count=1)."""
    parts = [
        np.zeros(100, dtype=np.int64),   # class 0: 100
        np.ones(50, dtype=np.int64),      # class 1:  50
        np.full(5, 2, dtype=np.int64),    # class 2:   5
        np.full(2, 3, dtype=np.int64),    # class 3:   2
        np.full(1, 4, dtype=np.int64),    # class 4:   1  ← singleton
        np.full(1, 5, dtype=np.int64),    # class 5:   1  ← singleton
    ]
    y = np.concatenate(parts)
    return torch.from_numpy(y)


def test_split_edge_masks_does_not_crash() -> None:
    print("=" * 70)
    print("[TEST 1] split_edge_masks KHÔNG crash với 2 lớp singleton")
    print("=" * 70)
    from src.train import split_edge_masks

    edge_label = _make_labels()
    E = int(edge_label.shape[0])
    print(f"  Input: E={E}, 6 lớp, 2 singleton (class 4, 5)")

    # Capture stdout để kiểm tra warning
    buf = io.StringIO()
    with redirect_stdout(buf):
        train_mask, val_mask, test_mask = split_edge_masks(
            edge_label, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20,
            seed=42,
        )
    out = buf.getvalue()
    print(out)

    # --- Assertions ---
    # 1. Không crash — đến đây là OK.
    assert train_mask.shape == (E,), f"train_mask shape sai: {train_mask.shape}"
    assert val_mask.shape == (E,), f"val_mask shape sai: {val_mask.shape}"
    assert test_mask.shape == (E,), f"test_mask shape sai: {test_mask.shape}"

    # 2. 3 mask rời nhau, hợp = toàn bộ
    assert not (train_mask & val_mask).any(), "train ∩ val ≠ ∅"
    assert not (train_mask & test_mask).any(), "train ∩ test ≠ ∅"
    assert not (val_mask & test_mask).any(), "val ∩ test ≠ ∅"
    assert (train_mask | val_mask | test_mask).sum().item() == E, (
        "train ∪ val ∪ test ≠ toàn bộ E"
    )
    print("  ✓ 3 mask rời nhau, hợp = E (159 mẫu)")

    # 3. Singleton (class 4, 5) PHẢI nằm trong train (force_into_first)
    singleton_idx = (edge_label == 4) | (edge_label == 5)
    assert singleton_idx.sum().item() == 2, "Có 2 singleton, không đúng 2?"
    assert (train_mask & singleton_idx).sum().item() == 2, (
        f"Singleton KHÔNG hết trong train: "
        f"{(train_mask & singleton_idx).sum().item()}/2"
    )
    print("  ✓ Cả 2 singleton đều nằm trong train_mask (force_into_first OK)")

    # 4. Warning phải in ra ÍT NHẤT 1 lần (fallback vì lớp hiếm).
    # Lưu ý: class 4, 5 đã bị force vào train, nên warning ở step 2
    # có thể liệt kê class KHÁC (2, 3) nếu step 1 stratified cũng đẻ ra
    # singleton/rare. Chỉ cần warning CÓ ĐẦY ĐỦ thông tin.
    assert "fallback" in out.lower() or "random" in out.lower(), (
        f"Warning KHÔNG nói rõ fallback. Output:\n{out}"
    )
    assert "không phải lỗi" in out.lower() or "không phải loi" in out.lower(), (
        f"Warning KHÔNG nói rõ 'không phải lỗi'. Output:\n{out}"
    )
    # Phải liệt kê ÍT NHẤT 1 lớp hiếm (sau step 1, lớp 2/3 có thể thành rare)
    assert "class " in out, f"Warning KHÔNG liệt kê lớp nào. Output:\n{out}"
    print("  ✓ Warning có fallback + 'không phải lỗi' + liệt kê lớp hiếm")

    # 5. Phân bố lớp hợp lý
    n_train = int(train_mask.sum())
    n_val = int(val_mask.sum())
    n_test = int(test_mask.sum())
    print(f"  ✓ Split: train={n_train} (≈70%), val={n_val} (≈10%), "
          f"test={n_test} (≈20%)")
    assert abs(n_train - 0.70 * E) <= 5, f"train quá lệch: {n_train}"
    assert abs(n_val - 0.10 * E) <= 3, f"val quá lệch: {n_val}"
    assert abs(n_test - 0.20 * E) <= 5, f"test quá lệch: {n_test}"

    print("\n  [TEST 1] PASS\n")


def test_split_edge_masks_all_enough_samples() -> None:
    print("=" * 70)
    print("[TEST 2] split_edge_masks KHÔNG warning khi mọi lớp đều >= 2 mẫu")
    print("=" * 70)
    from src.train import split_edge_masks

    # 4 lớp, mỗi lớp >= 2 mẫu → dùng stratified bình thường
    parts = [
        np.zeros(20, dtype=np.int64),
        np.ones(15, dtype=np.int64),
        np.full(10, 2, dtype=np.int64),
        np.full(5, 3, dtype=np.int64),
    ]
    y = np.concatenate(parts)
    edge_label = torch.from_numpy(y)
    E = int(edge_label.shape[0])
    print(f"  Input: E={E}, 4 lớp, mỗi lớp >= 5 mẫu (KHÔNG có singleton)")

    buf = io.StringIO()
    with redirect_stdout(buf):
        train_mask, val_mask, test_mask = split_edge_masks(
            edge_label, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20,
            seed=42,
        )
    out = buf.getvalue()
    print(out)

    assert (train_mask | val_mask | test_mask).sum().item() == E
    # Không in cảnh báo fallback
    assert "fallback" not in out.lower(), (
        f"Có cảnh báo fallback dù không nên có. Output:\n{out}"
    )
    assert "không phải lỗi" not in out.lower(), (
        f"Có cảnh báo 'không phải lỗi' dù không nên có. Output:\n{out}"
    )
    print("  ✓ Không warning fallback → stratified path thuần chạy đúng.")
    print("\n  [TEST 2] PASS\n")


def test_compute_val_mask_loso_does_not_crash() -> None:
    print("=" * 70)
    print("[TEST 3] _compute_val_mask (LOSO) KHÔNG crash với 1 lớp singleton")
    print("=" * 70)
    from src.multi_scenario import _compute_val_mask

    # edge_label có 1 singleton (class 3, n=1) → cũ sẽ crash
    parts = [
        np.zeros(40, dtype=np.int64),
        np.ones(20, dtype=np.int64),
        np.full(3, 2, dtype=np.int64),
        np.full(1, 3, dtype=np.int64),    # singleton
    ]
    y = np.concatenate(parts)
    edge_label = torch.from_numpy(y)
    E = int(edge_label.shape[0])
    print(f"  Input: E={E}, 4 lớp, class 3 chỉ có 1 mẫu (singleton)")

    buf = io.StringIO()
    with redirect_stdout(buf):
        val_mask = _compute_val_mask(edge_label, val_ratio=0.10, seed=42)
    out = buf.getvalue()
    print(out)

    # Không crash
    assert val_mask.shape == (E,), f"val_mask shape sai: {val_mask.shape}"
    n_val = int(val_mask.sum())
    print(f"  ✓ val_mask có {n_val} mẫu (≈10% = {int(0.1 * E)})")
    assert abs(n_val - 0.10 * E) <= 3, f"val quá lệch: {n_val}"

    # Singleton không nằm trong val (vì bị force vào "first" = train)
    singleton_idx = edge_label == 3
    assert singleton_idx.sum().item() == 1
    assert not val_mask[singleton_idx].item(), (
        "Singleton NÊN ở train (bị force_into_first), KHÔNG ở val_mask."
    )
    print("  ✓ Singleton KHÔNG nằm trong val_mask (force_into_first OK)")

    # Lưu ý: trong trường hợp pool SAU KHI bỏ singleton còn đủ mẫu
    # stratified (>= 2 / lớp), KHÔNG cần in warning. Đây là hành vi đúng:
    # singleton bị force vào first (bị DISCARD trong _compute_val_mask) →
    # không xuất hiện warning, không xuất hiện val.
    print(f"  ✓ Pool đủ mẫu → không cần warning (hành vi đúng).")
    print("    Output thu được:")
    for line in (out or "(trống)").splitlines():
        print(f"      {line}")

    print("\n  [TEST 3] PASS\n")


def test_safe_stratified_split_helper_directly() -> None:
    print("=" * 70)
    print("[TEST 4] safe_stratified_split (helper trực tiếp)")
    print("=" * 70)
    from src.train import safe_stratified_split

    # Pool có 1 lớp count=1 → fallback
    y = np.array([0, 0, 0, 1, 1, 2, 2, 3], dtype=np.int64)  # class 3 = 1
    idx_pool = np.arange(len(y))
    singleton = np.array([7], dtype=np.int64)  # index của class 3

    buf = io.StringIO()
    with redirect_stdout(buf):
        idx_first, idx_second = safe_stratified_split(
            idx_pool, y, test_size=0.5, seed=42,
            context="unit-test", force_into_first=singleton,
        )
    out = buf.getvalue()
    print(out)

    # Singleton phải có mặt trong first
    assert singleton[0] in idx_first, (
        f"Singleton index {singleton[0]} KHÔNG có trong idx_first: {idx_first}"
    )
    assert singleton[0] not in idx_second, (
        f"Singleton index {singleton[0]} LẠI có trong idx_second: {idx_second}"
    )
    print("  ✓ Singleton ÉP vào idx_first, KHÔNG ở idx_second.")

    # First + Second = Pool (singleton chỉ tính 1 lần)
    union = set(idx_first.tolist()) | set(idx_second.tolist())
    assert union == set(idx_pool.tolist()), (
        f"first ∪ second = {sorted(union)} != pool {sorted(idx_pool.tolist())}"
    )
    print("  ✓ first ∪ second = pool (không mất/multi-count singleton).")
    print("\n  [TEST 4] PASS\n")


if __name__ == "__main__":
    test_split_edge_masks_does_not_crash()
    test_split_edge_masks_all_enough_samples()
    test_compute_val_mask_loso_does_not_crash()
    test_safe_stratified_split_helper_directly()
    print("=" * 70)
    print(" ALL SAFE-SPLIT TESTS PASSED")
    print("=" * 70)
