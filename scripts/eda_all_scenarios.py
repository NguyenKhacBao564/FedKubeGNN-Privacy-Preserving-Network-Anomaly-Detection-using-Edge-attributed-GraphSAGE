"""
eda_all_scenarios.py — EDA toàn bộ scenario TRƯỚC khi train (Giai đoạn 1).

Mục đích
--------
Chạy TRƛC khi train thật trên GPU vast.ai để:
  1. Kiểm tra dataset đã tải về ĐỦ chưa (thiếu scenario nào → tải lại).
  2. Nắm phân bố ``detailed-label`` / ``label`` của TỪNG scenario → quyết
     định ``cap_per_class`` hợp lý cho thí nghiệm chính.
  3. Phát hiện lớp PRIVATE (chỉ xuất hiện ở 1 scenario) — ảnh hưởng trực
     tiếp đến LOSO (lớp đó F1=0 trong held-out round tương ứng).
  4. Phát hiện lớp HIẾM toàn cục → biết cần ``class_weight`` / ``undersample``.

KHÔNG train gì. Chỉ đọc file + in bảng + ghi 1 CSV tóm tắt.

Đầu vào
-------
- ``config.yaml`` (block ``experiments.scenarios``): danh sách scenario +
  đường dẫn ``conn.log.labeled``.
- ``--config``: đường dẫn config (mặc định ``config.yaml``).
- ``--cap``: cap per-class cho mỗi scenario khi đọc (mặc định 50000 — đủ
  để thấy phân bố đại diện, RAM ổn).

Đầu ra
------
- Bảng in ra console: tóm tắt phân bố + ma trận hiện diện + gợi ý cap.
- ``artifacts/eda_summary.csv``: bảng tổng hợp (1 dòng / scenario × lớp).

Chạy
----
    /Users/nguyen_bao/Projects/AIproject/FedKube-IDS/.venv/bin/python \\
        scripts/eda_all_scenarios.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

# ---- Repo root + sys.path ---------------------------------------------------
_REPO_ROOT = "/Users/nguyen_bao/Projects/AIproject/FedKube-IDS"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


logger = logging.getLogger("eda_all_scenarios")


# ============================================================================
# Helpers
# ============================================================================

def _print_header(s: str, ch: str = "=") -> None:
    line = ch * 70
    print()
    print(line)
    print(f" {s}")
    print(line)


def _load_scenario_with_cap(
    path: str,
    cap_per_class: int,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Load + clean 1 file ``conn.log.labeled`` với cap per-class cao (để EDA
    có mẫu đại diện mà không OOM với scenario lớn).

    Tái sử dụng ``src.multi_scenario.load_all_scenarios`` (đã có sẵn
    logic cap + chunk).
    """
    from src.multi_scenario import load_all_scenarios

    return load_all_scenarios(
        {"_one": path}, cap_per_class=cap_per_class, chunksize=chunksize,
    )["_one"]


def _resolve_scenarios(cfg_path: str) -> Dict[str, str]:
    """Đọc ``config['experiments']['scenarios']`` → ``{name: path}``."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    out: Dict[str, str] = {}
    for sc in (cfg.get("experiments", {}) or {}).get("scenarios", []) or []:
        n, p = sc.get("name"), sc.get("path")
        if n and p:
            out[str(n)] = str(p)
    if not out:
        raise FileNotFoundError(
            "Không tìm thấy experiments.scenarios trong config.yaml."
        )
    return out


def _suggest_cap(dist: pd.Series) -> Tuple[Optional[int], str]:
    """
    Đề xuất ``cap_per_class`` dựa trên phân bố thật.

    Quy tắc đơn giản:
      - Nếu lớp đa số / lớp hiếm ≤ 100×: cap = min(median * 5, max).
      - Nếu > 100×: cap = max(median, 5000) (đủ đa dạng, RAM ổn).
      - Nếu dataset quá nhỏ (tổng < 1000): None = đọc nguyên.

    Trả về (cap_suggested, lý_do).
    """
    if dist.empty or int(dist.sum()) < 1000:
        return None, "dataset quá nhỏ → đọc nguyên"

    n_classes = int(dist.shape[0])
    max_c = int(dist.iloc[0])  # đa số (đã sort)
    min_c = int(dist.iloc[-1])  # hiếm nhất
    median_c = float(dist.median())
    ratio = (max_c / max(min_c, 1)) if min_c > 0 else float("inf")

    if ratio <= 100.0:
        # Mất cân bằng vừa phải → cap theo median * 5 (giữ lớp hiếm).
        cap = int(min(median_c * 5, max_c))
        cap = max(cap, 1000)  # tối thiểu 1000 dòng / lớp
        reason = (
            f"ratio max/min={ratio:.1f}× (vừa phải) → "
            f"cap≈{cap} (median*5, min=1000)"
        )
    else:
        # Mất cân bằng cực đoan → cap = max(median, 5000) — chừng đủ để
        # đa dạng lớp hiếm mà RAM không phình.
        cap = int(max(median_c, 5000.0))
        cap = max(cap, 1000)
        reason = (
            f"ratio max/min={ratio:.0f}× (cực đoan) → "
            f"cap={cap} (max(median,5000))"
        )

    if n_classes <= 2:
        # Binary — cap có thể cao hơn.
        cap = max(cap, 10000)
        reason += f"; n_classes={n_classes} (binary → nâng cap tối thiểu 10000)"

    return cap, reason


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EDA toàn bộ scenario trước khi train — KHÔNG train. "
            "In phân bố label, ma trận hiện diện lớp, gợi ý cap_per_class."
        ),
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Đường dẫn config.yaml (mặc định: config.yaml).",
    )
    parser.add_argument(
        "--cap", type=int, default=50_000,
        help=(
            "Số flow TỐI ĐA mỗi lớp khi đọc để EDA (mặc định 50000). "
            "Đặt nhỏ nếu RAM hẹp; None nghĩa là đọc nguyên (KHÔNG khuyến nghị "
            "với 39-1 ~10GB)."
        ),
    )
    parser.add_argument(
        "--out", default="artifacts/eda_summary.csv",
        help="File CSV tổng hợp đầu ra (mặc định: artifacts/eda_summary.csv).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    cfg_path = args.config
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Thiếu config: {cfg_path}")

    scenario_paths = _resolve_scenarios(cfg_path)
    print("=" * 70)
    print(" EDA  ·  IoT-23 scenarios (trước khi train)")
    print("=" * 70)
    print(f"  config      : {os.path.abspath(cfg_path)}")
    print(f"  #scenarios  : {len(scenario_paths)}")
    print(f"  cap_per_class (EDA): {args.cap}")
    print(f"  out CSV     : {args.out}")
    print()

    # ---- 1) Load + clean từng scenario ----
    _print_header("[1] Load + clean từng scenario (chunked, cap=" + str(args.cap) + ")")
    all_dfs: Dict[str, pd.DataFrame] = {}
    for name, path in sorted(scenario_paths.items()):
        if not os.path.isfile(path):
            print(f"  [{name}]  ⚠ THIẾU FILE: {path}")
            print(f"            → chạy lại scripts/download_all.sh")
            continue
        try:
            df = _load_scenario_with_cap(
                path, cap_per_class=args.cap, chunksize=200_000,
            )
            all_dfs[name] = df
            n_class = df["detailed-label"].nunique()
            print(
                f"  [{name:<25s}]  rows={df.shape[0]:>9,}   "
                f"#class={n_class:>3d}   "
                f"path={path}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [{name}]  ⚠ LỖI load: {e}")

    if not all_dfs:
        raise SystemExit("Không scenario nào load được. Kiểm tra file path.")

    # ---- 2) Phân bố label per-scenario ----
    _print_header("[2] Phân bố detailed-label / label per scenario")
    rows: List[Dict[str, Any]] = []
    for name in sorted(all_dfs.keys()):
        df = all_dfs[name]
        n_total = int(df.shape[0])
        print()
        print(f"--- {name}  (n={n_total:,} rows) ---")
        if "label" in df.columns:
            vc_label = df["label"].astype(str).value_counts(dropna=False)
            print(f"  [label nhị phân]")
            for k, v in vc_label.items():
                pct = (100.0 * v / max(n_total, 1))
                print(f"    {k:<15s}  {v:>10,}  ({pct:5.2f}%)")
        vc_det = df["detailed-label"].astype(str).value_counts(dropna=False)
        print(f"  [detailed-label]  ({vc_det.shape[0]} lớp)")
        for k, v in vc_det.items():
            pct = (100.0 * v / max(n_total, 1))
            print(f"    {k:<40s}  {v:>10,}  ({pct:5.2f}%)")
            rows.append({
                "scenario": name,
                "detailed-label": str(k),
                "count": int(v),
                "percent": float(pct),
            })

    # ---- 3) Ma trận hiện diện lớp × scenario ----
    _print_header("[3] MA TRẬN HIỆN DIỆN LỚP × SCENARIO")
    all_classes = sorted({
        c for df in all_dfs.values() for c in df["detailed-label"].astype(str).unique()
    })
    sc_names = sorted(all_dfs.keys())
    name_w = max((len(n) for n in sc_names), default=8)
    header_cells = [f"{n[:name_w]:>{name_w}s}" for n in sc_names]
    print(f"  Tổng số lớp union: {len(all_classes)}")
    print(f"  Số scenario     : {len(sc_names)}")
    print()
    print(f"  {'detailed-label':<40s}  " + "  ".join(header_cells))
    print("  " + "-" * (40 + 2 + (name_w + 2) * len(sc_names)))
    private_classes: List[str] = []
    rare_global_classes: List[str] = []
    for c in all_classes:
        cells = []
        n_present = 0
        for n in sc_names:
            sub = all_dfs[n]
            cnt = int((sub["detailed-label"].astype(str) == c).sum())
            if cnt > 0:
                cells.append(f"{cnt:>{name_w},d}")
                n_present += 1
            else:
                cells.append(f"{'.':>{name_w}s}")
        line = f"  {c:<40s}  " + "  ".join(cells)
        if n_present == 1:
            private_classes.append(c)
            line += "   ← PRIVATE (chỉ ở 1 scenario, ảnh hưởng LOSO)"
        print(line)
    print()
    # Tổng per-class
    print("  Tổng số flow per lớp (sum mọi scenario):")
    for c in all_classes:
        total = sum(
            int((all_dfs[n]["detailed-label"].astype(str) == c).sum())
            for n in sc_names if n in all_dfs
        )
        print(f"    {c:<40s}  {total:>10,}")
        if 0 < total < 1000:
            rare_global_classes.append(c)
    if private_classes:
        print()
        print(f"  ⚠ {len(private_classes)} LỚP PRIVATE "
              f"(chỉ xuất hiện ở 1 scenario):")
        for c in private_classes:
            print(f"    - {c}")
        print(
            "    → Khi LOSO dùng scenario đó làm held-out, F1 lớp này = 0.\n"
            "      Báo cáo phải nói rõ đây là giới hạn inductive, "
            "KHÔNG phải lỗi model."
        )
    if rare_global_classes:
        print()
        print(f"  ⚠ {len(rare_global_classes)} LỚP HIẾM TOÀN CỤC "
              f"(< 1000 flow tổng):")
        for c in rare_global_classes:
            print(f"    - {c}")
        print(
            "    → Cân nhắc dùng class_weight hoặc undersample; "
            "nếu không, model sẽ gần như bỏ qua lớp này."
        )

    # ---- 4) Đề xuất cap_per_class ----
    _print_header("[4] ĐỀ XUẤT cap_per_class per scenario")
    for name in sorted(all_dfs.keys()):
        df = all_dfs[name]
        vc = df["detailed-label"].astype(str).value_counts(dropna=False)
        cap, reason = _suggest_cap(vc)
        print(f"  [{name:<25s}]  cap_per_class = {cap}   ← {reason}")

    # ---- 5) Save CSV ----
    _print_header("[5] Save CSV")
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(out_path, index=False)
    print(f"  → {out_path}  ({df_summary.shape[0]} dòng)")
    print()
    print("=" * 70)
    print(" EDA DONE. Bước tiếp theo:")
    print("   1. Nếu thiếu file scenario → chạy lại scripts/download_all.sh")
    print("   2. Chốt cap_per_class (tham khảo gợi ý ở mục [4])")
    print("   3. Cập nhật config.yaml block experiments.cap_per_class nếu cần")
    print("   4. Chạy bash scripts/run_full_gpu.sh trên GPU vast.ai")
    print("=" * 70)


if __name__ == "__main__":
    main()