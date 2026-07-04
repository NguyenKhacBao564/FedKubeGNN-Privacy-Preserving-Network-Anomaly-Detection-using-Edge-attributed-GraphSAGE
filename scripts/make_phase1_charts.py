#!/usr/bin/env python3
"""Tạo biểu đồ tổng hợp cho PHASE1_REPORT.md.

Đọc artifacts/phase1_results/results_summary.csv, xuất ra:
  - docs/figures/pooled_phase_b_models.png  : bar chart so sánh 5 model trên pooled
  - docs/figures/loso_phase_b_models.png    : bar chart so sánh 5 model trên LOSO
  - docs/figures/loso_egraphsage_by_scenario.png : bar chart LOSO theo từng scenario
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # không cần display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "artifacts/phase1_results/results_summary.csv"
FIG_DIR = REPO / "docs/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    return df


def chart_pooled_phase_b(df: pd.DataFrame) -> Path:
    """Bar chart: 5 model trên pooled (macro-F1)."""
    sub = df[df["protocol"] == "pooled"]
    sub = sub[sub["scenario"] == "POOLED"]
    sub = sub.sort_values("macro_f1", ascending=False)

    out = FIG_DIR / "pooled_phase_b_models.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71" if m == "egraphsage" else "#3498db" for m in sub["model"]]
    bars = ax.bar(sub["model"], sub["macro_f1"], color=colors, edgecolor="black")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Phase B (Pooled) — 5 models, mode=class_weight\n(n_scenarios=6)")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.8773, color="#2ecc71", linestyle="--", alpha=0.5,
               label="E-GraphSAGE: 0.8773")

    # Annotate bars
    for bar, val in zip(bars, sub["macro_f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def chart_loso_phase_b(df: pd.DataFrame) -> Path:
    """Bar chart: 5 model trên LOSO (mean macro-F1)."""
    sub = df[df["protocol"] == "loso"]
    sub = sub[sub["scenario"] == "MEAN"]
    sub = sub.sort_values("macro_f1", ascending=False)

    out = FIG_DIR / "loso_phase_b_models.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if m == "egraphsage" else "#95a5a6" for m in sub["model"]]
    bars = ax.bar(sub["model"], sub["macro_f1"], color=colors, edgecolor="black")
    ax.set_ylabel("Mean Macro-F1")
    ax.set_title("Phase B (LOSO) — 5 models, mean over 6 held-out rounds")
    ax.set_ylim(0, 0.5)
    ax.axhline(y=0.2334, color="#e74c3c", linestyle="--", alpha=0.5,
               label="E-GraphSAGE: 0.2334")

    for bar, val in zip(bars, sub["macro_f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def chart_loso_egraphsage_by_scenario(df: pd.DataFrame) -> Path:
    """Bar chart: LOSO theo từng scenario held-out cho E-GraphSAGE."""
    sub = df[df["protocol"] == "loso"]
    sub = sub[sub["model"] == "egraphsage"]
    sub = sub[sub["imbalance_mode"] == "none"]
    sub = sub[sub["scenario"] != "MEAN"]

    # Sắp xếp theo macro_f1 giảm dần
    sub = sub.sort_values("macro_f1", ascending=True)

    out = FIG_DIR / "loso_egraphsage_by_scenario.png"

    fig, ax = plt.subplots(figsize=(9, 5))
    # Màu: đỏ = có private classes (n_unseen > 0), xanh = không có
    colors = ["#e74c3c" if n > 0 else "#3498db"
              for n in sub["n_unseen_in_train"]]
    bars = ax.barh(sub["scenario"], sub["macro_f1"], color=colors,
                   edgecolor="black")

    ax.set_xlabel("Macro-F1")
    ax.set_title("LOSO E-GraphSAGE — by held-out scenario\n"
                 "(red = has private classes unseen in train)")
    ax.set_xlim(0, 0.5)

    # Annotate bars
    for bar, val, n_unseen in zip(bars, sub["macro_f1"], sub["n_unseen_in_train"]):
        label = f"{val:.3f}" + (f"  (n_unseen={int(n_unseen)})"
                                 if n_unseen > 0 else "")
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                label, ha="left", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def chart_pooled_vs_loso_egraphsage(df: pd.DataFrame) -> Path:
    """So sánh trực quan pooled vs LOSO cho E-GraphSAGE — thể hiện khoảng cách lớn."""
    pooled = df[(df["protocol"] == "pooled") & (df["scenario"] == "POOLED")
                & (df["model"] == "egraphsage") & (df["imbalance_mode"] == "class_weight")]
    loso = df[(df["protocol"] == "loso") & (df["scenario"] == "MEAN")
              & (df["model"] == "egraphsage") & (df["imbalance_mode"] == "none")]

    pooled_val = float(pooled["macro_f1"].iloc[0]) if len(pooled) else 0
    loso_val = float(loso["macro_f1"].iloc[0]) if len(loso) else 0

    out = FIG_DIR / "pooled_vs_loso_egraphsage.png"
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = ["Pooled\n(transductive)", "LOSO\n(inductive, mean)"]
    values = [pooled_val, loso_val]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black")

    ax.set_ylabel("Macro-F1")
    ax.set_title("E-GraphSAGE: Pooled vs LOSO\n"
                 f"(gap = {pooled_val - loso_val:.3f})")
    ax.set_ylim(0, 1.0)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Vẽ mũi tên thể hiện khoảng cách
    ax.annotate("",
                xy=(1, loso_val + 0.05), xytext=(1, pooled_val - 0.02),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(1.25, (pooled_val + loso_val) / 2, "Inductive gap",
            ha="left", va="center", fontsize=10, color="black")

    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    p1 = chart_pooled_phase_b(df)
    print(f"Wrote: {p1}")
    p2 = chart_loso_phase_b(df)
    print(f"Wrote: {p2}")
    p3 = chart_loso_egraphsage_by_scenario(df)
    print(f"Wrote: {p3}")
    p4 = chart_pooled_vs_loso_egraphsage(df)
    print(f"Wrote: {p4}")


if __name__ == "__main__":
    main()