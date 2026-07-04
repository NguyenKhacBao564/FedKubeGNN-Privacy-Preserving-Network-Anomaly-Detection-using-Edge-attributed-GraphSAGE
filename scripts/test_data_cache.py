"""
test_data_cache.py — Smoke test cho ``DataCache`` trong ``src.run_experiments``.

Mục đích
--------
Xác nhận cache hoạt động đúng: lần thứ 2 gọi với CÙNG khóa dữ liệu →
CACHE HIT (KHÔNG load lại từ đĩa), lần đầu → CACHE MISS.

Kịch bản test
-------------
1. Tạo ``DataCache`` mới.
2. Gọi ``get_clean_dfs`` lần đầu → cache MISS (có log).
3. Gọi ``get_clean_dfs`` lần hai với CÙNG params → cache HIT (có log).
4. Kiểm tra ``print_stats()`` in đúng số hit/miss.
5. Kiểm tra graph cache: lần đầu build → MISS, lần hai → HIT.
6. Kiểm tra ``clear_graphs`` xóa tier 2 mà giữ tier 1.
7. Kiểm tra đổi ``cap_per_class`` → cache MISS (key khác).

Chạy
----
    .venv/bin/python scripts/test_data_cache.py
"""

from __future__ import annotations

import io
import os
import sys
import time
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import torch

REPO_ROOT = "/Users/nguyen_bao/Projects/AIproject/FedKube-IDS"
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# ---- Small data (34-1 có sẵn, chỉ cần 1 file) ----
LOG_34_1 = REPO_ROOT + "/data/CTU-IoT-Malware-Capture-34-1/conn.log.labeled"


def test_clean_dfs_cache_hit_miss() -> None:
    """Lần 1 = miss, lần 2 = hit, in đúng log."""
    from src.run_experiments import DataCache

    print("=" * 70)
    print("[TEST 1] clean_dfs: miss → hit")
    print("=" * 70)

    cache = DataCache()
    scenario_paths = {"34-1": LOG_34_1}
    cap = 2000
    chunksize = 100_000

    # Lần 1: cache MISS
    buf = io.StringIO()
    with redirect_stdout(buf):
        dfs1 = cache.get_clean_dfs(scenario_paths, cap_per_class=cap,
                                   chunksize=chunksize)
    out1 = buf.getvalue()
    assert "[CACHE MISS]" in out1, f"Lần 1 phải là MISS.\n{out1}"
    assert "34-1" in dfs1, "dfs1 phải chứa key '34-1'"
    print(f"  ✓ Lần 1: [CACHE MISS] — OK")

    # Lần 2: cache HIT
    buf = io.StringIO()
    with redirect_stdout(buf):
        dfs2 = cache.get_clean_dfs(scenario_paths, cap_per_class=cap,
                                   chunksize=chunksize)
    out2 = buf.getvalue()
    assert "[CACHE HIT]" in out2, f"Lần 2 phải là HIT.\n{out2}"
    # Object identity: dfs1 và dfs2 phải là cùng dict
    assert dfs1 is dfs2, "dfs2 phải là object reference giống dfs1"
    print(f"  ✓ Lần 2: [CACHE HIT] — cùng object reference ✓")

    # Print stats
    buf = io.StringIO()
    with redirect_stdout(buf):
        cache.print_stats()
    stats_out = buf.getvalue()
    print(stats_out)
    assert "1/2 hits" in stats_out or "1/2" in stats_out, (
        f"Stats phải ghi 1 hit / 2 total.\n{stats_out}"
    )
    print(f"  ✓ print_stats() in đúng hit/miss.")
    print(f"\n  [TEST 1] PASS\n")


def test_graph_cache_hit_miss() -> None:
    """Graph cache: lần build = MISS, lần 2 = HIT."""
    from src.run_experiments import DataCache

    print("=" * 70)
    print("[TEST 2] graph: miss → hit")
    print("=" * 70)

    cache = DataCache()
    scenario_name = "34-1"
    mode = "none"
    cap = 2000

    build_count = 0

    def build_fn():
        nonlocal build_count
        build_count += 1
        # Trả 1 Data object giả lập (PyG Data)
        g = torch_geometric_data_stub()
        return g

    # Lần 1: MISS
    buf = io.StringIO()
    with redirect_stdout(buf):
        g1 = cache.get_graph(scenario_name, mode, cap, build_fn)
    out1 = buf.getvalue()
    assert "[CACHE MISS]" in out1, f"Lần 1 phải là MISS.\n{out1}"
    assert build_count == 1, f"build_fn phải được gọi 1 lần, got {build_count}"
    print(f"  ✓ Lần 1: [CACHE MISS] — build_fn called once")

    # Lần 2: HIT
    buf = io.StringIO()
    with redirect_stdout(buf):
        g2 = cache.get_graph(scenario_name, mode, cap, build_fn)
    out2 = buf.getvalue()
    assert "[CACHE HIT]" in out2, f"Lần 2 phải là HIT.\n{out2}"
    assert g1 is g2, "g2 phải là cùng object reference"
    assert build_count == 1, (
        f"build_fn không được gọi thêm lần nào, got {build_count}"
    )
    print(f"  ✓ Lần 2: [CACHE HIT] — build_fn NOT called ✓")

    print(f"\n  [TEST 2] PASS\n")


def test_clear_graphs_preserves_clean() -> None:
    """clear_graphs() xóa tier 2, giữ nguyên tier 1."""
    from src.run_experiments import DataCache

    print("=" * 70)
    print("[TEST 3] clear_graphs() xóa graph, giữ clean_dfs")
    print("=" * 70)

    cache = DataCache()
    scenario_paths = {"34-1": LOG_34_1}
    cap = 2000
    scenario_name = "34-1"
    mode = "none"
    build_count = 0

    def build_fn():
        nonlocal build_count
        build_count += 1
        return torch_geometric_data_stub()

    # Load clean_dfs + 1 graph
    buf = io.StringIO()
    with redirect_stdout(buf):
        cache.get_clean_dfs(scenario_paths, cap_per_class=cap,
                            chunksize=100_000)
        cache.get_graph(scenario_name, mode, cap, build_fn)

    assert len(cache._clean_dfs) == 1
    assert len(cache._graphs) == 1

    # Clear graphs
    buf = io.StringIO()
    with redirect_stdout(buf):
        cache.clear_graphs()
    assert len(cache._clean_dfs) == 1, "clean_dfs phải giữ nguyên"
    assert len(cache._graphs) == 0, "graphs phải bị xóa"
    print(f"  ✓ clear_graphs() giữ tier 1, xóa tier 2.")

    # Gọi lại graph → build_fn phải được gọi lại (MISS)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cache.get_graph(scenario_name, mode, cap, build_fn)
    assert build_count == 2, (
        f"Sau clear, build_fn phải được gọi lại, got {build_count}"
    )
    print(f"  ✓ Sau clear, graph rebuild → MISS (build_fn called again).")
    print(f"\n  [TEST 3] PASS\n")


def test_different_cap_is_miss() -> None:
    """Đổi cap_per_class → key khác → cache MISS."""
    from src.run_experiments import DataCache

    print("=" * 70)
    print("[TEST 4] cap_per_class khác → cache MISS (key riêng biệt)")
    print("=" * 70)

    cache = DataCache()
    scenario_paths = {"34-1": LOG_34_1}
    chunksize = 100_000

    buf = io.StringIO()
    with redirect_stdout(buf):
        cache.get_clean_dfs(scenario_paths, cap_per_class=2000,
                            chunksize=chunksize)
    assert len(cache._clean_dfs) == 1

    # Gọi với cap_per_class KHÁC → MISS, key mới
    buf = io.StringIO()
    with redirect_stdout(buf):
        cache.get_clean_dfs(scenario_paths, cap_per_class=500,
                            chunksize=chunksize)
    assert len(cache._clean_dfs) == 2, (
        f"Phải có 2 key riêng biệt trong _clean_dfs, got {len(cache._clean_dfs)}"
    )
    print(f"  ✓ cap=2000 và cap=500 tạo 2 key riêng → cache miss riêng.")
    print(f"\n  [TEST 4] PASS\n")


def test_cache_hits_actually_skip_load() -> None:
    """Real timing: lần 2 nhanh hơn rõ rệt so với lần 1."""
    from src.run_experiments import DataCache

    print("=" * 70)
    print("[TEST 5] Timing: lần 2 (HIT) nhanh hơn lần 1 (MISS)")
    print("=" * 70)

    scenario_paths = {"34-1": LOG_34_1}
    cap = 2000
    chunksize = 100_000

    # Lần 1: MISS (load từ đĩa)
    cache = DataCache()
    t1_start = time.perf_counter()
    cache.get_clean_dfs(scenario_paths, cap_per_class=cap,
                        chunksize=chunksize)
    t1 = time.perf_counter() - t1_start

    # Lần 2: HIT (từ bộ nhớ)
    t2_start = time.perf_counter()
    cache.get_clean_dfs(scenario_paths, cap_per_class=cap,
                        chunksize=chunksize)
    t2 = time.perf_counter() - t2_start

    print(f"  Lần 1 (MISS): {t1:.3f}s")
    print(f"  Lần 2 (HIT) : {t2:.3f}s")
    if t1 > 0.01:
        speedup = t1 / max(t2, 0.0001)
        print(f"  Speedup      : {speedup:.1f}x")
        assert t2 < t1, f"HIT ({t2:.3f}s) phải nhanh hơn MISS ({t1:.3f}s)"
    print(f"  ✓ HIT nhanh hơn MISS (như kỳ vọng)")
    print(f"\n  [TEST 5] PASS\n")


# ---- Helper: stub PyG Data object ----

def torch_geometric_data_stub():
    """Tạo 1 object giả lập ``torch_geometric.data.Data`` (chỉ đủ field
    cần cho ``DataCache.get_graph`` log in)."""
    from torch_geometric.data import Data
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_label = torch.tensor([0, 1], dtype=torch.long)
    node_feature = torch.ones(2, 3)
    return Data(
        x=node_feature,
        edge_index=edge_index,
        edge_label=edge_label,
        num_nodes=2,
        feature_dim=3,
        num_classes=2,
    )


if __name__ == "__main__":
    test_clean_dfs_cache_hit_miss()
    test_graph_cache_hit_miss()
    test_clear_graphs_preserves_clean()
    test_different_cap_is_miss()
    test_cache_hits_actually_skip_load()

    print("=" * 70)
    print(" ALL DATACACHE TESTS PASSED")
    print("=" * 70)
