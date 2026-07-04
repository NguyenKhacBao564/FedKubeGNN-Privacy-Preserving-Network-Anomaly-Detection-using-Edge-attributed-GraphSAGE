"""
test_egraphsage_layer.py — Sanity check cho 1 layer EGraphSAGELayer đơn lẻ.

Mục đích (Task 1.9, ★★★★★):
    Trước khi ghép thành model đầy đủ, xác nhận cơ chế cốt lõi của
    `EGraphSAGELayer` đúng như thiết kế:

    1. forward cho shape đúng [N, out_dim].
    2. Không sinh NaN.
    3. `edge_attr` THỰC SỰ được dùng trong message (nếu không → vô dụng,
       giống GraphSAGE gốc).
    4. message dùng `x_j` (node NGUỒN theo `edge_index[0]`), KHÔNG dùng
       `x_i` (node ĐÍCH). Đồ thị bất đối xứng có hướng → chứng minh
       được bằng cách đổi embedding của 1 node nguồn và quan sát
       embedding node đích tương ứng.

Chạy:
    /Users/nguyen_bao/Projects/AIproject/FedKube-IDS/.venv/bin/python \\
        scripts/test_egraphsage_layer.py
"""

import sys
import torch

# Import layer (không qua __init__ chính của model đầy đủ).
sys.path.insert(0, "/Users/nguyen_bao/Projects/AIproject/FedKube-IDS")
from src.model import EGraphSAGELayer


def main():
    torch.manual_seed(42)  # reproducible

    N = 4
    E = 5
    in_dim = 1
    edge_dim = 3
    out_dim = 8

    # ---- Mock graph ----
    # Đồ thị CÓ HƯỚNG BẤT ĐỐI XỨNG để phân biệt source vs target.
    # Layout:
    #   0 -> 1         (out[0] = source của (0,1); out[1] = target duy nhất của (0,1))
    #   1 -> 2
    #   2 -> 3
    #   3 -> 0         (tạo chu trình 0→1→2→3→0)
    #   1 -> 3         (out[1] có 2 outgoing; node 3 có 2 incoming)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 1],   # nguồn
        [1, 2, 3, 0, 3],   # đích
    ], dtype=torch.long)

    # Node init: vector hằng all-ones (theo tinh thần E-GraphSAGE — phân biệt
    # nằm ở cạnh, không ở node).
    x = torch.ones(N, in_dim)
    edge_attr = torch.rand(E, edge_dim)

    layer = EGraphSAGELayer(in_dim=in_dim, out_dim=out_dim, edge_dim=edge_dim)
    layer.eval()  # tắt dropout-like effects (không có nhưng cho chắc)

    print(f"Layer: in_dim={in_dim}, out_dim={out_dim}, edge_dim={edge_dim}, "
          f"aggr={layer.aggr}, flow={layer.flow}")
    print(f"Mock graph: N={N}, E={E}")
    print(f"edge_index =\n{edge_index.numpy()}")
    print()

    # ============================================================
    # [1] Shape forward pass
    # ============================================================
    out = layer(x, edge_index, edge_attr)
    print(f"[1] out.shape = {tuple(out.shape)}  (kỳ vọng: ({N}, {out_dim}))")
    assert out.shape == (N, out_dim), \
        f"shape sai: {out.shape} != ({N}, {out_dim})"
    print("    PASS")
    print()

    # ============================================================
    # [2] Không có NaN
    # ============================================================
    has_nan = torch.isnan(out).any().item()
    print(f"[2] NaN trong output? {has_nan}")
    assert not has_nan, "output chứa NaN — kiểm tra Linear init hoặc ReLU"
    print("    PASS")
    print()

    # ============================================================
    # [3] edge_attr thật sự được dùng trong message
    #     Nếu layer VÔ TÌNH bỏ qua edge_attr, out sẽ giống hệt khi
    #     đổi edge_attr (vì chỉ x và edge_index không đổi).
    # ============================================================
    edge_attr_alt = torch.rand(E, edge_dim) * 100.0  # rất khác về scale
    out_alt = layer(x, edge_index, edge_attr_alt)
    diff_attr = (out - out_alt).abs().max().item()
    print(f"[3] max |out(edge_attr) - out(edge_attr_alt)| = {diff_attr:.6f}")
    assert diff_attr > 0.0, (
        "edge_attr bị BỎ QUA — output giống nhau dù đổi toàn bộ edge_attr. "
        "Kiểm tra message() có thật sự concat edge_attr."
    )
    print("    PASS — edge_attr có ảnh hưởng đến output.")
    print()

    # ============================================================
    # [4] message dùng x_j (SOURCE) chứ không x_i (TARGET)
    #
    #     Logic:
    #     - Đổi embedding của node 0 từ 1.0 -> 7.0.
    #     - Node 1 là target DUY NHẤT của edge 0→1, và input x[1] không đổi.
    #     - Nếu message lấy x_j (source=0) → thông điệp tới node 1 thay đổi
    #       → aggr_out[1] thay đổi → out[1] THAY ĐỔI.
    #     - Nếu message (sai) lấy x_i (target=1) → thông điệp không đổi
    #       (vì target=1 mang x[1] không đổi) → out[1] KHÔNG đổi.
    # ============================================================
    x_alt = x.clone()
    x_alt[0] = torch.tensor([7.0])  # chỉ đổi node 0
    out_src = layer(x_alt, edge_index, edge_attr)

    diff_out_0 = (out[0] - out_src[0]).abs().max().item()  # TRIVIAL: own x in update
    diff_out_1 = (out[1] - out_src[1]).abs().max().item()  # QUAN TRỌNG — node 1 là target của 0→1

    print(f"[4] Đổi x[0] từ 1.0 -> 7.0:")
    print(f"    |Δout[0]| = {diff_out_0:.6f}  (trivial: chính x[0] cũng xuất hiện trong update)")
    print(f"    |Δout[1]| = {diff_out_1:.6f}  (target duy nhất của edge 0→1; nếu >0 → dùng x_j)")
    assert diff_out_1 > 0.0, (
        "out[1] KHÔNG đổi khi đổi x[0]. Có nghĩa message đang dùng x_i (target) "
        "thay vì x_j (source). SAI cơ chế E-GraphSAGE."
    )
    print("    PASS — embedding node 1 phản ứng với thay đổi của node nguồn 0.")
    print()

    # ============================================================
    # Tổng kết
    # ============================================================
    print("=" * 60)
    print("ALL 4 CHECKS PASSED.")
    print("EGraphSAGELayer (Task 1.9) đã đúng cơ chế cốt lõi.")
    print("Bước tiếp theo: ghép thành EGraphSAGE đầy đủ trong src/model.py.")
    print("=" * 60)


if __name__ == "__main__":
    main()
