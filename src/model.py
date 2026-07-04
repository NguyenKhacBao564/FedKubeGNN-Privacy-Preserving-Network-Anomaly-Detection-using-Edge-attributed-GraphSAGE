"""
model.py — E-GraphSAGE (chính) + baselines GCN / GraphSAGE cho edge classification.

E-GraphSAGE (Task 1.9, độ khó ★★★★★):
    PyG KHÔNG có sẵn. Phải tự viết lớp kế thừa
    `torch_geometric.nn.MessagePassing`. Trong hàm `message()` GHÉP (concat)
    đặc trưng node nguồn + đặc trưng cạnh rồi mới aggregate. Đây là phần
    tinh vi nhất — test shape từng bước khi triển khai.

Baselines (Task 1.11):
    • GCN     — dùng `GCNConv` có sẵn; ghép embedding 2 đầu cạnh + edge_attr
                trước classifier head.
    • GraphSAGE — dùng `SAGEConv` có sẵn; tương tự GCN.

Phương án B (nếu E-GraphSAGE quá khó):
    Dùng `SAGEConv` sẵn có + nối đặc trưng cạnh vào đặc trưng node TRƯỚC khi
    vào layer. Ghi rõ trong báo cáo đây là bản "gần đúng" của E-GraphSAGE.

Lưu ý hạ tầng: mọi layer phải đặt trên `device` do train.py truyền vào;
không hardcode `.cuda()`.
"""


class EGraphSAGEConv(torch.nn.Module):
    """Lớp MessagePassing tự viết: message = concat(x_src, edge_attr) (placeholder)."""
    raise NotImplementedError("EGraphSAGEConv sẽ triển khai ở task sau.")


class EGraphSAGE(torch.nn.Module):
    """Mô hình E-GraphSAGE đầy đủ cho edge classification (placeholder)."""
    raise NotImplementedError("EGraphSAGE sẽ triển khai ở task sau.")


class GCNBaseline(torch.nn.Module):
    """Baseline GCN cho edge classification (placeholder)."""
    raise NotImplementedError("GCNBaseline sẽ triển khai ở task sau.")


class GraphSAGEBaseline(torch.nn.Module):
    """Baseline GraphSAGE cho edge classification (placeholder)."""
    raise NotImplementedError("GraphSAGEBaseline sẽ triển khai ở task sau.")


def build_model(model_type: str, in_node_dim: int, in_edge_dim: int, num_classes: int,
                hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.5):
    """Factory: trả về model phù hợp với `model_type` (placeholder)."""
    raise NotImplementedError("build_model sẽ triển khai ở task sau.")