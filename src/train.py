"""
train.py — Vòng train device-agnostic cho edge classification.

Nguyên tắc bắt buộc (CLAUDE.md mục 2):

    • Device-agnostic: `device = "cuda" if torch.cuda.is_available() else "cpu"`,
      mọi tensor/model đưa vào `.to(device)`. KHÔNG hardcode `.cuda()`.
    • Reproducible: set seed numpy + torch + random NGAY ĐẦU chương trình
      (seed đọc từ config.yaml).
    • Đọc file trung gian từ artifacts/, KHÔNG xử lý lại dataset từ đầu —
      preprocess đã chạy 1 lần trên CPU rồi.
    • Checkpoint model tốt nhất (theo macro-F1 trên val) vào checkpoints/.
    • Hỗ trợ class-weighted CrossEntropyLoss (đọc weights từ imbalance.py).
    • Có early stopping & grad clipping để tránh exploding gradient.
"""


def set_seed(seed: int):
    """Set seed cho numpy + torch + random (placeholder)."""
    raise NotImplementedError("train.set_seed sẽ triển khai ở task sau.")


def get_device():
    """Trả về 'cuda' nếu có, ngược lại 'cpu' (placeholder)."""
    raise NotImplementedError("train.get_device sẽ triển khai ở task sau.")


def load_graph(artifact_path: str):
    """Đọc file .pt trả về torch_geometric.data.Data (placeholder)."""
    raise NotImplementedError("train.load_graph sẽ triển khai ở task sau.")


def train_one_epoch(model, data, optimizer, criterion, device, grad_clip: float = 1.0):
    """Train 1 epoch full-batch; trả về loss trung bình (placeholder)."""
    raise NotImplementedError("train.train_one_epoch sẽ triển khai ở task sau.")


def evaluate(model, data, criterion, device, mask=None):
    """Đánh giá trên 1 tập (train/val/test); trả về (loss, logits, labels) (placeholder)."""
    raise NotImplementedError("train.evaluate sẽ triển khai ở task sau.")


def run_training(config_path: str):
    """Vòng train đầy đủ: đọc config → load graph → train → checkpoint (placeholder)."""
    raise NotImplementedError("train.run_training sẽ triển khai ở task sau.")