"""
evaluate.py — Đánh giá model edge classification đã train.

Quy tắc đánh giá (CLAUDE.md mục 8):

    • Do mất cân bằng cực đoan, accuracy tổng thể GẦN NHƯ VÔ NGHĨA (đoán
      toàn lớp đa số vẫn cao). Chỉ số chính = MACRO-F1 và F1 THEO TỪNG LỚP.
    • Bắt buộc có CONFUSION MATRIX, chú ý riêng các lớp hiếm.
    • Cũng ghi Precision/Recall theo lớp và AUC-ROC (one-vs-rest) nếu áp dụng.

Input : checkpoint + file .pt của đồ thị (test split).
Output: artifacts/metrics.json + artifacts/confusion_matrix.png.
"""


def compute_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """Trả về dict gồm macro-F1, per-class P/R/F1, accuracy, AUC-ROC (placeholder)."""
    raise NotImplementedError("evaluate.compute_metrics sẽ triển khai ở task sau.")


def plot_confusion_matrix(y_true, y_pred, class_names, out_path: str):
    """Vẽ & lưu confusion matrix ra file PNG (placeholder)."""
    raise NotImplementedError("evaluate.plot_confusion_matrix sẽ triển khai ở task sau.")


def save_metrics_json(metrics: dict, out_path: str):
    """Ghi dict metrics ra file JSON (placeholder)."""
    raise NotImplementedError("evaluate.save_metrics_json sẽ triển khai ở task sau.")


def run_evaluation(config_path: str, checkpoint_path: str):
    """Pipeline đầy đủ: load model + graph → predict → metrics + confusion matrix (placeholder)."""
    raise NotImplementedError("evaluate.run_evaluation sẽ triển khai ở task sau.")