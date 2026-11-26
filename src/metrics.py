import numpy as np

def compute_mining_metrics(est_counts: np.ndarray, true_counts: np.ndarray, n_txns: int, theta: float = 0.01) -> dict:
    min_cnt = theta * n_txns
    true_freq = true_counts >= min_cnt
    est_freq = est_counts >= min_cnt

    tp = int(np.sum(true_freq & est_freq))
    fp = int(np.sum((~true_freq) & est_freq))
    fn = int(np.sum(true_freq & (~est_freq)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    support_diffs = np.abs((est_counts / n_txns) - (true_counts / n_txns))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae_support": float(np.mean(support_diffs)),
        "max_support_error": float(np.max(support_diffs)),
        "survivors": int(np.sum(est_freq)),
        "true_frequent_count": int(np.sum(true_freq)),
    }

def evaluate_two_pass_verification(est_counts: np.ndarray, true_counts: np.ndarray, n_txns: int, theta: float = 0.01) -> dict:
    min_cnt = theta * n_txns
    survivor_mask = est_counts >= min_cnt
    true_freq = true_counts >= min_cnt
    verified_freq = survivor_mask & true_freq

    tp = int(np.sum(verified_freq))
    fn = int(np.sum(true_freq & (~verified_freq)))
    precision = 1.0 if tp > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    pruned_pct = (1.0 - (np.sum(survivor_mask) / len(true_counts))) * 100.0
    return {
        "verified_precision": precision,
        "verified_recall": recall,
        "verified_f1": f1,
        "candidates_pruned_pct": float(pruned_pct),
        "survivor_count": int(np.sum(survivor_mask)),
    }
