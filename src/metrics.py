"""
Evaluation metrics for HS classification: Top-K accuracy, F1, latency.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class StageMetrics:
    """Evaluation metrics for a single stage."""
    stage_name: str
    n_samples: int = 0
    n_classes: int = 0

    # Accuracy
    top1: float = 0.0
    top3: float = 0.0
    top5: float = 0.0

    # F1
    f1_macro: float = 0.0
    f1_weighted: float = 0.0

    # Latency (ms)
    latency_mean: float = 0.0
    latency_median: float = 0.0
    latency_p95: float = 0.0

    # Per-sample details
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stage_name': self.stage_name,
            'n_samples': self.n_samples,
            'n_classes': self.n_classes,
            'top1': round(self.top1, 4),
            'top3': round(self.top3, 4),
            'top5': round(self.top5, 4),
            'f1_macro': round(self.f1_macro, 4),
            'f1_weighted': round(self.f1_weighted, 4),
            'latency_mean': round(self.latency_mean, 1),
            'latency_median': round(self.latency_median, 1),
            'latency_p95': round(self.latency_p95, 1),
        }


def compute_topk_accuracy(
    true_labels: List[str],
    predictions: List[List[str]],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """Compute Top-K accuracy."""
    if k_values is None:
        k_values = [1, 3, 5]
    n = len(true_labels)
    if n == 0:
        return {f"top{k}": 0.0 for k in k_values}

    results = {}
    for k in k_values:
        correct = sum(1 for t, p in zip(true_labels, predictions) if t in p[:k])
        results[f"top{k}"] = correct / n
    return results


def compute_f1(
    true_labels: List[str],
    pred_labels: List[str],
) -> Dict[str, float]:
    """Compute macro and weighted F1 scores."""
    try:
        from sklearn.metrics import f1_score
        return {
            'f1_macro': f1_score(true_labels, pred_labels, average='macro', zero_division=0),
            'f1_weighted': f1_score(true_labels, pred_labels, average='weighted', zero_division=0),
        }
    except ImportError:
        # Manual macro F1 fallback
        return _manual_f1(true_labels, pred_labels)


def _manual_f1(true_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
    """Manual F1 computation without sklearn."""
    classes = sorted(set(true_labels) | set(pred_labels))
    f1_scores = []
    weights = []

    for cls in classes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        support = sum(1 for t in true_labels if t == cls)
        f1_scores.append(f1)
        weights.append(support)

    total = sum(weights)
    macro = np.mean(f1_scores) if f1_scores else 0.0
    weighted = sum(f * w for f, w in zip(f1_scores, weights)) / total if total > 0 else 0.0
    return {'f1_macro': float(macro), 'f1_weighted': float(weighted)}


def compute_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """Compute latency statistics."""
    if not latencies_ms:
        return {'mean': 0.0, 'median': 0.0, 'p95': 0.0}
    arr = np.array(latencies_ms)
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'p95': float(np.percentile(arr, 95)),
    }


def compute_chapter_accuracy(
    true_labels: List[str],
    pred_labels: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Per-chapter (HS2) accuracy breakdown."""
    chapter_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for true, pred in zip(true_labels, pred_labels):
        ch = true[:2]
        chapter_stats[ch]['total'] += 1
        if true == pred:
            chapter_stats[ch]['correct'] += 1

    for ch in chapter_stats:
        s = chapter_stats[ch]
        s['accuracy'] = s['correct'] / s['total'] if s['total'] > 0 else 0.0
    return dict(chapter_stats)
