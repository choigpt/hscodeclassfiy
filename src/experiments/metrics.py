"""
Evaluation Metrics Module

Core metrics for HS4 classification evaluation:
- Top-K Accuracy
- Macro/Weighted F1
- Coverage
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass
class MetricResult:
    """평가 결과"""
    name: str
    value: float
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"name": self.name, "value": round(self.value, 4)}
        if self.details:
            result["details"] = self.details
        return result


def compute_topk_accuracy(
    true_labels: List[str],
    predictions: List[List[str]],
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    """
    Top-K Accuracy 계산

    Args:
        true_labels: 실제 라벨 리스트
        predictions: 예측 결과 (각 샘플의 top-k 예측 리스트)
        k_values: 계산할 k 값들

    Returns:
        {f"top{k}": accuracy} 딕셔너리
    """
    n = len(true_labels)
    if n == 0:
        return {f"top{k}": 0.0 for k in k_values}

    results = {}

    for k in k_values:
        correct = 0
        for true_label, preds in zip(true_labels, predictions):
            if true_label in preds[:k]:
                correct += 1
        results[f"top{k}"] = correct / n

    return results


def compute_f1_scores(
    true_labels: List[str],
    pred_labels: List[str],
    average: str = "macro"
) -> Dict[str, float]:
    """
    F1 Score 계산

    Args:
        true_labels: 실제 라벨
        pred_labels: 예측 라벨 (top-1)
        average: 'macro', 'weighted', 'micro'

    Returns:
        F1 점수들
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    # 모든 클래스 집합
    all_classes = sorted(set(true_labels) | set(pred_labels))

    # zero_division=0으로 경고 방지
    results = {
        f"f1_{average}": f1_score(true_labels, pred_labels, average=average, zero_division=0),
        f"precision_{average}": precision_score(true_labels, pred_labels, average=average, zero_division=0),
        f"recall_{average}": recall_score(true_labels, pred_labels, average=average, zero_division=0),
    }

    return results


def compute_coverage(
    pred_labels: List[str],
    all_classes: List[str]
) -> float:
    """
    Coverage 계산: 모델이 예측한 클래스의 비율

    Args:
        pred_labels: 예측 라벨 (top-1)
        all_classes: 전체 클래스 목록

    Returns:
        예측된 클래스 비율
    """
    if not all_classes:
        return 0.0

    predicted_classes = set(pred_labels)
    return len(predicted_classes) / len(all_classes)


def compute_per_class_accuracy(
    true_labels: List[str],
    pred_labels: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    클래스별 정확도 계산

    Args:
        true_labels: 실제 라벨
        pred_labels: 예측 라벨

    Returns:
        {class: {correct, total, accuracy}} 딕셔너리
    """
    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for true_label, pred_label in zip(true_labels, pred_labels):
        class_stats[true_label]["total"] += 1
        if true_label == pred_label:
            class_stats[true_label]["correct"] += 1

    # 정확도 계산
    for cls in class_stats:
        total = class_stats[cls]["total"]
        correct = class_stats[cls]["correct"]
        class_stats[cls]["accuracy"] = correct / total if total > 0 else 0.0

    return dict(class_stats)


def compute_chapter_accuracy(
    true_labels: List[str],
    pred_labels: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    류(Chapter)별 정확도 계산 (HS 2자리)

    Args:
        true_labels: 실제 HS4 라벨
        pred_labels: 예측 HS4 라벨

    Returns:
        {chapter: {correct, total, accuracy, classes}} 딕셔너리
    """
    chapter_stats = defaultdict(lambda: {
        "correct": 0,
        "total": 0,
        "classes": set()
    })

    for true_label, pred_label in zip(true_labels, pred_labels):
        chapter = true_label[:2]
        chapter_stats[chapter]["total"] += 1
        chapter_stats[chapter]["classes"].add(true_label)

        if true_label == pred_label:
            chapter_stats[chapter]["correct"] += 1

    # 정확도 계산 및 클래스 수 변환
    result = {}
    for chapter, stats in chapter_stats.items():
        result[chapter] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
            "num_classes": len(stats["classes"])
        }

    return result


def compute_metrics(
    true_labels: List[str],
    predictions: List[List[Tuple[str, float]]],  # [(hs4, score), ...]
    all_classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    전체 평가 지표 계산

    Args:
        true_labels: 실제 라벨
        predictions: 예측 결과 리스트 [(hs4, score), ...]
        all_classes: 전체 클래스 목록 (coverage 계산용)

    Returns:
        전체 평가 결과 딕셔너리
    """
    # 예측 라벨 추출
    pred_labels_only = [[p[0] for p in preds] for preds in predictions]
    top1_preds = [preds[0] if preds else "" for preds in pred_labels_only]

    # 확률 추출 (calibration용)
    top1_scores = [preds[0][1] if preds else 0.0 for preds in predictions]

    # Top-K Accuracy
    topk_acc = compute_topk_accuracy(true_labels, pred_labels_only, k_values=[1, 3, 5])

    # F1 Scores
    f1_macro = compute_f1_scores(true_labels, top1_preds, average="macro")
    f1_weighted = compute_f1_scores(true_labels, top1_preds, average="weighted")

    # Coverage
    coverage = compute_coverage(top1_preds, all_classes or list(set(true_labels)))

    # Chapter-level accuracy
    chapter_acc = compute_chapter_accuracy(true_labels, top1_preds)

    # 결과 집계
    results = {
        "n_samples": len(true_labels),
        "n_classes": len(set(true_labels)),

        # Core metrics
        "top1_accuracy": topk_acc["top1"],
        "top3_accuracy": topk_acc["top3"],
        "top5_accuracy": topk_acc["top5"],

        # F1 scores
        "macro_f1": f1_macro["f1_macro"],
        "macro_precision": f1_macro["precision_macro"],
        "macro_recall": f1_macro["recall_macro"],
        "weighted_f1": f1_weighted["f1_weighted"],

        # Coverage
        "coverage": coverage,

        # Chapter breakdown
        "chapter_accuracy": chapter_acc,

        # Top1 scores for calibration
        "top1_scores": top1_scores,
        "top1_correct": [t == p for t, p in zip(true_labels, top1_preds)],
    }

    return results


def compute_improvement(
    baseline_metrics: Dict[str, float],
    model_metrics: Dict[str, float],
    metric_names: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    베이스라인 대비 개선 폭 계산

    Args:
        baseline_metrics: 베이스라인 결과
        model_metrics: 모델 결과
        metric_names: 비교할 메트릭 이름들

    Returns:
        {metric: {baseline, model, absolute_improvement, relative_improvement}}
    """
    if metric_names is None:
        metric_names = ["top1_accuracy", "top3_accuracy", "macro_f1", "weighted_f1"]

    results = {}

    for metric in metric_names:
        baseline_val = baseline_metrics.get(metric, 0)
        model_val = model_metrics.get(metric, 0)

        absolute_imp = model_val - baseline_val
        relative_imp = (model_val - baseline_val) / baseline_val if baseline_val > 0 else 0

        results[metric] = {
            "baseline": baseline_val,
            "model": model_val,
            "absolute_improvement": absolute_imp,
            "relative_improvement": relative_imp,
        }

    return results


def format_metrics_table(
    results_dict: Dict[str, Dict[str, Any]],
    metric_names: List[str] = None
) -> str:
    """
    결과를 표 형태로 포맷팅

    Args:
        results_dict: {model_name: metrics_dict}
        metric_names: 표시할 메트릭 이름들

    Returns:
        포맷팅된 문자열
    """
    if metric_names is None:
        metric_names = ["top1_accuracy", "top3_accuracy", "top5_accuracy", "macro_f1", "weighted_f1"]

    # 헤더
    header = ["Model"] + metric_names
    rows = [header]

    # 데이터 행
    for model_name, metrics in results_dict.items():
        row = [model_name]
        for metric in metric_names:
            value = metrics.get(metric, 0)
            row.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        rows.append(row)

    # 열 너비 계산
    col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(header))]

    # 테이블 생성
    lines = []
    for i, row in enumerate(rows):
        line = " | ".join(str(cell).ljust(col_widths[j]) for j, cell in enumerate(row))
        lines.append(line)
        if i == 0:  # 헤더 후 구분선
            lines.append("-" * len(line))

    return "\n".join(lines)


# 테스트
if __name__ == "__main__":
    # 테스트 데이터
    true_labels = ["0201", "0201", "0203", "8528", "8528"]
    predictions = [
        [("0201", 0.8), ("0203", 0.1), ("0301", 0.05)],
        [("0203", 0.5), ("0201", 0.3), ("0301", 0.1)],  # 오답
        [("0203", 0.9), ("0201", 0.05), ("0301", 0.03)],
        [("8528", 0.7), ("8529", 0.2), ("8530", 0.1)],
        [("8529", 0.4), ("8528", 0.35), ("8530", 0.2)],  # 오답 (top1), 정답 (top2)
    ]

    results = compute_metrics(true_labels, predictions)

    print("=== Evaluation Results ===")
    print(f"Top-1 Accuracy: {results['top1_accuracy']:.4f}")
    print(f"Top-3 Accuracy: {results['top3_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    print(f"Coverage: {results['coverage']:.4f}")

    print("\n=== Chapter Accuracy ===")
    for chapter, stats in results['chapter_accuracy'].items():
        print(f"  Chapter {chapter}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
