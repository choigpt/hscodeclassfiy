"""
Metrics Module - 정량 지표 정의 및 계산

지표:
A) Top-k Accuracy (k=1,3,5)
B) Macro F1
C) Candidate Recall@K
D) Calibration: ECE, Brier Score
E) Routing: AUTO/ASK/REVIEW/ABSTAIN 비율
F) Legal Conflict Rate
G) Fact Missing Stats (8축 분포)
H) Confusion Pairs Top 20
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np


@dataclass
class ECEBin:
    """ECE 계산용 bin"""
    bin_id: int
    count: int
    accuracy: float
    avg_confidence: float
    ece_contribution: float


@dataclass
class MetricsSummary:
    """평가 지표 요약"""
    # A) Top-k Accuracy
    top1_accuracy: float = 0.0
    top3_accuracy: float = 0.0
    top5_accuracy: float = 0.0

    # B) Macro F1
    macro_f1: float = 0.0
    weighted_f1: float = 0.0

    # C) Candidate Recall
    candidate_recall_5: float = 0.0
    candidate_recall_10: float = 0.0
    candidate_recall_20: float = 0.0

    # D) Calibration
    ece: float = 0.0
    brier_score: float = 0.0
    ece_bins: List[ECEBin] = field(default_factory=list)

    # E) Routing
    auto_rate: float = 0.0
    ask_rate: float = 0.0
    review_rate: float = 0.0
    abstain_rate: float = 0.0

    # F) Legal Conflict Rate
    legal_conflict_rate: float = 0.0
    legal_conflict_count: int = 0

    # G) Fact Missing Stats
    fact_missing_rate: float = 0.0
    missing_hard_axis_dist: Dict[str, int] = field(default_factory=dict)
    missing_soft_axis_dist: Dict[str, int] = field(default_factory=dict)

    # H) Confusion Pairs
    confusion_pairs_top20: List[Dict[str, Any]] = field(default_factory=list)

    # 메타 정보
    total_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_samples': self.total_samples,
            'top1_accuracy': round(self.top1_accuracy, 4),
            'top3_accuracy': round(self.top3_accuracy, 4),
            'top5_accuracy': round(self.top5_accuracy, 4),
            'macro_f1': round(self.macro_f1, 4),
            'weighted_f1': round(self.weighted_f1, 4),
            'candidate_recall_5': round(self.candidate_recall_5, 4),
            'candidate_recall_10': round(self.candidate_recall_10, 4),
            'candidate_recall_20': round(self.candidate_recall_20, 4),
            'ece': round(self.ece, 4),
            'brier_score': round(self.brier_score, 4),
            'auto_rate': round(self.auto_rate, 4),
            'ask_rate': round(self.ask_rate, 4),
            'review_rate': round(self.review_rate, 4),
            'abstain_rate': round(self.abstain_rate, 4),
            'legal_conflict_rate': round(self.legal_conflict_rate, 4),
            'legal_conflict_count': self.legal_conflict_count,
            'fact_missing_rate': round(self.fact_missing_rate, 4),
            'missing_hard_axis_dist': self.missing_hard_axis_dist,
            'missing_soft_axis_dist': self.missing_soft_axis_dist,
            'confusion_pairs_top20': self.confusion_pairs_top20,
        }


def compute_top_k_accuracy(
    predictions: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5]
) -> Dict[int, float]:
    """
    Top-k Accuracy 계산

    Args:
        predictions: per-sample 예측 결과 (true_hs4, topk 포함)
        k_values: k 값 리스트

    Returns:
        {k: accuracy}
    """
    if not predictions:
        return {k: 0.0 for k in k_values}

    results = {k: 0 for k in k_values}

    for pred in predictions:
        true_hs4 = pred.get('true_hs4')
        topk = pred.get('topk', [])

        if not true_hs4:
            continue

        # topk는 [{'hs4': ..., 'score': ...}, ...] 형태
        pred_hs4s = [c['hs4'] for c in topk]

        for k in k_values:
            if true_hs4 in pred_hs4s[:k]:
                results[k] += 1

    total = len(predictions)
    return {k: count / total for k, count in results.items()}


def compute_macro_f1(
    predictions: List[Dict[str, Any]]
) -> Tuple[float, float]:
    """
    Macro F1 및 Weighted F1 계산

    Args:
        predictions: per-sample 예측 결과

    Returns:
        (macro_f1, weighted_f1)
    """
    from sklearn.metrics import f1_score

    y_true = []
    y_pred = []

    for pred in predictions:
        true_hs4 = pred.get('true_hs4')
        topk = pred.get('topk', [])

        if not true_hs4 or not topk:
            continue

        y_true.append(true_hs4)
        y_pred.append(topk[0]['hs4'])  # Top-1

    if not y_true:
        return 0.0, 0.0

    # Macro F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Weighted F1
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return float(macro_f1), float(weighted_f1)


def compute_candidate_recall(
    predictions: List[Dict[str, Any]],
    k_values: List[int] = [5, 10, 20]
) -> Dict[int, float]:
    """
    Candidate Recall@K 계산

    정의: Retrieval 단계 후보에 정답이 포함되는 비율

    Args:
        predictions: per-sample 예측 결과 (topk 포함)
        k_values: k 값 리스트

    Returns:
        {k: recall}
    """
    if not predictions:
        return {k: 0.0 for k in k_values}

    results = {k: 0 for k in k_values}

    for pred in predictions:
        true_hs4 = pred.get('true_hs4')
        topk = pred.get('topk', [])

        if not true_hs4:
            continue

        pred_hs4s = [c['hs4'] for c in topk]

        for k in k_values:
            if true_hs4 in pred_hs4s[:k]:
                results[k] += 1

    total = len(predictions)
    return {k: count / total for k, count in results.items()}


def compute_ece_and_brier(
    predictions: List[Dict[str, Any]],
    n_bins: int = 10
) -> Tuple[float, float, List[ECEBin]]:
    """
    ECE (Expected Calibration Error) 및 Brier Score 계산

    Args:
        predictions: per-sample 예측 결과 (confidence, top1_correct 포함)
        n_bins: ECE bin 개수

    Returns:
        (ece, brier_score, bins)
    """
    if not predictions:
        return 0.0, 0.0, []

    confidences = []
    correctness = []

    for pred in predictions:
        topk = pred.get('topk', [])
        true_hs4 = pred.get('true_hs4')

        if not topk or not true_hs4:
            continue

        # Top-1 confidence (score_total)
        conf = topk[0].get('score_total', 0.0)

        # Confidence가 확률(0~1) 범위인지 점검
        # score_total이 큰 값이면 softmax 적용
        if conf > 1.0 or conf < 0.0:
            # Softmax 또는 sigmoid 적용
            # 일단 sigmoid로 변환
            conf = 1.0 / (1.0 + np.exp(-conf))

        # Clamp to [0, 1]
        conf = np.clip(conf, 0.0, 1.0)

        confidences.append(conf)

        # Top-1 correct
        is_correct = (topk[0]['hs4'] == true_hs4)
        correctness.append(1.0 if is_correct else 0.0)

    if not confidences:
        return 0.0, 0.0, []

    confidences = np.array(confidences)
    correctness = np.array(correctness)

    # ECE 계산
    bins = []
    ece = 0.0

    for i in range(n_bins):
        bin_lower = i / n_bins
        bin_upper = (i + 1) / n_bins

        # bin에 속하는 샘플
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        if i == n_bins - 1:  # 마지막 bin은 upper 포함
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)

        bin_count = in_bin.sum()

        if bin_count > 0:
            bin_acc = correctness[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            bin_ece = abs(bin_acc - bin_conf) * (bin_count / len(confidences))
            ece += bin_ece

            bins.append(ECEBin(
                bin_id=i,
                count=int(bin_count),
                accuracy=float(bin_acc),
                avg_confidence=float(bin_conf),
                ece_contribution=float(bin_ece)
            ))
        else:
            bins.append(ECEBin(
                bin_id=i,
                count=0,
                accuracy=0.0,
                avg_confidence=0.0,
                ece_contribution=0.0
            ))

    # Brier Score 계산
    brier_score = float(((confidences - correctness) ** 2).mean())

    return float(ece), brier_score, bins


def compute_routing_stats(
    predictions: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Routing 통계 계산 (AUTO/ASK/REVIEW/ABSTAIN 비율)

    Args:
        predictions: per-sample 예측 결과 (decision 포함)

    Returns:
        {status: rate}
    """
    if not predictions:
        return {'auto': 0.0, 'ask': 0.0, 'review': 0.0, 'abstain': 0.0}

    counter = Counter()

    for pred in predictions:
        decision = pred.get('decision', {})
        status = decision.get('status', 'ABSTAIN').lower()
        counter[status] += 1

    total = len(predictions)

    return {
        'auto': counter['auto'] / total,
        'ask': counter['ask'] / total,
        'review': counter['review'] / total,
        'abstain': counter['abstain'] / total,
    }


def compute_legal_conflict_rate(
    predictions: List[Dict[str, Any]]
) -> Tuple[float, int]:
    """
    Legal Conflict Rate 계산

    정의: 최종 예측이 LegalGate의 hard-exclude/redirect 규범과 충돌하는 비율

    Args:
        predictions: per-sample 예측 결과 (legal_gate_debug 포함)

    Returns:
        (conflict_rate, conflict_count)
    """
    if not predictions:
        return 0.0, 0

    conflict_count = 0

    for pred in predictions:
        topk = pred.get('topk', [])
        debug = pred.get('debug', {})
        legal_gate = debug.get('legal_gate', {})

        if not topk:
            continue

        # Top-1 예측
        pred_hs4 = topk[0]['hs4']

        # LegalGate 결과
        results = legal_gate.get('results', {})
        pred_result = results.get(pred_hs4, {})

        # passed=False면 conflict
        if not pred_result.get('passed', True):
            conflict_count += 1

    total = len(predictions)
    conflict_rate = conflict_count / total if total > 0 else 0.0

    return conflict_rate, conflict_count


def compute_fact_missing_stats(
    predictions: List[Dict[str, Any]]
) -> Tuple[float, Dict[str, int], Dict[str, int]]:
    """
    Fact Missing 통계 계산

    Args:
        predictions: per-sample 예측 결과 (fact_check 포함)

    Returns:
        (missing_rate, hard_axis_dist, soft_axis_dist)
    """
    if not predictions:
        return 0.0, {}, {}

    missing_count = 0
    hard_axis_counter = Counter()
    soft_axis_counter = Counter()

    for pred in predictions:
        debug = pred.get('debug', {})
        fact_check = debug.get('fact_check', {})

        if not fact_check.get('sufficient', True):
            missing_count += 1

            # axis 분포 수집
            candidates_missing = fact_check.get('candidates_missing', {})
            for hs4, missing_info in candidates_missing.items():
                # missing_hard
                for fact in missing_info.get('missing_hard', []):
                    axis = fact.get('axis', 'unknown')
                    hard_axis_counter[axis] += 1

                # missing_soft
                for fact in missing_info.get('missing_soft', []):
                    axis = fact.get('axis', 'unknown')
                    soft_axis_counter[axis] += 1

    total = len(predictions)
    missing_rate = missing_count / total if total > 0 else 0.0

    return missing_rate, dict(hard_axis_counter), dict(soft_axis_counter)


def compute_confusion_pairs(
    predictions: List[Dict[str, Any]],
    topk: int = 20
) -> List[Dict[str, Any]]:
    """
    Confusion Pairs Top-K 계산

    Args:
        predictions: per-sample 예측 결과
        topk: 반환할 상위 개수

    Returns:
        [{true_hs4, pred_hs4, count}, ...]
    """
    pair_counter = Counter()

    for pred in predictions:
        true_hs4 = pred.get('true_hs4')
        topk_cands = pred.get('topk', [])

        if not true_hs4 or not topk_cands:
            continue

        pred_hs4 = topk_cands[0]['hs4']

        # 오분류만
        if true_hs4 != pred_hs4:
            pair_counter[(true_hs4, pred_hs4)] += 1

    # Top-K
    top_pairs = pair_counter.most_common(topk)

    return [
        {
            'true_hs4': pair[0],
            'pred_hs4': pair[1],
            'count': count
        }
        for pair, count in top_pairs
    ]


def compute_all_metrics(
    predictions: List[Dict[str, Any]]
) -> MetricsSummary:
    """
    모든 지표 계산

    Args:
        predictions: per-sample 예측 결과 리스트

    Returns:
        MetricsSummary
    """
    summary = MetricsSummary(total_samples=len(predictions))

    # A) Top-k Accuracy
    topk_acc = compute_top_k_accuracy(predictions, k_values=[1, 3, 5])
    summary.top1_accuracy = topk_acc[1]
    summary.top3_accuracy = topk_acc[3]
    summary.top5_accuracy = topk_acc[5]

    # B) Macro F1
    macro_f1, weighted_f1 = compute_macro_f1(predictions)
    summary.macro_f1 = macro_f1
    summary.weighted_f1 = weighted_f1

    # C) Candidate Recall
    recall = compute_candidate_recall(predictions, k_values=[5, 10, 20])
    summary.candidate_recall_5 = recall[5]
    summary.candidate_recall_10 = recall[10]
    summary.candidate_recall_20 = recall[20]

    # D) Calibration
    ece, brier, bins = compute_ece_and_brier(predictions, n_bins=10)
    summary.ece = ece
    summary.brier_score = brier
    summary.ece_bins = bins

    # E) Routing
    routing = compute_routing_stats(predictions)
    summary.auto_rate = routing['auto']
    summary.ask_rate = routing['ask']
    summary.review_rate = routing['review']
    summary.abstain_rate = routing['abstain']

    # F) Legal Conflict
    conflict_rate, conflict_count = compute_legal_conflict_rate(predictions)
    summary.legal_conflict_rate = conflict_rate
    summary.legal_conflict_count = conflict_count

    # G) Fact Missing
    missing_rate, hard_dist, soft_dist = compute_fact_missing_stats(predictions)
    summary.fact_missing_rate = missing_rate
    summary.missing_hard_axis_dist = hard_dist
    summary.missing_soft_axis_dist = soft_dist

    # H) Confusion Pairs
    confusion = compute_confusion_pairs(predictions, topk=20)
    summary.confusion_pairs_top20 = confusion

    return summary


def save_metrics(
    summary: MetricsSummary,
    output_dir: str
) -> None:
    """
    지표 저장

    Args:
        summary: MetricsSummary
        output_dir: 출력 디렉토리
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. metrics_summary.json
    with open(output_path / 'metrics_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)

    # 2. metrics_table.csv
    with open(output_path / 'metrics_table.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Samples', summary.total_samples])
        writer.writerow(['Top-1 Accuracy', f"{summary.top1_accuracy:.4f}"])
        writer.writerow(['Top-3 Accuracy', f"{summary.top3_accuracy:.4f}"])
        writer.writerow(['Top-5 Accuracy', f"{summary.top5_accuracy:.4f}"])
        writer.writerow(['Macro F1', f"{summary.macro_f1:.4f}"])
        writer.writerow(['Weighted F1', f"{summary.weighted_f1:.4f}"])
        writer.writerow(['Candidate Recall@5', f"{summary.candidate_recall_5:.4f}"])
        writer.writerow(['Candidate Recall@10', f"{summary.candidate_recall_10:.4f}"])
        writer.writerow(['Candidate Recall@20', f"{summary.candidate_recall_20:.4f}"])
        writer.writerow(['ECE', f"{summary.ece:.4f}"])
        writer.writerow(['Brier Score', f"{summary.brier_score:.4f}"])
        writer.writerow(['AUTO Rate', f"{summary.auto_rate:.4f}"])
        writer.writerow(['ASK Rate', f"{summary.ask_rate:.4f}"])
        writer.writerow(['REVIEW Rate', f"{summary.review_rate:.4f}"])
        writer.writerow(['ABSTAIN Rate', f"{summary.abstain_rate:.4f}"])
        writer.writerow(['Legal Conflict Rate', f"{summary.legal_conflict_rate:.4f}"])
        writer.writerow(['Fact Missing Rate', f"{summary.fact_missing_rate:.4f}"])

    # 3. ece_bins.csv
    with open(output_path / 'ece_bins.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Bin', 'Count', 'Accuracy', 'Avg Confidence', 'ECE Contribution'])
        for bin_data in summary.ece_bins:
            writer.writerow([
                bin_data.bin_id,
                bin_data.count,
                f"{bin_data.accuracy:.4f}",
                f"{bin_data.avg_confidence:.4f}",
                f"{bin_data.ece_contribution:.4f}"
            ])

    # 4. confusion_pairs.csv
    with open(output_path / 'confusion_pairs.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['True HS4', 'Pred HS4', 'Count'])
        for pair in summary.confusion_pairs_top20:
            writer.writerow([pair['true_hs4'], pair['pred_hs4'], pair['count']])

    print(f"Metrics saved to {output_path}")
