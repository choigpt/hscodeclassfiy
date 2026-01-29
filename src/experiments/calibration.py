"""
Calibration Metrics Module

모델 신뢰도 평가:
- ECE (Expected Calibration Error)
- Brier Score
- Reliability Curve
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Calibration 평가 결과"""
    ece: float  # Expected Calibration Error
    brier_score: float
    reliability_curve: Dict[str, List[float]]
    n_bins: int
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ece": round(self.ece, 6),
            "brier_score": round(self.brier_score, 6),
            "reliability_curve": self.reliability_curve,
            "n_bins": self.n_bins,
            "details": self.details
        }


def compute_ece(
    confidences: List[float],
    correctness: List[bool],
    n_bins: int = 10
) -> Tuple[float, Dict[str, Any]]:
    """
    Expected Calibration Error (ECE) 계산

    ECE는 예측 신뢰도와 실제 정확도 간의 가중 평균 차이.
    잘 보정된 모델은 ECE가 0에 가깝다.

    Args:
        confidences: 예측 신뢰도 (확률) 리스트
        correctness: 정답 여부 리스트
        n_bins: 구간 수

    Returns:
        (ECE 값, 세부 정보)
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness).astype(float)

    n = len(confidences)
    if n == 0:
        return 0.0, {"bins": [], "n_samples": 0}

    # 구간 경계
    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        # 구간에 속하는 샘플 찾기
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:  # 마지막 구간은 upper 포함
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        bin_count = mask.sum()

        if bin_count > 0:
            bin_confidence = confidences[mask].mean()
            bin_accuracy = correctness[mask].mean()
            bin_ece = abs(bin_accuracy - bin_confidence) * (bin_count / n)
            ece += bin_ece
        else:
            bin_confidence = (lower + upper) / 2
            bin_accuracy = 0.0
            bin_ece = 0.0

        bin_details.append({
            "bin_index": i,
            "lower": lower,
            "upper": upper,
            "count": int(bin_count),
            "avg_confidence": float(bin_confidence),
            "accuracy": float(bin_accuracy),
            "contribution": float(bin_ece) if bin_count > 0 else 0.0
        })

    details = {
        "bins": bin_details,
        "n_samples": n,
        "n_bins": n_bins
    }

    return float(ece), details


def compute_brier_score(
    confidences: List[float],
    correctness: List[bool]
) -> float:
    """
    Brier Score 계산

    Brier Score는 확률 예측의 정확도 측정.
    0에 가까울수록 좋다.

    Args:
        confidences: 예측 신뢰도
        correctness: 정답 여부

    Returns:
        Brier score
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness).astype(float)

    if len(confidences) == 0:
        return 0.0

    # Brier score = mean((confidence - correct)^2)
    brier = np.mean((confidences - correctness) ** 2)

    return float(brier)


def reliability_curve(
    confidences: List[float],
    correctness: List[bool],
    n_bins: int = 10
) -> Dict[str, List[float]]:
    """
    Reliability Curve 데이터 생성

    Args:
        confidences: 예측 신뢰도
        correctness: 정답 여부
        n_bins: 구간 수

    Returns:
        {
            "bin_centers": 구간 중심점,
            "accuracies": 각 구간의 정확도,
            "counts": 각 구간의 샘플 수,
            "perfect_calibration": 완벽 보정 라인 (대각선)
        }
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    accuracies = []
    counts = []

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        center = (lower + upper) / 2
        bin_centers.append(center)

        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        bin_count = mask.sum()
        counts.append(int(bin_count))

        if bin_count > 0:
            accuracies.append(float(correctness[mask].mean()))
        else:
            accuracies.append(0.0)

    return {
        "bin_centers": bin_centers,
        "accuracies": accuracies,
        "counts": counts,
        "perfect_calibration": bin_centers.copy()  # 대각선
    }


def compute_calibration(
    confidences: List[float],
    correctness: List[bool],
    n_bins: int = 10
) -> CalibrationResult:
    """
    전체 Calibration 평가 수행

    Args:
        confidences: 예측 신뢰도
        correctness: 정답 여부
        n_bins: 구간 수

    Returns:
        CalibrationResult
    """
    ece, ece_details = compute_ece(confidences, correctness, n_bins)
    brier = compute_brier_score(confidences, correctness)
    rel_curve = reliability_curve(confidences, correctness, n_bins)

    return CalibrationResult(
        ece=ece,
        brier_score=brier,
        reliability_curve=rel_curve,
        n_bins=n_bins,
        details=ece_details
    )


def compute_confidence_histogram(
    confidences: List[float],
    correctness: List[bool],
    n_bins: int = 20
) -> Dict[str, Any]:
    """
    신뢰도 히스토그램 생성

    정답/오답별 신뢰도 분포 비교

    Args:
        confidences: 예측 신뢰도
        correctness: 정답 여부
        n_bins: 구간 수

    Returns:
        히스토그램 데이터
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness)

    correct_confs = confidences[correctness]
    incorrect_confs = confidences[~correctness]

    bin_edges = np.linspace(0, 1, n_bins + 1)

    correct_hist, _ = np.histogram(correct_confs, bins=bin_edges)
    incorrect_hist, _ = np.histogram(incorrect_confs, bins=bin_edges)

    return {
        "bin_edges": bin_edges.tolist(),
        "correct_counts": correct_hist.tolist(),
        "incorrect_counts": incorrect_hist.tolist(),
        "total_correct": len(correct_confs),
        "total_incorrect": len(incorrect_confs),
        "mean_correct_confidence": float(correct_confs.mean()) if len(correct_confs) > 0 else 0,
        "mean_incorrect_confidence": float(incorrect_confs.mean()) if len(incorrect_confs) > 0 else 0,
    }


def assess_overconfidence(
    confidences: List[float],
    correctness: List[bool],
    threshold: float = 0.8
) -> Dict[str, Any]:
    """
    과신(Overconfidence) 분석

    높은 신뢰도에서 오답인 비율 분석

    Args:
        confidences: 예측 신뢰도
        correctness: 정답 여부
        threshold: 과신 기준 신뢰도

    Returns:
        과신 분석 결과
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness)

    high_conf_mask = confidences >= threshold
    n_high_conf = high_conf_mask.sum()

    if n_high_conf == 0:
        return {
            "threshold": threshold,
            "n_high_confidence": 0,
            "overconfidence_rate": 0.0,
            "high_conf_accuracy": 0.0
        }

    high_conf_correct = correctness[high_conf_mask].sum()
    high_conf_incorrect = n_high_conf - high_conf_correct

    return {
        "threshold": threshold,
        "n_high_confidence": int(n_high_conf),
        "n_high_conf_correct": int(high_conf_correct),
        "n_high_conf_incorrect": int(high_conf_incorrect),
        "overconfidence_rate": float(high_conf_incorrect / n_high_conf),
        "high_conf_accuracy": float(high_conf_correct / n_high_conf),
    }


# 테스트
if __name__ == "__main__":
    import random

    # 테스트 데이터 생성
    random.seed(42)
    n = 100

    # 잘 보정된 모델 시뮬레이션
    confidences = []
    correctness = []

    for _ in range(n):
        conf = random.random()
        # 신뢰도에 비례하여 정답
        correct = random.random() < conf
        confidences.append(conf)
        correctness.append(correct)

    print("=== Calibration Test ===")

    # ECE
    ece, ece_details = compute_ece(confidences, correctness)
    print(f"ECE: {ece:.4f}")

    # Brier Score
    brier = compute_brier_score(confidences, correctness)
    print(f"Brier Score: {brier:.4f}")

    # Reliability Curve
    rel_curve = reliability_curve(confidences, correctness)
    print("\nReliability Curve:")
    print(f"  Bin Centers: {rel_curve['bin_centers'][:5]}...")
    print(f"  Accuracies:  {[f'{a:.2f}' for a in rel_curve['accuracies'][:5]]}...")

    # 전체 결과
    result = compute_calibration(confidences, correctness)
    print(f"\nFull Result: {result.to_dict()['ece']:.4f} ECE, {result.to_dict()['brier_score']:.4f} Brier")

    # 과신 분석
    overconf = assess_overconfidence(confidences, correctness, threshold=0.8)
    print(f"\nOverconfidence (>0.8): {overconf['overconfidence_rate']:.2%} error rate")
