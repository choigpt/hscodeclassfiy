"""
Routing Analysis Module

저신뢰도 라우팅 분석:
- AUTO/ASK/REVIEW 분류
- 라우팅별 정확도
- 저신뢰도 Top-3 Hit Rate
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict


@dataclass
class RoutingDecision:
    """라우팅 결정"""
    sample_id: str
    route: str  # "AUTO", "ASK", "REVIEW"
    confidence: float
    top1_prediction: str
    top3_predictions: List[str]
    true_label: str
    is_correct: bool
    top3_hit: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "route": self.route,
            "confidence": round(self.confidence, 4),
            "top1_prediction": self.top1_prediction,
            "top3_predictions": self.top3_predictions,
            "true_label": self.true_label,
            "is_correct": self.is_correct,
            "top3_hit": self.top3_hit,
        }


@dataclass
class RoutingStats:
    """라우팅 통계"""
    total: int
    auto_count: int
    ask_count: int
    review_count: int
    auto_accuracy: float
    ask_accuracy: float
    review_accuracy: float
    ask_top3_hit_rate: float
    review_top3_hit_rate: float
    overall_accuracy: float
    thresholds: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "counts": {
                "auto": self.auto_count,
                "ask": self.ask_count,
                "review": self.review_count,
            },
            "ratios": {
                "auto": self.auto_count / self.total if self.total > 0 else 0,
                "ask": self.ask_count / self.total if self.total > 0 else 0,
                "review": self.review_count / self.total if self.total > 0 else 0,
            },
            "accuracy": {
                "auto": round(self.auto_accuracy, 4),
                "ask": round(self.ask_accuracy, 4),
                "review": round(self.review_accuracy, 4),
                "overall": round(self.overall_accuracy, 4),
            },
            "top3_hit_rate": {
                "ask": round(self.ask_top3_hit_rate, 4),
                "review": round(self.review_top3_hit_rate, 4),
            },
            "thresholds": self.thresholds,
        }


class RoutingAnalyzer:
    """
    라우팅 분석기

    신뢰도 기반 라우팅:
    - AUTO (>= auto_threshold): 자동 분류
    - ASK (review_threshold ~ auto_threshold): 추가 정보 요청
    - REVIEW (< review_threshold): 전문가 검토
    """

    def __init__(
        self,
        auto_threshold: float = 0.7,
        review_threshold: float = 0.4
    ):
        self.auto_threshold = auto_threshold
        self.review_threshold = review_threshold

    def decide_route(self, confidence: float) -> str:
        """
        신뢰도 기반 라우팅 결정

        Args:
            confidence: 예측 신뢰도

        Returns:
            라우팅 결정 ("AUTO", "ASK", "REVIEW")
        """
        if confidence >= self.auto_threshold:
            return "AUTO"
        elif confidence >= self.review_threshold:
            return "ASK"
        else:
            return "REVIEW"

    def analyze(
        self,
        sample_ids: List[str],
        true_labels: List[str],
        predictions: List[List[Tuple[str, float]]],  # [(hs4, score), ...]
    ) -> Tuple[List[RoutingDecision], RoutingStats]:
        """
        전체 라우팅 분석

        Args:
            sample_ids: 샘플 ID 리스트
            true_labels: 실제 라벨
            predictions: 예측 결과 [(hs4, score), ...]

        Returns:
            (라우팅 결정 리스트, 통계)
        """
        decisions = []

        for sid, true_label, preds in zip(sample_ids, true_labels, predictions):
            if not preds:
                continue

            top1_pred = preds[0][0]
            confidence = preds[0][1]
            top3_preds = [p[0] for p in preds[:3]]

            route = self.decide_route(confidence)
            is_correct = top1_pred == true_label
            top3_hit = true_label in top3_preds

            decisions.append(RoutingDecision(
                sample_id=sid,
                route=route,
                confidence=confidence,
                top1_prediction=top1_pred,
                top3_predictions=top3_preds,
                true_label=true_label,
                is_correct=is_correct,
                top3_hit=top3_hit
            ))

        # 통계 계산
        stats = self._compute_stats(decisions)

        return decisions, stats

    def _compute_stats(self, decisions: List[RoutingDecision]) -> RoutingStats:
        """라우팅 통계 계산"""
        total = len(decisions)

        if total == 0:
            return RoutingStats(
                total=0,
                auto_count=0, ask_count=0, review_count=0,
                auto_accuracy=0, ask_accuracy=0, review_accuracy=0,
                ask_top3_hit_rate=0, review_top3_hit_rate=0,
                overall_accuracy=0,
                thresholds={"auto": self.auto_threshold, "review": self.review_threshold}
            )

        # 라우팅별 분류
        auto_decisions = [d for d in decisions if d.route == "AUTO"]
        ask_decisions = [d for d in decisions if d.route == "ASK"]
        review_decisions = [d for d in decisions if d.route == "REVIEW"]

        # 정확도 계산
        def calc_accuracy(decs: List[RoutingDecision]) -> float:
            if not decs:
                return 0.0
            return sum(1 for d in decs if d.is_correct) / len(decs)

        def calc_top3_hit_rate(decs: List[RoutingDecision]) -> float:
            if not decs:
                return 0.0
            return sum(1 for d in decs if d.top3_hit) / len(decs)

        return RoutingStats(
            total=total,
            auto_count=len(auto_decisions),
            ask_count=len(ask_decisions),
            review_count=len(review_decisions),
            auto_accuracy=calc_accuracy(auto_decisions),
            ask_accuracy=calc_accuracy(ask_decisions),
            review_accuracy=calc_accuracy(review_decisions),
            ask_top3_hit_rate=calc_top3_hit_rate(ask_decisions),
            review_top3_hit_rate=calc_top3_hit_rate(review_decisions),
            overall_accuracy=calc_accuracy(decisions),
            thresholds={"auto": self.auto_threshold, "review": self.review_threshold}
        )

    def find_optimal_thresholds(
        self,
        true_labels: List[str],
        predictions: List[List[Tuple[str, float]]],
        target_auto_accuracy: float = 0.9,
        min_auto_ratio: float = 0.3
    ) -> Dict[str, Any]:
        """
        최적 임계값 탐색

        목표:
        - AUTO에서 높은 정확도 유지
        - AUTO 비율 최대화

        Args:
            true_labels: 실제 라벨
            predictions: 예측 결과
            target_auto_accuracy: 목표 AUTO 정확도
            min_auto_ratio: 최소 AUTO 비율

        Returns:
            최적 임계값 및 결과
        """
        results = []

        # 임계값 그리드
        for auto_th in np.arange(0.5, 0.95, 0.05):
            for review_th in np.arange(0.2, auto_th, 0.05):
                # 임시 분석
                temp_analyzer = RoutingAnalyzer(auto_th, review_th)
                _, stats = temp_analyzer.analyze(
                    [str(i) for i in range(len(true_labels))],
                    true_labels,
                    predictions
                )

                auto_ratio = stats.auto_count / stats.total if stats.total > 0 else 0

                # 조건 검사
                if stats.auto_accuracy >= target_auto_accuracy and auto_ratio >= min_auto_ratio:
                    results.append({
                        "auto_threshold": auto_th,
                        "review_threshold": review_th,
                        "auto_accuracy": stats.auto_accuracy,
                        "auto_ratio": auto_ratio,
                        "ask_top3_hit_rate": stats.ask_top3_hit_rate,
                        "review_top3_hit_rate": stats.review_top3_hit_rate,
                    })

        # AUTO 비율 최대화하는 결과 선택
        if results:
            best = max(results, key=lambda x: x["auto_ratio"])
        else:
            # 조건 만족하는 결과 없으면 기본값
            best = {
                "auto_threshold": self.auto_threshold,
                "review_threshold": self.review_threshold,
                "note": "No threshold satisfies target accuracy"
            }

        return best

    def generate_routing_report(
        self,
        decisions: List[RoutingDecision],
        stats: RoutingStats
    ) -> str:
        """라우팅 리포트 생성"""
        lines = [
            "=" * 60,
            "Routing Analysis Report",
            "=" * 60,
            "",
            f"Total Samples: {stats.total}",
            f"Thresholds: AUTO >= {stats.thresholds['auto']:.2f}, REVIEW < {stats.thresholds['review']:.2f}",
            "",
            "--- Distribution ---",
            f"  AUTO:   {stats.auto_count:5d} ({stats.auto_count/stats.total*100:5.1f}%)",
            f"  ASK:    {stats.ask_count:5d} ({stats.ask_count/stats.total*100:5.1f}%)",
            f"  REVIEW: {stats.review_count:5d} ({stats.review_count/stats.total*100:5.1f}%)",
            "",
            "--- Accuracy (Top-1) ---",
            f"  AUTO:   {stats.auto_accuracy*100:5.1f}%",
            f"  ASK:    {stats.ask_accuracy*100:5.1f}%",
            f"  REVIEW: {stats.review_accuracy*100:5.1f}%",
            f"  Overall:{stats.overall_accuracy*100:5.1f}%",
            "",
            "--- Top-3 Hit Rate (for low confidence) ---",
            f"  ASK:    {stats.ask_top3_hit_rate*100:5.1f}%",
            f"  REVIEW: {stats.review_top3_hit_rate*100:5.1f}%",
            "",
        ]

        # 오분류 샘플 (AUTO에서 오답)
        auto_errors = [d for d in decisions if d.route == "AUTO" and not d.is_correct]
        if auto_errors:
            lines.append("--- AUTO Errors (High Confidence Mistakes) ---")
            for d in auto_errors[:10]:
                lines.append(f"  [{d.confidence:.3f}] {d.true_label} -> {d.top1_prediction}")
            if len(auto_errors) > 10:
                lines.append(f"  ... and {len(auto_errors) - 10} more")
            lines.append("")

        # REVIEW에서 Top-3 miss
        review_misses = [d for d in decisions if d.route == "REVIEW" and not d.top3_hit]
        if review_misses:
            lines.append("--- REVIEW Top-3 Misses (Hardest Cases) ---")
            for d in review_misses[:10]:
                lines.append(f"  [{d.confidence:.3f}] {d.true_label} not in {d.top3_predictions}")
            if len(review_misses) > 10:
                lines.append(f"  ... and {len(review_misses) - 10} more")

        return "\n".join(lines)


def analyze_confidence_distribution(
    predictions: List[List[Tuple[str, float]]],
    true_labels: List[str]
) -> Dict[str, Any]:
    """
    신뢰도 분포 분석

    Args:
        predictions: 예측 결과
        true_labels: 실제 라벨

    Returns:
        분포 분석 결과
    """
    correct_confidences = []
    incorrect_confidences = []

    for preds, true_label in zip(predictions, true_labels):
        if not preds:
            continue

        top1_pred = preds[0][0]
        confidence = preds[0][1]

        if top1_pred == true_label:
            correct_confidences.append(confidence)
        else:
            incorrect_confidences.append(confidence)

    return {
        "correct": {
            "count": len(correct_confidences),
            "mean": float(np.mean(correct_confidences)) if correct_confidences else 0,
            "std": float(np.std(correct_confidences)) if correct_confidences else 0,
            "median": float(np.median(correct_confidences)) if correct_confidences else 0,
            "percentiles": {
                "25": float(np.percentile(correct_confidences, 25)) if correct_confidences else 0,
                "75": float(np.percentile(correct_confidences, 75)) if correct_confidences else 0,
            }
        },
        "incorrect": {
            "count": len(incorrect_confidences),
            "mean": float(np.mean(incorrect_confidences)) if incorrect_confidences else 0,
            "std": float(np.std(incorrect_confidences)) if incorrect_confidences else 0,
            "median": float(np.median(incorrect_confidences)) if incorrect_confidences else 0,
            "percentiles": {
                "25": float(np.percentile(incorrect_confidences, 25)) if incorrect_confidences else 0,
                "75": float(np.percentile(incorrect_confidences, 75)) if incorrect_confidences else 0,
            }
        },
        "separation": {
            # 정답과 오답의 신뢰도 분리 정도
            "mean_diff": (
                (np.mean(correct_confidences) if correct_confidences else 0) -
                (np.mean(incorrect_confidences) if incorrect_confidences else 0)
            ),
        }
    }


# 테스트
if __name__ == "__main__":
    import random

    # 테스트 데이터
    random.seed(42)
    n = 100

    sample_ids = [f"s_{i}" for i in range(n)]
    true_labels = [f"hs{random.randint(0, 9):04d}" for _ in range(n)]

    predictions = []
    for true_label in true_labels:
        # 랜덤 신뢰도
        conf = random.random()
        # 신뢰도에 비례하여 정답
        if random.random() < conf:
            top1 = true_label
        else:
            top1 = f"hs{random.randint(0, 9):04d}"

        preds = [(top1, conf)]
        # Top 2, 3
        for _ in range(2):
            preds.append((f"hs{random.randint(0, 9):04d}", conf * random.random()))
        predictions.append(preds)

    # 분석
    analyzer = RoutingAnalyzer(auto_threshold=0.7, review_threshold=0.4)
    decisions, stats = analyzer.analyze(sample_ids, true_labels, predictions)

    # 리포트 출력
    report = analyzer.generate_routing_report(decisions, stats)
    print(report)

    # 최적 임계값 탐색
    print("\n=== Optimal Threshold Search ===")
    optimal = analyzer.find_optimal_thresholds(
        true_labels, predictions,
        target_auto_accuracy=0.8,
        min_auto_ratio=0.2
    )
    print(f"Optimal: {optimal}")
