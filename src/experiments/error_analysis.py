"""
Error Analysis Module

오류 분석:
- Confusion Pairs (혼동 쌍)
- 축별 실패 케이스
- Chapter별 성능
"""

import json
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass
class FailureCase:
    """실패 케이스"""
    sample_id: str
    text: str
    true_hs4: str
    pred_hs4: str
    confidence: float
    top3_predictions: List[str]
    error_type: str  # "chapter_error", "heading_error", "subheading_error"
    chapter_error: bool
    features: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "text": self.text,
            "true_hs4": self.true_hs4,
            "pred_hs4": self.pred_hs4,
            "confidence": round(self.confidence, 4),
            "top3_predictions": self.top3_predictions,
            "error_type": self.error_type,
            "chapter_error": self.chapter_error,
            "features": self.features,
        }


@dataclass
class ConfusionPair:
    """혼동 쌍"""
    true_hs4: str
    pred_hs4: str
    count: int
    examples: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "true_hs4": self.true_hs4,
            "pred_hs4": self.pred_hs4,
            "count": self.count,
            "examples": self.examples[:5],
        }


class ErrorAnalyzer:
    """
    오류 분석기

    - Confusion pairs 추출
    - 축별 실패 분석
    - Chapter별 오류 패턴
    """

    def __init__(self):
        self.failure_cases: List[FailureCase] = []
        self.confusion_pairs: List[ConfusionPair] = []

    def analyze(
        self,
        sample_ids: List[str],
        texts: List[str],
        true_labels: List[str],
        predictions: List[List[Tuple[str, float]]],  # [(hs4, score), ...]
        features: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        전체 오류 분석

        Args:
            sample_ids: 샘플 ID
            texts: 입력 텍스트
            true_labels: 실제 라벨
            predictions: 예측 결과
            features: 피처 정보 (옵션)

        Returns:
            분석 결과
        """
        self.failure_cases = []
        confusion_counter = defaultdict(lambda: {"count": 0, "examples": []})

        for i, (sid, text, true_label, preds) in enumerate(zip(
            sample_ids, texts, true_labels, predictions
        )):
            if not preds:
                continue

            pred_label = preds[0][0]
            confidence = preds[0][1]
            top3 = [p[0] for p in preds[:3]]

            # 정답이면 스킵
            if pred_label == true_label:
                continue

            # 오류 유형 분류
            error_type = self._classify_error_type(true_label, pred_label)
            chapter_error = true_label[:2] != pred_label[:2]

            # 피처 정보
            feat = features[i] if features and i < len(features) else None

            # 실패 케이스 기록
            self.failure_cases.append(FailureCase(
                sample_id=sid,
                text=text,
                true_hs4=true_label,
                pred_hs4=pred_label,
                confidence=confidence,
                top3_predictions=top3,
                error_type=error_type,
                chapter_error=chapter_error,
                features=feat
            ))

            # Confusion 쌍 기록
            pair_key = (true_label, pred_label)
            confusion_counter[pair_key]["count"] += 1
            if len(confusion_counter[pair_key]["examples"]) < 5:
                confusion_counter[pair_key]["examples"].append(text[:100])

        # Confusion pairs 정렬
        self.confusion_pairs = [
            ConfusionPair(
                true_hs4=pair[0],
                pred_hs4=pair[1],
                count=data["count"],
                examples=data["examples"]
            )
            for pair, data in sorted(
                confusion_counter.items(),
                key=lambda x: -x[1]["count"]
            )
        ]

        # 결과 집계
        return self._aggregate_results()

    def _classify_error_type(self, true_hs4: str, pred_hs4: str) -> str:
        """오류 유형 분류"""
        if true_hs4[:2] != pred_hs4[:2]:
            return "chapter_error"  # 류(2자리) 오류
        elif true_hs4[:4] != pred_hs4[:4]:
            return "heading_error"  # 호(4자리) 오류
        else:
            return "subheading_error"  # 소호 오류 (이론상 HS4에서는 없음)

    def _aggregate_results(self) -> Dict[str, Any]:
        """결과 집계"""
        total_errors = len(self.failure_cases)

        if total_errors == 0:
            return {
                "total_errors": 0,
                "error_types": {},
                "chapter_error_rate": 0,
                "top_confusion_pairs": [],
                "chapter_breakdown": {},
            }

        # 오류 유형별 통계
        error_types = Counter(fc.error_type for fc in self.failure_cases)
        chapter_errors = sum(1 for fc in self.failure_cases if fc.chapter_error)

        # Chapter별 오류 분석
        chapter_errors_breakdown = defaultdict(lambda: {
            "total": 0,
            "chapter_error": 0,
            "examples": []
        })

        for fc in self.failure_cases:
            chapter = fc.true_hs4[:2]
            chapter_errors_breakdown[chapter]["total"] += 1
            if fc.chapter_error:
                chapter_errors_breakdown[chapter]["chapter_error"] += 1
            if len(chapter_errors_breakdown[chapter]["examples"]) < 3:
                chapter_errors_breakdown[chapter]["examples"].append({
                    "text": fc.text[:50],
                    "true": fc.true_hs4,
                    "pred": fc.pred_hs4
                })

        # 신뢰도 분석
        high_conf_errors = [fc for fc in self.failure_cases if fc.confidence >= 0.7]
        low_conf_errors = [fc for fc in self.failure_cases if fc.confidence < 0.4]

        return {
            "total_errors": total_errors,
            "error_types": dict(error_types),
            "chapter_error_rate": chapter_errors / total_errors,
            "heading_error_rate": error_types.get("heading_error", 0) / total_errors,

            "confidence_analysis": {
                "high_conf_errors": len(high_conf_errors),
                "low_conf_errors": len(low_conf_errors),
                "avg_error_confidence": sum(fc.confidence for fc in self.failure_cases) / total_errors,
            },

            "top_confusion_pairs": [cp.to_dict() for cp in self.confusion_pairs[:20]],

            "chapter_breakdown": {
                chapter: {
                    "total_errors": data["total"],
                    "chapter_errors": data["chapter_error"],
                    "chapter_error_rate": data["chapter_error"] / data["total"] if data["total"] > 0 else 0,
                    "examples": data["examples"]
                }
                for chapter, data in sorted(chapter_errors_breakdown.items(), key=lambda x: -x[1]["total"])
            },
        }

    def get_confusion_pairs(self, topk: int = 20) -> List[ConfusionPair]:
        """상위 혼동 쌍 반환"""
        return self.confusion_pairs[:topk]

    def get_failure_cases(
        self,
        n: int = 20,
        error_type: Optional[str] = None,
        high_confidence: bool = False
    ) -> List[FailureCase]:
        """
        실패 케이스 반환

        Args:
            n: 반환 개수
            error_type: 특정 오류 유형만
            high_confidence: 고신뢰도 오류만

        Returns:
            실패 케이스 리스트
        """
        cases = self.failure_cases

        if error_type:
            cases = [fc for fc in cases if fc.error_type == error_type]

        if high_confidence:
            cases = [fc for fc in cases if fc.confidence >= 0.7]

        # 신뢰도 내림차순 (고신뢰도 오류가 더 중요)
        cases = sorted(cases, key=lambda x: -x.confidence)

        return cases[:n]

    def analyze_by_chapter(self) -> Dict[str, Dict[str, Any]]:
        """Chapter별 상세 분석"""
        chapter_stats = defaultdict(lambda: {
            "errors": [],
            "confusion_matrix": defaultdict(int)
        })

        for fc in self.failure_cases:
            true_chapter = fc.true_hs4[:2]
            pred_chapter = fc.pred_hs4[:2]

            chapter_stats[true_chapter]["errors"].append(fc)
            chapter_stats[true_chapter]["confusion_matrix"][pred_chapter] += 1

        result = {}
        for chapter, stats in chapter_stats.items():
            result[chapter] = {
                "total_errors": len(stats["errors"]),
                "avg_confidence": sum(e.confidence for e in stats["errors"]) / len(stats["errors"]),
                "top_confusions": dict(
                    sorted(stats["confusion_matrix"].items(), key=lambda x: -x[1])[:5]
                ),
                "sample_errors": [
                    {"text": e.text[:50], "true": e.true_hs4, "pred": e.pred_hs4}
                    for e in stats["errors"][:3]
                ]
            }

        return dict(sorted(result.items(), key=lambda x: -x[1]["total_errors"]))

    def analyze_by_attribute(
        self,
        attribute_key: str = "f_material_match"
    ) -> Dict[str, Any]:
        """
        속성별 오류 분석

        Args:
            attribute_key: 분석할 피처 키

        Returns:
            속성별 오류 통계
        """
        with_attr = []
        without_attr = []

        for fc in self.failure_cases:
            if fc.features is None:
                continue

            attr_value = fc.features.get(attribute_key, 0)
            if attr_value > 0:
                with_attr.append(fc)
            else:
                without_attr.append(fc)

        return {
            "attribute": attribute_key,
            "with_attribute": {
                "count": len(with_attr),
                "avg_confidence": sum(e.confidence for e in with_attr) / len(with_attr) if with_attr else 0,
            },
            "without_attribute": {
                "count": len(without_attr),
                "avg_confidence": sum(e.confidence for e in without_attr) / len(without_attr) if without_attr else 0,
            }
        }

    def export_failure_cases(self, path: str, n: int = None):
        """실패 케이스 JSONL 저장"""
        cases = self.failure_cases[:n] if n else self.failure_cases

        with open(path, 'w', encoding='utf-8') as f:
            for fc in cases:
                f.write(json.dumps(fc.to_dict(), ensure_ascii=False) + '\n')

        print(f"[ErrorAnalyzer] 실패 케이스 저장: {path} ({len(cases)}개)")

    def export_confusion_pairs(self, path: str, topk: int = 20):
        """혼동 쌍 CSV 저장"""
        import csv

        pairs = self.confusion_pairs[:topk]

        with open(path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["true_hs4", "pred_hs4", "count", "example"])

            for cp in pairs:
                example = cp.examples[0] if cp.examples else ""
                writer.writerow([cp.true_hs4, cp.pred_hs4, cp.count, example])

        print(f"[ErrorAnalyzer] 혼동 쌍 저장: {path} ({len(pairs)}개)")

    def generate_report(self) -> str:
        """오류 분석 리포트 생성"""
        results = self._aggregate_results()

        lines = [
            "=" * 60,
            "Error Analysis Report",
            "=" * 60,
            "",
            f"Total Errors: {results['total_errors']}",
            "",
            "--- Error Types ---",
        ]

        for error_type, count in results['error_types'].items():
            rate = count / results['total_errors'] * 100 if results['total_errors'] > 0 else 0
            lines.append(f"  {error_type}: {count} ({rate:.1f}%)")

        lines.extend([
            "",
            f"Chapter Error Rate: {results['chapter_error_rate']*100:.1f}%",
            "",
            "--- Confidence Analysis ---",
            f"  High-conf errors (>=0.7): {results['confidence_analysis']['high_conf_errors']}",
            f"  Low-conf errors (<0.4):  {results['confidence_analysis']['low_conf_errors']}",
            f"  Avg error confidence:    {results['confidence_analysis']['avg_error_confidence']:.4f}",
            "",
            "--- Top Confusion Pairs ---",
        ])

        for i, cp in enumerate(results['top_confusion_pairs'][:10], 1):
            lines.append(f"  {i}. {cp['true_hs4']} -> {cp['pred_hs4']}: {cp['count']} times")

        lines.extend([
            "",
            "--- Chapter Breakdown (Top 5) ---",
        ])

        chapter_items = list(results['chapter_breakdown'].items())[:5]
        for chapter, data in chapter_items:
            lines.append(
                f"  Chapter {chapter}: {data['total_errors']} errors "
                f"({data['chapter_error_rate']*100:.1f}% cross-chapter)"
            )

        return "\n".join(lines)


# 테스트
if __name__ == "__main__":
    import random

    # 테스트 데이터
    random.seed(42)
    n = 50

    sample_ids = [f"s_{i}" for i in range(n)]
    texts = [f"테스트 품명 {i}" for i in range(n)]
    true_labels = [f"{random.randint(1, 97):02d}{random.randint(0, 99):02d}" for _ in range(n)]

    predictions = []
    for true_label in true_labels:
        conf = random.random() * 0.8 + 0.1

        # 일부는 정답, 일부는 오답
        if random.random() < 0.6:
            pred = true_label
        else:
            # 랜덤 오답
            if random.random() < 0.3:
                # 다른 류
                pred = f"{random.randint(1, 97):02d}{random.randint(0, 99):02d}"
            else:
                # 같은 류 다른 호
                pred = f"{true_label[:2]}{random.randint(0, 99):02d}"

        preds = [(pred, conf)]
        for _ in range(2):
            preds.append((f"{random.randint(1, 97):02d}{random.randint(0, 99):02d}", conf * 0.5))
        predictions.append(preds)

    # 분석
    analyzer = ErrorAnalyzer()
    results = analyzer.analyze(sample_ids, texts, true_labels, predictions)

    # 리포트
    report = analyzer.generate_report()
    print(report)

    print("\n=== High Confidence Errors ===")
    high_conf_errors = analyzer.get_failure_cases(n=5, high_confidence=True)
    for fc in high_conf_errors:
        print(f"  [{fc.confidence:.3f}] {fc.text[:30]}... : {fc.true_hs4} -> {fc.pred_hs4}")
