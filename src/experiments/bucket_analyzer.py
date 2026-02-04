"""
Bucket Analyzer - 성능 분석을 위한 데이터 버킷 분류

데이터를 다양한 특성별로 분류하여 성능 분석:
1. fact-insufficient: 사실 정보 부족
2. legal-conflict: LegalGate에서 정답이 제외됨
3. short-text: 짧은 텍스트 (< 20자)
4. ambiguous: 후보 점수 차이 작음
5. rare-class: 학습 데이터가 적은 클래스
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter

from .data_split import DataSample


@dataclass
class BucketStats:
    """버킷 통계"""
    bucket_name: str
    total_samples: int
    top1_correct: int = 0
    top3_correct: int = 0
    top5_correct: int = 0

    @property
    def top1_accuracy(self) -> float:
        return self.top1_correct / self.total_samples if self.total_samples > 0 else 0.0

    @property
    def top3_accuracy(self) -> float:
        return self.top3_correct / self.total_samples if self.total_samples > 0 else 0.0

    @property
    def top5_accuracy(self) -> float:
        return self.top5_correct / self.total_samples if self.total_samples > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'bucket_name': self.bucket_name,
            'total_samples': self.total_samples,
            'top1_correct': self.top1_correct,
            'top3_correct': self.top3_correct,
            'top5_correct': self.top5_correct,
            'top1_accuracy': round(self.top1_accuracy, 4),
            'top3_accuracy': round(self.top3_accuracy, 4),
            'top5_accuracy': round(self.top5_accuracy, 4),
        }


@dataclass
class BucketAssignment:
    """샘플의 버킷 할당"""
    sample_id: str
    buckets: List[str] = field(default_factory=list)  # 복수 버킷 가능
    metadata: Dict[str, Any] = field(default_factory=dict)


class BucketAnalyzer:
    """
    데이터 버킷 분류 및 성능 분석

    버킷 종류:
    - fact_insufficient: FactCheck에서 missing_hard 있음
    - legal_conflict: LegalGate에서 정답 제외
    - short_text: 텍스트 길이 < 20자
    - ambiguous: Top1과 Top2 점수 차이 < 0.1
    - rare_class: 학습 샘플 < 5개
    """

    def __init__(
        self,
        short_text_threshold: int = 20,
        ambiguous_threshold: float = 0.1,
        rare_class_threshold: int = 5
    ):
        """
        Args:
            short_text_threshold: 짧은 텍스트 임계값 (문자 수)
            ambiguous_threshold: 모호성 임계값 (점수 차이)
            rare_class_threshold: 희소 클래스 임계값 (샘플 수)
        """
        self.short_text_threshold = short_text_threshold
        self.ambiguous_threshold = ambiguous_threshold
        self.rare_class_threshold = rare_class_threshold

        # 학습 데이터 클래스 분포 (rare_class 판단용)
        self.train_class_counts: Dict[str, int] = {}

    def set_train_distribution(self, train_samples: List[DataSample]) -> None:
        """
        학습 데이터 분포 설정 (rare_class 판단용)

        Args:
            train_samples: 학습 데이터
        """
        self.train_class_counts = Counter([s.hs4 for s in train_samples])

    def classify_sample(
        self,
        sample: DataSample,
        prediction_result: Optional[Dict[str, Any]] = None,
        pipeline_debug: Optional[Dict[str, Any]] = None
    ) -> BucketAssignment:
        """
        단일 샘플을 버킷에 할당

        Args:
            sample: 데이터 샘플
            prediction_result: 예측 결과 (점수 포함)
            pipeline_debug: 파이프라인 디버그 정보

        Returns:
            BucketAssignment
        """
        buckets = []
        metadata = {}

        # 1. Short Text
        text_len = len(sample.text)
        if text_len < self.short_text_threshold:
            buckets.append('short_text')
            metadata['text_length'] = text_len

        # 2. Rare Class
        train_count = self.train_class_counts.get(sample.hs4, 0)
        if train_count < self.rare_class_threshold:
            buckets.append('rare_class')
            metadata['train_count'] = train_count

        # 3. Ambiguous (예측 결과 필요)
        if prediction_result and 'predictions' in prediction_result:
            preds = prediction_result['predictions']
            if len(preds) >= 2:
                top1_score = preds[0][1] if len(preds[0]) > 1 else 0.0
                top2_score = preds[1][1] if len(preds[1]) > 1 else 0.0
                score_diff = top1_score - top2_score

                if score_diff < self.ambiguous_threshold:
                    buckets.append('ambiguous')
                    metadata['score_diff'] = round(score_diff, 4)

        # 4. Fact Insufficient (파이프라인 디버그 필요)
        if pipeline_debug and 'fact_check' in pipeline_debug:
            fact_check = pipeline_debug['fact_check']
            if not fact_check.get('sufficient', True):
                buckets.append('fact_insufficient')
                metadata['missing_hard_count'] = len(fact_check.get('missing_hard', []))

        # 5. Legal Conflict (파이프라인 디버그 필요)
        if pipeline_debug and 'legal_gate' in pipeline_debug:
            legal_gate = pipeline_debug['legal_gate']
            results = legal_gate.get('results', {})

            # 정답 HS4가 LegalGate에서 제외되었는지 확인
            answer_result = results.get(sample.hs4)
            if answer_result and not answer_result.get('passed', True):
                buckets.append('legal_conflict')
                metadata['exclude_conflict_score'] = answer_result.get('exclude_conflict_score', 0.0)

        return BucketAssignment(
            sample_id=sample.id,
            buckets=buckets,
            metadata=metadata
        )

    def compute_bucket_stats(
        self,
        assignments: List[BucketAssignment],
        sample_map: Dict[str, DataSample],
        predictions: List[List[Tuple[str, float]]]
    ) -> Dict[str, BucketStats]:
        """
        버킷별 통계 계산

        Args:
            assignments: 버킷 할당 목록
            sample_map: {sample_id: DataSample}
            predictions: 예측 결과 [(hs4, score), ...]

        Returns:
            {bucket_name: BucketStats}
        """
        # 버킷별 샘플 수집
        bucket_samples: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        for i, assign in enumerate(assignments):
            for bucket in assign.buckets:
                bucket_samples[bucket].append((assign.sample_id, i))

        # 버킷별 통계 계산
        bucket_stats = {}

        for bucket_name, samples in bucket_samples.items():
            stats = BucketStats(
                bucket_name=bucket_name,
                total_samples=len(samples)
            )

            for sample_id, pred_idx in samples:
                sample = sample_map[sample_id]
                pred = predictions[pred_idx]

                # Top-K correct 확인
                pred_hs4s = [p[0] for p in pred]

                if sample.hs4 in pred_hs4s[:1]:
                    stats.top1_correct += 1
                if sample.hs4 in pred_hs4s[:3]:
                    stats.top3_correct += 1
                if sample.hs4 in pred_hs4s[:5]:
                    stats.top5_correct += 1

            bucket_stats[bucket_name] = stats

        return bucket_stats

    def analyze(
        self,
        samples: List[DataSample],
        predictions: List[List[Tuple[str, float]]],
        pipeline_debugs: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[BucketAssignment], Dict[str, BucketStats]]:
        """
        전체 데이터에 대한 버킷 분석

        Args:
            samples: 데이터 샘플
            predictions: 예측 결과
            pipeline_debugs: 파이프라인 디버그 정보 (선택적)

        Returns:
            (assignments, bucket_stats)
        """
        # 샘플 맵 생성
        sample_map = {s.id: s for s in samples}

        # 각 샘플을 버킷에 할당
        assignments = []

        for i, sample in enumerate(samples):
            pred_result = {
                'predictions': predictions[i]
            }

            pipeline_debug = None
            if pipeline_debugs and i < len(pipeline_debugs):
                pipeline_debug = pipeline_debugs[i]

            assign = self.classify_sample(sample, pred_result, pipeline_debug)
            assignments.append(assign)

        # 버킷별 통계 계산
        bucket_stats = self.compute_bucket_stats(assignments, sample_map, predictions)

        return assignments, bucket_stats

    def save_analysis(
        self,
        assignments: List[BucketAssignment],
        bucket_stats: Dict[str, BucketStats],
        output_path: str
    ) -> None:
        """
        분석 결과 저장

        Args:
            assignments: 버킷 할당
            bucket_stats: 버킷 통계
            output_path: 출력 경로
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. 버킷 할당 저장
        assignments_file = output_path / "bucket_assignments.jsonl"
        with open(assignments_file, 'w', encoding='utf-8') as f:
            for assign in assignments:
                f.write(json.dumps({
                    'sample_id': assign.sample_id,
                    'buckets': assign.buckets,
                    'metadata': assign.metadata
                }, ensure_ascii=False) + '\n')

        # 2. 버킷 통계 저장
        stats_file = output_path / "bucket_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                name: stats.to_dict()
                for name, stats in bucket_stats.items()
            }, f, ensure_ascii=False, indent=2)

        # 3. 요약 CSV
        csv_file = output_path / "bucket_stats.csv"
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(['Bucket', 'Total', 'Top1_Acc', 'Top3_Acc', 'Top5_Acc'])

            for name, stats in sorted(bucket_stats.items()):
                writer.writerow([
                    name,
                    stats.total_samples,
                    f"{stats.top1_accuracy:.4f}",
                    f"{stats.top3_accuracy:.4f}",
                    f"{stats.top5_accuracy:.4f}",
                ])


def analyze_buckets(
    samples: List[DataSample],
    predictions: List[List[Tuple[str, float]]],
    train_samples: Optional[List[DataSample]] = None,
    pipeline_debugs: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str] = None
) -> Dict[str, BucketStats]:
    """
    버킷 분석 편의 함수

    Args:
        samples: 데이터 샘플
        predictions: 예측 결과
        train_samples: 학습 데이터 (rare_class 분석용)
        pipeline_debugs: 파이프라인 디버그 정보
        output_path: 저장 경로 (선택적)

    Returns:
        {bucket_name: BucketStats}
    """
    analyzer = BucketAnalyzer()

    if train_samples:
        analyzer.set_train_distribution(train_samples)

    assignments, bucket_stats = analyzer.analyze(samples, predictions, pipeline_debugs)

    if output_path:
        analyzer.save_analysis(assignments, bucket_stats, output_path)

    return bucket_stats
