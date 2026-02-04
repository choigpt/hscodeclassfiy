"""
Enhanced Evaluator - 4개 모델 비교 + Bucket 분석

사용자 요구사항:
1. 4개 모델 비교: TFIDF+LR, ST+LR, KB-only, Hybrid(LegalGate+Ranker)
2. 지표: Top1/Top3/MacroF1/ECE, AUTO/ASK/REVIEW/ABSTAIN 비율, 후보 리콜
3. Bucket별 성능: fact-insufficient, legal-conflict, short-text
"""

import json
import time
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .data_split import DataSample, DataSplitter
from .baselines import TFIDFBaseline, SBertBaseline, Prediction
from .kb_only_model import KBOnlyModel
from .metrics import compute_metrics
from .calibration import compute_calibration
from .routing import RoutingAnalyzer
from .bucket_analyzer import BucketAnalyzer, BucketStats


@dataclass
class ModelResult:
    """단일 모델 평가 결과"""
    model_name: str
    model_type: str

    # 기본 지표
    top1_accuracy: float = 0.0
    top3_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0

    # Calibration
    ece: float = 0.0
    brier_score: float = 0.0

    # Routing (AUTO/ASK/REVIEW/ABSTAIN)
    auto_rate: float = 0.0
    ask_rate: float = 0.0
    review_rate: float = 0.0
    abstain_rate: float = 0.0

    # 후보 리콜 (정답이 Top-K 후보에 포함된 비율)
    candidate_recall_top5: float = 0.0
    candidate_recall_top10: float = 0.0
    candidate_recall_top20: float = 0.0

    # 런타임
    runtime_seconds: float = 0.0

    # Bucket별 성능
    bucket_stats: Dict[str, BucketStats] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'top1_accuracy': round(self.top1_accuracy, 4),
            'top3_accuracy': round(self.top3_accuracy, 4),
            'top5_accuracy': round(self.top5_accuracy, 4),
            'macro_f1': round(self.macro_f1, 4),
            'weighted_f1': round(self.weighted_f1, 4),
            'ece': round(self.ece, 4),
            'brier_score': round(self.brier_score, 4),
            'auto_rate': round(self.auto_rate, 4),
            'ask_rate': round(self.ask_rate, 4),
            'review_rate': round(self.review_rate, 4),
            'abstain_rate': round(self.abstain_rate, 4),
            'candidate_recall_top5': round(self.candidate_recall_top5, 4),
            'candidate_recall_top10': round(self.candidate_recall_top10, 4),
            'candidate_recall_top20': round(self.candidate_recall_top20, 4),
            'runtime_seconds': round(self.runtime_seconds, 2),
            'bucket_stats': {
                name: stats.to_dict()
                for name, stats in self.bucket_stats.items()
            }
        }


class EnhancedEvaluator:
    """
    4개 모델 비교 평가기

    모델:
    1. TFIDF+LR: TF-IDF 벡터화 + Logistic Regression
    2. ST+LR: Sentence Transformer + Logistic Regression
    3. KB-only: LegalGate + KB 매칭 (ML 없음)
    4. Hybrid: LegalGate + Ranker (전체 파이프라인)
    """

    def __init__(
        self,
        benchmark_dir: str = "data/benchmarks",
        output_dir: str = "artifacts/evaluation",
        auto_threshold: float = 0.7,
        review_threshold: float = 0.4
    ):
        """
        Args:
            benchmark_dir: 벤치마크 데이터 디렉토리
            output_dir: 출력 디렉토리
            auto_threshold: AUTO 라우팅 임계값
            review_threshold: REVIEW 라우팅 임계값
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.auto_threshold = auto_threshold
        self.review_threshold = review_threshold

        # 데이터
        self.train_data: List[DataSample] = []
        self.val_data: List[DataSample] = []
        self.test_data: List[DataSample] = []

        # 평가 결과
        self.results: Dict[str, ModelResult] = {}

        # Bucket Analyzer
        self.bucket_analyzer = BucketAnalyzer()

    def load_data(self) -> None:
        """벤치마크 데이터 로드"""
        print("데이터 로드 중...")

        self.train_data = DataSplitter.load_split(
            str(self.benchmark_dir / "hs4_train.jsonl")
        )
        self.val_data = DataSplitter.load_split(
            str(self.benchmark_dir / "hs4_val.jsonl")
        )
        self.test_data = DataSplitter.load_split(
            str(self.benchmark_dir / "hs4_test.jsonl")
        )

        print(f"  Train: {len(self.train_data)}")
        print(f"  Val: {len(self.val_data)}")
        print(f"  Test: {len(self.test_data)}")

        # Bucket Analyzer에 학습 데이터 분포 설정
        self.bucket_analyzer.set_train_distribution(self.train_data)

    def evaluate_model(
        self,
        model_name: str,
        model_type: str,
        model: Any,
        train_data: List[DataSample],
        test_data: List[DataSample],
        skip_training: bool = False
    ) -> ModelResult:
        """
        단일 모델 평가

        Args:
            model_name: 모델 이름
            model_type: 모델 타입
            model: 모델 인스턴스
            train_data: 학습 데이터
            test_data: 테스트 데이터
            skip_training: 학습 스킵 (KB-only 등)

        Returns:
            ModelResult
        """
        print(f"\n[{model_name}] 평가 중...")

        start_time = time.time()

        # 1. 학습 (필요한 경우)
        if not skip_training:
            print(f"  학습 중...")
            train_texts = [s.text for s in train_data]
            train_labels = [s.hs4 for s in train_data]
            model.fit(train_texts, train_labels)

        # 2. 예측
        print(f"  예측 중...")
        test_texts = [s.text for s in test_data]
        test_labels = [s.hs4 for s in test_data]

        predictions = []
        predictions_extended = []  # Top-20 for recall

        for text in test_texts:
            preds_5 = model.predict(text, topk=5)
            preds_20 = model.predict(text, topk=20)

            predictions.append([(p.hs4, p.score) for p in preds_5])
            predictions_extended.append([(p.hs4, p.score) for p in preds_20])

        runtime = time.time() - start_time

        # 3. 기본 지표 계산
        all_classes = list(set(train_labels + test_labels))
        metrics = compute_metrics(test_labels, predictions, all_classes)

        # 4. Calibration
        cal_result = compute_calibration(
            metrics['top1_scores'],
            metrics['top1_correct'],
            n_bins=10
        )

        # 5. Routing 분석
        routing_analyzer = RoutingAnalyzer(
            self.auto_threshold,
            self.review_threshold
        )
        _, routing_stats = routing_analyzer.analyze(
            [s.id for s in test_data],
            test_labels,
            predictions
        )

        # 6. 후보 리콜 계산
        recall_5 = self._compute_recall(test_labels, predictions, k=5)
        recall_10 = self._compute_recall(test_labels, predictions_extended, k=10)
        recall_20 = self._compute_recall(test_labels, predictions_extended, k=20)

        # 7. Bucket 분석
        bucket_assignments, bucket_stats = self.bucket_analyzer.analyze(
            test_data,
            predictions,
            pipeline_debugs=None  # 파이프라인 디버그는 Hybrid에서만 사용
        )

        # 결과 생성
        result = ModelResult(
            model_name=model_name,
            model_type=model_type,
            top1_accuracy=metrics['top1_accuracy'],
            top3_accuracy=metrics['top3_accuracy'],
            top5_accuracy=metrics['top5_accuracy'],
            macro_f1=metrics['macro_f1'],
            weighted_f1=metrics['weighted_f1'],
            ece=cal_result.ece,
            brier_score=cal_result.brier_score,
            auto_rate=routing_stats.auto_rate,
            ask_rate=routing_stats.ask_rate,
            review_rate=routing_stats.review_rate,
            abstain_rate=routing_stats.abstain_rate,
            candidate_recall_top5=recall_5,
            candidate_recall_top10=recall_10,
            candidate_recall_top20=recall_20,
            runtime_seconds=runtime,
            bucket_stats=bucket_stats
        )

        print(f"  Top1: {result.top1_accuracy:.4f}, "
              f"Top3: {result.top3_accuracy:.4f}, "
              f"F1: {result.macro_f1:.4f}, "
              f"ECE: {result.ece:.4f}")

        return result

    def _compute_recall(
        self,
        labels: List[str],
        predictions: List[List[Tuple[str, float]]],
        k: int
    ) -> float:
        """
        후보 리콜 계산 (정답이 Top-K 후보에 포함된 비율)

        Args:
            labels: 정답 라벨
            predictions: 예측 결과
            k: Top-K

        Returns:
            리콜 (0.0 ~ 1.0)
        """
        correct = 0
        for label, preds in zip(labels, predictions):
            pred_hs4s = [p[0] for p in preds[:k]]
            if label in pred_hs4s:
                correct += 1

        return correct / len(labels) if len(labels) > 0 else 0.0

    def run_all(
        self,
        skip_sbert: bool = False,
        skip_hybrid: bool = False
    ) -> Dict[str, ModelResult]:
        """
        4개 모델 전체 평가

        Args:
            skip_sbert: SBert 스킵 (시간 절약)
            skip_hybrid: Hybrid 스킵

        Returns:
            {model_name: ModelResult}
        """
        print("=" * 60)
        print("Enhanced Evaluation - 4 Models Comparison")
        print("=" * 60)

        # 1. TFIDF+LR
        print("\n[1/4] TFIDF+LR")
        tfidf_model = TFIDFBaseline()
        self.results['TFIDF+LR'] = self.evaluate_model(
            'TFIDF+LR',
            'tfidf_lr',
            tfidf_model,
            self.train_data,
            self.test_data
        )

        # 2. ST+LR (SBert+LR)
        if not skip_sbert:
            print("\n[2/4] ST+LR (SBert)")
            sbert_model = SBertBaseline()
            self.results['ST+LR'] = self.evaluate_model(
                'ST+LR',
                'sbert_lr',
                sbert_model,
                self.train_data,
                self.test_data
            )
        else:
            print("\n[2/4] ST+LR (SBert) - SKIPPED")

        # 3. KB-only
        print("\n[3/4] KB-only")
        kb_only_model = KBOnlyModel(use_legal_gate=True, kb_topk=30)
        self.results['KB-only'] = self.evaluate_model(
            'KB-only',
            'kb_only',
            kb_only_model,
            self.train_data,
            self.test_data,
            skip_training=True  # KB-only는 학습 불필요
        )

        # 4. Hybrid (LegalGate + Ranker)
        if not skip_hybrid:
            print("\n[4/4] Hybrid (LegalGate + Ranker)")
            # TODO: Hybrid 모델 구현 (파이프라인 사용)
            print("  [TODO] Hybrid 모델 구현 필요")
        else:
            print("\n[4/4] Hybrid - SKIPPED")

        return self.results

    def save_results(self) -> None:
        """결과 저장"""
        print("\n결과 저장 중...")

        # 1. JSON 저장
        json_file = self.output_dir / "evaluation_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                name: result.to_dict()
                for name, result in self.results.items()
            }, f, ensure_ascii=False, indent=2)
        print(f"  저장: {json_file}")

        # 2. 비교 테이블 CSV
        csv_file = self.output_dir / "comparison_table.csv"
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Model', 'Type', 'Top1', 'Top3', 'Top5', 'MacroF1',
                'ECE', 'AUTO%', 'ASK%', 'REVIEW%', 'Recall@5', 'Recall@10', 'Runtime(s)'
            ])

            for name, result in self.results.items():
                writer.writerow([
                    name,
                    result.model_type,
                    f"{result.top1_accuracy:.4f}",
                    f"{result.top3_accuracy:.4f}",
                    f"{result.top5_accuracy:.4f}",
                    f"{result.macro_f1:.4f}",
                    f"{result.ece:.4f}",
                    f"{result.auto_rate*100:.1f}",
                    f"{result.ask_rate*100:.1f}",
                    f"{result.review_rate*100:.1f}",
                    f"{result.candidate_recall_top5:.4f}",
                    f"{result.candidate_recall_top10:.4f}",
                    f"{result.runtime_seconds:.1f}",
                ])
        print(f"  저장: {csv_file}")

        # 3. Bucket 성능 CSV
        bucket_csv = self.output_dir / "bucket_performance.csv"
        with open(bucket_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Bucket', 'Total', 'Top1', 'Top3', 'Top5'])

            for model_name, result in self.results.items():
                for bucket_name, stats in result.bucket_stats.items():
                    writer.writerow([
                        model_name,
                        bucket_name,
                        stats.total_samples,
                        f"{stats.top1_accuracy:.4f}",
                        f"{stats.top3_accuracy:.4f}",
                        f"{stats.top5_accuracy:.4f}",
                    ])
        print(f"  저장: {bucket_csv}")

    def print_summary(self) -> None:
        """결과 요약 출력"""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        for name, result in self.results.items():
            print(f"\n[{name}] ({result.model_type})")
            print(f"  Top1: {result.top1_accuracy:.4f}, "
                  f"Top3: {result.top3_accuracy:.4f}, "
                  f"F1: {result.macro_f1:.4f}, "
                  f"ECE: {result.ece:.4f}")
            print(f"  Routing: AUTO={result.auto_rate*100:.1f}%, "
                  f"ASK={result.ask_rate*100:.1f}%, "
                  f"REVIEW={result.review_rate*100:.1f}%")
            print(f"  Recall: @5={result.candidate_recall_top5:.4f}, "
                  f"@10={result.candidate_recall_top10:.4f}")
            print(f"  Runtime: {result.runtime_seconds:.1f}s")

            if result.bucket_stats:
                print(f"  Buckets:")
                for bucket_name, stats in result.bucket_stats.items():
                    print(f"    - {bucket_name}: {stats.total_samples} samples, "
                          f"Top1={stats.top1_accuracy:.4f}")


def main():
    """CLI 엔트리포인트"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced Evaluation - 4 Models Comparison"
    )
    parser.add_argument("--benchmark-dir", default="data/benchmarks",
                        help="벤치마크 데이터 디렉토리")
    parser.add_argument("--output-dir", default="artifacts/evaluation",
                        help="출력 디렉토리")
    parser.add_argument("--skip-sbert", action="store_true",
                        help="SBert 스킵")
    parser.add_argument("--skip-hybrid", action="store_true",
                        help="Hybrid 스킵")

    args = parser.parse_args()

    evaluator = EnhancedEvaluator(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir
    )

    evaluator.load_data()
    evaluator.run_all(
        skip_sbert=args.skip_sbert,
        skip_hybrid=args.skip_hybrid
    )
    evaluator.save_results()
    evaluator.print_summary()

    print("\n[완료]")


if __name__ == "__main__":
    main()
