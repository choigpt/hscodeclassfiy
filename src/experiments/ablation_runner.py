"""
Ablation Runner Module

Ablation 실험 실행:
- P1-P6 파이프라인 변형 실행
- 피처 토글 관리
- 결과 수집
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# 상대 임포트를 절대 임포트로 변경 (모듈 실행 시)
try:
    from .data_split import DataSample, DataSplitter
    from .metrics import compute_metrics
    from .calibration import compute_calibration
    from .routing import RoutingAnalyzer
    from .error_analysis import ErrorAnalyzer
except ImportError:
    from data_split import DataSample, DataSplitter
    from metrics import compute_metrics
    from calibration import compute_calibration
    from routing import RoutingAnalyzer
    from error_analysis import ErrorAnalyzer


@dataclass
class AblationConfig:
    """Ablation 설정"""
    name: str
    description: str
    use_gri: bool = False
    use_8axis: bool = False
    use_rules: bool = False
    use_ranker: bool = False
    use_questions: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "use_gri": self.use_gri,
            "use_8axis": self.use_8axis,
            "use_rules": self.use_rules,
            "use_ranker": self.use_ranker,
            "use_questions": self.use_questions,
        }


@dataclass
class AblationResult:
    """Ablation 실험 결과"""
    config: AblationConfig
    metrics: Dict[str, Any]
    calibration: Dict[str, Any]
    routing: Dict[str, Any]
    error_analysis: Dict[str, Any]
    runtime_seconds: float
    predictions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "calibration": self.calibration,
            "routing": self.routing,
            "error_analysis": self.error_analysis,
            "runtime_seconds": round(self.runtime_seconds, 2),
        }


# 기본 Ablation 설정들
DEFAULT_ABLATIONS = {
    "P1_ml_only": AblationConfig(
        name="P1_ml_only",
        description="ML only (no KB)",
        use_gri=False, use_8axis=False, use_rules=False, use_ranker=False, use_questions=False
    ),
    "P2_plus_gri": AblationConfig(
        name="P2_plus_gri",
        description="ML + GRI signals",
        use_gri=True, use_8axis=False, use_rules=False, use_ranker=False, use_questions=False
    ),
    "P3_plus_8axis": AblationConfig(
        name="P3_plus_8axis",
        description="ML + GRI + 8-axis attributes",
        use_gri=True, use_8axis=True, use_rules=False, use_ranker=False, use_questions=False
    ),
    "P4_plus_rules": AblationConfig(
        name="P4_plus_rules",
        description="ML + GRI + 8-axis + KB rules",
        use_gri=True, use_8axis=True, use_rules=True, use_ranker=False, use_questions=False
    ),
    "P5_plus_ranker": AblationConfig(
        name="P5_plus_ranker",
        description="ML + GRI + 8-axis + rules + LightGBM ranker",
        use_gri=True, use_8axis=True, use_rules=True, use_ranker=True, use_questions=False
    ),
    "P6_full": AblationConfig(
        name="P6_full",
        description="Full pipeline with question generation",
        use_gri=True, use_8axis=True, use_rules=True, use_ranker=True, use_questions=True
    ),
}


class AblationRunner:
    """
    Ablation 실험 실행기

    파이프라인 컴포넌트 토글 및 실험 실행
    """

    def __init__(
        self,
        ranker_model_path: Optional[str] = None,
        auto_threshold: float = 0.7,
        review_threshold: float = 0.4
    ):
        self.ranker_model_path = ranker_model_path
        self.auto_threshold = auto_threshold
        self.review_threshold = review_threshold

        self.pipeline = None
        self.routing_analyzer = RoutingAnalyzer(auto_threshold, review_threshold)
        self.error_analyzer = ErrorAnalyzer()

    def _init_pipeline(self, config: AblationConfig):
        """파이프라인 초기화 (토글 적용)"""
        # 지연 임포트
        from src.classifier.pipeline import HSPipeline
        from src.classifier.retriever import HSRetriever
        from src.classifier.reranker import HSReranker
        from src.classifier.clarify import HSClarifier

        # Ranker 모델 경로
        ranker_path = self.ranker_model_path if config.use_ranker else None

        self.pipeline = HSPipeline(
            retriever=HSRetriever(),
            reranker=HSReranker(),
            clarifier=HSClarifier(),
            ranker_model_path=ranker_path
        )

        # 설정 저장 (classify에서 참조)
        self._current_config = config

    def _classify_with_ablation(
        self,
        text: str,
        config: AblationConfig,
        topk: int = 5
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        """
        Ablation 설정에 따른 분류

        Args:
            text: 입력 텍스트
            config: Ablation 설정
            topk: 반환할 후보 수

        Returns:
            (예측 리스트 [(hs4, score), ...], 피처 정보)
        """
        from src.classifier.gri_signals import detect_gri_signals
        from src.classifier.attribute_extract import extract_attributes, extract_attributes_8axis

        # GRI 신호 탐지 (토글)
        gri_signals = detect_gri_signals(text) if config.use_gri else None

        # 8축 속성 추출 (토글)
        input_attrs = extract_attributes(text)
        input_attrs_8axis = extract_attributes_8axis(text) if config.use_8axis else None

        # ML 후보 생성
        ml_candidates = []
        if self.pipeline.retriever.is_ready():
            ml_candidates = self.pipeline.retriever.predict_topk(text, k=50)

        # KB 후보 및 Rerank (토글)
        if config.use_rules:
            kb_candidates = self.pipeline.reranker.retrieve_from_kb(
                text, topk=30,
                gri_signals=gri_signals,
                input_attrs=input_attrs,
                input_attrs_8axis=input_attrs_8axis
            )
            candidates = self.pipeline._merge_candidates(ml_candidates, kb_candidates)

            reranked, stats = self.pipeline.reranker.rerank(
                text, candidates, topk=topk,
                gri_signals=gri_signals,
                input_attrs=input_attrs,
                input_attrs_8axis=input_attrs_8axis,
                model_classes=self.pipeline.get_model_classes(),
                ranker_model=self.pipeline.ranker_model if config.use_ranker else None
            )

            predictions = [(c.hs4, c.score_total) for c in reranked]
            features = reranked[0].features if reranked else {}
        else:
            # KB 없이 ML만
            predictions = [(c.hs4, c.score_ml) for c in ml_candidates[:topk]]
            features = {}
            stats = {}

        return predictions, features

    def run_single(
        self,
        config: AblationConfig,
        samples: List[DataSample],
        topk: int = 5
    ) -> AblationResult:
        """
        단일 Ablation 실험 실행

        Args:
            config: Ablation 설정
            samples: 테스트 샘플
            topk: Top-K 예측

        Returns:
            실험 결과
        """
        print(f"\n[Ablation] {config.name}: {config.description}")
        print(f"  GRI={config.use_gri}, 8axis={config.use_8axis}, "
              f"rules={config.use_rules}, ranker={config.use_ranker}")

        # 파이프라인 초기화
        self._init_pipeline(config)

        # 예측 수행
        start_time = time.time()

        sample_ids = []
        texts = []
        true_labels = []
        all_predictions = []
        all_features = []

        for i, sample in enumerate(samples):
            if i % 100 == 0:
                print(f"  처리 중: {i}/{len(samples)}")

            predictions, features = self._classify_with_ablation(
                sample.text, config, topk
            )

            sample_ids.append(sample.id)
            texts.append(sample.text)
            true_labels.append(sample.hs4)
            all_predictions.append(predictions)
            all_features.append(features)

        runtime = time.time() - start_time
        print(f"  완료: {runtime:.1f}초")

        # 평가 메트릭 계산
        all_classes = list(set(true_labels))
        metrics = compute_metrics(true_labels, all_predictions, all_classes)

        # Calibration
        cal_result = compute_calibration(
            metrics['top1_scores'],
            metrics['top1_correct'],
            n_bins=10
        )

        # Routing 분석
        _, routing_stats = self.routing_analyzer.analyze(
            sample_ids, true_labels, all_predictions
        )

        # Error 분석
        error_results = self.error_analyzer.analyze(
            sample_ids, texts, true_labels, all_predictions, all_features
        )

        return AblationResult(
            config=config,
            metrics={
                "n_samples": metrics["n_samples"],
                "n_classes": metrics["n_classes"],
                "top1_accuracy": metrics["top1_accuracy"],
                "top3_accuracy": metrics["top3_accuracy"],
                "top5_accuracy": metrics["top5_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "coverage": metrics["coverage"],
            },
            calibration=cal_result.to_dict(),
            routing=routing_stats.to_dict(),
            error_analysis=error_results,
            runtime_seconds=runtime,
            predictions=[
                {
                    "sample_id": sid,
                    "true_hs4": true,
                    "predictions": preds
                }
                for sid, true, preds in zip(sample_ids, true_labels, all_predictions)
            ]
        )

    def run_all(
        self,
        samples: List[DataSample],
        ablation_names: Optional[List[str]] = None,
        topk: int = 5
    ) -> Dict[str, AblationResult]:
        """
        전체 Ablation 실험 실행

        Args:
            samples: 테스트 샘플
            ablation_names: 실행할 Ablation 이름 (None이면 전체)
            topk: Top-K 예측

        Returns:
            {ablation_name: result}
        """
        if ablation_names is None:
            ablation_names = list(DEFAULT_ABLATIONS.keys())

        results = {}

        for name in ablation_names:
            if name not in DEFAULT_ABLATIONS:
                print(f"[경고] 알 수 없는 Ablation: {name}")
                continue

            config = DEFAULT_ABLATIONS[name]
            result = self.run_single(config, samples, topk)
            results[name] = result

        return results

    def compare_results(
        self,
        results: Dict[str, AblationResult],
        baseline_name: str = "P1_ml_only"
    ) -> Dict[str, Any]:
        """
        Ablation 결과 비교

        Args:
            results: Ablation 결과들
            baseline_name: 베이스라인 이름

        Returns:
            비교 결과
        """
        if baseline_name not in results:
            print(f"[경고] 베이스라인 {baseline_name} 없음")
            return {}

        baseline = results[baseline_name]
        comparison = {}

        metrics_to_compare = [
            "top1_accuracy", "top3_accuracy", "top5_accuracy",
            "macro_f1", "weighted_f1"
        ]

        for name, result in results.items():
            if name == baseline_name:
                continue

            improvements = {}
            for metric in metrics_to_compare:
                base_val = baseline.metrics.get(metric, 0)
                curr_val = result.metrics.get(metric, 0)
                abs_imp = curr_val - base_val
                rel_imp = abs_imp / base_val if base_val > 0 else 0

                improvements[metric] = {
                    "baseline": base_val,
                    "current": curr_val,
                    "absolute": abs_imp,
                    "relative": rel_imp,
                }

            comparison[name] = {
                "config": result.config.to_dict(),
                "improvements": improvements,
                "calibration_ece": result.calibration.get("ece", 0),
                "routing_auto_accuracy": result.routing.get("accuracy", {}).get("auto", 0),
            }

        return comparison


def generate_ablation_table(
    results: Dict[str, AblationResult],
    output_path: Optional[str] = None
) -> str:
    """
    Ablation 결과 테이블 생성

    Args:
        results: Ablation 결과
        output_path: CSV 저장 경로 (옵션)

    Returns:
        포맷팅된 테이블 문자열
    """
    import csv
    from io import StringIO

    # 헤더
    headers = [
        "Model", "GRI", "8axis", "Rules", "Ranker",
        "Top1", "Top3", "Top5", "Macro-F1", "ECE", "Runtime(s)"
    ]

    rows = []
    for name, result in results.items():
        cfg = result.config
        rows.append([
            name,
            "O" if cfg.use_gri else "X",
            "O" if cfg.use_8axis else "X",
            "O" if cfg.use_rules else "X",
            "O" if cfg.use_ranker else "X",
            f"{result.metrics['top1_accuracy']:.4f}",
            f"{result.metrics['top3_accuracy']:.4f}",
            f"{result.metrics['top5_accuracy']:.4f}",
            f"{result.metrics['macro_f1']:.4f}",
            f"{result.calibration['ece']:.4f}",
            f"{result.runtime_seconds:.1f}",
        ])

    # CSV 저장
    if output_path:
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"[Ablation] 테이블 저장: {output_path}")

    # 문자열 테이블 생성
    output = StringIO()
    writer = csv.writer(output, delimiter='|')
    writer.writerow(headers)
    writer.writerow(['-' * len(h) for h in headers])
    writer.writerows(rows)

    return output.getvalue()


# CLI
if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Ablation 실험")
    parser.add_argument("--config", default="configs/benchmark.yaml", help="설정 파일")
    parser.add_argument("--ablation", help="특정 Ablation만 실행")
    parser.add_argument("--test-data", default="data/benchmarks/hs4_test.jsonl", help="테스트 데이터")
    parser.add_argument("--output-dir", default="artifacts/reports", help="출력 디렉토리")
    parser.add_argument("--limit", type=int, help="샘플 수 제한 (테스트용)")
    args = parser.parse_args()

    # 테스트 데이터 로드
    print(f"[Ablation] 테스트 데이터 로드: {args.test_data}")
    samples = DataSplitter.load_split(args.test_data)

    if args.limit:
        samples = samples[:args.limit]
        print(f"  샘플 수 제한: {args.limit}")

    print(f"  샘플 수: {len(samples)}")

    # Ablation 실행
    runner = AblationRunner()

    if args.ablation:
        ablation_names = [args.ablation]
    else:
        ablation_names = None

    results = runner.run_all(samples, ablation_names)

    # 결과 저장
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 테이블 생성
    table = generate_ablation_table(results, output_dir / "ablation_table.csv")
    print("\n" + "=" * 60)
    print("Ablation Results")
    print("=" * 60)
    print(table)

    # 비교 결과
    comparison = runner.compare_results(results)
    with open(output_dir / "ablation_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n비교 결과 저장: {output_dir / 'ablation_comparison.json'}")
