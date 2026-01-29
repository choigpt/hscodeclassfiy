"""
Benchmark Runner - 단일 엔트리포인트

전체 벤치마크 실험 실행:
1. 데이터 분할
2. 베이스라인 학습/평가
3. Ablation 실험
4. 결과 저장
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import yaml
import csv

# 모듈 임포트
from .data_split import DataSplitter, DataSample
from .baselines import TFIDFBaseline, SBertBaseline, BM25Baseline, create_baseline, Prediction
from .metrics import compute_metrics, compute_improvement, format_metrics_table
from .calibration import compute_calibration
from .routing import RoutingAnalyzer
from .error_analysis import ErrorAnalyzer
from .ablation_runner import AblationRunner, DEFAULT_ABLATIONS, generate_ablation_table


class BenchmarkRunner:
    """
    벤치마크 실험 통합 실행기
    """

    def __init__(self, config_path: str = "configs/benchmark.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

        # 경로 설정
        self.data_config = self.config.get('data', {})
        self.source_path = self.data_config.get('source_path', 'data/ruling_cases/all_cases_full_v7.json')
        self.benchmark_dir = self.data_config.get('benchmark_dir', 'data/benchmarks')
        self.output_dir = self.config.get('experiment', {}).get('output_dir', 'artifacts/reports')

        # 결과 저장
        self.results = {}
        self.baseline_results = {}
        self.ablation_results = {}

    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def prepare_data(self, force: bool = False) -> Dict[str, Any]:
        """
        Phase 1: 데이터 준비

        Args:
            force: 기존 데이터 덮어쓰기

        Returns:
            분할 통계
        """
        print("=" * 60)
        print("Phase 1: Data Preparation")
        print("=" * 60)

        # 기존 데이터 확인
        train_path = Path(self.benchmark_dir) / "hs4_train.jsonl"
        if train_path.exists() and not force:
            print(f"[Data] 기존 데이터 사용: {self.benchmark_dir}")

            # 통계 로드
            splits_path = Path(self.benchmark_dir) / "splits.json"
            if splits_path.exists():
                with open(splits_path, 'r', encoding='utf-8') as f:
                    return json.load(f).get('stats', {})
            return {}

        # 데이터 분할
        split_config = self.data_config.get('split', {})
        splitter = DataSplitter(
            train_ratio=split_config.get('train_ratio', 0.70),
            val_ratio=split_config.get('val_ratio', 0.15),
            test_ratio=split_config.get('test_ratio', 0.15),
            min_samples_per_class=split_config.get('min_samples_per_class', 3),
            seed=self.config.get('experiment', {}).get('seed', 42)
        )

        stats = splitter.split(self.source_path, self.benchmark_dir)
        return stats.to_dict()

    def load_data(self) -> Dict[str, List[DataSample]]:
        """데이터 로드"""
        return {
            'train': DataSplitter.load_split(f"{self.benchmark_dir}/hs4_train.jsonl"),
            'val': DataSplitter.load_split(f"{self.benchmark_dir}/hs4_val.jsonl"),
            'test': DataSplitter.load_split(f"{self.benchmark_dir}/hs4_test.jsonl"),
        }

    def run_baselines(
        self,
        train_data: List[DataSample],
        test_data: List[DataSample],
        skip_sbert: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Phase 2: 베이스라인 실험

        Args:
            train_data: 학습 데이터
            test_data: 테스트 데이터
            skip_sbert: SBert 스킵 (시간 절약)

        Returns:
            베이스라인 결과
        """
        print("\n" + "=" * 60)
        print("Phase 2: Baseline Experiments")
        print("=" * 60)

        baselines_config = self.config.get('models', {}).get('baselines', {})
        results = {}

        # 텍스트와 라벨 추출
        train_texts = [s.text for s in train_data]
        train_labels = [s.hs4 for s in train_data]
        test_texts = [s.text for s in test_data]
        test_labels = [s.hs4 for s in test_data]
        all_classes = list(set(train_labels + test_labels))

        for name, cfg in baselines_config.items():
            if not cfg.get('enabled', True):
                print(f"\n[Baseline] {name}: 스킵 (disabled)")
                continue

            if skip_sbert and cfg.get('type') == 'sbert_lr':
                print(f"\n[Baseline] {name}: 스킵 (--skip-sbert)")
                continue

            model_type = cfg.get('type')
            params = cfg.get('params', {})

            print(f"\n[Baseline] {name} ({model_type})")
            start_time = time.time()

            try:
                # 모델 생성 및 학습
                model = create_baseline(model_type, params)
                model.fit(train_texts, train_labels)

                # 예측
                predictions = []
                for text in test_texts:
                    preds = model.predict(text, topk=5)
                    predictions.append([(p.hs4, p.score) for p in preds])

                runtime = time.time() - start_time

                # 평가
                metrics = compute_metrics(test_labels, predictions, all_classes)

                # Calibration
                cal_result = compute_calibration(
                    metrics['top1_scores'],
                    metrics['top1_correct'],
                    n_bins=10
                )

                # Routing 분석
                routing_config = self.config.get('evaluation', {}).get('routing', {})
                routing_analyzer = RoutingAnalyzer(
                    auto_threshold=routing_config.get('thresholds', {}).get('auto', 0.7),
                    review_threshold=routing_config.get('thresholds', {}).get('review', 0.4)
                )
                _, routing_stats = routing_analyzer.analyze(
                    [s.id for s in test_data],
                    test_labels,
                    predictions
                )

                results[name] = {
                    'type': model_type,
                    'params': params,
                    'metrics': {
                        'top1_accuracy': metrics['top1_accuracy'],
                        'top3_accuracy': metrics['top3_accuracy'],
                        'top5_accuracy': metrics['top5_accuracy'],
                        'macro_f1': metrics['macro_f1'],
                        'weighted_f1': metrics['weighted_f1'],
                        'coverage': metrics['coverage'],
                    },
                    'calibration': {
                        'ece': cal_result.ece,
                        'brier_score': cal_result.brier_score,
                    },
                    'routing': routing_stats.to_dict(),
                    'runtime_seconds': runtime,
                }

                print(f"  Top-1: {metrics['top1_accuracy']:.4f}, "
                      f"Top-3: {metrics['top3_accuracy']:.4f}, "
                      f"F1: {metrics['macro_f1']:.4f}, "
                      f"ECE: {cal_result.ece:.4f}")

                # 모델 저장
                model_dir = Path(self.output_dir) / "models"
                model_dir.mkdir(parents=True, exist_ok=True)
                model.save(str(model_dir / f"{name}.pkl"))

            except Exception as e:
                print(f"  오류: {e}")
                import traceback
                traceback.print_exc()
                results[name] = {'error': str(e)}

        self.baseline_results = results
        return results

    def run_ablations(
        self,
        test_data: List[DataSample],
        ablation_names: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Phase 3: Ablation 실험

        Args:
            test_data: 테스트 데이터
            ablation_names: 실행할 Ablation (None이면 전체)
            limit: 샘플 수 제한

        Returns:
            Ablation 결과
        """
        print("\n" + "=" * 60)
        print("Phase 3: Ablation Experiments")
        print("=" * 60)

        if limit:
            test_data = test_data[:limit]
            print(f"[Ablation] 샘플 제한: {limit}")

        # Ranker 모델 경로
        ranker_path = None
        ranker_file = Path("artifacts/ranker/ranker_model.txt")
        if ranker_file.exists():
            ranker_path = str(ranker_file)
            print(f"[Ablation] Ranker 모델: {ranker_path}")

        # Routing 설정
        routing_config = self.config.get('evaluation', {}).get('routing', {})

        runner = AblationRunner(
            ranker_model_path=ranker_path,
            auto_threshold=routing_config.get('thresholds', {}).get('auto', 0.7),
            review_threshold=routing_config.get('thresholds', {}).get('review', 0.4)
        )

        results = runner.run_all(test_data, ablation_names)

        # 결과 변환
        self.ablation_results = {
            name: result.to_dict()
            for name, result in results.items()
        }

        return self.ablation_results

    def save_results(self):
        """
        Phase 4: 결과 저장
        """
        print("\n" + "=" * 60)
        print("Phase 4: Saving Results")
        print("=" * 60)

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 전체 결과 요약
        summary = {
            'timestamp': timestamp,
            'config': self.config_path,
            'baselines': self.baseline_results,
            'ablations': self.ablation_results,
        }

        with open(output_path / "benchmark_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  저장: benchmark_summary.json")

        # 2. 베이스라인 결과 CSV
        if self.baseline_results:
            with open(output_path / "benchmark_summary.csv", 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Model", "Type", "Top1", "Top3", "Top5",
                    "Macro-F1", "Weighted-F1", "ECE", "Runtime(s)"
                ])

                for name, result in self.baseline_results.items():
                    if 'error' in result:
                        continue
                    metrics = result.get('metrics', {})
                    cal = result.get('calibration', {})
                    writer.writerow([
                        name,
                        result.get('type', ''),
                        f"{metrics.get('top1_accuracy', 0):.4f}",
                        f"{metrics.get('top3_accuracy', 0):.4f}",
                        f"{metrics.get('top5_accuracy', 0):.4f}",
                        f"{metrics.get('macro_f1', 0):.4f}",
                        f"{metrics.get('weighted_f1', 0):.4f}",
                        f"{cal.get('ece', 0):.4f}",
                        f"{result.get('runtime_seconds', 0):.1f}",
                    ])
            print(f"  저장: benchmark_summary.csv")

        # 3. Ablation 테이블 CSV
        if self.ablation_results:
            with open(output_path / "ablation_table.csv", 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Model", "use_gri", "use_8axis", "use_rules", "use_ranker",
                    "Top1", "Top3", "Top5", "Macro-F1", "ECE", "Runtime(s)"
                ])

                for name, result in self.ablation_results.items():
                    cfg = result.get('config', {})
                    metrics = result.get('metrics', {})
                    cal = result.get('calibration', {})
                    writer.writerow([
                        name,
                        "O" if cfg.get('use_gri') else "X",
                        "O" if cfg.get('use_8axis') else "X",
                        "O" if cfg.get('use_rules') else "X",
                        "O" if cfg.get('use_ranker') else "X",
                        f"{metrics.get('top1_accuracy', 0):.4f}",
                        f"{metrics.get('top3_accuracy', 0):.4f}",
                        f"{metrics.get('top5_accuracy', 0):.4f}",
                        f"{metrics.get('macro_f1', 0):.4f}",
                        f"{cal.get('ece', 0):.4f}",
                        f"{result.get('runtime_seconds', 0):.1f}",
                    ])
            print(f"  저장: ablation_table.csv")

        # 4. Calibration 데이터
        calibration_data = {
            'baselines': {
                name: result.get('calibration', {})
                for name, result in self.baseline_results.items()
                if 'error' not in result
            },
            'ablations': {
                name: result.get('calibration', {})
                for name, result in self.ablation_results.items()
            }
        }
        with open(output_path / "calibration.json", 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, ensure_ascii=False, indent=2)
        print(f"  저장: calibration.json")

    def print_summary(self):
        """결과 요약 출력"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        # 베이스라인 결과
        if self.baseline_results:
            print("\n--- Baselines ---")
            for name, result in self.baseline_results.items():
                if 'error' in result:
                    print(f"  {name}: ERROR - {result['error']}")
                    continue
                metrics = result.get('metrics', {})
                print(f"  {name}: Top1={metrics.get('top1_accuracy', 0):.4f}, "
                      f"Top3={metrics.get('top3_accuracy', 0):.4f}, "
                      f"F1={metrics.get('macro_f1', 0):.4f}")

        # Ablation 결과
        if self.ablation_results:
            print("\n--- Ablations ---")
            for name, result in self.ablation_results.items():
                metrics = result.get('metrics', {})
                print(f"  {name}: Top1={metrics.get('top1_accuracy', 0):.4f}, "
                      f"Top3={metrics.get('top3_accuracy', 0):.4f}, "
                      f"F1={metrics.get('macro_f1', 0):.4f}")

            # B1 vs P5 비교 (있으면)
            if 'B1_sbert_lr' in self.baseline_results and 'P5_plus_ranker' in self.ablation_results:
                b1 = self.baseline_results['B1_sbert_lr'].get('metrics', {})
                p5 = self.ablation_results['P5_plus_ranker'].get('metrics', {})

                print("\n--- B1 vs P5 Improvement ---")
                for metric in ['top1_accuracy', 'top3_accuracy', 'macro_f1']:
                    b1_val = b1.get(metric, 0)
                    p5_val = p5.get(metric, 0)
                    imp = p5_val - b1_val
                    print(f"  {metric}: {b1_val:.4f} -> {p5_val:.4f} (+{imp:.4f})")


def main():
    """CLI 엔트리포인트"""
    parser = argparse.ArgumentParser(
        description="HS4 Classification Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 전체 벤치마크 실행
  python -m src.experiments.run_benchmark

  # 데이터만 준비
  python -m src.experiments.run_benchmark --phase data

  # 베이스라인만 실행
  python -m src.experiments.run_benchmark --phase baselines

  # 특정 Ablation만 실행
  python -m src.experiments.run_benchmark --phase ablations --ablation P5_plus_ranker

  # 샘플 제한 (테스트용)
  python -m src.experiments.run_benchmark --limit 100
        """
    )

    parser.add_argument("--config", default="configs/benchmark.yaml", help="설정 파일 경로")
    parser.add_argument("--phase", choices=['all', 'data', 'baselines', 'ablations'],
                        default='all', help="실행할 단계")
    parser.add_argument("--ablation", help="특정 Ablation만 실행")
    parser.add_argument("--skip-sbert", action="store_true", help="SBert 베이스라인 스킵")
    parser.add_argument("--limit", type=int, help="샘플 수 제한 (테스트용)")
    parser.add_argument("--force-split", action="store_true", help="데이터 강제 재분할")

    args = parser.parse_args()

    print("=" * 60)
    print("HS4 Classification Benchmark")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Phase: {args.phase}")

    # 벤치마크 실행기 생성
    runner = BenchmarkRunner(args.config)

    # Phase 1: 데이터 준비
    if args.phase in ['all', 'data']:
        runner.prepare_data(force=args.force_split)

    # 데이터 로드
    if args.phase in ['all', 'baselines', 'ablations']:
        data = runner.load_data()
        print(f"\nData loaded: train={len(data['train'])}, "
              f"val={len(data['val'])}, test={len(data['test'])}")

    # Phase 2: 베이스라인
    if args.phase in ['all', 'baselines']:
        runner.run_baselines(
            data['train'],
            data['test'],
            skip_sbert=args.skip_sbert
        )

    # Phase 3: Ablation
    if args.phase in ['all', 'ablations']:
        ablation_names = [args.ablation] if args.ablation else None
        runner.run_ablations(
            data['test'],
            ablation_names=ablation_names,
            limit=args.limit
        )

    # Phase 4: 결과 저장
    if args.phase != 'data':
        runner.save_results()
        runner.print_summary()

    print("\n[완료]")


if __name__ == "__main__":
    main()
