"""
Full Evaluation Runner - 평가/진단/설명 통합 실행

실행 내용:
1. 4개 모델 비교 평가 (TFIDF+LR, ST+LR, KB-only, Hybrid)
2. Bucket별 성능 분석 (fact-insufficient, legal-conflict, short-text)
3. 진단 분석 (LegalGate 제외, Missing Facts, Confusion Pairs)
4. 설명 생성 (분류 근거 요약)
"""

import argparse
from pathlib import Path

from src.experiments.enhanced_evaluator import EnhancedEvaluator
from src.experiments.enhanced_diagnostics import EnhancedDiagnostics
from src.classifier.explanation_generator import ExplanationGenerator


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="Full Evaluation - 평가/진단/설명 통합",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 전체 실행
  python scripts/run_full_evaluation.py

  # TFIDF+LR과 KB-only만
  python scripts/run_full_evaluation.py --skip-sbert --skip-hybrid

  # 빠른 테스트
  python scripts/run_full_evaluation.py --skip-sbert --skip-hybrid --limit 100
        """
    )

    parser.add_argument("--benchmark-dir", default="data/benchmarks",
                        help="벤치마크 데이터 디렉토리")
    parser.add_argument("--output-dir", default="artifacts/full_evaluation",
                        help="출력 디렉토리")
    parser.add_argument("--skip-sbert", action="store_true",
                        help="SBert 스킵 (시간 절약)")
    parser.add_argument("--skip-hybrid", action="store_true",
                        help="Hybrid 스킵 (미구현)")
    parser.add_argument("--limit", type=int,
                        help="테스트 샘플 수 제한")

    args = parser.parse_args()

    print("=" * 80)
    print("Full Evaluation - 평가/진단/설명 통합")
    print("=" * 80)
    print(f"Benchmark Dir: {args.benchmark_dir}")
    print(f"Output Dir: {args.output_dir}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ========================================
    # Phase 1: 4개 모델 비교 평가
    # ========================================
    print("\n" + "=" * 80)
    print("Phase 1: 4 Models Comparison")
    print("=" * 80)

    evaluator = EnhancedEvaluator(
        benchmark_dir=args.benchmark_dir,
        output_dir=str(output_path / "evaluation")
    )

    evaluator.load_data()

    # 샘플 제한 (테스트용)
    if args.limit:
        evaluator.test_data = evaluator.test_data[:args.limit]
        print(f"[테스트 모드] 샘플 제한: {args.limit}")

    evaluator.run_all(
        skip_sbert=args.skip_sbert,
        skip_hybrid=args.skip_hybrid
    )

    evaluator.save_results()
    evaluator.print_summary()

    # ========================================
    # Phase 2: 진단 분석
    # ========================================
    print("\n" + "=" * 80)
    print("Phase 2: Diagnostics Analysis")
    print("=" * 80)

    # TODO: 파이프라인 실행 결과에서 진단 정보 수집
    # 현재는 평가 결과만 있으므로 스킵
    print("[TODO] 파이프라인 디버그 정보 필요 (Hybrid 모델 실행 필요)")

    # ========================================
    # Phase 3: 설명 생성
    # ========================================
    print("\n" + "=" * 80)
    print("Phase 3: Explanation Generation")
    print("=" * 80)

    # TODO: 분류 결과에서 설명 생성
    print("[TODO] 분류 결과에서 설명 생성 (Hybrid 모델 실행 필요)")

    # ========================================
    # 완료
    # ========================================
    print("\n" + "=" * 80)
    print("Full Evaluation Complete")
    print("=" * 80)
    print(f"결과 저장: {output_path}")
    print("\n생성된 파일:")
    print(f"  - {output_path / 'evaluation' / 'evaluation_results.json'}")
    print(f"  - {output_path / 'evaluation' / 'comparison_table.csv'}")
    print(f"  - {output_path / 'evaluation' / 'bucket_performance.csv'}")


if __name__ == "__main__":
    main()
