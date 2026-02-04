"""
Evaluation Runner - 평가 실행기

모드:
- kb_only: ML retriever OFF, ranker OFF, LegalGate+FactCheck+KB만
- hybrid: ML retriever ON, LegalGate+FactCheck+Ranker ON
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import random

from .metrics import compute_all_metrics, save_metrics
from .usage_audit import UsageAuditor
from ..pipeline import HSPipeline
from ..retriever import HSRetriever
from ..reranker import HSReranker
from ..clarify import HSClarifier


class EvalRunner:
    """
    평가 실행기

    데이터셋 로드 → 분류 실행 → 지표 계산 → 리포트 생성
    """

    def __init__(
        self,
        mode: str = 'kb_only',
        seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        topk: int = 100,
        output_base_dir: str = 'artifacts/eval'
    ):
        """
        Args:
            mode: 'kb_only' | 'hybrid'
            seed: random seed
            train_ratio: 학습 데이터 비율
            val_ratio: 검증 데이터 비율
            test_ratio: 테스트 데이터 비율
            topk: 후보 K
            output_base_dir: 출력 기본 디렉토리
        """
        self.mode = mode
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.topk = topk
        self.output_base_dir = output_base_dir

        # Run ID 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{mode}_{timestamp}"
        self.output_dir = Path(output_base_dir) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline 초기화
        self.pipeline = self._init_pipeline()

        # 데이터
        self.train_data: List[Dict[str, Any]] = []
        self.val_data: List[Dict[str, Any]] = []
        self.test_data: List[Dict[str, Any]] = []

    def _init_pipeline(self) -> HSPipeline:
        """
        모드에 따라 파이프라인 초기화

        Returns:
            HSPipeline
        """
        if self.mode == 'kb_only':
            # ML retriever OFF, ranker OFF
            print("[KB-only 모드] ML retriever와 Ranker를 비활성화합니다.")
            pipeline = HSPipeline(
                retriever=None,  # ML retriever 사용 안 함
                reranker=HSReranker(),
                clarifier=HSClarifier(),
                use_gri=True,
                use_legal_gate=True,
                use_8axis=True,
                use_rules=True,
                use_ranker=False,  # Ranker OFF
                use_questions=True
            )
        elif self.mode == 'hybrid':
            # 전체 파이프라인 (Retriever + Ranker 필수)
            print("[Hybrid 모드] ML retriever와 Ranker를 활성화합니다.")

            # Retriever 로드 (필수)
            if not self._has_ml_retriever():
                raise RuntimeError(
                    "Hybrid 모드는 ML retriever가 필수입니다. "
                    "sentence-transformers를 설치하세요: pip install sentence-transformers"
                )
            retriever = HSRetriever()
            print("  [OK] ML Retriever 로드 완료")

            # Ranker 로드 (필수)
            ranker_path = "artifacts/ranker_legal/model_legal.txt"
            ranker_file = Path(ranker_path)
            if not ranker_file.exists():
                # fallback 경로 시도
                ranker_path_fallback = "artifacts/ranker/model.txt"
                ranker_file_fallback = Path(ranker_path_fallback)
                if ranker_file_fallback.exists():
                    ranker_path = ranker_path_fallback
                    print(f"  [OK] Ranker 모델 로드: {ranker_path} (fallback)")
                else:
                    raise RuntimeError(
                        f"Hybrid 모드는 Ranker 모델이 필수입니다. "
                        f"다음 경로에 모델이 없습니다: {ranker_path}, {ranker_path_fallback}\n"
                        f"Ranker 학습을 먼저 실행하세요: python -m src.classifier.rank.train_ranker_legal --build"
                    )
            else:
                print(f"  [OK] Ranker 모델 로드: {ranker_path}")

            pipeline = HSPipeline(
                retriever=retriever,
                reranker=HSReranker(),
                clarifier=HSClarifier(),
                ranker_model_path=ranker_path,
                use_gri=True,
                use_legal_gate=True,
                use_8axis=True,
                use_rules=True,
                use_ranker=True,  # Ranker ON
                use_questions=True
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return pipeline

    def _has_ml_retriever(self) -> bool:
        """ML retriever 사용 가능 여부 확인"""
        try:
            # Sentence Transformer 사용 가능한지 확인
            from sentence_transformers import SentenceTransformer
            return True
        except:
            return False

    def load_dataset(
        self,
        dataset_path: str,
        limit: Optional[int] = None
    ) -> None:
        """
        데이터셋 로드 및 분할

        Args:
            dataset_path: 결정사례 JSON 경로
            limit: 샘플 수 제한 (테스트용)
        """
        print(f"데이터셋 로드 중: {dataset_path}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        loaded_total = len(data)
        print(f"  로드된 전체 샘플: {loaded_total}")

        # Seed 고정
        random.seed(self.seed)
        random.shuffle(data)

        # 분할
        total = len(data)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        self.train_data = data[:train_end]
        self.val_data = data[train_end:val_end]
        self.test_data = data[val_end:]

        # limit 적용 (test_data에만)
        test_selected_n = len(self.test_data)
        if limit:
            if limit > loaded_total:
                raise ValueError(
                    f"--limit {limit}가 전체 데이터 {loaded_total}보다 큽니다. "
                    f"limit을 줄이거나 더 많은 데이터를 준비하세요."
                )
            # test에서 최소 limit만큼 확보
            required_test = limit
            if test_selected_n < required_test:
                raise ValueError(
                    f"Test split ({test_selected_n})이 --limit {limit}보다 작습니다. "
                    f"전체 데이터({loaded_total})에서 test_ratio를 높이거나 limit을 줄이세요."
                )
            self.test_data = self.test_data[:limit]
            test_selected_n = len(self.test_data)
            print(f"  샘플 제한 적용: Test {test_selected_n}")

        print(f"  Train: {len(self.train_data)}")
        print(f"  Val: {len(self.val_data)}")
        print(f"  Test (평가 대상): {len(self.test_data)}")

        # Split 정보 저장
        split_info = {
            'loaded_total': loaded_total,
            'seed': self.seed,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'train_n': len(self.train_data),
            'val_n': len(self.val_data),
            'test_n': len(self.test_data),
            'test_selected_n': len(self.test_data),
            'limit_requested': limit,
        }

        with open(self.output_dir / 'split_info.json', 'w', encoding='utf-8') as f:
            json.dump(split_info, f, ensure_ascii=False, indent=2)

    def run_evaluation(
        self,
        split: str = 'test'
    ) -> List[Dict[str, Any]]:
        """
        평가 실행

        Args:
            split: 'train' | 'val' | 'test'

        Returns:
            per-sample 예측 결과 리스트
        """
        if split == 'train':
            data = self.train_data
        elif split == 'val':
            data = self.val_data
        else:
            data = self.test_data

        print(f"\n{'='*60}")
        print(f"평가 실행: {self.mode} mode, {split} split ({len(data)} samples)")
        print(f"{'='*60}")

        predictions = []
        start_time = time.time()

        for i, sample in enumerate(data):
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  진행: {i+1}/{len(data)} ({elapsed:.1f}s)")

            # 샘플 정보
            sample_id = sample.get('id', f"sample_{i}")
            text = sample.get('product_name', '')
            true_hs4 = sample.get('hs_heading', '')

            if not text or not true_hs4 or len(true_hs4) != 4:
                continue

            # 분류 실행
            result = self.pipeline.classify(text)

            # Debug 검증 (mode별)
            debug = result.debug

            # retriever_used/ranker_used 검증
            retriever_used = debug.get('ml_used', False) or (self.pipeline.retriever is not None)
            ranker_used = debug.get('ranker_applied', False)

            # kb_only 모드에서 ranker_used=True면 에러
            if self.mode == 'kb_only' and ranker_used:
                raise AssertionError(
                    f"KB-only 모드에서 ranker_used=True 발생! sample_id={sample_id}"
                )

            # LegalGate 후보 수 기록
            legal_gate_debug = debug.get('legal_gate', {})
            candidates_before_legal = debug.get('after_generation_count', 0)
            candidates_after_legal = debug.get('after_legal_gate_count', candidates_before_legal)

            # Enhanced debug
            # features_count_for_ranker는 ranker가 실제 사용될 때만 설정
            features_count_for_audit = 0
            if ranker_used and result.topk and hasattr(result.topk[0], 'features'):
                features_count_for_audit = len(result.topk[0].features)

            enhanced_debug = {
                **debug,
                'retriever_used': retriever_used,
                'ranker_used': ranker_used,
                'candidates_before_legal': candidates_before_legal,
                'candidates_after_legal': candidates_after_legal,
                'features_count_for_ranker': features_count_for_audit,
            }

            # 결과 저장
            pred_dict = {
                'sample_id': sample_id,
                'text': text,
                'true_hs4': true_hs4,
                'y_true_hs4': true_hs4,  # Alias for compatibility
                'topk': [c.to_dict() for c in result.topk[:self.topk]],
                'decision': result.decision.to_dict(),
                'questions': result.questions,
                'debug': enhanced_debug,
            }

            predictions.append(pred_dict)

        elapsed = time.time() - start_time
        print(f"  완료: {len(predictions)} samples in {elapsed:.1f}s")

        # 샘플 수 검증
        expected_count = len(data)
        actual_count = len(predictions)
        if actual_count != expected_count:
            print(f"  [Warning] 예상({expected_count}) != 실제({actual_count})")
            print(f"  일부 샘플이 스킵되었습니다 (true_hs4 없음 등)")

        # 모드별 강제 검증
        self._validate_mode_separation(predictions)

        return predictions

    def _validate_mode_separation(self, predictions: List[Dict[str, Any]]) -> None:
        """
        모드별 retriever/ranker 사용 검증 (강제)

        Raises:
            RuntimeError: 모드 분리 위반 시
        """
        total = len(predictions)
        if total == 0:
            return

        # retriever_used 카운트
        retriever_used_count = sum(
            1 for p in predictions
            if p.get('debug', {}).get('retriever_used', False)
        )

        # score_ml nonzero 카운트
        ml_nonzero_count = sum(
            1 for p in predictions
            if p.get('topk') and len(p['topk']) > 0 and p['topk'][0].get('score_ml', 0) > 0
        )

        # ranker_applied 카운트
        ranker_applied_count = sum(
            1 for p in predictions
            if p.get('debug', {}).get('ranker_applied', False)
        )

        print(f"\n{'='*60}")
        print(f"모드 분리 검증 ({self.mode} mode)")
        print(f"{'='*60}")
        print(f"  Total samples: {total}")
        print(f"  retriever_used=True: {retriever_used_count}/{total} ({retriever_used_count/total:.1%})")
        print(f"  score_ml nonzero: {ml_nonzero_count}/{total} ({ml_nonzero_count/total:.1%})")
        print(f"  ranker_applied=True: {ranker_applied_count}/{total} ({ranker_applied_count/total:.1%})")

        # KB-only 모드 검증
        if self.mode == 'kb_only':
            violations = []

            if retriever_used_count > 0:
                violations.append(
                    f"retriever_used=True가 {retriever_used_count}개 샘플에서 발견됨 (기대: 0)"
                )

            if ml_nonzero_count > 0:
                violations.append(
                    f"score_ml > 0인 샘플이 {ml_nonzero_count}개 발견됨 (기대: 0)"
                )

            if ranker_applied_count > 0:
                violations.append(
                    f"ranker_applied=True가 {ranker_applied_count}개 샘플에서 발견됨 (기대: 0)"
                )

            if violations:
                error_msg = "\n".join([
                    "[ERROR] KB-only 모드 분리 위반!",
                    "",
                    "위반 사항:",
                    *[f"  - {v}" for v in violations],
                    "",
                    "KB-only 모드에서는 ML retriever와 ranker를 절대 사용하면 안 됩니다.",
                    "src/classifier/eval/run_eval.py의 pipeline 생성 코드를 확인하세요.",
                ])
                raise RuntimeError(error_msg)

            print(f"  [PASS] KB-only 모드: ML retriever/ranker 사용 없음")

        # Hybrid 모드 검증
        elif self.mode == 'hybrid':
            violations = []

            if retriever_used_count == 0:
                violations.append(
                    f"retriever_used=True가 없음 (기대: {total}개 전부)"
                )

            if ranker_applied_count == 0:
                violations.append(
                    f"ranker_applied=True가 없음 (기대: {total}개 전부)"
                )

            if violations:
                error_msg = "\n".join([
                    "[ERROR] Hybrid 모드 분리 위반!",
                    "",
                    "위반 사항:",
                    *[f"  - {v}" for v in violations],
                    "",
                    "Hybrid 모드에서는 ML retriever와 ranker를 반드시 사용해야 합니다.",
                    "src/classifier/eval/run_eval.py의 pipeline 생성 코드를 확인하세요.",
                ])
                raise RuntimeError(error_msg)

            print(f"  [PASS] Hybrid 모드: ML retriever/ranker 정상 사용")

        print(f"{'='*60}\n")

    def save_predictions(
        self,
        predictions: List[Dict[str, Any]],
        split: str = 'test'
    ) -> None:
        """
        예측 결과 저장

        Args:
            predictions: per-sample 예측 결과
            split: 'train' | 'val' | 'test'
        """
        output_file = self.output_dir / f'predictions_{split}.jsonl'

        with open(output_file, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + '\n')

        # 파일 검증: 실제 라인 수 확인
        with open(output_file, 'r', encoding='utf-8') as f:
            actual_lines = sum(1 for _ in f)

        expected_lines = len(predictions)
        if actual_lines != expected_lines:
            raise AssertionError(
                f"저장된 predictions 라인 수 불일치: "
                f"expected={expected_lines}, actual={actual_lines}, file={output_file}"
            )

        print(f"예측 저장: {output_file} ({actual_lines} lines)")

    def run_full_eval(
        self,
        dataset_path: str,
        limit: Optional[int] = None
    ) -> None:
        """
        전체 평가 실행

        Args:
            dataset_path: 데이터셋 경로
            limit: 샘플 수 제한
        """
        print("\n" + "="*80)
        print(f"Full Evaluation: {self.mode} mode")
        print("="*80)

        # 1. 데이터 로드
        self.load_dataset(dataset_path, limit=limit)

        # 2. Test split 평가
        predictions = self.run_evaluation(split='test')

        # 3. 예측 저장
        self.save_predictions(predictions, split='test')

        # 4. Metrics 계산
        print("\n지표 계산 중...")
        metrics_summary = compute_all_metrics(predictions)
        save_metrics(metrics_summary, str(self.output_dir))

        # 5. Usage Audit
        print("\nUsage audit 중...")
        auditor = UsageAuditor()
        auditor.audit_all(predictions)
        auditor.save_audit(str(self.output_dir))

        # 6. 실험 설정 저장
        # LegalGate heading_terms 길이 확인
        heading_terms_len = 0
        if self.pipeline.legal_gate and hasattr(self.pipeline.legal_gate, 'heading_terms'):
            heading_terms_len = len(self.pipeline.legal_gate.heading_terms)

        config = {
            'run_id': self.run_id,
            'mode': self.mode,
            'seed': self.seed,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'topk': self.topk,
            'dataset_path': dataset_path,
            'limit': limit,
            'pipeline_config': {
                'use_gri': self.pipeline.use_gri,
                'use_legal_gate': self.pipeline.use_legal_gate,
                'use_8axis': self.pipeline.use_8axis,
                'use_rules': self.pipeline.use_rules,
                'use_ranker': self.pipeline.use_ranker,
                'use_questions': self.pipeline.use_questions,
                # 추가: retriever/ranker 실제 상태
                'retriever_present': self.pipeline.retriever is not None,
                'ranker_model_loaded': self.pipeline.ranker_model is not None,
                'heading_terms_len': heading_terms_len,
                'ml_topk': self.pipeline.ml_topk,
                'kb_topk': self.pipeline.kb_topk,
            }
        }

        with open(self.output_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print("\n" + "="*80)
        print("평가 완료")
        print("="*80)
        print(f"Run ID: {self.run_id}")
        print(f"Output Dir: {self.output_dir}")
        print("\n생성된 파일:")
        print(f"  - predictions_test.jsonl")
        print(f"  - metrics_summary.json")
        print(f"  - metrics_table.csv")
        print(f"  - ece_bins.csv")
        print(f"  - confusion_pairs.csv")
        print(f"  - usage_audit.jsonl")
        print(f"  - usage_summary.json")
        print(f"  - config.json")


def main():
    """CLI 엔트리포인트"""
    parser = argparse.ArgumentParser(
        description="HS4 Classification Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # KB-only 모드 (ML 없이 KB만)
  python -m src.classifier.eval.run_eval --mode kb_only

  # Hybrid 모드 (전체 파이프라인)
  python -m src.classifier.eval.run_eval --mode hybrid

  # Seed 및 K 지정
  python -m src.classifier.eval.run_eval --mode hybrid --seed 42 --k 120

  # 샘플 제한 (smoke test)
  python -m src.classifier.eval.run_eval --mode kb_only --limit 200
        """
    )

    parser.add_argument('--mode', choices=['kb_only', 'hybrid'], default='kb_only',
                        help='평가 모드')
    parser.add_argument('--dataset', default='data/ruling_cases/all_cases_full_v7.json',
                        help='데이터셋 경로')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--k', type=int, default=100,
                        help='후보 K')
    parser.add_argument('--limit', type=int,
                        help='샘플 수 제한 (테스트용)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='학습 데이터 비율')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='검증 데이터 비율')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='테스트 데이터 비율')
    parser.add_argument('--output-dir', default='artifacts/eval',
                        help='출력 디렉토리')

    args = parser.parse_args()

    # Runner 생성
    runner = EvalRunner(
        mode=args.mode,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        topk=args.k,
        output_base_dir=args.output_dir
    )

    # 평가 실행
    runner.run_full_eval(
        dataset_path=args.dataset,
        limit=args.limit
    )


if __name__ == '__main__':
    main()
