"""
HS Classification Pipeline (GRI 통칙 + 전역 속성 통합판)

End-to-end 분류 파이프라인:
1. GRI 신호 탐지 + 전역 속성 추출
2. ML Top-50 + KB Top-30 + 속성 기반 후보 생성 (합집합)
3. 카드/규칙 기반 Rerank (GRI + 속성 피처 포함)
4. 저신뢰도 판정 + 속성 충돌 기반 질문 생성
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Set, List, Any

from .types import ClassificationResult, Candidate, DecisionStatus, StageInfo
from .retriever import HSRetriever
from .reranker import HSReranker
from .clarify import HSClarifier
from .gri_signals import GRISignals, detect_gri_signals, detect_parts_signal
from .attribute_extract import GlobalAttributes, extract_attributes
from .legal_gate import LegalGate
from .fact_checker import FactSufficiencyChecker


class HSPipeline:
    """
    HS 품목분류 파이프라인 (GRI 통칙 통합판)
    - GRI 신호 탐지 및 질문 라우팅
    - ML + KB 합집합 후보 생성 (GRI 기반 조정)
    - GRI 피처 기반 Reranking
    - Label space 진단

    Ablation 토글 파라미터:
    - use_gri: GRI 신호 탐지 사용 여부
    - use_legal_gate: LegalGate (GRI 1) 사용 여부
    - use_8axis: 8축 전역 속성 사용 여부
    - use_rules: KB 규칙 매칭 사용 여부
    - use_ranker: LightGBM ranker 사용 여부
    - use_questions: 저신뢰도 질문 생성 여부
    """

    def __init__(
        self,
        retriever: Optional[HSRetriever] = None,
        reranker: Optional[HSReranker] = None,
        clarifier: Optional[HSClarifier] = None,
        ml_topk: int = 50,
        kb_topk: int = 30,
        ranker_model_path: Optional[str] = None,
        # Ablation 토글 파라미터
        use_gri: bool = True,
        use_legal_gate: bool = True,
        use_8axis: bool = True,
        use_rules: bool = True,
        use_ranker: bool = True,
        use_questions: bool = True
    ):
        # KB-only 모드 지원: retriever=None 허용 (fallback 생성 금지)
        self.retriever = retriever
        self.reranker = reranker or HSReranker()
        self.clarifier = clarifier or HSClarifier()
        self.legal_gate = LegalGate() if use_legal_gate else None
        self.fact_checker = FactSufficiencyChecker() if use_legal_gate else None  # FactCheck는 LegalGate와 함께 사용

        self.ml_topk = ml_topk
        self.kb_topk = kb_topk

        # Ablation 토글 설정
        self.use_gri = use_gri
        self.use_legal_gate = use_legal_gate
        self.use_8axis = use_8axis
        self.use_rules = use_rules
        self.use_ranker = use_ranker
        self.use_questions = use_questions

        # Label space 캐시
        self._model_classes: Optional[Set[str]] = None
        self._kb_classes: Optional[Set[str]] = None

        # LightGBM ranker (옵션)
        self.ranker_model = None
        if ranker_model_path and use_ranker:
            self._load_ranker(ranker_model_path)

    def _load_ranker(self, path: str):
        """LightGBM ranker 모델 로드"""
        try:
            import lightgbm as lgb
            model_file = Path(path)
            if model_file.exists():
                self.ranker_model = lgb.Booster(model_file=str(model_file))
                print(f"[Pipeline] Ranker 모델 로드: {path}")
        except Exception as e:
            print(f"[Pipeline] Ranker 로드 실패: {e}")

    def get_model_classes(self) -> Set[str]:
        """모델이 알고 있는 HS4 집합"""
        if self._model_classes is None:
            if self.retriever:
                self._model_classes = self.retriever.get_model_classes()
            else:
                # KB-only 모드: 빈 set 반환
                self._model_classes = set()
        return self._model_classes

    def get_kb_classes(self) -> Set[str]:
        """KB가 알고 있는 HS4 집합"""
        if self._kb_classes is None:
            self._kb_classes = self.reranker.get_kb_hs4_set()
        return self._kb_classes

    def has_hs4_in_model(self, hs4: str) -> bool:
        """특정 HS4가 모델에 있는지 확인"""
        return hs4 in self.get_model_classes()

    def diagnose_label_space(self) -> Dict[str, Any]:
        """
        Label space 불일치 진단

        Returns:
            진단 결과 딕셔너리
        """
        model_classes = self.get_model_classes()
        kb_classes = self.get_kb_classes()

        missing_in_model = kb_classes - model_classes
        missing_in_kb = model_classes - kb_classes
        common = model_classes & kb_classes

        diagnosis = {
            'model_classes_count': len(model_classes),
            'kb_classes_count': len(kb_classes),
            'common_count': len(common),
            'missing_in_model_count': len(missing_in_model),
            'missing_in_kb_count': len(missing_in_kb),
            'missing_in_model': sorted(list(missing_in_model)),
            'missing_in_kb': sorted(list(missing_in_kb)),
            'coverage_model': len(common) / len(kb_classes) if kb_classes else 0,
            'coverage_kb': len(common) / len(model_classes) if model_classes else 0,
        }

        return diagnosis

    def _merge_candidates(
        self,
        ml_candidates: List[Candidate],
        kb_candidates: List[Candidate],
        merge_debug: dict
    ) -> List[Candidate]:
        """
        KB + ML 후보 합집합 (KB-first 전략)

        Args:
            ml_candidates: ML 기반 후보
            kb_candidates: KB 기반 후보
            merge_debug: 병합 통계 기록용 dict

        Returns:
            합집합 (중복 제거, KB 우선 + ML 보강)
        """
        seen_hs4 = set()
        merged = []

        # [변경] KB 후보 먼저 (KB-first strategy)
        kb_only_count = 0
        for c in kb_candidates:
            if c.hs4 not in seen_hs4:
                seen_hs4.add(c.hs4)
                c.features['source'] = 'kb'  # Track source
                merged.append(c)
                kb_only_count += 1

        # ML 후보 추가 (KB에 없는 것만 - recall 보강)
        ml_only_count = 0
        intersection_count = 0
        for c in ml_candidates:
            if c.hs4 in seen_hs4:
                intersection_count += 1
                # KB와 ML 모두에 있는 후보: ML score를 KB 후보에 추가
                for kb_cand in merged:
                    if kb_cand.hs4 == c.hs4:
                        kb_cand.score_ml = c.score_ml
                        kb_cand.features['source'] = 'kb+ml'
                        break
            else:
                seen_hs4.add(c.hs4)
                c.features['source'] = 'ml'
                # ML 전용 후보임을 evidence에 표시
                from .types import Evidence
                c.evidence.insert(0, Evidence(
                    kind="ml_only",
                    source_id=c.hs4,
                    text="KB에 없는 후보 (ML로 보강)",
                    weight=0.1,
                    meta={'ml_recall': True}
                ))
                merged.append(c)
                ml_only_count += 1

        # 병합 통계 기록
        merge_debug['kb_only_count'] = kb_only_count
        merge_debug['ml_only_count'] = ml_only_count
        merge_debug['intersection_count'] = intersection_count
        merge_debug['total_merged'] = len(merged)

        return merged

    def classify(
        self,
        text: str,
        topk: int = 5,
        include_kb_candidates: bool = True
    ) -> ClassificationResult:
        """
        품목분류 실행 (GRI + 전역 속성 통합판)

        Args:
            text: 입력 텍스트 (품명)
            topk: 반환할 상위 후보 수
            include_kb_candidates: KB 후보 포함 여부

        Returns:
            ClassificationResult
        """
        debug: Dict[str, Any] = {}

        # Step 0: GRI 신호 탐지 + 전역 속성 추출 (Ablation 토글)
        if self.use_gri:
            gri_signals = detect_gri_signals(text)
            parts_signal = detect_parts_signal(text)
        else:
            gri_signals = GRISignals()  # 빈 신호
            parts_signal = {'is_parts': False, 'matched': []}

        input_attrs = extract_attributes(text)

        debug['gri_signals'] = gri_signals.to_dict()
        debug['parts_signal'] = parts_signal
        debug['active_gri'] = gri_signals.active_signals()
        debug['input_attrs'] = input_attrs.to_dict()
        debug['attrs_summary'] = input_attrs.summary()
        debug['ablation'] = {
            'use_gri': self.use_gri,
            'use_legal_gate': self.use_legal_gate,
            'use_8axis': self.use_8axis,
            'use_rules': self.use_rules,
            'use_ranker': self.use_ranker,
            'use_questions': self.use_questions,
        }

        # Step 1: ML Top-K 후보 생성
        ml_candidates = []
        if self.retriever and self.retriever.is_ready():
            ml_candidates = self.retriever.predict_topk(text, k=self.ml_topk)

        debug['ml_candidates_count'] = len(ml_candidates)
        debug['ml_top5'] = [
            {'hs4': c.hs4, 'score_ml': round(c.score_ml, 4)}
            for c in ml_candidates[:5]
        ]
        debug['ml_used'] = bool(self.retriever and ml_candidates)

        # Step 2: KB 후보 생성 (GRI + 속성 기반 조정) - Ablation 토글
        kb_candidates = []
        if include_kb_candidates and self.use_rules:
            # GRI 신호에 따라 KB topk 조정
            actual_kb_topk = self.kb_topk
            if gri_signals.gri2a_incomplete:
                actual_kb_topk += 20  # 미조립품: 완제품 후보 확대
            if gri_signals.gri2b_mixtures:
                actual_kb_topk += 10  # 혼합물: 재질 후보 확대

            # 속성 기반 확장
            if input_attrs.has_quant:
                actual_kb_topk += 10  # 정량 조건 있으면 확장
            if len(input_attrs.materials) > 1:
                actual_kb_topk += 5  # 복합 재질이면 확장

            kb_candidates = self.reranker.retrieve_from_kb(
                text,
                topk=actual_kb_topk,
                gri_signals=gri_signals,
                input_attrs=input_attrs
            )

        debug['kb_candidates_count'] = len(kb_candidates)
        debug['kb_top5'] = [
            {'hs4': c.hs4, 'kb_score': c.evidence[0].meta.get('kb_score', 0) if c.evidence else 0}
            for c in kb_candidates[:5]
        ]

        # Step 3: 합집합 (KB-first + ML recall)
        merge_debug = {}
        candidates = self._merge_candidates(ml_candidates, kb_candidates, merge_debug)
        debug['merged_candidates_count'] = len(candidates)
        debug['merge_stats'] = merge_debug

        # KB confidence gate 계산 (rerank 전 pre-score 기반)
        kb_margin = 0.0
        kb_locked = False
        kb_top1_hs4 = None
        kb_top1_card_count = 0

        if len(kb_candidates) >= 2:
            # KB top1과 top2의 KB score로 confidence 판단
            kb_top1 = kb_candidates[0]
            kb_top2 = kb_candidates[1]
            kb_top1_hs4 = kb_top1.hs4

            # KB retrieval evidence의 meta에서 kb_score 가져오기
            kb_top1_score = 0.0
            kb_top2_score = 0.0
            for ev in kb_top1.evidence:
                if ev.kind == 'kb_retrieval' and 'kb_score' in ev.meta:
                    kb_top1_score = ev.meta['kb_score']
                    break
            for ev in kb_top2.evidence:
                if ev.kind == 'kb_retrieval' and 'kb_score' in ev.meta:
                    kb_top2_score = ev.meta['kb_score']
                    break

            # KB score 차이
            kb_margin = kb_top1_score - kb_top2_score
            kb_top1_card_count = int(kb_top1_score)  # For debug logging

            # Lock 조건: KB top1 score >= 10.0 AND top2보다 3.0 이상 높음
            KB_SCORE_THRESHOLD = 10.0
            KB_MARGIN_THRESHOLD = 3.0
            if kb_top1_score >= KB_SCORE_THRESHOLD and kb_margin >= KB_MARGIN_THRESHOLD:
                kb_locked = True

        elif len(kb_candidates) == 1:
            # KB 후보가 1개만 있으면 그것의 score로 판단
            kb_top1_hs4 = kb_candidates[0].hs4
            kb_top1_score = 0.0
            for ev in kb_candidates[0].evidence:
                if ev.kind == 'kb_retrieval' and 'kb_score' in ev.meta:
                    kb_top1_score = ev.meta['kb_score']
                    break
            kb_margin = kb_top1_score
            kb_top1_card_count = int(kb_top1_score)  # For debug logging
            # Single KB candidate: lock if score >= 15.0 (very high confidence)
            if kb_top1_score >= 15.0:
                kb_locked = True  # 단독이므로 margin = card count

        debug['kb_margin'] = round(kb_margin, 2)
        debug['kb_locked'] = kb_locked
        debug['kb_top1_hs4'] = kb_top1_hs4
        debug['kb_top1_card_count'] = kb_top1_card_count
        # Debug: show card counts for top2 as well
        if len(kb_candidates) >= 2:
            kb_top2_card_count_debug = sum(1 for ev in kb_candidates[1].evidence if ev.kind in ['card_keyword', 'card_exact'])
            debug['kb_top2_card_count'] = kb_top2_card_count_debug

        # not_in_model 표시
        model_classes = self.get_model_classes()
        not_in_model_list = []
        for c in candidates:
            if c.hs4 not in model_classes:
                not_in_model_list.append(c.hs4)
        if not_in_model_list:
            debug['not_in_model_hs4'] = not_in_model_list

        # Step 3.5: LegalGate (GRI 1 기반 법적 필터링) - Ablation 토글
        if self.use_legal_gate and self.legal_gate:
            candidates, redirect_hs4s, legal_gate_debug = self.legal_gate.apply(text, candidates)
            debug['legal_gate'] = legal_gate_debug

            # 리다이렉트 HS4 추가 (새로운 후보)
            if redirect_hs4s:
                for rhs4 in redirect_hs4s:
                    # 간단한 Candidate 생성 (reranker에서 점수 계산)
                    from .types import Evidence
                    redirect_cand = Candidate(
                        hs4=rhs4,
                        score_ml=0.0,
                        evidence=[Evidence(
                            kind="legal_redirect",
                            source_id=rhs4,
                            text=f"주규정에 의한 리다이렉트: 제{rhs4}호",
                            weight=0.8
                        )]
                    )
                    candidates.append(redirect_cand)

            debug['after_legal_gate_count'] = len(candidates)

            # LegalGate 통과 후 후보가 1개면 즉시 반환 (GRI 1로 확정)
            if len(candidates) == 1:
                single_cand = candidates[0]
                single_cand.score_total = 1.0
                single_cand.features['gri1_definitive'] = True
                debug['gri1_single_candidate'] = True
                debug['gri_decision'] = 'GRI 1 - 단일 호로 확정'
                return ClassificationResult(
                    input_text=text,
                    topk=[single_cand],
                    decision=DecisionStatus(status="AUTO", reason="GRI 1 - 단일 호로 확정", confidence=1.0),
                    questions=[],
                    debug=debug
                )
            elif len(candidates) == 0:
                # LegalGate에서 모든 후보가 제외됨 → 질문 필요
                debug['gri1_all_excluded'] = True
                debug['gri_decision'] = 'GRI 1 - 모든 후보 제외됨'
                return ClassificationResult(
                    input_text=text,
                    topk=[],
                    decision=DecisionStatus(status="ASK", reason="GRI 1 - 모든 후보 제외됨", confidence=0.0),
                    questions=["입력 정보로는 분류할 수 없습니다. 추가 정보를 제공해주세요."],
                    debug=debug
                )
            else:
                # 복수 후보가 남음 → GRI 2/3/5 적용
                debug['gri_decision'] = f'GRI 1 불충분 ({len(candidates)}개 후보) → GRI 2/3/5 적용'

        # Step 4: Rerank (GRI + 속성 피처 포함) - Ablation 토글
        # GRI 2/3/5는 LegalGate 통과 후 복수 후보가 남은 경우에만 활성화
        # 8축 속성은 use_8axis 플래그에 따라 전달
        input_attrs_8axis = None
        if self.use_8axis:
            from .attribute_extract import extract_attributes_8axis
            input_attrs_8axis = extract_attributes_8axis(text)

        # GRI 2/3/5 적용 여부 결정
        # - LegalGate를 사용하지 않거나
        # - LegalGate 통과 후 복수 후보가 남은 경우에만 GRI 2/3/5 적용
        apply_gri_235 = (
            not self.use_legal_gate or
            (self.use_legal_gate and len(candidates) > 1)
        )

        # Conditional ML weight 계산
        # ML을 강하게 사용할 조건: short text, fact insufficient, ambiguous
        w_ml = 0.2  # 기본 낮은 가중치
        use_strong_ml = False

        # 조건 1: 텍스트가 짧음 (< 20자)
        if len(text) < 20:
            w_ml = 0.4
            use_strong_ml = True

        # 조건 2: Fact insufficient (질문 생성이 필요한 경우)
        if debug.get('questions_generated_count', 0) > 0:
            w_ml = 0.5
            use_strong_ml = True

        # 조건 3: KB 후보가 너무 적음 (< 5개)
        if len(kb_candidates) < 5:
            w_ml = 0.4
            use_strong_ml = True

        # 조건 4: KB locked - ML을 더 약하게
        if kb_locked:
            w_ml = 0.05  # KB가 확신있으면 ML 거의 무시
            use_strong_ml = False

        debug['w_ml'] = round(w_ml, 2)
        debug['use_strong_ml'] = use_strong_ml

        # ML weight를 candidates에 적용 (score_ml에 가중치)
        for cand in candidates:
            if hasattr(cand, 'score_ml') and cand.score_ml > 0:
                cand.features['w_ml'] = w_ml
                # score_ml을 가중치 적용한 값으로 조정 (reranker에서 사용)
                cand.features['weighted_ml_score'] = cand.score_ml * w_ml

        ranker_model_to_use = self.ranker_model if self.use_ranker else None
        reranked, rerank_stats = self.reranker.rerank(
            text,
            candidates,
            topk=topk,
            gri_signals=gri_signals if (self.use_gri and apply_gri_235) else None,
            input_attrs=input_attrs,
            input_attrs_8axis=input_attrs_8axis,
            model_classes=model_classes,
            ranker_model=ranker_model_to_use
        )
        rerank_stats['gri_235_applied'] = apply_gri_235

        # KB lock 적용: KB top1이 locked이면 최종 top1을 강제로 kb_top1으로
        top1_source = "ranker"
        if kb_locked and kb_top1_hs4:
            # reranked에서 kb_top1_hs4 찾기
            kb_top1_in_reranked = None
            kb_top1_rank = -1
            for i, cand in enumerate(reranked):
                if cand.hs4 == kb_top1_hs4:
                    kb_top1_in_reranked = cand
                    kb_top1_rank = i
                    break

            # KB top1이 reranked top-5 안에 있으면 top1으로 이동
            if kb_top1_in_reranked and kb_top1_rank < 10:  # top-10 이내만 override
                # kb_top1을 맨 앞으로
                reranked.remove(kb_top1_in_reranked)
                reranked.insert(0, kb_top1_in_reranked)
                top1_source = "kb_locked"
                debug['kb_lock_applied'] = True
                debug['kb_top1_original_rank'] = kb_top1_rank

        debug['top1_source'] = top1_source

        # Retriever 사용 여부 명시적 기록
        debug['retriever_used'] = debug.get('ml_used', False)  # ML 후보 생성 여부

        # Ranker 사용 여부 명시적 기록 (3-level tracking)
        debug['use_ranker'] = self.use_ranker  # Config: ranker 활성화 여부
        debug['ranker_loaded'] = bool(self.ranker_model)  # Model: 모델 로드 성공 여부
        debug['ranker_applied'] = (self.use_ranker and bool(self.ranker_model) and len(candidates) > 0)  # Actual: 실제 적용 여부
        # Backward compatibility
        debug['ranker_used'] = debug['ranker_applied']

        debug['rerank_stats'] = rerank_stats
        debug['reranked_top5'] = [
            {
                'hs4': c.hs4,
                'score_total': round(c.score_total, 4),
                'score_ml': round(c.score_ml, 4),
                'score_card': round(c.score_card, 4),
                'score_rule': round(c.score_rule, 4),
                'evidence_count': len(c.evidence),
                'in_model': c.hs4 in model_classes,
                'features': c.features
            }
            for c in reranked
        ]

        # Step 5: 저신뢰도 판정
        scores = [c.score_total for c in reranked]
        low_confidence = self.clarifier.is_low_confidence(scores)
        debug['low_confidence_check'] = {
            'p1': round(scores[0], 4) if scores else 0,
            'p2': round(scores[1], 4) if len(scores) > 1 else 0,
            'result': low_confidence
        }

        # no_hits 체크
        if rerank_stats.get('no_hits', False):
            low_confidence = True
            debug['no_kb_hits'] = True

        # Step 6: GRI + 속성 충돌 기반 질문 생성 (Ablation 토글)
        questions = []
        if low_confidence and self.use_questions:
            hs4_list = [c.hs4 for c in reranked]
            questions = self.clarifier.get_questions_with_context(
                hs4_list,
                gri_signals=gri_signals,
                input_attrs=input_attrs,
                no_kb_hits=rerank_stats.get('no_hits', False),
                has_parts_signal=parts_signal['is_parts'],
                reranker=self.reranker,
                max_questions=3
            )

        # Decision 생성
        if low_confidence:
            status = "ASK"
            reason = "저신뢰도"
            confidence = scores[0] if scores else 0.0
        else:
            status = "AUTO"
            reason = "정상 분류"
            confidence = scores[0] if scores else 0.0

        return ClassificationResult(
            input_text=text,
            topk=reranked,
            decision=DecisionStatus(status=status, reason=reason, confidence=confidence),
            questions=questions,
            debug=debug
        )

    def save_diagnostics(self, output_dir: str = "artifacts/diagnostics"):
        """진단 결과 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        diagnosis = self.diagnose_label_space()

        diag_file = output_path / "space_diff.json"
        with open(diag_file, 'w', encoding='utf-8') as f:
            json.dump(diagnosis, f, ensure_ascii=False, indent=2)

        print(f"진단 결과 저장: {diag_file}")
        return diagnosis


def main():
    """CLI 진입점"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.classifier.pipeline <입력텍스트>")
        print("  python -m src.classifier.pipeline --diagnose")
        print("  python -m src.classifier.pipeline --smoke-test")
        print("  python -m src.classifier.pipeline --gri-test")
        print()
        print("Examples:")
        print("  python -m src.classifier.pipeline \"냉동 돼지 삼겹살\"")
        print("  python -m src.classifier.pipeline \"자동차 CKD 부품 세트\"")
        sys.exit(1)

    arg = sys.argv[1]

    # 진단 모드
    if arg == "--diagnose":
        print("=" * 60)
        print("Label Space 진단")
        print("=" * 60)

        pipeline = HSPipeline()
        diagnosis = pipeline.save_diagnostics()

        print(f"\n모델 클래스: {diagnosis['model_classes_count']}개")
        print(f"KB 클래스: {diagnosis['kb_classes_count']}개")
        print(f"공통: {diagnosis['common_count']}개")
        print(f"모델에 없음 (KB에만): {diagnosis['missing_in_model_count']}개")
        print(f"KB에 없음 (모델에만): {diagnosis['missing_in_kb_count']}개")
        print(f"\n모델 커버리지: {diagnosis['coverage_model']*100:.1f}%")
        print(f"KB 커버리지: {diagnosis['coverage_kb']*100:.1f}%")

        if diagnosis['missing_in_model_count'] > 0:
            print(f"\n모델에 없는 HS4 (상위 20개):")
            for hs4 in diagnosis['missing_in_model'][:20]:
                print(f"  {hs4}")

        sys.exit(0)

    # 스모크 테스트 모드
    if arg == "--smoke-test":
        from .diagnostics import run_smoke_test
        run_smoke_test()
        sys.exit(0)

    # GRI 테스트 모드
    if arg == "--gri-test":
        print("=" * 60)
        print("GRI 신호 테스트")
        print("=" * 60)

        test_cases = [
            "자동차 CKD 부품 세트",
            "면 60% 폴리에스터 40% 혼방 직물",
            "스마트폰 전용 케이스",
            "미조립 가구 키트",
            "세트 구성: 칼, 포크, 숟가락",
            "냉동 돼지 삼겹살",
            "LED TV 55인치",
        ]

        for text in test_cases:
            gri = detect_gri_signals(text)
            parts = detect_parts_signal(text)
            print(f"\n입력: {text}")
            print(f"  GRI: {gri.active_signals()}")
            print(f"  부품: {parts['is_parts']}")
            if gri.matched_keywords:
                for sig, kws in gri.matched_keywords.items():
                    print(f"    {sig}: {kws}")

        sys.exit(0)

    # 일반 분류 모드
    input_text = arg

    print("=" * 60)
    print("HS 품목분류 파이프라인 (GRI 통합)")
    print("=" * 60)
    print(f"\n입력: {input_text}\n")

    print("[초기화 중...]")
    try:
        pipeline = HSPipeline()
    except RuntimeError as e:
        print(f"\n오류: {e}")
        print("\n모델이 없습니다. 먼저 학습을 실행하세요:")
        print("  python -c \"from src.classifier.retriever import HSRetriever; r = HSRetriever(); r.train_model()\"")
        sys.exit(1)

    print("\n[분류 중...]\n")
    result = pipeline.classify(input_text)

    # GRI 신호 출력
    gri_signals = result.debug.get('active_gri', [])
    if gri_signals:
        print(f"[GRI 신호 감지] {', '.join(gri_signals)}")

    parts_signal = result.debug.get('parts_signal', {})
    if parts_signal.get('is_parts'):
        print(f"[부품 신호 감지] 매칭: {parts_signal.get('matched', [])}")

    # 전역 속성 출력
    attrs_summary = result.debug.get('attrs_summary', '')
    if attrs_summary:
        print(f"[전역 속성] {attrs_summary}")

    # 결과 출력
    print("\n" + "-" * 60)
    print("결과")
    print("-" * 60)

    for i, cand in enumerate(result.topk, 1):
        in_model = cand.hs4 in pipeline.get_model_classes()
        model_tag = "" if in_model else " [NOT_IN_MODEL]"

        print(f"\n[{i}] HS {cand.hs4}{model_tag}")
        print(f"    총점: {cand.score_total:.4f} (ML: {cand.score_ml:.4f}, Card: {cand.score_card:.4f}, Rule: {cand.score_rule:.4f})")

        # 피처 breakdown
        if cand.features:
            f = cand.features
            print(f"    피처: specificity={f.get('f_specificity', 0):.2f}, exc_conflict={f.get('f_exclude_conflict', 0)}, parts={f.get('f_is_parts_candidate', 0)}")

        print(f"    근거: {len(cand.evidence)}개")
        if cand.evidence:
            for ev in cand.evidence[:3]:
                print(f"      - [{ev.kind}] {ev.text[:80]}...")

    print(f"\n저신뢰도: {result.low_confidence}")
    if result.debug.get('no_kb_hits'):
        print("  (KB 매칭 없음)")

    if result.questions:
        print("\n[GRI 기반 질문]")
        for q in result.questions:
            print(f"  - {q}")

    # 통계
    stats = result.debug.get('rerank_stats', {})
    print(f"\n[통계]")
    print(f"  ML 후보: {result.debug.get('ml_candidates_count', 0)}개")
    print(f"  KB 후보: {result.debug.get('kb_candidates_count', 0)}개")
    print(f"  합집합: {result.debug.get('merged_candidates_count', 0)}개")
    print(f"  Card hit rate: {stats.get('card_hit_rate', 0)*100:.1f}%")
    print(f"  Rule hit rate: {stats.get('rule_hit_rate', 0)*100:.1f}%")
    print(f"  Exclude conflicts: {stats.get('exclude_conflict_count', 0)}")

    # JSON 출력
    print("\n" + "-" * 60)
    print("JSON 출력")
    print("-" * 60)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
