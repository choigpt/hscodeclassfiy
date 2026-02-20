"""
GRI 순차 오케스트레이터 (GRI Sequential Orchestrator)

GRI 통칙 1→2→3→5→6 순차 적용:
Step 1: GRI 1 — 호의 용어 + 주규정으로 Chapter/Heading 결정 (LegalGate)
Step 2: GRI 2 — 미완성/혼합물 처리
Step 3: GRI 3 — 복수 후보 해소 (3a: 구체적 호, 3b: EC, 3c: 수순서)
Step 4: GRI 5 — 용기/포장 처리
Step 5: GRI 6 — 소호 세분화 (HS4→HS6)
Step 6: 리스크 평가

기존 HSPipeline.classify()의 Step 3~끝 로직을 완전 대체.
"""

from typing import List, Tuple, Optional, Dict, Any

from .types import (
    ClassificationResult, Candidate, DecisionStatus, StageInfo,
    GRIApplication, EssentialCharacterResult, RiskAssessment,
    RuleReference, CaseEvidence,
)
from .input_model import ClassificationInput
from .gri_signals import GRISignals, detect_gri_signals
from .attribute_extract import (
    GlobalAttributes, GlobalAttributes8Axis,
    extract_attributes, extract_attributes_8axis,
)
from .legal_gate import LegalGate
from .essential_character import EssentialCharacterModule
from .subheading_resolver import SubheadingResolver
from .risk_assessor import RiskAssessor


class GRIOrchestrator:
    """GRI 순차 처리 오케스트레이터"""

    def __init__(
        self,
        legal_gate: Optional[LegalGate] = None,
        reranker=None,
        ec_module: Optional[EssentialCharacterModule] = None,
        subheading_resolver: Optional[SubheadingResolver] = None,
        risk_assessor: Optional[RiskAssessor] = None,
        ec_weights: Optional[Dict[str, float]] = None,
    ):
        self.legal_gate = legal_gate
        self.reranker = reranker
        self.ec_module = ec_module or EssentialCharacterModule(reranker=reranker)
        self.subheading_resolver = subheading_resolver or SubheadingResolver()
        self.risk_assessor = risk_assessor or RiskAssessor()

        if ec_weights:
            self.ec_module.WEIGHTS = ec_weights

    def classify(
        self,
        input_data,
        ml_candidates: List[Candidate],
        kb_candidates: List[Candidate],
        merged_candidates: List[Candidate],
        gri_signals: Optional[GRISignals] = None,
        input_attrs: Optional[GlobalAttributes] = None,
        input_attrs_8axis: Optional[GlobalAttributes8Axis] = None,
        rerank_fn=None,
        debug: Optional[Dict[str, Any]] = None,
    ) -> ClassificationResult:
        """
        GRI 순차 분류 실행

        Args:
            input_data: ClassificationInput 또는 str
            ml_candidates: ML 후보
            kb_candidates: KB 후보
            merged_candidates: 합집합 후보
            gri_signals: GRI 신호 (없으면 탐지)
            input_attrs: 7축 속성
            input_attrs_8axis: 8축 속성
            rerank_fn: rerank 함수 (callable)
            debug: 디버그 dict (기존 파이프라인에서 전달)

        Returns:
            ClassificationResult
        """
        if debug is None:
            debug = {}

        # 텍스트 추출
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, ClassificationInput):
            text = input_data.to_enriched_text()
        else:
            text = str(input_data)

        # 속성 추출 (없으면)
        if gri_signals is None:
            gri_signals = detect_gri_signals(text)
        if input_attrs is None:
            input_attrs = extract_attributes(text)
        if input_attrs_8axis is None:
            input_attrs_8axis = extract_attributes_8axis(text)

        candidates = list(merged_candidates)
        applied_gri: List[GRIApplication] = []
        stages: List[StageInfo] = []
        ec_result: Optional[EssentialCharacterResult] = None
        rule_references: List[RuleReference] = []
        case_evidence: List[CaseEvidence] = []

        # ============================================================
        # Step 1: GRI 1 — 호의 용어 + 주규정
        # ============================================================
        candidates, gri1_app, gri1_stage = self._apply_gri1(
            text, candidates, gri_signals
        )
        applied_gri.append(gri1_app)
        stages.append(gri1_stage)

        # 단일 후보 시 즉시 확정
        if len(candidates) == 1:
            candidates[0].score_total = 1.0
            candidates[0].features['gri1_definitive'] = True
            debug['gri_decision'] = 'GRI 1 - 단일 호로 확정'

            return self._build_result(
                text, candidates, applied_gri, stages,
                ec_result, None, rule_references, case_evidence,
                debug, reason="GRI 1 - 단일 호로 확정",
                input_attrs_8axis=input_attrs_8axis,
            )
        elif len(candidates) == 0:
            debug['gri_decision'] = 'GRI 1 - 모든 후보 제외됨'
            return ClassificationResult(
                input_text=text,
                topk=[],
                decision=DecisionStatus(status="ASK", reason="GRI 1 - 모든 후보 제외됨", confidence=0.0),
                questions=["입력 정보로는 분류할 수 없습니다. 추가 정보를 제공해주세요."],
                applied_gri=applied_gri,
                stages=stages,
                debug=debug,
            )

        # ============================================================
        # Step 2: GRI 2 — 미완성/혼합물 처리
        # ============================================================
        candidates, gri2_app, gri2_stage = self._apply_gri2(
            text, candidates, gri_signals, input_attrs_8axis
        )
        applied_gri.append(gri2_app)
        stages.append(gri2_stage)

        # ============================================================
        # Step 3: GRI 3 — 복수 후보 해소
        # ============================================================
        if len(candidates) > 1:
            candidates, gri3_app, gri3_stage, ec_result = self._apply_gri3(
                text, candidates, gri_signals, input_attrs_8axis, rerank_fn
            )
            applied_gri.append(gri3_app)
            stages.append(gri3_stage)

        # ============================================================
        # Step 4: GRI 5 — 용기/포장 처리
        # ============================================================
        candidates, gri5_app, gri5_stage = self._apply_gri5(
            text, candidates, gri_signals
        )
        applied_gri.append(gri5_app)
        stages.append(gri5_stage)

        # ============================================================
        # Step 5: GRI 6 — 소호 세분화 (HS4→HS6)
        # ============================================================
        candidates, gri6_app, gri6_stage = self._apply_gri6(
            input_data, candidates, input_attrs_8axis
        )
        applied_gri.append(gri6_app)
        stages.append(gri6_stage)

        # ============================================================
        # Step 6: 리스크 평가
        # ============================================================
        risk = self.risk_assessor.assess(
            candidates=candidates,
            applied_gri=applied_gri,
            ec_result=ec_result,
            input_attrs=input_attrs_8axis,
            input_data=input_data,
        )

        # Rule references 수집 (evidence에서)
        rule_references = self._collect_rule_references(candidates)

        debug['gri_decision'] = f'GRI 순차 완료 ({len(applied_gri)} steps)'

        return self._build_result(
            text, candidates, applied_gri, stages,
            ec_result, risk, rule_references, case_evidence,
            debug, input_attrs_8axis=input_attrs_8axis,
        )

    # ============================================================
    # GRI 1: 호의 용어 + 주규정
    # ============================================================
    def _apply_gri1(
        self, text: str, candidates: List[Candidate], gri_signals: GRISignals
    ) -> Tuple[List[Candidate], GRIApplication, StageInfo]:
        """GRI 1 적용: LegalGate 기반 hard exclude + redirect"""
        gri_app = GRIApplication(
            gri_id="GRI1",
            applied=False,
            candidates_before=len(candidates),
        )
        stage = StageInfo(stage_name="gri1_legal_gate", passed=True)

        if not self.legal_gate:
            gri_app.result_summary = "LegalGate 비활성"
            gri_app.candidates_after = len(candidates)
            return candidates, gri_app, stage

        gri_app.applied = True
        filtered, redirect_hs4s, lg_debug = self.legal_gate.apply(text, candidates)

        # 리다이렉트 후보 추가
        if redirect_hs4s:
            from .types import Evidence
            for rhs4 in redirect_hs4s:
                redirect_cand = Candidate(
                    hs4=rhs4,
                    score_ml=0.0,
                    evidence=[Evidence(
                        kind="legal_redirect",
                        source_id=rhs4,
                        text=f"주규정에 의한 리다이렉트: 제{rhs4}호",
                        weight=0.8,
                    )]
                )
                filtered.append(redirect_cand)

        gri_app.candidates_after = len(filtered)
        gri_app.result_summary = (
            f"통과 {lg_debug.get('passed', 0)}, "
            f"제외 {lg_debug.get('excluded', 0)}, "
            f"리다이렉트 {lg_debug.get('redirects_added', 0)}"
        )

        stage.passed = len(filtered) > 0
        stage.details = {
            'legal_gate': lg_debug,
            'candidates_after': len(filtered),
        }

        return filtered, gri_app, stage

    # ============================================================
    # GRI 2: 미완성/혼합물
    # ============================================================
    def _apply_gri2(
        self,
        text: str,
        candidates: List[Candidate],
        gri_signals: GRISignals,
        input_attrs_8axis: Optional[GlobalAttributes8Axis],
    ) -> Tuple[List[Candidate], GRIApplication, StageInfo]:
        """GRI 2 적용: 미완성/혼합물 처리"""
        gri_app = GRIApplication(
            gri_id="GRI2",
            applied=False,
            candidates_before=len(candidates),
        )
        stage = StageInfo(stage_name="gri2_incomplete_mixtures", passed=True)

        applied_2a = False
        applied_2b = False

        # GRI 2a: 미조립/불완전 → 완성품 호에 분류 가능 확장
        if gri_signals.gri2a_incomplete:
            applied_2a = True
            # 미조립품은 완성품 호에도 분류 가능 → 후보 유지 (제거하지 않음)
            # 부품 전용 호를 패널티 부여
            for cand in candidates:
                if self.reranker and self.reranker.is_parts_candidate(cand.hs4):
                    # 부품 전용 호: 약간의 패널티 (미조립이면 완성품 호가 우선)
                    cand.features['gri2a_parts_penalty'] = True

        # GRI 2b: 혼합물 → 구성 재질별 후보 확장
        if gri_signals.gri2b_mixtures:
            applied_2b = True
            # 혼합물은 주된 재질의 호에 분류
            # 이미 KB retrieval에서 재질 기반 확장되어 있음
            # 여기서는 재질 매칭 부스트만 적용
            if input_attrs_8axis and self.reranker:
                input_materials = set(input_attrs_8axis.material.values)
                for cand in candidates:
                    card_materials = self.reranker.card_attrs_8axis.get(
                        cand.hs4, {}
                    ).get('material', set())
                    if input_materials & card_materials:
                        cand.features['gri2b_material_match'] = True

        gri_app.applied = applied_2a or applied_2b
        gri_app.candidates_after = len(candidates)
        parts = []
        if applied_2a:
            parts.append("2a(미조립)")
        if applied_2b:
            parts.append("2b(혼합물)")
        gri_app.result_summary = f"적용: {', '.join(parts)}" if parts else "미적용"

        stage.details = {
            'gri2a_applied': applied_2a,
            'gri2b_applied': applied_2b,
        }

        return candidates, gri_app, stage

    # ============================================================
    # GRI 3: 복수 후보 해소
    # ============================================================
    def _apply_gri3(
        self,
        text: str,
        candidates: List[Candidate],
        gri_signals: GRISignals,
        input_attrs_8axis: Optional[GlobalAttributes8Axis],
        rerank_fn=None,
    ) -> Tuple[List[Candidate], GRIApplication, StageInfo, Optional[EssentialCharacterResult]]:
        """GRI 3 적용: 3a→3b→3c 순차"""
        gri_app = GRIApplication(
            gri_id="GRI3",
            applied=True,
            candidates_before=len(candidates),
        )
        stage = StageInfo(stage_name="gri3_resolution", passed=True)
        ec_result = None
        sub_step = ""

        # --- GRI 3(a): 가장 구체적 호 (specificity scoring) ---
        if rerank_fn and len(candidates) > 1:
            # Rerank로 specificity 반영
            candidates = rerank_fn(candidates)
            sub_step = "3a(specificity)"

        # 여전히 Top-2 간 격차가 작으면 3(b) 적용
        if len(candidates) >= 2:
            gap = candidates[0].score_total - candidates[1].score_total
            needs_3b = gap < 0.15 or (gri_signals.gri3_multi_candidate and gap < 0.30)

            if needs_3b:
                # --- GRI 3(b): Essential Character ---
                ec_result = self.ec_module.evaluate(
                    text, candidates[:5], input_attrs_8axis
                )

                if ec_result.applicable and ec_result.winner_hs4:
                    # EC winner를 top1으로 이동
                    winner_idx = None
                    for i, c in enumerate(candidates):
                        if c.hs4 == ec_result.winner_hs4:
                            winner_idx = i
                            break

                    if winner_idx is not None and winner_idx != 0:
                        winner = candidates.pop(winner_idx)
                        candidates.insert(0, winner)

                    candidates[0].features['gri3b_ec_winner'] = True
                    sub_step = "3b(essential_character)"
                else:
                    # --- GRI 3(c): 수 순서 최말위 (fallback) ---
                    # 동점인 후보 중 HS4 번호가 가장 큰 것을 선택
                    if len(candidates) >= 2:
                        tied = [c for c in candidates if abs(c.score_total - candidates[0].score_total) < 0.05]
                        if len(tied) >= 2:
                            tied.sort(key=lambda c: c.hs4, reverse=True)
                            # 최말위를 top1으로
                            last_hs4 = tied[0]
                            candidates.remove(last_hs4)
                            candidates.insert(0, last_hs4)
                            candidates[0].features['gri3c_last_in_order'] = True
                            sub_step = "3c(last_in_order)"

        gri_app.candidates_after = len(candidates)
        gri_app.result_summary = f"적용: {sub_step}" if sub_step else "미적용"

        stage.details = {
            'sub_step': sub_step,
            'ec_applied': ec_result is not None and ec_result.applicable,
            'ec_winner': ec_result.winner_hs4 if ec_result else None,
        }

        return candidates, gri_app, stage, ec_result

    # ============================================================
    # GRI 5: 용기/포장
    # ============================================================
    def _apply_gri5(
        self, text: str, candidates: List[Candidate], gri_signals: GRISignals
    ) -> Tuple[List[Candidate], GRIApplication, StageInfo]:
        """GRI 5 적용: 용기/포장 처리"""
        gri_app = GRIApplication(
            gri_id="GRI5",
            applied=False,
            candidates_before=len(candidates),
        )
        stage = StageInfo(stage_name="gri5_containers", passed=True)

        if not gri_signals.gri5_containers:
            gri_app.candidates_after = len(candidates)
            gri_app.result_summary = "미적용 (용기 신호 없음)"
            return candidates, gri_app, stage

        gri_app.applied = True

        # 전용 케이스/용기 신호 시 내용물 기준 분류
        # 용기 전용 호(4202 등)가 아닌 내용물 호를 우선
        container_hs4s = {'4202', '4602', '7010', '7612', '8304'}
        content_candidates = [c for c in candidates if c.hs4 not in container_hs4s]

        if content_candidates:
            # 내용물 후보가 있으면 내용물 우선
            container_cands = [c for c in candidates if c.hs4 in container_hs4s]
            candidates = content_candidates + container_cands
            gri_app.result_summary = f"내용물 기준 분류 (용기 {len(container_cands)}개 후순위)"
        else:
            gri_app.result_summary = "내용물 후보 없음, 용기 호 유지"

        gri_app.candidates_after = len(candidates)
        stage.details = {'gri5_applied': True}

        return candidates, gri_app, stage

    # ============================================================
    # GRI 6: 소호 세분화
    # ============================================================
    def _apply_gri6(
        self,
        input_data,
        candidates: List[Candidate],
        input_attrs_8axis: Optional[GlobalAttributes8Axis],
    ) -> Tuple[List[Candidate], GRIApplication, StageInfo]:
        """GRI 6 적용: HS4→HS6 서브헤딩 결정"""
        gri_app = GRIApplication(
            gri_id="GRI6",
            applied=False,
            candidates_before=len(candidates),
        )
        stage = StageInfo(stage_name="gri6_subheading", passed=True)

        if not self.subheading_resolver:
            gri_app.candidates_after = len(candidates)
            gri_app.result_summary = "SubheadingResolver 없음"
            return candidates, gri_app, stage

        resolved_count = 0
        for cand in candidates:
            if cand.hs6:
                continue  # 이미 결정됨

            sub_candidates = self.subheading_resolver.resolve(
                input_data, cand.hs4, input_attrs_8axis, topk=3
            )

            if sub_candidates:
                cand.hs6 = sub_candidates[0].hs6
                cand.features['hs6_candidates'] = [
                    sc.to_dict() for sc in sub_candidates
                ]
                resolved_count += 1

        gri_app.applied = resolved_count > 0
        gri_app.candidates_after = len(candidates)
        gri_app.result_summary = f"HS6 결정: {resolved_count}개"

        stage.details = {'resolved_count': resolved_count}

        return candidates, gri_app, stage

    # ============================================================
    # Helper methods
    # ============================================================
    def _collect_rule_references(self, candidates: List[Candidate]) -> List[RuleReference]:
        """후보 evidence에서 rule reference 수집"""
        refs = []
        seen_ids = set()

        for cand in candidates[:3]:
            for ev in cand.evidence:
                rule_id = ev.meta.get('rule_id', '')
                if rule_id and rule_id not in seen_ids:
                    seen_ids.add(rule_id)
                    refs.append(RuleReference(
                        rule_id=rule_id,
                        source=ev.meta.get('source', ''),
                        hs_version=ev.meta.get('hs_version', '2022'),
                        text_snippet=ev.text,
                    ))

        return refs

    def _build_result(
        self,
        text: str,
        candidates: List[Candidate],
        applied_gri: List[GRIApplication],
        stages: List[StageInfo],
        ec_result: Optional[EssentialCharacterResult],
        risk: Optional[RiskAssessment],
        rule_references: List[RuleReference],
        case_evidence: List[CaseEvidence],
        debug: Dict[str, Any],
        reason: str = "",
        input_attrs_8axis: Optional[GlobalAttributes8Axis] = None,
    ) -> ClassificationResult:
        """최종 ClassificationResult 생성"""
        # Confidence & Decision
        scores = [c.score_total for c in candidates]
        confidence = scores[0] if scores else 0.0

        if not reason:
            if risk and risk.level == "HIGH":
                status = "REVIEW"
                reason = f"리스크 HIGH ({risk.score:.1f})"
            elif not candidates:
                status = "ASK"
                reason = "후보 없음"
            else:
                status = "AUTO"
                reason = "GRI 순차 분류 완료"

        status = "AUTO" if reason.startswith("GRI 1 - 단일") else (
            "ASK" if "제외" in reason or "없음" in reason else "AUTO"
        )

        # 리스크가 HIGH면 REVIEW로 override
        if risk and risk.level == "HIGH":
            status = "REVIEW"
            reason = f"리스크 HIGH ({risk.score:.1f})"

        return ClassificationResult(
            input_text=text,
            topk=candidates[:5],
            decision=DecisionStatus(
                status=status,
                reason=reason,
                confidence=confidence,
            ),
            questions=[],
            stages=stages,
            debug=debug,
            applied_gri=applied_gri,
            essential_character=ec_result,
            risk=risk,
            rule_references=rule_references,
            case_evidence=case_evidence,
        )
