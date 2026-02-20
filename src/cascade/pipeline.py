"""
Cascade Pipeline: Hybrid-First + RAG Escalation

전략:
1. Hybrid Pipeline 먼저 실행 (빠름, CPU, 0.5s)
2. 고신뢰 → Hybrid 결과 즉시 반환
3. 저신뢰 → RAG Pipeline으로 에스컬레이션 (GPU, ~11s)
4. RAG 결과와 Hybrid 결과를 비교하여 최종 결정

실험 근거 (200-sample evaluation):
- Hybrid only: 63.0% Top-1
- RAG only: 63.5% Top-1 (유효 ~70.6%)
- Oracle combination: 86.0% (두 시스템 중 하나라도 맞으면)
- 오답 패턴이 상호보완적: Hybrid만 정답 45건, RAG만 정답 46건
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from ..classifier.pipeline import HSPipeline
from ..classifier.retriever import HSRetriever
from ..classifier.reranker import HSReranker
from ..classifier.clarify import HSClarifier
from ..classifier.types import ClassificationResult, DecisionStatus, Candidate
from ..rag.pipeline import RAGPipeline
from ..rag.types import RAGResult


@dataclass
class CascadeResult:
    """Cascade 파이프라인 결과"""
    final_result: ClassificationResult
    source: str  # "hybrid" | "rag" | "rag_confirmed"
    hybrid_result: ClassificationResult = None
    rag_result: Optional[RAGResult] = None
    escalated: bool = False
    escalation_reason: str = ""
    debug: Dict[str, Any] = field(default_factory=dict)

    @property
    def best_hs4(self) -> str:
        if self.final_result.topk:
            return self.final_result.topk[0].hs4
        return ""

    @property
    def confidence(self) -> float:
        return self.final_result.decision.confidence

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "source": self.source,
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason,
            "best_hs4": self.best_hs4,
            "confidence": round(self.confidence, 4),
            "final": self.final_result.to_dict(),
        }
        if self.rag_result:
            result["rag"] = self.rag_result.to_dict()
        result["debug"] = self.debug
        return result


class CascadePipeline:
    """
    Hybrid-First + RAG Escalation Pipeline

    Usage:
        pipe = CascadePipeline()
        result = pipe.classify("냉동 돼지 삼겹살")
        print(result.best_hs4, result.source, result.confidence)
    """

    def __init__(
        self,
        # Hybrid pipeline args
        hybrid_mode: str = "hybrid",
        # RAG pipeline args
        ollama_model: str = "qwen2.5:7b",
        # Cascade thresholds
        escalation_threshold_top1: float = 0.50,
        escalation_threshold_gap: float = 0.15,
        # Merge strategy
        rag_min_confidence: float = 0.3,
        cross_validation_boost: float = 0.15,
    ):
        self.escalation_threshold_top1 = escalation_threshold_top1
        self.escalation_threshold_gap = escalation_threshold_gap
        self.rag_min_confidence = rag_min_confidence
        self.cross_validation_boost = cross_validation_boost

        # Lazy init (무거운 모델 로딩은 첫 사용 시)
        self._hybrid_pipe = None
        self._rag_pipe = None
        self._hybrid_mode = hybrid_mode
        self._ollama_model = ollama_model

    def _get_hybrid(self) -> HSPipeline:
        if self._hybrid_pipe is None:
            if self._hybrid_mode == "hybrid":
                self._hybrid_pipe = HSPipeline(
                    retriever=HSRetriever(),
                    reranker=HSReranker(),
                    clarifier=HSClarifier(),
                    ranker_model_path="artifacts/ranker_legal/model_legal.txt",
                    use_gri=True, use_legal_gate=True, use_8axis=True,
                    use_rules=True, use_ranker=True, use_questions=True,
                )
            else:
                # kb_only mode
                self._hybrid_pipe = HSPipeline(
                    retriever=None,
                    reranker=HSReranker(),
                    clarifier=HSClarifier(),
                    use_gri=True, use_legal_gate=True, use_8axis=True,
                    use_rules=True, use_ranker=False, use_questions=True,
                )
        return self._hybrid_pipe

    def _get_rag(self) -> RAGPipeline:
        if self._rag_pipe is None:
            self._rag_pipe = RAGPipeline(ollama_model=self._ollama_model)
        return self._rag_pipe

    def _should_escalate(self, result: ClassificationResult) -> tuple:
        """
        에스컬레이션 여부 판단

        Returns:
            (should_escalate: bool, reason: str)
        """
        # Case 1: 결과 없음
        if not result.topk:
            return True, "no_candidates"

        # Case 2: REVIEW/ABSTAIN → 무조건 에스컬레이션
        if result.decision.status in ("REVIEW", "ABSTAIN"):
            return True, f"decision_{result.decision.status.lower()}"

        scores = [c.score_total for c in result.topk]
        p1 = scores[0]
        p2 = scores[1] if len(scores) > 1 else 0.0

        # Case 3: 1위 점수 낮음
        if p1 < self.escalation_threshold_top1:
            return True, f"low_top1({p1:.3f}<{self.escalation_threshold_top1})"

        # Case 4: 1위-2위 차이 작음 (경쟁 상태)
        gap = p1 - p2
        if gap < self.escalation_threshold_gap:
            return True, f"small_gap({gap:.3f}<{self.escalation_threshold_gap})"

        # Case 5: ASK 상태 (질문이 필요한 경우)
        if result.decision.status == "ASK":
            return True, "decision_ask"

        return False, ""

    def _merge_results(
        self,
        hybrid_result: ClassificationResult,
        rag_result: RAGResult,
    ) -> tuple:
        """
        Hybrid + RAG 결과 병합

        Returns:
            (final_result: ClassificationResult, source: str, debug: dict)
        """
        merge_debug = {}

        # RAG fallback/에러 → Hybrid 유지
        if rag_result.is_fallback:
            merge_debug["merge_reason"] = "rag_fallback"
            return hybrid_result, "hybrid", merge_debug

        # RAG best_hs4가 비어있으면 → Hybrid 유지
        if not rag_result.best_hs4:
            merge_debug["merge_reason"] = "rag_no_answer"
            return hybrid_result, "hybrid", merge_debug

        rag_hs4 = rag_result.best_hs4
        rag_conf = rag_result.confidence.calibrated
        hybrid_top5 = [c.hs4 for c in hybrid_result.topk[:5]]
        hybrid_conf = hybrid_result.decision.confidence

        merge_debug["rag_hs4"] = rag_hs4
        merge_debug["rag_confidence"] = round(rag_conf, 4)
        merge_debug["hybrid_top1"] = hybrid_top5[0] if hybrid_top5 else ""
        merge_debug["hybrid_confidence"] = round(hybrid_conf, 4)
        merge_debug["rag_in_hybrid_top5"] = rag_hs4 in hybrid_top5

        # RAG 신뢰도 최소 기준 미달 → Hybrid 유지
        if rag_conf < self.rag_min_confidence:
            merge_debug["merge_reason"] = f"rag_low_conf({rag_conf:.3f})"
            return hybrid_result, "hybrid", merge_debug

        # Case A: RAG 결과가 Hybrid Top-5에 포함 (교차 검증 성공)
        if rag_hs4 in hybrid_top5:
            # Hybrid Top-5에 있지만 1위가 아닌 경우 → RAG가 재정렬
            if rag_hs4 == hybrid_top5[0]:
                # 동일 답변 → Hybrid 결과 + 신뢰도 부스트
                boosted_conf = min(hybrid_conf + self.cross_validation_boost, 1.0)
                boosted_result = ClassificationResult(
                    input_text=hybrid_result.input_text,
                    topk=hybrid_result.topk,
                    decision=DecisionStatus(
                        status="AUTO",
                        reason=f"교차검증 확인 (Hybrid+RAG 일치)",
                        confidence=boosted_conf,
                    ),
                    questions=[],
                    stages=hybrid_result.stages,
                    debug=hybrid_result.debug,
                )
                merge_debug["merge_reason"] = "cross_validated_same"
                merge_debug["boosted_confidence"] = round(boosted_conf, 4)
                return boosted_result, "rag_confirmed", merge_debug
            else:
                # RAG가 Hybrid Top-5 내 다른 후보를 선택
                # → RAG 결과로 재정렬
                reordered = self._reorder_candidates(hybrid_result.topk, rag_hs4)
                new_conf = max(rag_conf, hybrid_conf)
                reordered_result = ClassificationResult(
                    input_text=hybrid_result.input_text,
                    topk=reordered,
                    decision=DecisionStatus(
                        status="AUTO",
                        reason=f"RAG 재정렬 (Top-5 내 교차검증)",
                        confidence=new_conf,
                    ),
                    questions=[],
                    stages=hybrid_result.stages,
                    debug=hybrid_result.debug,
                )
                merge_debug["merge_reason"] = "cross_validated_reorder"
                return reordered_result, "rag", merge_debug

        # Case B: RAG 결과가 Hybrid Top-5에 없음 (완전 새로운 답변)
        if rag_conf > 0.7:
            # RAG 고신뢰 → RAG 채택
            rag_candidate = Candidate(
                hs4=rag_hs4,
                score_total=rag_conf,
            )
            rag_adapted = ClassificationResult(
                input_text=hybrid_result.input_text,
                topk=[rag_candidate] + hybrid_result.topk[:4],
                decision=DecisionStatus(
                    status="AUTO",
                    reason=f"RAG 고신뢰 답변 채택 (conf={rag_conf:.3f})",
                    confidence=rag_conf,
                ),
                questions=[],
                stages=hybrid_result.stages,
                debug=hybrid_result.debug,
            )
            merge_debug["merge_reason"] = "rag_novel_high_conf"
            return rag_adapted, "rag", merge_debug
        else:
            # RAG 저신뢰 + 새로운 답변 → REVIEW
            merge_debug["merge_reason"] = "rag_novel_low_conf"
            review_result = ClassificationResult(
                input_text=hybrid_result.input_text,
                topk=hybrid_result.topk,
                decision=DecisionStatus(
                    status="REVIEW",
                    reason=f"Hybrid-RAG 불일치 (RAG={rag_hs4}, conf={rag_conf:.3f})",
                    confidence=hybrid_conf,
                ),
                questions=[],
                stages=hybrid_result.stages,
                debug=hybrid_result.debug,
            )
            return review_result, "hybrid", merge_debug

    def _reorder_candidates(
        self, candidates: List[Candidate], target_hs4: str
    ) -> List[Candidate]:
        """target_hs4를 1위로 재정렬"""
        target = None
        others = []
        for c in candidates:
            if c.hs4 == target_hs4 and target is None:
                target = c
            else:
                others.append(c)
        if target:
            return [target] + others
        return candidates

    def classify(self, text: str, topk: int = 5) -> CascadeResult:
        """
        Cascade 분류 실행

        1. Hybrid 먼저 실행
        2. 저신뢰 시 RAG로 에스컬레이션
        3. 결과 병합

        Args:
            text: 입력 품명
            topk: 반환할 상위 후보 수

        Returns:
            CascadeResult
        """
        debug = {}
        t0 = time.time()

        # Step 1: Hybrid
        hybrid_pipe = self._get_hybrid()
        t_hybrid_start = time.time()
        hybrid_result = hybrid_pipe.classify(text, topk=topk)
        t_hybrid = time.time() - t_hybrid_start
        debug["hybrid_sec"] = round(t_hybrid, 3)
        debug["hybrid_top1"] = hybrid_result.topk[0].hs4 if hybrid_result.topk else ""
        debug["hybrid_confidence"] = round(hybrid_result.decision.confidence, 4)
        debug["hybrid_status"] = hybrid_result.decision.status

        # Step 2: 에스컬레이션 판단
        should_escalate, reason = self._should_escalate(hybrid_result)
        debug["escalated"] = should_escalate
        debug["escalation_reason"] = reason

        if not should_escalate:
            # 고신뢰 → Hybrid 결과 즉시 반환
            debug["total_sec"] = round(time.time() - t0, 3)
            return CascadeResult(
                final_result=hybrid_result,
                source="hybrid",
                hybrid_result=hybrid_result,
                rag_result=None,
                escalated=False,
                escalation_reason="",
                debug=debug,
            )

        # Step 3: RAG 에스컬레이션
        rag_pipe = self._get_rag()
        t_rag_start = time.time()
        try:
            rag_result = rag_pipe.classify(text)
        except Exception as e:
            # RAG 실패 → Hybrid 결과 유지
            debug["rag_error"] = str(e)[:100]
            debug["total_sec"] = round(time.time() - t0, 3)
            return CascadeResult(
                final_result=hybrid_result,
                source="hybrid",
                hybrid_result=hybrid_result,
                rag_result=None,
                escalated=True,
                escalation_reason=reason,
                debug=debug,
            )
        t_rag = time.time() - t_rag_start
        debug["rag_sec"] = round(t_rag, 3)
        debug["rag_top1"] = rag_result.best_hs4
        debug["rag_confidence"] = round(rag_result.confidence.calibrated, 4)
        debug["rag_is_fallback"] = rag_result.is_fallback

        # Step 4: 결과 병합
        final_result, source, merge_debug = self._merge_results(
            hybrid_result, rag_result
        )
        debug["merge"] = merge_debug
        debug["total_sec"] = round(time.time() - t0, 3)

        return CascadeResult(
            final_result=final_result,
            source=source,
            hybrid_result=hybrid_result,
            rag_result=rag_result,
            escalated=True,
            escalation_reason=reason,
            debug=debug,
        )
