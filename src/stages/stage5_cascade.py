"""
Stage 5: Cascade (Hybrid -> RAG Escalation)

Hybrid-first, escalate to RAG when confidence is low.

Pipeline:
  Input -> Stage 4 (Hybrid) execute
       -> confidence >= 0.50 AND gap >= 0.15 -> return immediately
       -> otherwise -> Stage 3 (LLM/RAG) execute
       -> Merge results (cross-validation / adopt / keep)
       -> StageResult
"""

from typing import Optional, Dict, Any, List

from ..types import BaseClassifier, StageResult, Prediction, StageID


class CascadeClassifier(BaseClassifier):
    """Cascade: Hybrid-first + RAG escalation."""

    def __init__(
        self,
        hybrid_classifier: Optional['BaseClassifier'] = None,
        llm_classifier: Optional['BaseClassifier'] = None,
        escalation_conf_threshold: float = 0.50,
        escalation_gap_threshold: float = 0.15,
        rag_min_confidence: float = 0.30,
        cross_validation_boost: float = 0.15,
    ):
        self._hybrid = hybrid_classifier
        self._llm = llm_classifier
        self._conf_threshold = escalation_conf_threshold
        self._gap_threshold = escalation_gap_threshold
        self._rag_min_conf = rag_min_confidence
        self._cv_boost = cross_validation_boost

    def _init_hybrid(self):
        if self._hybrid is None:
            from .stage4_hybrid import HybridClassifier
            self._hybrid = HybridClassifier()

    def _init_llm(self):
        if self._llm is None:
            from .stage3_llm import LLMClassifier
            self._llm = LLMClassifier()

    @property
    def name(self) -> str:
        return "Cascade (Hybrid+RAG)"

    @property
    def stage_id(self) -> StageID:
        return StageID.CASCADE

    def _should_escalate(self, result: StageResult) -> tuple:
        """Determine if RAG escalation is needed. Returns (should_escalate, reason)."""
        if not result.predictions:
            return True, "no_candidates"

        p1 = result.predictions[0].score
        p2 = result.predictions[1].score if len(result.predictions) > 1 else 0.0

        # Decision status from hybrid
        status = result.metadata.get('status', '')
        if status in ('REVIEW', 'ABSTAIN'):
            return True, f"decision_{status.lower()}"

        if p1 < self._conf_threshold:
            return True, f"low_top1({p1:.3f}<{self._conf_threshold})"

        gap = p1 - p2
        if gap < self._gap_threshold:
            return True, f"small_gap({gap:.3f}<{self._gap_threshold})"

        if status == 'ASK':
            return True, "decision_ask"

        return False, ""

    def _merge_results(
        self, hybrid_result: StageResult, rag_result: StageResult
    ) -> tuple:
        """Merge hybrid + RAG results. Returns (merged_result, source, merge_debug)."""
        merge_debug: Dict[str, Any] = {}

        # RAG failed
        if not rag_result.predictions:
            merge_debug['reason'] = 'rag_no_results'
            return hybrid_result, 'hybrid', merge_debug

        if rag_result.metadata.get('is_fallback'):
            merge_debug['reason'] = 'rag_fallback'
            return hybrid_result, 'hybrid', merge_debug

        rag_hs4 = rag_result.top1
        rag_conf = rag_result.confidence
        hybrid_top5 = hybrid_result.topk_codes[:5]
        hybrid_conf = hybrid_result.confidence

        merge_debug['rag_hs4'] = rag_hs4
        merge_debug['rag_conf'] = round(rag_conf, 4)
        merge_debug['hybrid_top1'] = hybrid_result.top1
        merge_debug['hybrid_conf'] = round(hybrid_conf, 4)

        # RAG confidence too low
        if rag_conf < self._rag_min_conf:
            merge_debug['reason'] = f'rag_low_conf({rag_conf:.3f})'
            return hybrid_result, 'hybrid', merge_debug

        # Case A: RAG agrees with hybrid top-5 (cross-validation)
        if rag_hs4 in hybrid_top5:
            if rag_hs4 == hybrid_result.top1:
                # Same answer -> boost confidence
                boosted_conf = min(hybrid_conf + self._cv_boost, 1.0)
                boosted = StageResult(
                    input_text=hybrid_result.input_text,
                    predictions=hybrid_result.predictions,
                    confidence=boosted_conf,
                    metadata={
                        **hybrid_result.metadata,
                        'cascade_source': 'cross_validated',
                    },
                )
                merge_debug['reason'] = 'cross_validated_same'
                return boosted, 'rag_confirmed', merge_debug
            else:
                # RAG picks different candidate within top-5 -> reorder
                reordered = self._reorder(hybrid_result.predictions, rag_hs4)
                new_conf = max(rag_conf, hybrid_conf)
                reordered_result = StageResult(
                    input_text=hybrid_result.input_text,
                    predictions=reordered,
                    confidence=new_conf,
                    metadata={
                        **hybrid_result.metadata,
                        'cascade_source': 'rag_reordered',
                    },
                )
                merge_debug['reason'] = 'cross_validated_reorder'
                return reordered_result, 'rag', merge_debug

        # Case B: RAG gives completely new answer
        if rag_conf > 0.7:
            # High confidence RAG -> adopt RAG
            merged_preds = [
                Prediction(hs4=rag_hs4, score=rag_conf, rank=1),
            ] + [
                Prediction(hs4=p.hs4, score=p.score, rank=i + 2)
                for i, p in enumerate(hybrid_result.predictions[:4])
            ]
            rag_adopted = StageResult(
                input_text=hybrid_result.input_text,
                predictions=merged_preds,
                confidence=rag_conf,
                metadata={
                    **hybrid_result.metadata,
                    'cascade_source': 'rag_novel_adopted',
                },
            )
            merge_debug['reason'] = 'rag_novel_high_conf'
            return rag_adopted, 'rag', merge_debug
        else:
            # Low confidence + novel -> keep hybrid, flag for review
            merge_debug['reason'] = 'rag_novel_low_conf'
            review_result = StageResult(
                input_text=hybrid_result.input_text,
                predictions=hybrid_result.predictions,
                confidence=hybrid_conf,
                metadata={
                    **hybrid_result.metadata,
                    'cascade_source': 'hybrid_kept',
                    'status': 'REVIEW',
                },
            )
            return review_result, 'hybrid', merge_debug

    @staticmethod
    def _reorder(predictions: List[Prediction], target_hs4: str) -> List[Prediction]:
        """Move target_hs4 to rank 1."""
        target = None
        others = []
        for p in predictions:
            if p.hs4 == target_hs4 and target is None:
                target = p
            else:
                others.append(p)
        if target:
            result = [target] + others
        else:
            result = predictions
        for i, p in enumerate(result):
            p.rank = i + 1
        return result

    def classify(self, text: str, topk: int = 5) -> StageResult:
        debug: Dict[str, Any] = {}

        # Step 1: Hybrid
        self._init_hybrid()
        hybrid_result = self._hybrid.classify_timed(text, topk=topk)
        debug['hybrid_top1'] = hybrid_result.top1
        debug['hybrid_conf'] = round(hybrid_result.confidence, 4)
        debug['hybrid_ms'] = round(hybrid_result.latency_ms, 1)

        # Step 2: Escalation check
        should_escalate, reason = self._should_escalate(hybrid_result)
        debug['escalated'] = should_escalate
        debug['escalation_reason'] = reason

        if not should_escalate:
            hybrid_result.metadata = {
                **hybrid_result.metadata,
                'cascade_source': 'hybrid_direct',
                'cascade_debug': debug,
            }
            return hybrid_result

        # Step 3: RAG escalation
        self._init_llm()
        try:
            rag_result = self._llm.classify_timed(text, topk=topk)
            debug['rag_top1'] = rag_result.top1
            debug['rag_conf'] = round(rag_result.confidence, 4)
            debug['rag_ms'] = round(rag_result.latency_ms, 1)
        except Exception as e:
            debug['rag_error'] = str(e)[:100]
            hybrid_result.metadata = {
                **hybrid_result.metadata,
                'cascade_source': 'hybrid_rag_failed',
                'cascade_debug': debug,
            }
            return hybrid_result

        # Step 4: Merge
        merged, source, merge_debug = self._merge_results(hybrid_result, rag_result)
        debug['merge'] = merge_debug
        debug['final_source'] = source

        merged.metadata = {
            **merged.metadata,
            'cascade_debug': debug,
        }
        return merged
