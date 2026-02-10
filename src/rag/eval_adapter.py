"""
기존 eval 프레임워크 호환 어댑터

RAGResult → ClassificationResult 변환
"""

from typing import Optional

from .types import RAGResult
from .pipeline import RAGPipeline

from src.classifier.types import (
    ClassificationResult,
    Candidate,
    Evidence,
    DecisionStatus,
    StageInfo,
)


class RAGPipelineAdapter:
    """RAGPipeline을 기존 eval 프레임워크에 연결하는 어댑터"""

    def __init__(
        self,
        index_dir: str = "artifacts/rag_index",
        ollama_base_url: str = "http://localhost:11434/v1",
        ollama_model: str = "qwen2.5:7b",
    ):
        self.pipeline = RAGPipeline(
            index_dir=index_dir,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
        )

    def classify(self, text: str) -> ClassificationResult:
        """
        기존 파이프라인과 동일한 인터페이스

        Args:
            text: 입력 품명

        Returns:
            ClassificationResult (기존 eval과 호환)
        """
        rag_result = self.pipeline.classify(text)
        return self.convert(rag_result)

    @staticmethod
    def convert(rag_result: RAGResult) -> ClassificationResult:
        """RAGResult → ClassificationResult 변환"""

        # Candidate 변환
        topk = []
        for rc in rag_result.candidates:
            evidence_list = [
                Evidence(
                    kind=e.kind,
                    source_id=e.source_id,
                    text=e.text[:160],
                    weight=e.score,
                )
                for e in rc.evidence[:5]
            ]

            cand = Candidate(
                hs4=rc.hs4,
                score_ml=rc.score_llm,
                score_card=rc.score_retrieval,
                score_rule=0.0,
                score_total=rc.score_llm + rc.score_retrieval,
                evidence=evidence_list,
                features={
                    'score_llm': rc.score_llm,
                    'score_retrieval': rc.score_retrieval,
                    'rank': rc.rank,
                },
            )
            topk.append(cand)

        # DecisionStatus 변환
        conf = rag_result.confidence.calibrated
        if rag_result.need_info:
            status = "ASK"
            reason = "추가 정보 필요"
        elif conf >= 0.7:
            status = "AUTO"
            reason = "RAG 자동 분류"
        elif conf >= 0.4:
            status = "REVIEW"
            reason = "RAG 검토 필요"
        else:
            status = "ABSTAIN"
            reason = "RAG 신뢰도 부족"

        decision = DecisionStatus(
            status=status,
            reason=reason,
            confidence=conf,
        )

        # 질문 변환
        questions = [
            {'question': q, 'source_ref': 'rag_llm'}
            for q in rag_result.questions
        ]

        # StageInfo
        stages = [
            StageInfo(
                stage_name='rag_retrieval',
                passed=len(rag_result.candidates) > 0,
                details={
                    'retrieval_count': rag_result.debug.get('retrieval_count', 0),
                    'context_chars': rag_result.debug.get('context_chars', 0),
                },
            ),
            StageInfo(
                stage_name='rag_llm',
                passed=rag_result.debug.get('llm_success', False),
                details={
                    'is_fallback': rag_result.is_fallback,
                    'elapsed_sec': rag_result.debug.get('elapsed_sec', 0),
                },
            ),
        ]

        # GRI stage
        gri_active = rag_result.gri_signals.get('active_gri', [])
        if gri_active:
            stages.append(StageInfo(
                stage_name='gri_signals',
                passed=True,
                details={'active': gri_active},
            ))

        # Debug
        debug = {
            **rag_result.debug,
            'rag_mode': True,
            'ml_used': False,
            'ranker_applied': False,
            'confidence_breakdown': rag_result.confidence.to_dict(),
        }

        return ClassificationResult(
            input_text=rag_result.input_text,
            topk=topk,
            decision=decision,
            questions=questions,
            stages=stages,
            debug=debug,
        )
