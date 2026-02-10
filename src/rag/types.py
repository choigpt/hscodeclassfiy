"""
RAG 파이프라인 데이터 타입 정의
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RAGEvidence:
    """RAG 분류 근거"""
    kind: str  # "retrieval", "llm_reasoning", "gri_signal", "legal_gate", "ruling_case"
    source_id: str  # hs4, chunk_id, case_ref 등
    text: str
    score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Confidence:
    """신뢰도 정보"""
    raw_llm: float = 0.0       # LLM 자가 평가 (0~1)
    retrieval: float = 0.0     # 검색 점수 기반 (0~1)
    legal_gate: float = 0.0    # LegalGate 통과 점수 (0~1)
    calibrated: float = 0.0    # 최종 보정 신뢰도 (0~1)

    def to_dict(self) -> Dict[str, float]:
        return {
            'raw_llm': round(self.raw_llm, 4),
            'retrieval': round(self.retrieval, 4),
            'legal_gate': round(self.legal_gate, 4),
            'calibrated': round(self.calibrated, 4),
        }


@dataclass
class RAGCandidate:
    """RAG HS4 후보"""
    hs4: str
    rank: int = 0
    score_retrieval: float = 0.0   # 검색 점수 (RRF)
    score_llm: float = 0.0        # LLM 평가 점수
    reasoning: str = ""            # LLM 추론 근거
    evidence: List[RAGEvidence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'hs4': self.hs4,
            'rank': self.rank,
            'score_retrieval': round(self.score_retrieval, 4),
            'score_llm': round(self.score_llm, 4),
            'reasoning': self.reasoning[:300],
            'evidence': [
                {
                    'kind': e.kind,
                    'source_id': e.source_id,
                    'text': e.text[:160],
                    'score': round(e.score, 3),
                }
                for e in self.evidence[:5]
            ],
        }


@dataclass
class RAGResult:
    """RAG 분류 결과"""
    input_text: str
    best_hs4: str = ""
    candidates: List[RAGCandidate] = field(default_factory=list)
    confidence: Confidence = field(default_factory=Confidence)
    reasoning: str = ""
    need_info: bool = False
    questions: List[str] = field(default_factory=list)
    gri_signals: Dict[str, Any] = field(default_factory=dict)
    legal_gate_debug: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)
    is_fallback: bool = False  # LLM 실패 시 검색 결과만 사용

    @property
    def topk_hs4(self) -> List[str]:
        """상위 후보 HS4 코드 리스트"""
        return [c.hs4 for c in self.candidates]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'input_text': self.input_text,
            'best_hs4': self.best_hs4,
            'candidates': [c.to_dict() for c in self.candidates],
            'confidence': self.confidence.to_dict(),
            'reasoning': self.reasoning[:500],
            'need_info': self.need_info,
            'questions': self.questions,
            'gri_signals': self.gri_signals,
            'is_fallback': self.is_fallback,
            'debug': self.debug,
        }
