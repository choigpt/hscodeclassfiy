"""
HS 분류 파이프라인 타입 정의
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Evidence:
    """분류 근거"""
    kind: str  # "card_keyword", "include_rule", "exclude_rule", "example"
    source_id: str  # hs4 또는 chunk_id
    text: str  # 근거 텍스트 (최대 160자)
    weight: float  # 가중치
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    """HS4 후보"""
    hs4: str
    score_ml: float = 0.0  # ML 모델 확률
    score_card: float = 0.0  # 카드 키워드 매칭 점수
    score_rule: float = 0.0  # 규칙 매칭 점수
    score_total: float = 0.0  # 총점
    evidence: List[Evidence] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)  # 피처 breakdown

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "hs4": self.hs4,
            "score_ml": round(self.score_ml, 4),
            "score_card": round(self.score_card, 4),
            "score_rule": round(self.score_rule, 4),
            "score_total": round(self.score_total, 4),
            "evidence": [
                {
                    "kind": e.kind,
                    "source_id": e.source_id,
                    "text": e.text[:160],
                    "weight": round(e.weight, 3)
                }
                for e in self.evidence[:5]  # 최대 5개
            ]
        }
        if self.features:
            result["features"] = self.features
        return result


@dataclass
class DecisionStatus:
    """분류 결정 상태"""
    status: str  # AUTO, ASK, REVIEW, ABSTAIN
    reason: str = ""  # 결정 이유
    confidence: float = 0.0  # 신뢰도

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "confidence": round(self.confidence, 4)
        }


@dataclass
class StageInfo:
    """파이프라인 단계 정보"""
    stage_name: str  # legal_gate_gri1, gri1_fact_check, gri2_rerank 등
    passed: bool  # 통과 여부
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "passed": self.passed,
            "details": self.details
        }


@dataclass
class ClassificationResult:
    """분류 결과 (GRI 1 중심 확장판)"""
    input_text: str
    topk: List[Candidate]
    decision: DecisionStatus  # AUTO/ASK/REVIEW/ABSTAIN
    questions: List[Dict[str, Any]] = field(default_factory=list)  # 구조화된 질문 (source_ref 포함)
    stages: List[StageInfo] = field(default_factory=list)  # 파이프라인 단계별 정보
    debug: Dict[str, Any] = field(default_factory=dict)

    # 하위호환을 위한 필드
    @property
    def low_confidence(self) -> bool:
        """하위호환: ASK/REVIEW/ABSTAIN이면 low_confidence"""
        return self.decision.status != "AUTO"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_text": self.input_text,
            "topk": [c.to_dict() for c in self.topk],
            "decision": self.decision.to_dict(),
            "questions": self.questions,
            "stages": [s.to_dict() for s in self.stages],
            "debug": self.debug,
            # 하위호환
            "low_confidence": self.low_confidence
        }
