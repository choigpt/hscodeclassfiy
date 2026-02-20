"""
HS 분류 파이프라인 타입 정의
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


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
    hs6: str = ""  # HS6 서브헤딩 코드 (GRI 6에서 결정)
    score_ml: float = 0.0  # ML 모델 확률
    score_card: float = 0.0  # 카드 키워드 매칭 점수
    score_rule: float = 0.0  # 규칙 매칭 점수
    score_total: float = 0.0  # 총점
    evidence: List[Evidence] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)  # 피처 breakdown

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "hs4": self.hs4,
            "hs6": self.hs6,
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


class RiskLevel(Enum):
    """리스크 레벨"""
    LOW = "LOW"
    MED = "MED"
    HIGH = "HIGH"


@dataclass
class GRIApplication:
    """GRI 통칙 적용 기록"""
    gri_id: str  # "GRI1", "GRI2a", "GRI2b", "GRI3a", "GRI3b", "GRI3c", "GRI5", "GRI6"
    applied: bool  # 실제 적용 여부
    result_summary: str = ""  # 적용 결과 요약
    candidates_before: int = 0  # 적용 전 후보 수
    candidates_after: int = 0  # 적용 후 후보 수

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gri_id": self.gri_id,
            "applied": self.applied,
            "result_summary": self.result_summary,
            "candidates_before": self.candidates_before,
            "candidates_after": self.candidates_after,
        }


@dataclass
class ECFactor:
    """Essential Character 개별 요소 점수"""
    name: str
    score: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "score": round(self.score, 4), "reasoning": self.reasoning}


@dataclass
class EssentialCharacterResult:
    """Essential Character (GRI 3b) 평가 결과"""
    applicable: bool = False
    core_function: ECFactor = field(default_factory=lambda: ECFactor(name="core_function"))
    user_perception: ECFactor = field(default_factory=lambda: ECFactor(name="user_perception"))
    area_volume: ECFactor = field(default_factory=lambda: ECFactor(name="area_volume"))
    structural: ECFactor = field(default_factory=lambda: ECFactor(name="structural"))
    winner_hs4: str = ""
    reasoning: str = ""
    candidate_scores: Dict[str, float] = field(default_factory=dict)  # hs4 -> weighted total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applicable": self.applicable,
            "core_function": self.core_function.to_dict(),
            "user_perception": self.user_perception.to_dict(),
            "area_volume": self.area_volume.to_dict(),
            "structural": self.structural.to_dict(),
            "winner_hs4": self.winner_hs4,
            "reasoning": self.reasoning,
            "candidate_scores": {k: round(v, 4) for k, v in self.candidate_scores.items()},
        }


@dataclass
class RiskAssessment:
    """리스크 평가 결과"""
    level: str = "LOW"  # LOW / MED / HIGH
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    score_gap: float = 0.0
    missing_info: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "score": round(self.score, 2),
            "reasons": self.reasons,
            "score_gap": round(self.score_gap, 4),
            "missing_info": self.missing_info,
        }


@dataclass
class RuleReference:
    """규칙 참조"""
    rule_id: str
    source: str = ""  # explanatory_note, chapter_note, heading_note, etc.
    hs_version: str = "2022"
    text_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "source": self.source,
            "hs_version": self.hs_version,
            "text_snippet": self.text_snippet[:200],
        }


@dataclass
class CaseEvidence:
    """판결 케이스 근거"""
    case_id: str
    jurisdiction: str = "KR"
    final_code: str = ""
    decisive_reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "jurisdiction": self.jurisdiction,
            "final_code": self.final_code,
            "decisive_reasoning": self.decisive_reasoning[:300],
        }


@dataclass
class ClassificationResult:
    """분류 결과 (GRI 순차 파이프라인 확장판)"""
    input_text: str
    topk: List[Candidate]
    decision: DecisionStatus  # AUTO/ASK/REVIEW/ABSTAIN
    questions: List[Dict[str, Any]] = field(default_factory=list)  # 구조화된 질문 (source_ref 포함)
    stages: List[StageInfo] = field(default_factory=list)  # 파이프라인 단계별 정보
    debug: Dict[str, Any] = field(default_factory=dict)

    # GRI 순차 파이프라인 확장 필드
    applied_gri: List[GRIApplication] = field(default_factory=list)
    essential_character: Optional[EssentialCharacterResult] = None
    risk: Optional[RiskAssessment] = None
    rule_references: List[RuleReference] = field(default_factory=list)
    case_evidence: List[CaseEvidence] = field(default_factory=list)

    # 하위호환을 위한 필드
    @property
    def low_confidence(self) -> bool:
        """하위호환: ASK/REVIEW/ABSTAIN이면 low_confidence"""
        return self.decision.status != "AUTO"

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "input_text": self.input_text,
            "topk": [c.to_dict() for c in self.topk],
            "decision": self.decision.to_dict(),
            "questions": self.questions,
            "stages": [s.to_dict() for s in self.stages],
            "debug": self.debug,
            # GRI 순차 파이프라인 확장
            "applied_gri": [g.to_dict() for g in self.applied_gri],
            "essential_character": self.essential_character.to_dict() if self.essential_character else None,
            "risk": self.risk.to_dict() if self.risk else None,
            "rule_references": [r.to_dict() for r in self.rule_references],
            "case_evidence": [c.to_dict() for c in self.case_evidence],
            # 하위호환
            "low_confidence": self.low_confidence,
        }
        return result
