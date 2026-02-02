"""
RequiredFact - 주규정/카드에서 추출한 요구 사실

HS 분류를 위해 반드시 확인해야 하는 사실들을 구조화:
- axis: 8축 속성 (object, material, processing, function, form, completeness, quant, legal)
- operator: eq, ne, gt, gte, lt, lte, contains, not_contains
- value: 요구되는 값
- basis: quant의 경우 기준 (weight, volume, count 등)
- hardness: hard (필수), soft (선호)
- source_ref: 출처 (note_id, card_field 등)
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum


class FactAxis(str, Enum):
    """요구 사실의 축"""
    OBJECT = "object"  # 물체 본질
    MATERIAL = "material"  # 재질
    PROCESSING = "processing"  # 가공 상태
    FUNCTION = "function"  # 기능/용도
    FORM = "form"  # 물리적 형태
    COMPLETENESS = "completeness"  # 완성도
    QUANT = "quant"  # 정량 규칙
    LEGAL = "legal"  # 법적 범위


class FactOperator(str, Enum):
    """요구 사실의 연산자"""
    EQ = "eq"  # 같음
    NE = "ne"  # 다름 (exclude)
    GT = "gt"  # 초과
    GTE = "gte"  # 이상
    LT = "lt"  # 미만
    LTE = "lte"  # 이하
    CONTAINS = "contains"  # 포함
    NOT_CONTAINS = "not_contains"  # 미포함 (exclude)


class FactHardness(str, Enum):
    """요구 사실의 강도"""
    HARD = "hard"  # 필수 (없으면 ASK)
    SOFT = "soft"  # 선호 (없어도 진행 가능)


@dataclass
class RequiredFact:
    """요구 사실"""
    axis: str  # FactAxis
    operator: str  # FactOperator
    value: str  # 요구되는 값
    hardness: str = "hard"  # FactHardness
    basis: Optional[str] = None  # quant의 경우 기준 (weight, volume 등)
    source_ref: str = ""  # 출처 참조
    confidence: float = 1.0  # 추출 신뢰도

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'RequiredFact':
        """딕셔너리로부터 생성"""
        return RequiredFact(
            axis=data['axis'],
            operator=data['operator'],
            value=data['value'],
            hardness=data.get('hardness', 'hard'),
            basis=data.get('basis'),
            source_ref=data.get('source_ref', ''),
            confidence=data.get('confidence', 1.0)
        )

    def __str__(self) -> str:
        """사람이 읽기 쉬운 형태"""
        op_symbol = {
            'eq': '=', 'ne': '≠', 'gt': '>', 'gte': '≥',
            'lt': '<', 'lte': '≤', 'contains': '∋', 'not_contains': '∌'
        }
        basis_str = f" (기준: {self.basis})" if self.basis else ""
        hardness_str = "[필수]" if self.hardness == "hard" else "[선호]"
        return f"{hardness_str} {self.axis} {op_symbol.get(self.operator, self.operator)} {self.value}{basis_str}"


@dataclass
class NoteRequirement:
    """주규정별 요구 사실 집합"""
    note_id: str  # section_1_1, chapter_4_2 등
    note_level: str  # section, chapter, subheading
    note_type: str  # include, exclude, redirect, definition
    hs_scope: str  # 적용 범위 (section_1, chapter_04, hs4_0401 등)
    required_facts: List[RequiredFact] = field(default_factory=list)
    raw_content: str = ""  # 원문 (디버깅용)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'note_id': self.note_id,
            'note_level': self.note_level,
            'note_type': self.note_type,
            'hs_scope': self.hs_scope,
            'required_facts': [f.to_dict() for f in self.required_facts],
            'raw_content': self.raw_content[:200]  # 최대 200자
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'NoteRequirement':
        """딕셔너리로부터 생성"""
        return NoteRequirement(
            note_id=data['note_id'],
            note_level=data['note_level'],
            note_type=data['note_type'],
            hs_scope=data['hs_scope'],
            required_facts=[RequiredFact.from_dict(f) for f in data.get('required_facts', [])],
            raw_content=data.get('raw_content', '')
        )


@dataclass
class HS4Requirements:
    """HS4별 요구 사실 집합"""
    hs4: str
    card_facts: List[RequiredFact] = field(default_factory=list)  # 카드에서 추출
    note_facts: List[RequiredFact] = field(default_factory=list)  # 주규정에서 추출

    def all_facts(self) -> List[RequiredFact]:
        """모든 요구 사실"""
        return self.card_facts + self.note_facts

    def hard_facts(self) -> List[RequiredFact]:
        """필수 사실만"""
        return [f for f in self.all_facts() if f.hardness == "hard"]

    def soft_facts(self) -> List[RequiredFact]:
        """선호 사실만"""
        return [f for f in self.all_facts() if f.hardness == "soft"]

    def facts_by_axis(self, axis: str) -> List[RequiredFact]:
        """특정 축의 사실만"""
        return [f for f in self.all_facts() if f.axis == axis]

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'hs4': self.hs4,
            'card_facts': [f.to_dict() for f in self.card_facts],
            'note_facts': [f.to_dict() for f in self.note_facts],
            'total_facts': len(self.all_facts()),
            'hard_facts': len(self.hard_facts()),
            'soft_facts': len(self.soft_facts())
        }
