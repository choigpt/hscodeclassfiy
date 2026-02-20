"""
8-axis attribute extraction for HS classification.

Axes:
1) object_nature  2) material  3) processing_state  4) function_use
5) physical_form  6) completeness  7) quantitative_rules  8) legal_scope
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Tuple

from ..text import normalize

# ---- Keyword dictionaries for each axis ----

OBJECT_NATURE_KEYWORDS = {
    'substance': ['물질', 'substance', '화합물', '용액', '시약'],
    'product': ['제품', 'product', '완제품', '기성품'],
    'organism': ['생물', '동물', '식물', '활어', '살아있는'],
    'mixture': ['혼합물', 'mixture', '합금', '혼방', '복합', '조제품', 'alloy', 'blend'],
    'set': ['세트', 'set', '키트', 'kit', '구성품'],
    'machine': ['기계', 'machine', '장치', 'apparatus', '장비', '기기', 'device'],
    'food': ['식품', '식료품', 'food', '음식'],
}

MATERIAL_KEYWORDS = {
    'metal': ['금속', 'metal', '철', 'iron', 'steel', '알루미늄', '구리', '스테인레스', '합금'],
    'plastic': ['플라스틱', 'plastic', '합성수지', '폴리에틸렌', '나일론', '폴리에스터', 'abs'],
    'rubber': ['고무', 'rubber', '실리콘', 'silicone', '라텍스'],
    'wood': ['목재', 'wood', '나무', '합판', '대나무'],
    'paper': ['종이', 'paper', '펄프', '판지'],
    'glass': ['유리', 'glass', '석영'],
    'textile': ['섬유', 'textile', '직물', '면', 'cotton', '모', 'wool', '편물', '부직포'],
    'leather': ['가죽', 'leather', '피혁', '모피'],
    'ceramic': ['세라믹', 'ceramic', '도자기', '타일'],
    'chemical': ['화학', 'chemical', '유기', '무기', '산', 'acid'],
    'composite': ['복합', 'composite', '적층', 'laminated', '코팅', 'coated', '도금'],
    'animal': ['동물성', '육류', '어류', '갑각류'],
    'vegetable': ['식물성', '채소', '과일', '곡물'],
}

PROCESSING_STATE_KEYWORDS = {
    'fresh': ['신선', 'fresh', '생', '살아있는'],
    'chilled': ['냉장', 'chilled'],
    'frozen': ['냉동', 'frozen', '동결'],
    'dried': ['건조', 'dried', '말린'],
    'salted': ['염장', 'salted', '절임'],
    'smoked': ['훈제', 'smoked'],
    'cooked': ['조리', 'cooked', '익힌', '가열'],
    'concentrated': ['농축', 'concentrated'],
    'fermented': ['발효', 'fermented', '숙성'],
    'raw': ['원료', 'raw', '미가공', '천연'],
    'refined': ['정제', 'refined', '정련'],
    'processed': ['가공', 'processed', '처리'],
    'assembled': ['조립', 'assembled'],
    'unassembled': ['미조립', 'unassembled', '분해', 'ckd', 'skd'],
    'coated': ['코팅', 'coated', '피복', '도금', 'plated'],
    'powder': ['분말', 'powder', '가루'],
    'liquid': ['액체', 'liquid', '용액'],
}

FUNCTION_USE_KEYWORDS = {
    'food': ['식품', 'food', '식용'],
    'medical': ['의료', 'medical', '의약', '치료'],
    'cosmetic': ['화장품', 'cosmetic', '미용'],
    'construction': ['건축', 'construction', '건설'],
    'machinery': ['기계', 'machine', '장치'],
    'packaging': ['포장', 'packaging', '용기'],
    'industrial': ['산업', 'industrial', '공업'],
    'household': ['가정', 'household', '생활용', '주방'],
    'automotive': ['자동차', 'automotive', '차량'],
    'agricultural': ['농업', 'agricultural'],
    'electrical': ['전기', 'electrical', '전자'],
    'laboratory': ['실험', 'laboratory', '시험'],
    'sport': ['스포츠', 'sport', '운동'],
}

PHYSICAL_FORM_KEYWORDS = {
    'powder_form': ['분말', 'powder', '가루'],
    'granule': ['과립', 'granule', '펠릿', 'pellet'],
    'plate': ['판', 'plate', '시트', 'sheet'],
    'bar': ['봉', 'bar', 'rod', '막대'],
    'wire': ['선', 'wire', '케이블'],
    'tube': ['관', 'tube', 'pipe', '파이프'],
    'film': ['필름', 'film', '막'],
    'bottle': ['병', 'bottle'],
    'container': ['용기', 'container', '탱크'],
    'fabric': ['직물', 'fabric', '천', '원단'],
    'clothing': ['의복', 'clothing', '의류'],
}

COMPLETENESS_KEYWORDS = {
    'complete': ['완제품', 'complete', '완성품'],
    'incomplete': ['미완성', 'incomplete', '반제품'],
    'parts': ['부품', 'part', 'parts', '부분품', '부속품'],
    'accessory': ['액세서리', 'accessory'],
    'component': ['구성품', 'component', '모듈', 'module'],
    'set': ['세트', 'set', '세트품'],
    'kit': ['키트', 'kit', '조립키트'],
    'ckd': ['ckd', '완전분해'],
    'skd': ['skd', '반분해'],
    'unassembled': ['미조립', 'unassembled', '분해', '조립전'],
    'assembled': ['조립', 'assembled'],
}

LEGAL_SCOPE_KEYWORDS = {
    'gri1': ['주', 'note', '호의 용어'],
    'gri2a': ['미조립', 'unassembled', 'CKD'],
    'gri2b': ['혼합물', '혼방', '복합', '본질적 특성'],
    'gri3': ['세트', '본질적 특성'],
    'gri5': ['케이스', '용기', '포장'],
    'include': ['포함', 'include', '해당'],
    'exclude': ['제외', 'exclude', '해당하지'],
}

QUANT_PATTERNS = [
    (r'(\d+(?:\.\d+)?)\s*(%|퍼센트|percent)', 'percent', None),
    (r'함량\s*[:\s]*(\d+(?:\.\d+)?)\s*(%|g|mg|kg)?', 'content', None),
    (r'순도\s*[:\s]*(\d+(?:\.\d+)?)\s*%?', 'purity', 'percent'),
    (r'중량\s*[:\s]*(\d+(?:\.\d+)?)\s*(g|kg|mg)?', 'weight', None),
    (r'(\d+(?:\.\d+)?)\s*(mm|cm|m|인치|inch)', 'dimension', None),
]


# ---- Data structures ----

@dataclass
class QuantFact:
    value: float
    unit: Optional[str] = None
    property: Optional[str] = None
    operator: str = '='
    raw_span: str = ''


@dataclass
class AxisAttributes:
    axis: str
    values: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class Attributes8Axis:
    """8-axis global attributes."""
    object_nature: AxisAttributes = field(default_factory=lambda: AxisAttributes(axis='object_nature'))
    material: AxisAttributes = field(default_factory=lambda: AxisAttributes(axis='material'))
    processing_state: AxisAttributes = field(default_factory=lambda: AxisAttributes(axis='processing_state'))
    function_use: AxisAttributes = field(default_factory=lambda: AxisAttributes(axis='function_use'))
    physical_form: AxisAttributes = field(default_factory=lambda: AxisAttributes(axis='physical_form'))
    completeness: AxisAttributes = field(default_factory=lambda: AxisAttributes(axis='completeness'))
    quantitative_rules: List[QuantFact] = field(default_factory=list)
    legal_scope: AxisAttributes = field(default_factory=lambda: AxisAttributes(axis='legal_scope'))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'object_nature': {'values': self.object_nature.values, 'confidence': self.object_nature.confidence},
            'material': {'values': self.material.values, 'confidence': self.material.confidence},
            'processing_state': {'values': self.processing_state.values, 'confidence': self.processing_state.confidence},
            'function_use': {'values': self.function_use.values, 'confidence': self.function_use.confidence},
            'physical_form': {'values': self.physical_form.values, 'confidence': self.physical_form.confidence},
            'completeness': {'values': self.completeness.values, 'confidence': self.completeness.confidence},
            'quantitative_rules': [{'value': q.value, 'unit': q.unit, 'property': q.property} for q in self.quantitative_rules],
            'legal_scope': {'values': self.legal_scope.values, 'confidence': self.legal_scope.confidence},
        }

    def summary(self) -> str:
        parts = []
        for name in ['object_nature', 'material', 'processing_state', 'function_use',
                      'physical_form', 'completeness', 'legal_scope']:
            ax = getattr(self, name)
            if ax.values:
                parts.append(f"{name[:4]}:{','.join(ax.values[:2])}")
        if self.quantitative_rules:
            parts.append(f"quant:{len(self.quantitative_rules)}")
        return ' | '.join(parts) if parts else 'none'

    def get_axis(self, axis_id: str) -> AxisAttributes:
        axis_map = {
            'object_nature': self.object_nature,
            'material': self.material,
            'processing_state': self.processing_state,
            'function_use': self.function_use,
            'physical_form': self.physical_form,
            'completeness': self.completeness,
            'legal_scope': self.legal_scope,
        }
        return axis_map.get(axis_id, AxisAttributes(axis=axis_id))

    def is_parts(self) -> bool:
        return any(v in ('parts', 'component', 'accessory', 'module')
                   for v in self.completeness.values)

    def is_set(self) -> bool:
        return any(v in ('set', 'kit') for v in self.completeness.values)

    def has_quant(self) -> bool:
        return len(self.quantitative_rules) > 0


# ---- Legacy compatibility ----

@dataclass
class GlobalAttributes:
    """Legacy 7-axis attributes (for backward compat)."""
    states: Set[str] = field(default_factory=set)
    materials: Set[str] = field(default_factory=set)
    uses_functions: Set[str] = field(default_factory=set)
    forms: Set[str] = field(default_factory=set)
    is_parts: bool = False
    is_set: bool = False
    is_incomplete: bool = False
    has_quant: bool = False
    quant_facts: List[QuantFact] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'states': list(self.states),
            'materials': list(self.materials),
            'uses_functions': list(self.uses_functions),
            'forms': list(self.forms),
            'is_parts': self.is_parts,
            'is_set': self.is_set,
            'has_quant': self.has_quant,
        }

    def summary(self) -> str:
        parts = []
        if self.states: parts.append(f"state:{','.join(self.states)}")
        if self.materials: parts.append(f"mat:{','.join(list(self.materials)[:3])}")
        if self.uses_functions: parts.append(f"use:{','.join(list(self.uses_functions)[:2])}")
        if self.is_parts: parts.append("parts")
        if self.is_set: parts.append("set")
        return ' | '.join(parts) if parts else 'none'


# ---- Extraction functions ----

def _match_axis(text: str, keyword_dict: Dict[str, List[str]]) -> AxisAttributes:
    norm_text = normalize(text).lower()
    text_lower = text.lower()
    values = []
    for category, keywords in keyword_dict.items():
        for kw in keywords:
            if kw.lower() in norm_text or kw.lower() in text_lower:
                if category not in values:
                    values.append(category)
                break
    confidence = min(1.0, len(values) * 0.3 + 0.2) if values else 0.0
    return AxisAttributes(axis='', values=values, confidence=confidence)


def _extract_quant_facts(text: str) -> List[QuantFact]:
    norm_text = normalize(text)
    facts = []
    for pattern, prop, default_unit in QUANT_PATTERNS:
        for match in re.finditer(pattern, norm_text, re.IGNORECASE):
            try:
                value = float(match.group(1))
                unit = match.group(2) if len(match.groups()) > 1 else default_unit
                facts.append(QuantFact(
                    value=value, unit=unit, property=prop, raw_span=match.group(0),
                ))
            except (ValueError, IndexError):
                continue
    return facts


def extract_attributes(text: str) -> Attributes8Axis:
    """Extract 8-axis attributes from text."""
    if not text:
        return Attributes8Axis()

    attrs = Attributes8Axis()
    attrs.object_nature = _match_axis(text, OBJECT_NATURE_KEYWORDS)
    attrs.object_nature.axis = 'object_nature'
    attrs.material = _match_axis(text, MATERIAL_KEYWORDS)
    attrs.material.axis = 'material'
    attrs.processing_state = _match_axis(text, PROCESSING_STATE_KEYWORDS)
    attrs.processing_state.axis = 'processing_state'
    attrs.function_use = _match_axis(text, FUNCTION_USE_KEYWORDS)
    attrs.function_use.axis = 'function_use'
    attrs.physical_form = _match_axis(text, PHYSICAL_FORM_KEYWORDS)
    attrs.physical_form.axis = 'physical_form'
    attrs.completeness = _match_axis(text, COMPLETENESS_KEYWORDS)
    attrs.completeness.axis = 'completeness'
    attrs.quantitative_rules = _extract_quant_facts(text)
    attrs.legal_scope = _match_axis(text, LEGAL_SCOPE_KEYWORDS)
    attrs.legal_scope.axis = 'legal_scope'
    return attrs


def extract_legacy_attributes(text: str) -> GlobalAttributes:
    """Extract legacy 7-axis attributes."""
    if not text:
        return GlobalAttributes()

    attrs = GlobalAttributes()
    proc = _match_axis(text, PROCESSING_STATE_KEYWORDS)
    attrs.states = set(proc.values)
    mat = _match_axis(text, MATERIAL_KEYWORDS)
    attrs.materials = set(mat.values)
    use = _match_axis(text, FUNCTION_USE_KEYWORDS)
    attrs.uses_functions = set(use.values)
    form = _match_axis(text, PHYSICAL_FORM_KEYWORDS)
    attrs.forms = set(form.values)
    comp = _match_axis(text, COMPLETENESS_KEYWORDS)
    attrs.is_parts = any(v in ('parts', 'component', 'accessory') for v in comp.values)
    attrs.is_set = any(v in ('set', 'kit') for v in comp.values)
    attrs.is_incomplete = 'incomplete' in comp.values or 'unassembled' in comp.values
    attrs.quant_facts = _extract_quant_facts(text)
    attrs.has_quant = len(attrs.quant_facts) > 0
    return attrs
