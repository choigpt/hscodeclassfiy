"""
전역 속성 추출기 (Global Attribute Extractor)

HS 분류 8축 결정축(Decision Axes) 추출:
1) object_nature: 물체 본질 (물질/제품/생물/혼합물/세트/기계)
2) material: 재질/성분 (주성분, 혼합비)
3) processing_state: 가공상태 (물리/화학적 처리 상태)
4) function_use: 기능/용도 (주요 기능 및 사용처)
5) physical_form: 물리적 형태 (형상, 규격, 치수)
6) completeness: 완성도 (완제품/부품/세트)
7) quantitative_rules: 정량규칙 (함량/순도/임계값)
8) legal_scope: 법적범위 (GRI, 주규정, 포함/제외)

기존 7축 호환성 유지 (GlobalAttributes 클래스)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Tuple
from .utils_text import normalize, tokenize


# ============================================================
# 8축 축 ID 상수
# ============================================================
AXIS_IDS = [
    'object_nature',
    'material',
    'processing_state',
    'function_use',
    'physical_form',
    'completeness',
    'quantitative_rules',
    'legal_scope',
]


# ============================================================
# 8축 속성별 키워드/패턴 사전
# ============================================================

# 1) object_nature (물체 본질)
OBJECT_NATURE_KEYWORDS = {
    'substance': ['물질', 'substance', '원소', '화합물', '용액', '시약', '약품'],
    'product': ['제품', 'product', '상품', '용품', '완제품', '기성품'],
    'organism': ['생물', '동물', '식물', '생체', '생물체', 'living', '살아있는',
                 '활어', '활', '살아 있는'],
    'mixture': ['혼합물', 'mixture', '복합물', '합금', '혼방', '혼합', '복합', '조합물',
                '조제품', 'alloy', 'blend', '배합'],
    'set': ['세트', 'set', '키트', 'kit', '조합', '구성품', '세트품', '한벌'],
    'machine': ['기계', 'machine', '장치', 'apparatus', '장비', '설비', '기구', '기기',
                'device', 'equipment'],
    'material': ['재료', 'material', '원료', '소재', '원재료', '자재'],
    'food': ['식품', '식료품', 'food', '음식', '식량'],
}

# 2) material (재질/성분)
MATERIAL_KEYWORDS = {
    'metal': ['금속', 'metal', '철', 'iron', 'steel', '강', '스틸', '알루미늄', 'aluminum', 'aluminium',
              '구리', 'copper', '아연', 'zinc', '니켈', 'nickel', '주석', 'tin', '납', 'lead',
              '티타늄', 'titanium', '텅스텐', 'tungsten', '크롬', 'chrome', '스테인레스', 'stainless',
              '합금', 'alloy', '비철금속', '귀금속', '금', 'gold', '은', 'silver', '백금', 'platinum'],
    'plastic': ['플라스틱', 'plastic', '합성수지', '폴리에틸렌', 'polyethylene', 'pe', 'pp',
                '폴리프로필렌', 'polypropylene', 'pvc', '폴리염화비닐', '나일론', 'nylon',
                '폴리에스터', 'polyester', '아크릴', 'acrylic', 'abs', '폴리카보네이트',
                'polycarbonate', '에폭시', 'epoxy', '우레탄', 'urethane'],
    'rubber': ['고무', 'rubber', '합성고무', '천연고무', '라텍스', 'latex', '실리콘', 'silicone',
               '가황', 'vulcanized', '경화고무', '발포고무'],
    'wood': ['목재', 'wood', '나무', '원목', '합판', 'plywood', '목', 'wooden', '대나무', 'bamboo',
             '파티클보드', '섬유판', 'mdf', '코르크', 'cork'],
    'paper': ['종이', 'paper', '펄프', 'pulp', '판지', 'cardboard', '골판지', '크라프트지'],
    'glass': ['유리', 'glass', '글라스', '크리스탈', 'crystal', '석영', 'quartz', '광학유리'],
    'textile': ['섬유', 'textile', '직물', '면', 'cotton', '모', 'wool', '견', 'silk',
                '마', 'linen', '합성섬유', '편물', 'knitted', '직포', 'woven', '부직포',
                'nonwoven', '펠트', 'felt', '레이스', '자수'],
    'leather': ['가죽', 'leather', '피혁', '모피', 'fur', '스웨이드', 'suede', '인조가죽'],
    'ceramic': ['세라믹', 'ceramic', '도자기', '자기', 'porcelain', '도기', '타일', 'tile',
                '내화물', '벽돌'],
    'chemical': ['화학', 'chemical', '화합물', 'compound', '유기', 'organic', '무기', 'inorganic',
                 '산', 'acid', '알칼리', '염', 'salt', '알코올', 'alcohol', '에스테르', '케톤'],
    'composite': ['복합', 'composite', '적층', 'laminated', '코팅', 'coated',
                  '피복', '도금', 'plated', '접합'],
    'stone': ['석재', 'stone', '대리석', 'marble', '화강암', 'granite', '석회석', '슬레이트',
              '천연석', '인조석'],
    'concrete': ['콘크리트', 'concrete', '시멘트', 'cement', '몰탈', '석고', 'gypsum'],
    'animal': ['동물성', '육류', '어류', '갑각류', '연체동물', '젤라틴', '뼈', '가죽원료'],
    'vegetable': ['식물성', '채소', '과일', '곡물', '견과류', '유지', '목초', '건초'],
}

# 3) processing_state (가공상태)
PROCESSING_STATE_KEYWORDS = {
    # 생물/식품 가공상태
    'fresh': ['신선', 'fresh', '생', '살아있는', 'live', 'living', '생것'],
    'chilled': ['냉장', 'chilled', '냉각', '저온'],
    'frozen': ['냉동', 'frozen', '동결', '급속냉동'],
    'dried': ['건조', 'dried', 'dry', '말린', '건', '탈수', '건조된', '자연건조'],
    'salted': ['염장', 'salted', '절임', '소금절임', '염수', '염수장'],
    'smoked': ['훈제', 'smoked', '훈연'],
    'cooked': ['조리', 'cooked', '조리된', '익힌', '가열', '삶은', '찐', '구운', '튀긴'],
    'concentrated': ['농축', 'concentrated', '농축액', '진한'],
    'fermented': ['발효', 'fermented', '숙성', '양조'],
    'pickled': ['절인', 'pickled', '초절임', '피클', '장아찌'],
    'roasted': ['볶은', 'roasted', '로스팅', '배전'],
    'sterilized': ['멸균', 'sterilized', '살균', '저온살균', 'pasteurized'],

    # 산업/화학 가공상태
    'raw': ['원료', 'raw', '미가공', '천연', '원상태'],
    'refined': ['정제', 'refined', '정련', '정화', '순화'],
    'crude': ['조(粗)', 'crude', '조제', '미정제', '원유'],
    'processed': ['가공', 'processed', '처리', '처리된'],
    'semi_processed': ['반가공', 'semi-processed', '1차가공', '조가공'],

    # 기계/제품 가공상태
    'assembled': ['조립', 'assembled', '조립된', '완성'],
    'unassembled': ['미조립', 'unassembled', '분해', '조립전', 'ckd', 'skd', '녹다운'],
    'machined': ['기계가공', 'machined', '절삭', '연삭', '연마'],
    'cast': ['주조', 'cast', '주물', '다이캐스팅'],
    'forged': ['단조', 'forged', '압연', '압출'],
    'molded': ['성형', 'molded', '사출', '블로우', '압축성형'],
    'welded': ['용접', 'welded', '접합', '납땜', 'soldered'],
    'coated': ['코팅', 'coated', '피복', '도금', 'plated', '도장', 'painted'],
    'surface_treated': ['표면처리', 'surface treated', '양극산화', '열처리', 'heat treated'],

    # 형태변형 상태
    'powder': ['분말', 'powder', '가루', '분쇄'],
    'liquid': ['액체', 'liquid', '용액', 'solution', '액상'],
    'paste': ['페이스트', 'paste', '반죽'],
}

# 기존 호환성 유지
STATE_KEYWORDS = PROCESSING_STATE_KEYWORDS

# 4) function_use (기능/용도)
FUNCTION_USE_KEYWORDS = {
    'food': ['식품', 'food', '식용', 'edible', '먹는', '음식'],
    'feed': ['사료', 'feed', '동물용', '축산', '가축용'],
    'medical': ['의료', 'medical', '의약', 'pharmaceutical', '약품', 'drug', '치료',
                '의료용', '임상용', '진단용'],
    'cosmetic': ['화장품', 'cosmetic', '미용', 'beauty', '화장', '스킨케어'],
    'construction': ['건축', 'construction', '건설', '건자재', 'building', '토목'],
    'machinery': ['기계', 'machine', 'machinery', '장치', 'apparatus', '설비'],
    'measurement': ['측정', 'measurement', '계측', 'measuring', '계량', '시험용'],
    'communication': ['통신', 'communication', '전자', 'electronic', '무선', 'wireless', 'IT'],
    'packaging': ['포장', 'packaging', '용기', 'container', '병', 'bottle', '포장용'],
    'industrial': ['산업', 'industrial', '공업', '제조', '공업용', '산업용'],
    'household': ['가정', 'household', '생활용', '주방', '가정용', '주거용'],
    'automotive': ['자동차', 'automotive', '차량', 'vehicle', '자동차용', '수송'],
    'agricultural': ['농업', 'agricultural', '농산', '원예', '농업용', '영농'],
    'textile_use': ['섬유용', '의류용', 'clothing', '착용', '피복용'],
    'electrical': ['전기', 'electrical', 'electric', '전자', '전기용'],
    'military': ['군용', 'military', '국방', '방위'],
    'laboratory': ['실험', 'laboratory', '시험', '연구', '연구용'],
    'sanitary': ['위생', 'sanitary', '보건', '청결'],
    'sport': ['스포츠', 'sport', '운동', '레저', '레크리에이션'],
    'toy': ['장난감', 'toy', '완구', '놀이'],
}

# 기존 호환성 유지
USE_KEYWORDS = FUNCTION_USE_KEYWORDS

# 5) physical_form (물리적 형태)
PHYSICAL_FORM_KEYWORDS = {
    # 기본 형태
    'raw_material': ['원료', 'raw material', '원재료', '소재'],
    'semi_finished': ['반제품', 'semi-finished', '중간재', '반가공'],
    'finished': ['완제품', 'finished', '완성품', '제품'],

    # 상태 형태
    'powder_form': ['분말', 'powder', '가루', '미분'],
    'granule': ['과립', 'granule', '알갱이', '펠릿', 'pellet', '과립상'],
    'flake': ['플레이크', 'flake', '조각', '박편'],
    'block': ['괴', 'block', '덩어리', 'lump', 'ingot', '잉곳', '슬래브'],
    'liquid_form': ['액체', 'liquid', '액상', '용액'],

    # 단면 형태
    'plate': ['판', 'plate', '플레이트', '평판', '시트', 'sheet', '박판'],
    'bar': ['봉', 'bar', 'rod', '막대', '형강'],
    'wire': ['선', 'wire', '와이어', '코드', 'cord', '케이블'],
    'tube': ['관', 'tube', 'pipe', '파이프', '튜브'],
    'profile': ['형재', 'profile', '형', '앵글', '채널'],

    # 박막/필름
    'film': ['필름', 'film', '막', '박막', '필름형'],
    'foil': ['박', 'foil', '호일', '박막'],
    'tape': ['테이프', 'tape', '띠', '스트립', 'strip'],

    # 용기 형태
    'bottle': ['병', 'bottle', '보틀'],
    'container': ['용기', 'container', '그릇', '탱크', 'tank'],
    'bag': ['봉지', 'bag', '포대', '자루', '백'],
    'box': ['상자', 'box', '케이스', 'case', '박스'],

    # 의류 형태
    'clothing': ['의복', 'clothing', '의류', '옷'],
    'shirt': ['셔츠', 'shirt', '블라우스', 'blouse'],
    'pants': ['바지', 'pants', 'trousers', '팬츠'],
    'dress': ['드레스', 'dress', '원피스'],
    'jacket': ['자켓', 'jacket', '재킷', '상의'],

    # 직물 형태
    'fabric': ['직물', 'fabric', '천', '원단'],
    'woven': ['직포', 'woven', '제직'],
    'knitted': ['편물', 'knitted', '편직', '니트'],
}

# 기존 호환성 유지
FORM_KEYWORDS = PHYSICAL_FORM_KEYWORDS

# 6) completeness (완성도)
COMPLETENESS_KEYWORDS = {
    # 완성도 수준
    'complete': ['완제품', 'complete', '완성품', '완성', '완전한'],
    'incomplete': ['미완성', 'incomplete', '반제품', '불완전'],
    'parts': ['부품', 'part', 'parts', '부분품', '부속품', '부속'],
    'accessory': ['액세서리', 'accessory', '부착물', '첨부물', '악세사리'],
    'component': ['구성품', 'component', '부재', '요소'],
    'module': ['모듈', 'module', '유닛', 'unit'],

    # 세트/키트
    'set': ['세트', 'set', '세트품', '한벌'],
    'kit': ['키트', 'kit', '조립키트'],
    'assortment': ['조합', 'assortment', '모음', '구색'],

    # 조립 상태
    'ckd': ['ckd', 'completely knocked down', '완전분해'],
    'skd': ['skd', 'semi knocked down', '반분해'],
    'knockdown': ['녹다운', 'knockdown', '분해상태'],
    'unassembled': ['미조립', 'unassembled', '분해', '조립전'],
    'assembled': ['조립', 'assembled', '조립완료', '조립품'],

    # 전용성
    'solely': ['전용', 'solely', 'principally', '주로', '오로지'],
    'general_purpose': ['범용', 'general purpose', '다목적', '다용도'],
}

# 기존 호환성 유지
PARTS_KEYWORDS = COMPLETENESS_KEYWORDS

# 7) quantitative_rules (정량규칙) - 패턴 기반
QUANT_PATTERNS = [
    # 퍼센트
    (r'(\d+(?:\.\d+)?)\s*(%|퍼센트|percent)', 'percent', None),
    # 함량
    (r'함량\s*[:\s]*(\d+(?:\.\d+)?)\s*(%|g|mg|kg)?', 'content', None),
    (r'(\d+(?:\.\d+)?)\s*(%|g|mg|kg)\s*(?:이상|이하|초과|미만|함유)', 'content', None),
    # 농도
    (r'농도\s*[:\s]*(\d+(?:\.\d+)?)\s*(%|ppm|mol)?', 'concentration', None),
    # 순도
    (r'순도\s*[:\s]*(\d+(?:\.\d+)?)\s*%?', 'purity', 'percent'),
    # 중량
    (r'중량\s*[:\s]*(\d+(?:\.\d+)?)\s*(g|kg|mg|톤|ton)?', 'weight', None),
    (r'(\d+(?:\.\d+)?)\s*(g|kg|mg)\s*(?:이상|이하|초과|미만)', 'weight', None),
    # 부피
    (r'(\d+(?:\.\d+)?)\s*(ml|l|리터|cc)', 'volume', None),
    # 비율
    (r'(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)', 'ratio', None),
    # 두께/치수
    (r'(\d+(?:\.\d+)?)\s*(mm|cm|m|인치|inch)', 'dimension', None),
    # 밀도
    (r'밀도\s*[:\s]*(\d+(?:\.\d+)?)\s*(g/cm3|kg/m3)?', 'density', None),
]

QUANT_OPERATORS = {
    '이상': '>=',
    '이하': '<=',
    '초과': '>',
    '미만': '<',
    '이상이하': 'between',
}

# 8) legal_scope (법적 범위)
LEGAL_SCOPE_KEYWORDS = {
    # GRI 관련
    'gri1': ['주', 'note', '호의 용어', '류의 용어', '총설'],
    'gri2a': ['미조립', 'unassembled', 'CKD', 'SKD', '미완성', '녹다운'],
    'gri2b': ['혼합물', '혼방', '복합', '본질적 특성', '주된 성분'],
    'gri3': ['세트', '소매판매', '본질적 특성', '세트물품'],
    'gri5': ['케이스', '용기', '포장', '전용', '함께 제시'],
    'gri6': ['소호', '동일한 수준'],

    # 포함/제외
    'include': ['포함', 'include', '해당', '분류'],
    'exclude': ['제외', 'exclude', '해당하지', '분류하지'],

    # 특별규정
    'note_reference': ['제1호', '제2호', '제3호', '주1', '주2', '주3', '소호주'],
    'chapter_note': ['류주', '부주', '호주'],
}


# ============================================================
# 데이터 구조 정의
# ============================================================

@dataclass
class QuantFact:
    """정량 조건"""
    value: float
    unit: Optional[str] = None
    basis: Optional[str] = None  # 중량/부피/건조중량
    property: Optional[str] = None  # 함량/농도/순도
    operator: str = '='  # >=, <=, >, <, =
    raw_span: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'value': self.value,
            'unit': self.unit,
            'basis': self.basis,
            'property': self.property,
            'operator': self.operator,
            'raw_span': self.raw_span,
        }


@dataclass
class AttributeSpan:
    """속성 추출 span"""
    value: str           # 추출된 값 (카테고리)
    raw_span: str        # 원문 span
    start: int = 0       # 시작 위치
    end: int = 0         # 끝 위치
    confidence: float = 1.0  # 신뢰도 (0.0-1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'value': self.value,
            'raw_span': self.raw_span,
            'start': self.start,
            'end': self.end,
            'confidence': self.confidence,
        }


@dataclass
class AxisAttributes:
    """단일 축 속성"""
    axis: str                          # 축 ID
    values: List[str] = field(default_factory=list)  # 추출된 값들
    spans: List[AttributeSpan] = field(default_factory=list)  # span 정보
    confidence: float = 0.0            # 축 전체 신뢰도

    def to_dict(self) -> Dict[str, Any]:
        return {
            'axis': self.axis,
            'values': self.values,
            'spans': [s.to_dict() for s in self.spans],
            'confidence': round(self.confidence, 3),
        }


@dataclass
class GlobalAttributes8Axis:
    """8축 전역 속성"""
    object_nature: AxisAttributes = field(default_factory=lambda: AxisAttributes(axis='object_nature'))
    material: AxisAttributes = field(default_factory=lambda: AxisAttributes(axis='material'))
    processing_state: AxisAttributes = field(default_factory=lambda: AxisAttributes(axis='processing_state'))
    function_use: AxisAttributes = field(default_factory=lambda: AxisAttributes(axis='function_use'))
    physical_form: AxisAttributes = field(default_factory=lambda: AxisAttributes(axis='physical_form'))
    completeness: AxisAttributes = field(default_factory=lambda: AxisAttributes(axis='completeness'))
    quantitative_rules: List[QuantFact] = field(default_factory=list)
    legal_scope: AxisAttributes = field(default_factory=lambda: AxisAttributes(axis='legal_scope'))

    # 디버그용 원문
    raw_text: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'object_nature': self.object_nature.to_dict(),
            'material': self.material.to_dict(),
            'processing_state': self.processing_state.to_dict(),
            'function_use': self.function_use.to_dict(),
            'physical_form': self.physical_form.to_dict(),
            'completeness': self.completeness.to_dict(),
            'quantitative_rules': [q.to_dict() for q in self.quantitative_rules],
            'legal_scope': self.legal_scope.to_dict(),
        }

    def summary(self) -> str:
        """속성 요약 문자열"""
        parts = []
        if self.object_nature.values:
            parts.append(f"obj:{','.join(self.object_nature.values[:2])}")
        if self.material.values:
            parts.append(f"mat:{','.join(self.material.values[:2])}")
        if self.processing_state.values:
            parts.append(f"proc:{','.join(self.processing_state.values[:2])}")
        if self.function_use.values:
            parts.append(f"use:{','.join(self.function_use.values[:2])}")
        if self.physical_form.values:
            parts.append(f"form:{','.join(self.physical_form.values[:2])}")
        if self.completeness.values:
            parts.append(f"comp:{','.join(self.completeness.values[:2])}")
        if self.quantitative_rules:
            parts.append(f"quant:{len(self.quantitative_rules)}")
        if self.legal_scope.values:
            parts.append(f"legal:{','.join(self.legal_scope.values[:2])}")
        return ' | '.join(parts) if parts else 'none'

    def primary_axes(self) -> List[str]:
        """주요 활성 축 리스트"""
        axes = []
        if self.object_nature.values:
            axes.append('object_nature')
        if self.material.values:
            axes.append('material')
        if self.processing_state.values:
            axes.append('processing_state')
        if self.function_use.values:
            axes.append('function_use')
        if self.physical_form.values:
            axes.append('physical_form')
        if self.completeness.values:
            axes.append('completeness')
        if self.quantitative_rules:
            axes.append('quantitative_rules')
        if self.legal_scope.values:
            axes.append('legal_scope')
        return axes

    def get_axis(self, axis_id: str) -> AxisAttributes:
        """축 ID로 AxisAttributes 반환"""
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

    def has_quant(self) -> bool:
        """정량 조건 존재 여부"""
        return len(self.quantitative_rules) > 0

    def is_parts(self) -> bool:
        """부품 여부"""
        return any(v in ['parts', 'component', 'accessory', 'module']
                   for v in self.completeness.values)

    def is_set(self) -> bool:
        """세트 여부"""
        return any(v in ['set', 'kit', 'assortment'] for v in self.completeness.values)

    def is_ckd_skd(self) -> bool:
        """CKD/SKD 여부"""
        return any(v in ['ckd', 'skd', 'knockdown', 'unassembled']
                   for v in self.completeness.values)


# 기존 GlobalAttributes 클래스 (하위 호환성 유지)
@dataclass
class GlobalAttributes:
    """전역 속성 추출 결과 (기존 7축 호환)"""
    # 기본 속성
    states: Set[str] = field(default_factory=set)
    materials: Set[str] = field(default_factory=set)
    uses_functions: Set[str] = field(default_factory=set)
    forms: Set[str] = field(default_factory=set)

    # 부품/세트 관련
    parts_signals: Set[str] = field(default_factory=set)
    is_parts: bool = False
    is_set: bool = False
    is_ckd_skd: bool = False
    is_incomplete: bool = False

    # 정량 조건
    quant_facts: List[QuantFact] = field(default_factory=list)
    has_quant: bool = False

    # 디버그
    debug_spans: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'states': list(self.states),
            'materials': list(self.materials),
            'uses_functions': list(self.uses_functions),
            'forms': list(self.forms),
            'parts_signals': list(self.parts_signals),
            'is_parts': self.is_parts,
            'is_set': self.is_set,
            'is_ckd_skd': self.is_ckd_skd,
            'is_incomplete': self.is_incomplete,
            'quant_facts': [q.to_dict() for q in self.quant_facts],
            'has_quant': self.has_quant,
            'debug_spans': self.debug_spans,
        }

    def summary(self) -> str:
        """속성 요약 문자열"""
        parts = []
        if self.states:
            parts.append(f"state:{','.join(self.states)}")
        if self.materials:
            parts.append(f"mat:{','.join(list(self.materials)[:3])}")
        if self.uses_functions:
            parts.append(f"use:{','.join(list(self.uses_functions)[:2])}")
        if self.forms:
            parts.append(f"form:{','.join(list(self.forms)[:2])}")
        if self.is_parts:
            parts.append("parts")
        if self.is_set:
            parts.append("set")
        if self.has_quant:
            parts.append(f"quant:{len(self.quant_facts)}")
        return ' | '.join(parts) if parts else 'none'

    def primary_axes(self) -> List[str]:
        """주요 활성 축 리스트"""
        axes = []
        if self.states:
            axes.append('state')
        if self.materials:
            axes.append('material')
        if self.uses_functions:
            axes.append('use')
        if self.forms:
            axes.append('form')
        if self.is_parts or self.parts_signals:
            axes.append('parts')
        if self.has_quant:
            axes.append('quant')
        return axes


# ============================================================
# 추출 함수
# ============================================================

def _match_keywords_with_spans(
    text: str,
    keyword_dict: Dict[str, List[str]]
) -> Tuple[Set[str], Dict[str, List[str]]]:
    """
    키워드 매칭 및 span 추출

    Returns:
        (매칭된 카테고리 set, {카테고리: [매칭 span 리스트]})
    """
    norm_text = normalize(text).lower()
    text_lower = text.lower()

    matched_categories = set()
    spans = {}

    for category, keywords in keyword_dict.items():
        category_spans = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in norm_text or kw_lower in text_lower:
                matched_categories.add(category)
                category_spans.append(kw)

        if category_spans:
            spans[category] = category_spans

    return matched_categories, spans


def _match_keywords_with_axis_attrs(
    text: str,
    keyword_dict: Dict[str, List[str]],
    axis_id: str
) -> AxisAttributes:
    """
    키워드 매칭으로 AxisAttributes 생성

    Returns:
        AxisAttributes 객체
    """
    norm_text = normalize(text).lower()
    text_lower = text.lower()

    values = []
    spans = []

    for category, keywords in keyword_dict.items():
        for kw in keywords:
            kw_lower = kw.lower()
            # 원문에서 위치 찾기
            pos = text_lower.find(kw_lower)
            if pos == -1:
                pos = norm_text.find(kw_lower)

            if pos != -1 or kw_lower in norm_text or kw_lower in text_lower:
                if category not in values:
                    values.append(category)
                spans.append(AttributeSpan(
                    value=category,
                    raw_span=kw,
                    start=max(0, pos),
                    end=max(0, pos) + len(kw) if pos != -1 else 0,
                    confidence=1.0 if pos != -1 else 0.8
                ))

    # 신뢰도 계산 (매칭된 span 수 기반)
    confidence = min(1.0, len(values) * 0.3 + 0.2) if values else 0.0

    return AxisAttributes(
        axis=axis_id,
        values=values,
        spans=spans,
        confidence=confidence
    )


def _extract_quant_facts(text: str) -> List[QuantFact]:
    """정량 조건 추출"""
    facts = []
    norm_text = normalize(text)

    for pattern, prop, default_unit in QUANT_PATTERNS:
        for match in re.finditer(pattern, norm_text, re.IGNORECASE):
            try:
                value = float(match.group(1))
                unit = match.group(2) if len(match.groups()) > 1 else default_unit

                # 연산자 추출
                operator = '='
                for op_ko, op_sym in QUANT_OPERATORS.items():
                    if op_ko in norm_text[max(0, match.start()-10):match.end()+10]:
                        operator = op_sym
                        break

                facts.append(QuantFact(
                    value=value,
                    unit=unit,
                    property=prop,
                    operator=operator,
                    raw_span=match.group(0)
                ))
            except (ValueError, IndexError):
                continue

    return facts


def extract_attributes_8axis(text: str) -> GlobalAttributes8Axis:
    """
    8축 기반 전역 속성 추출

    Args:
        text: 입력 품명/설명

    Returns:
        GlobalAttributes8Axis 객체
    """
    if not text:
        return GlobalAttributes8Axis()

    attrs = GlobalAttributes8Axis(raw_text=text)

    # 1) object_nature
    attrs.object_nature = _match_keywords_with_axis_attrs(
        text, OBJECT_NATURE_KEYWORDS, 'object_nature')

    # 2) material
    attrs.material = _match_keywords_with_axis_attrs(
        text, MATERIAL_KEYWORDS, 'material')

    # 3) processing_state
    attrs.processing_state = _match_keywords_with_axis_attrs(
        text, PROCESSING_STATE_KEYWORDS, 'processing_state')

    # 4) function_use
    attrs.function_use = _match_keywords_with_axis_attrs(
        text, FUNCTION_USE_KEYWORDS, 'function_use')

    # 5) physical_form
    attrs.physical_form = _match_keywords_with_axis_attrs(
        text, PHYSICAL_FORM_KEYWORDS, 'physical_form')

    # 6) completeness
    attrs.completeness = _match_keywords_with_axis_attrs(
        text, COMPLETENESS_KEYWORDS, 'completeness')

    # 7) quantitative_rules
    attrs.quantitative_rules = _extract_quant_facts(text)

    # 8) legal_scope
    attrs.legal_scope = _match_keywords_with_axis_attrs(
        text, LEGAL_SCOPE_KEYWORDS, 'legal_scope')

    return attrs


def extract_attributes(text: str) -> GlobalAttributes:
    """
    텍스트에서 전역 속성 추출 (기존 7축 호환)

    Args:
        text: 입력 품명/설명

    Returns:
        GlobalAttributes 객체
    """
    if not text:
        return GlobalAttributes()

    attrs = GlobalAttributes()

    # A) 상태
    states, state_spans = _match_keywords_with_spans(text, PROCESSING_STATE_KEYWORDS)
    attrs.states = states
    if state_spans:
        attrs.debug_spans['state'] = [s for spans in state_spans.values() for s in spans]

    # B) 재질
    materials, mat_spans = _match_keywords_with_spans(text, MATERIAL_KEYWORDS)
    attrs.materials = materials
    if mat_spans:
        attrs.debug_spans['material'] = [s for spans in mat_spans.values() for s in spans]

    # C) 용도
    uses, use_spans = _match_keywords_with_spans(text, FUNCTION_USE_KEYWORDS)
    attrs.uses_functions = uses
    if use_spans:
        attrs.debug_spans['use'] = [s for spans in use_spans.values() for s in spans]

    # D) 형태
    forms, form_spans = _match_keywords_with_spans(text, PHYSICAL_FORM_KEYWORDS)
    attrs.forms = forms
    if form_spans:
        attrs.debug_spans['form'] = [s for spans in form_spans.values() for s in spans]

    # E) 부품
    parts, parts_spans = _match_keywords_with_spans(text, COMPLETENESS_KEYWORDS)
    attrs.parts_signals = parts
    if parts_spans:
        attrs.debug_spans['parts'] = [s for spans in parts_spans.values() for s in spans]

    # 부품 관련 플래그
    attrs.is_parts = 'parts' in parts or 'component' in parts or 'accessory' in parts
    attrs.is_set = 'set' in parts or 'kit' in parts
    attrs.is_ckd_skd = 'ckd' in parts or 'skd' in parts or 'unassembled' in parts
    attrs.is_incomplete = 'incomplete' in parts or 'unassembled' in parts

    # F) 정량
    attrs.quant_facts = _extract_quant_facts(text)
    attrs.has_quant = len(attrs.quant_facts) > 0
    if attrs.quant_facts:
        attrs.debug_spans['quant'] = [q.raw_span for q in attrs.quant_facts]

    return attrs


def convert_8axis_to_legacy(attrs8: GlobalAttributes8Axis) -> GlobalAttributes:
    """
    8축 속성을 기존 7축 포맷으로 변환

    Args:
        attrs8: GlobalAttributes8Axis 객체

    Returns:
        GlobalAttributes 객체
    """
    attrs = GlobalAttributes()

    # 상태 (processing_state)
    attrs.states = set(attrs8.processing_state.values)

    # 재질 (material)
    attrs.materials = set(attrs8.material.values)

    # 용도 (function_use)
    attrs.uses_functions = set(attrs8.function_use.values)

    # 형태 (physical_form)
    attrs.forms = set(attrs8.physical_form.values)

    # 부품 (completeness)
    attrs.parts_signals = set(attrs8.completeness.values)
    attrs.is_parts = attrs8.is_parts()
    attrs.is_set = attrs8.is_set()
    attrs.is_ckd_skd = attrs8.is_ckd_skd()
    attrs.is_incomplete = 'incomplete' in attrs8.completeness.values

    # 정량
    attrs.quant_facts = attrs8.quantitative_rules
    attrs.has_quant = len(attrs8.quantitative_rules) > 0

    return attrs


def get_attribute_keywords(axis: str) -> Dict[str, List[str]]:
    """특정 축의 키워드 사전 반환"""
    axis_map = {
        # 기존 호환
        'state': PROCESSING_STATE_KEYWORDS,
        'material': MATERIAL_KEYWORDS,
        'use': FUNCTION_USE_KEYWORDS,
        'form': PHYSICAL_FORM_KEYWORDS,
        'parts': COMPLETENESS_KEYWORDS,
        # 8축
        'object_nature': OBJECT_NATURE_KEYWORDS,
        'processing_state': PROCESSING_STATE_KEYWORDS,
        'function_use': FUNCTION_USE_KEYWORDS,
        'physical_form': PHYSICAL_FORM_KEYWORDS,
        'completeness': COMPLETENESS_KEYWORDS,
        'legal_scope': LEGAL_SCOPE_KEYWORDS,
    }
    return axis_map.get(axis, {})


def compute_attribute_overlap(
    attrs1: GlobalAttributes,
    attrs2_keywords: Set[str],
    axis: str
) -> int:
    """속성 간 overlap 계산"""
    if axis == 'state':
        return len(attrs1.states & attrs2_keywords)
    elif axis == 'material':
        return len(attrs1.materials & attrs2_keywords)
    elif axis == 'use':
        return len(attrs1.uses_functions & attrs2_keywords)
    elif axis == 'form':
        return len(attrs1.forms & attrs2_keywords)
    elif axis == 'parts':
        return len(attrs1.parts_signals & attrs2_keywords)
    return 0


# 테스트
if __name__ == "__main__":
    test_cases = [
        "냉동 돼지 삼겹살",
        "면 60% 폴리에스터 40% 혼방 직물",
        "스마트폰 전용 케이스",
        "자동차 CKD 부품 세트",
        "농도 70% 이상 에탄올",
        "순도 99.9% 금괴",
        "플라스틱 필름 두께 0.5mm",
        "건조 망고 500g",
        "스테인레스 스틸 파이프",
        "LED TV 55인치",
        "미조립 가구 키트",
        "의료용 실리콘 튜브",
    ]

    print("=" * 60)
    print("8축 전역 속성 추출 테스트")
    print("=" * 60)

    for text in test_cases:
        attrs8 = extract_attributes_8axis(text)
        attrs7 = extract_attributes(text)

        print(f"\n입력: {text}")
        print(f"  [8축] 요약: {attrs8.summary()}")
        print(f"  [8축] 활성축: {attrs8.primary_axes()}")
        print(f"  [7축] 요약: {attrs7.summary()}")
        if attrs8.quantitative_rules:
            print(f"  정량: {[q.to_dict() for q in attrs8.quantitative_rules]}")
