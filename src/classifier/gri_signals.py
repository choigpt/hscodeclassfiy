"""
GRI (General Rules of Interpretation) 신호 탐지 모듈

통칙 1/2a/2b/3/5 관련 신호를 입력 텍스트에서 탐지
"""

import re
from typing import Dict, List, Set, Any
from dataclasses import dataclass, field

from .utils_text import normalize, tokenize


@dataclass
class GRISignals:
    """GRI 신호 탐지 결과"""
    gri1_note_like: bool = False
    gri2a_incomplete: bool = False
    gri2b_mixtures: bool = False
    gri3_multi_candidate: bool = False
    gri5_containers: bool = False

    # 상세 정보
    matched_keywords: Dict[str, List[str]] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'gri1_note_like': self.gri1_note_like,
            'gri2a_incomplete': self.gri2a_incomplete,
            'gri2b_mixtures': self.gri2b_mixtures,
            'gri3_multi_candidate': self.gri3_multi_candidate,
            'gri5_containers': self.gri5_containers,
            'matched_keywords': self.matched_keywords,
            'confidence': self.confidence,
        }

    def any_signal(self) -> bool:
        """어떤 GRI 신호라도 감지되었는지"""
        return any([
            self.gri1_note_like,
            self.gri2a_incomplete,
            self.gri2b_mixtures,
            self.gri3_multi_candidate,
            self.gri5_containers
        ])

    def active_signals(self) -> List[str]:
        """활성화된 신호 리스트"""
        signals = []
        if self.gri1_note_like:
            signals.append('gri1')
        if self.gri2a_incomplete:
            signals.append('gri2a')
        if self.gri2b_mixtures:
            signals.append('gri2b')
        if self.gri3_multi_candidate:
            signals.append('gri3')
        if self.gri5_containers:
            signals.append('gri5')
        return signals


# GRI 신호별 키워드/패턴 정의
GRI_PATTERNS = {
    # GRI 1: 주(Note) 관련 - 분류에 법적 효력
    'gri1_note_like': {
        'keywords_ko': [
            '주', '류주', '장주', '호주', '소호주',
            '제1류', '제2류', '제3류', '제4류', '제5류',
            '제6류', '제7류', '제8류', '제9류', '제10류',
            '해설서', '품목분류', '통칙', '관세율표',
            '부의주', '류의주', '장의주',
        ],
        'keywords_en': [
            'note', 'chapter note', 'section note', 'heading note',
            'subheading note', 'legal note', 'explanatory note',
        ],
        'patterns': [
            r'제\s*\d+\s*류',
            r'제\s*\d+\s*장',
            r'\d+류\s*주',
            r'chapter\s*\d+',
            r'section\s*[ivx]+',
        ],
    },

    # GRI 2(a): 미완성품, 미조립품
    'gri2a_incomplete': {
        'keywords_ko': [
            '미조립', '조립용', '분해', '반제품', '미완성',
            '조립식', '조립품', '녹다운', '분리', '해체',
            '미가공', '반가공', '부분품', '구성품',
            '조립전', '조립 전', '분해상태', '분해 상태',
        ],
        'keywords_en': [
            'ckd', 'skd', 'knockdown', 'knock-down', 'knock down',
            'unassembled', 'disassembled', 'incomplete', 'unfinished',
            'semi-finished', 'semifinished', 'partially',
            'kit', 'kits', 'assembly', 'to be assembled',
        ],
        'patterns': [
            r'c\.?k\.?d',
            r's\.?k\.?d',
            r'미\s*조립',
            r'반\s*제품',
        ],
    },

    # GRI 2(b): 혼합물, 복합재료
    'gri2b_mixtures': {
        'keywords_ko': [
            '혼합', '블렌드', '함유', '합금', '코팅',
            '복합', '적층', '충전', '첨가', '배합',
            '성분', '재질', '소재', '원료',
            '피복', '도금', '접합', '라미네이트',
            '함량', '비율', '농도',
        ],
        'keywords_en': [
            'mixture', 'mixed', 'blend', 'blended', 'alloy',
            'coated', 'coating', 'composite', 'laminated', 'laminate',
            'filled', 'combined', 'compound', 'compounded',
            'containing', 'consists', 'made of', 'composed',
            'plated', 'clad', 'covered',
        ],
        'patterns': [
            r'\d+\s*%',
            r'\d+\s*퍼센트',
            r'함량\s*\d+',
            r'비율\s*\d+',
        ],
    },

    # GRI 3: 세트, 복수 후보
    'gri3_multi_candidate': {
        'keywords_ko': [
            '세트', '본질적', '혼용', '복수', '겸용',
            '다용도', '다기능', '복합기', '일체형',
            '구성품', '조합', '묶음', '패키지',
            '키트', '콤보', '셋트',
        ],
        'keywords_en': [
            'set', 'sets', 'essential', 'essential character',
            'multi-purpose', 'multipurpose', 'multi-function',
            'combo', 'combination', 'package', 'kit',
            'put up', 'retail sale', 'together',
        ],
        'patterns': [
            r'세트\s*구성',
            r'\d+\s*종\s*세트',
            r'본질적\s*특성',
        ],
    },

    # GRI 5: 포장용기, 케이스
    'gri5_containers': {
        'keywords_ko': [
            '케이스', '보관함', '전용케이스', '전용 케이스',
            '포장용기', '하드케이스', '소프트케이스',
            '가방', '파우치', '박스', '상자',
            '용기', '컨테이너', '보관',
        ],
        'keywords_en': [
            'case', 'cases', 'container', 'containers',
            'box', 'boxes', 'pouch', 'bag', 'bags',
            'packaging', 'packing', 'storage',
            'specially shaped', 'fitted',
        ],
        'patterns': [
            r'전용\s*케이스',
            r'보관\s*용',
            r'케이스\s*포함',
        ],
    },
}

# 부품 관련 키워드 (GRI 2a 관련 추가 판단용)
PARTS_KEYWORDS = {
    'keywords_ko': [
        '부품', '부속품', '부속', '액세서리', '교체품',
        '소모품', '예비품', '예비부품', '스페어',
    ],
    'keywords_en': [
        'part', 'parts', 'accessory', 'accessories',
        'component', 'components', 'spare', 'spares',
        'replacement', 'consumable',
    ],
}


def _match_keywords(text: str, keywords: List[str]) -> List[str]:
    """키워드 매칭"""
    norm_text = normalize(text)
    matched = []
    for kw in keywords:
        norm_kw = normalize(kw)
        if norm_kw and len(norm_kw) >= 2 and norm_kw in norm_text:
            matched.append(kw)
    return matched


def _match_patterns(text: str, patterns: List[str]) -> List[str]:
    """정규식 패턴 매칭"""
    norm_text = normalize(text)
    matched = []
    for pattern in patterns:
        if re.search(pattern, norm_text, re.IGNORECASE):
            matched.append(pattern)
    return matched


def detect_gri_signals(text: str) -> GRISignals:
    """
    입력 텍스트에서 GRI 신호 탐지

    Args:
        text: 입력 품명/설명

    Returns:
        GRISignals 객체
    """
    if not text:
        return GRISignals()

    signals = GRISignals()

    for signal_name, config in GRI_PATTERNS.items():
        matched_all = []

        # 한국어 키워드 매칭
        matched_ko = _match_keywords(text, config.get('keywords_ko', []))
        matched_all.extend(matched_ko)

        # 영어 키워드 매칭
        matched_en = _match_keywords(text, config.get('keywords_en', []))
        matched_all.extend(matched_en)

        # 패턴 매칭
        matched_patterns = _match_patterns(text, config.get('patterns', []))
        matched_all.extend(matched_patterns)

        # 결과 저장
        if matched_all:
            setattr(signals, signal_name, True)
            signals.matched_keywords[signal_name] = matched_all
            # confidence: 매칭 수 기반 (최대 1.0)
            signals.confidence[signal_name] = min(1.0, len(matched_all) * 0.3)
        else:
            signals.confidence[signal_name] = 0.0

    return signals


def detect_parts_signal(text: str) -> Dict[str, Any]:
    """
    부품 관련 신호 탐지 (GRI 2a 보조)

    Returns:
        {'is_parts': bool, 'matched': list, 'confidence': float}
    """
    if not text:
        return {'is_parts': False, 'matched': [], 'confidence': 0.0}

    matched = []
    matched.extend(_match_keywords(text, PARTS_KEYWORDS['keywords_ko']))
    matched.extend(_match_keywords(text, PARTS_KEYWORDS['keywords_en']))

    is_parts = len(matched) > 0
    confidence = min(1.0, len(matched) * 0.4) if matched else 0.0

    return {
        'is_parts': is_parts,
        'matched': matched,
        'confidence': confidence,
    }


def get_gri_questions(signals: GRISignals) -> List[str]:
    """
    GRI 신호 기반 질문 생성

    Args:
        signals: GRISignals 객체

    Returns:
        우선순위 정렬된 질문 리스트
    """
    questions = []

    # GRI 2(a): 미조립/미완성
    if signals.gri2a_incomplete:
        questions.append("미조립/분해 상태로 수입됩니까? 완제품으로 조립이 가능한 상태입니까?")
        questions.append("완제품입니까, 부품입니까? 어떤 기계/제품의 부품입니까?")

    # GRI 2(b): 혼합물
    if signals.gri2b_mixtures:
        questions.append("주요 성분/재질의 비율은 어떻게 됩니까? (예: 면 60%, 폴리에스터 40%)")
        questions.append("본질적 특성을 결정하는 주된 성분/재질은 무엇입니까?")

    # GRI 3: 세트
    if signals.gri3_multi_candidate:
        questions.append("세트로 판매됩니까? 구성품을 각각 개별 판매합니까?")
        questions.append("세트의 본질적 특성을 부여하는 주된 물품은 무엇입니까?")

    # GRI 5: 케이스/포장
    if signals.gri5_containers:
        questions.append("케이스/용기가 전용품이며 함께 제시/판매됩니까?")
        questions.append("케이스/용기가 단독으로 별도 판매되는 것입니까?")

    # GRI 1: 주/해설서 언급 시
    if signals.gri1_note_like:
        questions.append("해당 류/호의 주(Note)에서 특별히 규정하는 사항이 있습니까?")

    # 부품 의심 (gri2a와 연계)
    parts_signal = detect_parts_signal('')  # 이미 탐지된 경우 재사용

    return questions


def analyze_text_for_gri(text: str) -> Dict[str, Any]:
    """
    텍스트 전체 GRI 분석 (진단용)

    Returns:
        전체 분석 결과 딕셔너리
    """
    signals = detect_gri_signals(text)
    parts = detect_parts_signal(text)
    questions = get_gri_questions(signals)

    return {
        'text': text,
        'signals': signals.to_dict(),
        'parts_signal': parts,
        'suggested_questions': questions,
        'active_gri': signals.active_signals(),
        'any_signal': signals.any_signal(),
    }


# 테스트
if __name__ == "__main__":
    test_cases = [
        "자동차 CKD 부품 세트",
        "면 60% 폴리에스터 40% 혼방 직물",
        "스마트폰 전용 케이스",
        "미조립 가구 키트",
        "세트 구성: 칼, 포크, 숟가락",
        "냉동 돼지 삼겹살",
        "LED TV 55인치",
    ]

    for text in test_cases:
        print(f"\n입력: {text}")
        result = analyze_text_for_gri(text)
        print(f"  활성 GRI: {result['active_gri']}")
        print(f"  부품 신호: {result['parts_signal']['is_parts']}")
        if result['suggested_questions']:
            print(f"  제안 질문:")
            for q in result['suggested_questions'][:2]:
                print(f"    - {q}")
