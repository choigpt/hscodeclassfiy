"""
GRI (General Rules of Interpretation) signal detection.
Detects GRI 1/2a/2b/3/5 signals from input text.
"""

import re
from typing import Dict, List, Any
from dataclasses import dataclass, field

from ..text import normalize


@dataclass
class GRISignals:
    """GRI signal detection result."""
    gri1_note_like: bool = False
    gri2a_incomplete: bool = False
    gri2b_mixtures: bool = False
    gri3_multi_candidate: bool = False
    gri5_containers: bool = False
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
        return any([
            self.gri1_note_like, self.gri2a_incomplete,
            self.gri2b_mixtures, self.gri3_multi_candidate,
            self.gri5_containers,
        ])

    def active_signals(self) -> List[str]:
        signals = []
        if self.gri1_note_like: signals.append('gri1')
        if self.gri2a_incomplete: signals.append('gri2a')
        if self.gri2b_mixtures: signals.append('gri2b')
        if self.gri3_multi_candidate: signals.append('gri3')
        if self.gri5_containers: signals.append('gri5')
        return signals


GRI_PATTERNS = {
    'gri1_note_like': {
        'keywords_ko': [
            '류주', '장주', '호주', '소호주', '해설서', '품목분류', '통칙',
            '부의주', '류의주',
        ],
        'keywords_en': [
            'note', 'chapter note', 'section note', 'heading note',
            'subheading note', 'explanatory note',
        ],
        'patterns': [r'제\s*\d+\s*류', r'제\s*\d+\s*장', r'\d+류\s*주'],
    },
    'gri2a_incomplete': {
        'keywords_ko': [
            '미조립', '조립용', '분해', '반제품', '미완성', '조립식',
            '녹다운', '미가공', '반가공', '구성품',
        ],
        'keywords_en': [
            'ckd', 'skd', 'knockdown', 'unassembled', 'incomplete',
            'semi-finished', 'kit', 'kits', 'assembly',
        ],
        'patterns': [r'c\.?k\.?d', r's\.?k\.?d', r'미\s*조립'],
    },
    'gri2b_mixtures': {
        'keywords_ko': [
            '혼합', '블렌드', '함유', '합금', '코팅', '복합', '적층',
            '첨가', '배합', '성분', '재질', '소재', '피복', '도금',
            '함량', '비율',
        ],
        'keywords_en': [
            'mixture', 'mixed', 'blend', 'alloy', 'coated', 'composite',
            'laminated', 'filled', 'compound', 'containing', 'plated',
        ],
        'patterns': [r'\d+\s*%', r'\d+\s*퍼센트'],
    },
    'gri3_multi_candidate': {
        'keywords_ko': [
            '세트', '본질적', '겸용', '다용도', '다기능', '일체형',
            '조합', '묶음', '키트',
        ],
        'keywords_en': [
            'set', 'sets', 'essential character', 'multi-purpose',
            'combo', 'combination', 'package',
        ],
        'patterns': [r'세트\s*구성', r'\d+\s*종\s*세트'],
    },
    'gri5_containers': {
        'keywords_ko': [
            '케이스', '보관함', '전용케이스', '포장용기', '가방', '파우치',
        ],
        'keywords_en': [
            'case', 'cases', 'container', 'box', 'pouch', 'bag',
            'specially shaped', 'fitted',
        ],
        'patterns': [r'전용\s*케이스', r'보관\s*용'],
    },
}

PARTS_KEYWORDS_KO = [
    '부품', '부속품', '부속', '액세서리', '교체품', '소모품', '예비품',
]
PARTS_KEYWORDS_EN = [
    'part', 'parts', 'accessory', 'accessories', 'component', 'spare',
]


def _match_keywords(text: str, keywords: List[str]) -> List[str]:
    norm_text = normalize(text)
    return [kw for kw in keywords if normalize(kw) in norm_text and len(normalize(kw)) >= 2]


def _match_patterns(text: str, patterns: List[str]) -> List[str]:
    norm_text = normalize(text)
    return [p for p in patterns if re.search(p, norm_text, re.IGNORECASE)]


def detect_gri_signals(text: str) -> GRISignals:
    """Detect GRI signals from text."""
    if not text:
        return GRISignals()

    signals = GRISignals()
    for signal_name, config in GRI_PATTERNS.items():
        matched = []
        matched.extend(_match_keywords(text, config.get('keywords_ko', [])))
        matched.extend(_match_keywords(text, config.get('keywords_en', [])))
        matched.extend(_match_patterns(text, config.get('patterns', [])))

        if matched:
            setattr(signals, signal_name, True)
            signals.matched_keywords[signal_name] = matched
            signals.confidence[signal_name] = min(1.0, len(matched) * 0.3)
        else:
            signals.confidence[signal_name] = 0.0

    return signals


def detect_parts_signal(text: str) -> Dict[str, Any]:
    """Detect parts-related signal."""
    if not text:
        return {'is_parts': False, 'matched': [], 'confidence': 0.0}

    matched = _match_keywords(text, PARTS_KEYWORDS_KO) + _match_keywords(text, PARTS_KEYWORDS_EN)
    return {
        'is_parts': len(matched) > 0,
        'matched': matched,
        'confidence': min(1.0, len(matched) * 0.4) if matched else 0.0,
    }
