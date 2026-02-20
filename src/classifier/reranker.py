"""
HS Reranker - 카드/규칙 기반 후보 재정렬 + 전역 속성 매칭 + GRI 피처
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field

from .types import Candidate, Evidence
from .utils_text import normalize, simple_contains, extract_keywords, tokenize, fuzzy_match, token_overlap
from .gri_signals import GRISignals, detect_gri_signals, detect_parts_signal
from .attribute_extract import (
    GlobalAttributes, GlobalAttributes8Axis, AxisAttributes,
    extract_attributes, extract_attributes_8axis, get_attribute_keywords,
    convert_8axis_to_legacy, AXIS_IDS,
    OBJECT_NATURE_KEYWORDS, MATERIAL_KEYWORDS, PROCESSING_STATE_KEYWORDS,
    FUNCTION_USE_KEYWORDS, PHYSICAL_FORM_KEYWORDS, COMPLETENESS_KEYWORDS,
    LEGAL_SCOPE_KEYWORDS,
)


@dataclass
class CandidateFeatures:
    """후보별 피처 벡터 (8축 전역 속성 포함)"""
    hs4: str = ""

    # 기본 피처
    f_ml: float = 0.0
    f_lexical: float = 0.0
    f_card_hits: int = 0
    f_rule_inc_hits: int = 0
    f_rule_exc_hits: int = 0
    f_not_in_model: int = 0

    # 통칙 기반 피처 (입력 전역)
    f_gri2a_signal: int = 0
    f_gri2b_signal: int = 0
    f_gri3_signal: int = 0
    f_gri5_signal: int = 0

    # 후보별 피처
    f_specificity: float = 0.0
    f_exclude_conflict: int = 0
    f_is_parts_candidate: int = 0

    # 기존 전역 속성 매칭 피처 (호환성 유지)
    f_state_match: int = 0
    f_material_match: int = 0
    f_use_match: int = 0
    f_form_match: int = 0
    f_parts_mismatch: int = 0
    f_set_signal: int = 0
    f_incomplete_signal: int = 0
    f_quant_match_score: float = 0.0
    f_quant_hard_exclude: int = 0
    f_quant_missing_value: int = 0
    f_note_hard_exclude: int = 0
    f_note_support_sum: float = 0.0

    # 8축 매칭 피처 (NEW)
    f_object_match_score: float = 0.0      # 물체 본질 매칭
    f_material_match_score: float = 0.0    # 재질 매칭 (확장)
    f_processing_match_score: float = 0.0  # 가공상태 매칭
    f_function_match_score: float = 0.0    # 기능/용도 매칭
    f_form_match_score: float = 0.0        # 물리적 형태 매칭
    f_completeness_match_score: float = 0.0  # 완성도 매칭
    f_quant_rule_match_score: float = 0.0  # 정량규칙 매칭
    f_legal_scope_match_score: float = 0.0  # 법적범위 매칭

    # 충돌/불확실성 피처 (NEW)
    f_conflict_penalty: float = 0.0        # 후보간 속성 충돌
    f_uncertainty_penalty: float = 0.0     # 속성 추출 불확실성

    # LegalGate 피처 (학습 시 build_dataset_legal에서 추가)
    f_legal_heading_term: float = 0.0
    f_legal_include_support: float = 0.0
    f_legal_exclude_conflict: float = 0.0
    f_legal_redirect_penalty: float = 0.0

    # 최종 점수
    score_total: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'hs4': self.hs4,
            'f_ml': round(self.f_ml, 4),
            'f_lexical': round(self.f_lexical, 4),
            'f_card_hits': self.f_card_hits,
            'f_rule_inc_hits': self.f_rule_inc_hits,
            'f_rule_exc_hits': self.f_rule_exc_hits,
            'f_not_in_model': self.f_not_in_model,
            'f_gri2a_signal': self.f_gri2a_signal,
            'f_gri2b_signal': self.f_gri2b_signal,
            'f_gri3_signal': self.f_gri3_signal,
            'f_gri5_signal': self.f_gri5_signal,
            'f_specificity': round(self.f_specificity, 4),
            'f_exclude_conflict': self.f_exclude_conflict,
            'f_is_parts_candidate': self.f_is_parts_candidate,
            # 기존 전역 속성
            'f_state_match': self.f_state_match,
            'f_material_match': self.f_material_match,
            'f_use_match': self.f_use_match,
            'f_form_match': self.f_form_match,
            'f_parts_mismatch': self.f_parts_mismatch,
            'f_set_signal': self.f_set_signal,
            'f_incomplete_signal': self.f_incomplete_signal,
            'f_quant_match_score': round(self.f_quant_match_score, 4),
            'f_quant_hard_exclude': self.f_quant_hard_exclude,
            'f_quant_missing_value': self.f_quant_missing_value,
            'f_note_hard_exclude': self.f_note_hard_exclude,
            'f_note_support_sum': round(self.f_note_support_sum, 4),
            # 8축 피처 (NEW)
            'f_object_match_score': round(self.f_object_match_score, 4),
            'f_material_match_score': round(self.f_material_match_score, 4),
            'f_processing_match_score': round(self.f_processing_match_score, 4),
            'f_function_match_score': round(self.f_function_match_score, 4),
            'f_form_match_score': round(self.f_form_match_score, 4),
            'f_completeness_match_score': round(self.f_completeness_match_score, 4),
            'f_quant_rule_match_score': round(self.f_quant_rule_match_score, 4),
            'f_legal_scope_match_score': round(self.f_legal_scope_match_score, 4),
            'f_conflict_penalty': round(self.f_conflict_penalty, 4),
            'f_uncertainty_penalty': round(self.f_uncertainty_penalty, 4),
            # LegalGate 피처
            'f_legal_heading_term': round(self.f_legal_heading_term, 4),
            'f_legal_include_support': round(self.f_legal_include_support, 4),
            'f_legal_exclude_conflict': round(self.f_legal_exclude_conflict, 4),
            'f_legal_redirect_penalty': round(self.f_legal_redirect_penalty, 4),
            'score_total': round(self.score_total, 4),
        }

    def to_vector(self) -> List[float]:
        """LightGBM 학습용 벡터 - 8축 피처 포함"""
        return [
            self.f_ml,
            self.f_lexical,
            float(self.f_card_hits),
            float(self.f_rule_inc_hits),
            float(self.f_rule_exc_hits),
            float(self.f_not_in_model),
            float(self.f_gri2a_signal),
            float(self.f_gri2b_signal),
            float(self.f_gri3_signal),
            float(self.f_gri5_signal),
            self.f_specificity,
            float(self.f_exclude_conflict),
            float(self.f_is_parts_candidate),
            # 기존 전역 속성
            float(self.f_state_match),
            float(self.f_material_match),
            float(self.f_use_match),
            float(self.f_form_match),
            float(self.f_parts_mismatch),
            float(self.f_set_signal),
            float(self.f_incomplete_signal),
            self.f_quant_match_score,
            float(self.f_quant_hard_exclude),
            float(self.f_quant_missing_value),
            float(self.f_note_hard_exclude),
            self.f_note_support_sum,
            # 8축 피처 (NEW)
            self.f_object_match_score,
            self.f_material_match_score,
            self.f_processing_match_score,
            self.f_function_match_score,
            self.f_form_match_score,
            self.f_completeness_match_score,
            self.f_quant_rule_match_score,
            self.f_legal_scope_match_score,
            self.f_conflict_penalty,
            self.f_uncertainty_penalty,
            # LegalGate 피처
            self.f_legal_heading_term,
            self.f_legal_include_support,
            self.f_legal_exclude_conflict,
            self.f_legal_redirect_penalty,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        return [
            'f_ml', 'f_lexical', 'f_card_hits', 'f_rule_inc_hits', 'f_rule_exc_hits',
            'f_not_in_model', 'f_gri2a_signal', 'f_gri2b_signal', 'f_gri3_signal',
            'f_gri5_signal', 'f_specificity', 'f_exclude_conflict', 'f_is_parts_candidate',
            'f_state_match', 'f_material_match', 'f_use_match', 'f_form_match',
            'f_parts_mismatch', 'f_set_signal', 'f_incomplete_signal',
            'f_quant_match_score', 'f_quant_hard_exclude', 'f_quant_missing_value',
            'f_note_hard_exclude', 'f_note_support_sum',
            # 8축 피처
            'f_object_match_score', 'f_material_match_score', 'f_processing_match_score',
            'f_function_match_score', 'f_form_match_score', 'f_completeness_match_score',
            'f_quant_rule_match_score', 'f_legal_scope_match_score',
            'f_conflict_penalty', 'f_uncertainty_penalty',
            # LegalGate 피처
            'f_legal_heading_term', 'f_legal_include_support',
            'f_legal_exclude_conflict', 'f_legal_redirect_penalty',
        ]


# 부품 관련 키워드
PARTS_KEYWORDS = {'부품', '부속품', '부속', '액세서리', 'parts', 'part', 'accessory', 'accessories', 'component'}


class HSReranker:
    """
    HS4 카드 및 규칙 청크 기반 Reranker + KB Retrieval + 전역 속성 매칭
    """

    def __init__(
        self,
        cards_path: str = "kb/structured/hs4_cards.jsonl",
        rules_path: str = "kb/structured/hs4_rule_chunks.jsonl",
        thesaurus_path: str = "kb/structured/thesaurus_terms.jsonl",
        weight_ml: float = 1.0,
        weight_card: float = 0.3,
        weight_rule: float = 0.5,
        weight_specificity: float = 0.3,
        weight_attr_match: float = 0.2,
        exclude_penalty: float = -2.0,
        hard_exclude_penalty: float = -10.0,
        parts_penalty: float = -0.5,
    ):
        self.cards_path = cards_path
        self.rules_path = rules_path
        self.thesaurus_path = thesaurus_path

        # 점수 가중치
        self.weight_ml = weight_ml
        self.weight_card = weight_card
        self.weight_rule = weight_rule
        self.weight_specificity = weight_specificity
        self.weight_attr_match = weight_attr_match
        self.exclude_penalty = exclude_penalty
        self.hard_exclude_penalty = hard_exclude_penalty
        self.parts_penalty = parts_penalty

        # 데이터
        self.cards: Dict[str, Dict] = {}  # hs4 -> card
        self.rules: Dict[str, List[Dict]] = defaultdict(list)  # hs4 -> [rules]
        self.thesaurus: Dict[str, List[str]] = defaultdict(list)  # term -> [hs4]

        # 카드 속성 인덱스 (기존 호환)
        self.card_attrs: Dict[str, Dict[str, Set[str]]] = {}  # hs4 -> {axis: keywords}

        # 8축 카드 속성 인덱스 (NEW)
        self.card_attrs_8axis: Dict[str, Dict[str, Set[str]]] = {}  # hs4 -> {axis_id: values}

        # 8축 가중치 (NEW)
        self.weight_object = 0.15
        self.weight_material = 0.20
        self.weight_processing = 0.15
        self.weight_function = 0.15
        self.weight_form = 0.10
        self.weight_completeness = 0.15
        self.weight_quant = 0.20
        self.weight_legal = 0.10

        # IDF 계산용 (specificity)
        self.keyword_doc_freq: Dict[str, int] = defaultdict(int)
        self.total_cards: int = 0

        self._load_cards()
        self._load_rules()
        self._load_thesaurus()
        self._compute_idf()
        self._build_card_attr_index()
        self._build_card_attr_index_8axis()

    def _load_cards(self):
        """HS4 카드 로드"""
        cards_file = Path(self.cards_path)
        if not cards_file.exists():
            print(f"[Reranker] 경고: 카드 파일 없음: {cards_file}")
            return

        count = 0
        with open(cards_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    card = json.loads(line)
                    hs4 = card.get('hs4', '')
                    if hs4:
                        # 키워드 정규화
                        keywords = set()

                        # includes
                        for kw in card.get('includes', []):
                            if kw and len(kw) >= 2:
                                keywords.add(normalize(kw))

                        # scope
                        for kw in card.get('scope', []):
                            if kw and len(kw) >= 2:
                                keywords.add(normalize(kw))

                        # title에서 토큰화
                        title = card.get('title_ko', '')
                        title_tokens = extract_keywords(title, min_len=2, max_count=10)
                        keywords.update(title_tokens)

                        self.cards[hs4] = {
                            'title': title,
                            'keywords': list(keywords),
                            'excludes': card.get('excludes', []),
                            'includes': card.get('includes', []),
                            'decision_attributes': card.get('decision_attributes', {}),
                            'raw': card
                        }
                        count += 1
                except json.JSONDecodeError:
                    continue

        self.total_cards = count
        print(f"[Reranker] 카드 로드: {count}개")

    def _infer_rule_strength(self, text: str, chunk_type: str) -> str:
        """규칙 텍스트에서 hard/soft strength 추론

        Hard exclude 패턴:
        - "제외한다", "제외된다"
        - "해당하지 아니한다", "해당하지 않는다"
        - "분류하지 아니한다", "분류하지 않는다"
        - "포함하지 아니한다", "포함하지 않는다"
        - "적용하지 아니한다"
        - "이 류에서...제외"
        - "이 호에서...제외"
        """
        if chunk_type != 'exclude_rule':
            return 'soft'

        # Hard exclude 패턴
        hard_patterns = [
            '제외한다', '제외된다',
            '해당하지 아니한다', '해당하지 않는다', '해당되지 아니한다', '해당되지 않는다',
            '분류하지 아니한다', '분류하지 않는다', '분류되지 아니한다',
            '포함하지 아니한다', '포함하지 않는다', '포함되지 아니한다',
            '적용하지 아니한다', '적용하지 않는다',
            '이 류에서', '이 호에서', '이 절에서',
            '다음 각 목의 것은 제외',
            '다음의 것은 제외',
        ]

        text_lower = text.lower()
        for pattern in hard_patterns:
            if pattern in text_lower:
                return 'hard'

        return 'soft'

    def _load_rules(self):
        """규칙 청크 로드 (v2 파일 우선)"""
        # v2 파일 우선 로드
        v2_path = Path("kb/structured/hs4_rule_chunks_v2.jsonl")
        rules_file = v2_path if v2_path.exists() else Path(self.rules_path)
        if not rules_file.exists():
            print(f"[Reranker] 경고: 규칙 파일 없음: {rules_file}")
            return

        count = 0
        hard_count = 0
        with open(rules_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rule = json.loads(line)
                    hs4 = rule.get('hs4', '')
                    chunk_type = rule.get('chunk_type', 'general')
                    text = rule.get('text', '')

                    if not hs4:
                        continue

                    # signals 정규화
                    signals = rule.get('signals', [])
                    if not signals:
                        # text에서 키워드 추출 (fallback)
                        signals = extract_keywords(text, min_len=2, max_count=5)

                    signals = [normalize(s) for s in signals if s and len(s) >= 2]

                    # polarity 결정: chunk_type 기반
                    if 'exclude' in chunk_type:
                        polarity = 'exclude'
                    elif 'include' in chunk_type:
                        polarity = 'include'
                    else:
                        polarity = rule.get('polarity', 'neutral')

                    # strength 결정: KB 필드 없으면 텍스트에서 추론
                    strength = rule.get('strength')
                    if not strength:
                        strength = self._infer_rule_strength(text, chunk_type)

                    if strength == 'hard':
                        hard_count += 1

                    # 규칙 메타데이터 확장
                    self.rules[hs4].append({
                        'chunk_id': rule.get('rule_id', f"{hs4}_{idx}"),
                        'rule_id': rule.get('rule_id', f"{hs4}_{idx}"),
                        'chunk_type': chunk_type,
                        'signals': signals,
                        'text': text[:300],
                        'target_hs4': rule.get('target_hs4'),
                        'polarity': polarity,
                        'strength': strength,
                        'quant_rule': rule.get('quant_rule'),
                        'source': rule.get('source', ''),
                        'hs_version': rule.get('hs_version', '2022'),
                    })
                    count += 1
                except json.JSONDecodeError:
                    continue

        print(f"[Reranker] 규칙 로드: {count}개, {len(self.rules)}개 HS4 (hard exclude: {hard_count}개)")

    def _load_thesaurus(self):
        """용어 사전 로드"""
        thesaurus_file = Path(self.thesaurus_path)
        if not thesaurus_file.exists():
            print(f"[Reranker] 경고: 용어 사전 없음: {thesaurus_file}")
            return

        count = 0
        with open(thesaurus_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    term = normalize(item.get('term', ''))

                    hs4_list = item.get('hs4_candidates', [])
                    if not hs4_list:
                        hs4 = item.get('hs4', '')
                        if hs4:
                            hs4_list = [hs4]

                    if not term or len(term) < 2 or not hs4_list:
                        continue

                    for hs4 in hs4_list:
                        if hs4:
                            self.thesaurus[term].append(hs4)
                            count += 1

                    for alias in item.get('aliases', []):
                        norm_alias = normalize(alias)
                        if norm_alias and len(norm_alias) >= 2:
                            for hs4 in hs4_list:
                                if hs4:
                                    self.thesaurus[norm_alias].append(hs4)

                except json.JSONDecodeError:
                    continue

        print(f"[Reranker] 용어 사전 로드: {count}개 매핑, {len(self.thesaurus)}개 용어")

    def _compute_idf(self):
        """IDF 계산 (specificity용)"""
        for hs4, card in self.cards.items():
            for kw in card.get('keywords', []):
                self.keyword_doc_freq[kw] += 1

    def _build_card_attr_index(self):
        """카드 속성 인덱스 구축 (기존 전역 속성 매칭용 - 호환성 유지)"""
        for hs4, card in self.cards.items():
            keywords = card.get('keywords', [])
            title = card.get('title', '')
            full_text = ' '.join(keywords) + ' ' + title

            # 각 축에 대해 매칭되는 키워드 추출
            attrs = {}
            for axis, kw_dict in [
                ('state', PROCESSING_STATE_KEYWORDS),
                ('material', MATERIAL_KEYWORDS),
                ('use', FUNCTION_USE_KEYWORDS),
                ('form', PHYSICAL_FORM_KEYWORDS),
            ]:
                matched = set()
                for category, kw_list in kw_dict.items():
                    for kw in kw_list:
                        if kw.lower() in full_text.lower():
                            matched.add(category)
                attrs[axis] = matched

            # 부품 여부
            is_parts = any(pk in full_text.lower() for pk in PARTS_KEYWORDS)
            attrs['is_parts'] = is_parts

            self.card_attrs[hs4] = attrs

    def _build_card_attr_index_8axis(self):
        """8축 카드 속성 인덱스 구축"""
        axis_keyword_map = {
            'object_nature': OBJECT_NATURE_KEYWORDS,
            'material': MATERIAL_KEYWORDS,
            'processing_state': PROCESSING_STATE_KEYWORDS,
            'function_use': FUNCTION_USE_KEYWORDS,
            'physical_form': PHYSICAL_FORM_KEYWORDS,
            'completeness': COMPLETENESS_KEYWORDS,
            'legal_scope': LEGAL_SCOPE_KEYWORDS,
        }

        for hs4, card in self.cards.items():
            keywords = card.get('keywords', [])
            title = card.get('title', '')
            includes = card.get('includes', [])
            full_text = ' '.join(keywords + includes) + ' ' + title
            full_text_lower = full_text.lower()

            # 8축 각각에 대해 매칭되는 카테고리 추출
            attrs_8axis = {}
            for axis_id, kw_dict in axis_keyword_map.items():
                matched = set()
                for category, kw_list in kw_dict.items():
                    for kw in kw_list:
                        if kw.lower() in full_text_lower:
                            matched.add(category)
                            break  # 한 카테고리에 하나만 매칭되면 충분
                attrs_8axis[axis_id] = matched

            self.card_attrs_8axis[hs4] = attrs_8axis

    def compute_axis_match_score(
        self,
        input_attrs: GlobalAttributes8Axis,
        hs4: str,
        axis_id: str
    ) -> float:
        """
        8축 단일 축 매칭 점수 계산

        Args:
            input_attrs: 입력 8축 속성
            hs4: 후보 HS4
            axis_id: 축 ID

        Returns:
            매칭 점수 (0.0 ~ 1.0)
        """
        if hs4 not in self.card_attrs_8axis:
            return 0.0

        card_axis_values = self.card_attrs_8axis[hs4].get(axis_id, set())
        if not card_axis_values:
            return 0.0

        input_axis = input_attrs.get_axis(axis_id)
        if not input_axis.values:
            return 0.0

        # 교집합 계산
        input_values = set(input_axis.values)
        overlap = input_values & card_axis_values

        if not overlap:
            return 0.0

        # Jaccard-like score
        union = input_values | card_axis_values
        score = len(overlap) / len(union) if union else 0.0

        # 신뢰도 가중
        score *= input_axis.confidence

        return min(1.0, score)

    def compute_8axis_match_scores(
        self,
        input_attrs: GlobalAttributes8Axis,
        hs4: str
    ) -> Dict[str, float]:
        """
        8축 전체 매칭 점수 계산

        Args:
            input_attrs: 입력 8축 속성
            hs4: 후보 HS4

        Returns:
            {axis_id: score} dict
        """
        scores = {}
        for axis_id in ['object_nature', 'material', 'processing_state',
                        'function_use', 'physical_form', 'completeness', 'legal_scope']:
            scores[axis_id] = self.compute_axis_match_score(input_attrs, hs4, axis_id)

        # 정량 규칙 매칭 (별도 처리)
        quant_score = 0.0
        if input_attrs.quantitative_rules and hs4 in self.rules:
            for rule in self.rules[hs4]:
                if rule.get('quant_rule'):
                    quant_score = 0.5  # 정량 규칙이 있는 HS4에 기본 점수
                    for qf in input_attrs.quantitative_rules:
                        if qf.property == rule.get('quant_rule', {}).get('property'):
                            quant_score = 1.0
                            break
                    break
        scores['quantitative_rules'] = quant_score

        return scores

    def compute_conflict_penalty(
        self,
        input_attrs: GlobalAttributes8Axis,
        hs4: str
    ) -> float:
        """
        속성 충돌 패널티 계산

        입력 속성과 후보 속성 간 명시적 불일치 탐지

        Returns:
            0.0 (충돌 없음) ~ 1.0 (심각한 충돌)
        """
        if hs4 not in self.card_attrs_8axis:
            return 0.0

        penalty = 0.0
        card_attrs = self.card_attrs_8axis[hs4]

        # 완성도 충돌: 입력이 완제품인데 후보가 부품
        input_completeness = set(input_attrs.completeness.values)
        card_completeness = card_attrs.get('completeness', set())

        if 'complete' in input_completeness and 'parts' in card_completeness:
            penalty += 0.3
        if 'parts' in input_completeness and 'complete' in card_completeness:
            penalty += 0.2

        # 가공상태 충돌: 신선 vs 가공
        input_processing = set(input_attrs.processing_state.values)
        card_processing = card_attrs.get('processing_state', set())

        fresh_states = {'fresh', 'raw'}
        processed_states = {'cooked', 'processed', 'refined', 'assembled'}

        if (input_processing & fresh_states) and (card_processing & processed_states):
            penalty += 0.2
        if (input_processing & processed_states) and (card_processing & fresh_states):
            penalty += 0.2

        return min(1.0, penalty)

    def compute_uncertainty_penalty(
        self,
        input_attrs: GlobalAttributes8Axis
    ) -> float:
        """
        속성 추출 불확실성 패널티 계산

        주요 축의 신뢰도가 낮거나 추출된 값이 없으면 패널티

        Returns:
            0.0 (확실) ~ 1.0 (매우 불확실)
        """
        penalty = 0.0
        key_axes = ['material', 'processing_state', 'function_use', 'completeness']

        missing_count = 0
        low_confidence_count = 0

        for axis_id in key_axes:
            axis = input_attrs.get_axis(axis_id)
            if not axis.values:
                missing_count += 1
            elif axis.confidence < 0.5:
                low_confidence_count += 1

        # 주요 축 3개 이상 누락시 높은 패널티
        if missing_count >= 3:
            penalty += 0.4
        elif missing_count >= 2:
            penalty += 0.2

        # 낮은 신뢰도 축 패널티
        penalty += low_confidence_count * 0.1

        return min(1.0, penalty)

    def get_kb_hs4_set(self) -> Set[str]:
        """KB가 알고 있는 HS4 집합 반환"""
        cards_hs4 = set(self.cards.keys())
        rules_hs4 = set(self.rules.keys())
        return cards_hs4 | rules_hs4

    def compute_specificity(self, matched_keywords: List[str]) -> float:
        """IDF 기반 specificity 점수"""
        if not matched_keywords or self.total_cards == 0:
            return 0.0

        score = 0.0
        for kw in matched_keywords:
            df = self.keyword_doc_freq.get(kw, 1)
            idf = math.log(self.total_cards / df) if df > 0 else 0.0
            score += idf

        return score

    def is_parts_candidate(self, hs4: str) -> bool:
        """후보가 '부품' 성향인지 판단"""
        if hs4 in self.card_attrs:
            return self.card_attrs[hs4].get('is_parts', False)

        if hs4 not in self.cards:
            return False

        card = self.cards[hs4]
        title = card.get('title', '').lower()
        keywords = card.get('keywords', [])

        for pk in PARTS_KEYWORDS:
            if pk in title:
                return True
            for kw in keywords:
                if pk in kw:
                    return True

        return False

    def compute_attribute_match(
        self,
        input_attrs: GlobalAttributes,
        hs4: str
    ) -> Dict[str, int]:
        """입력 속성과 카드 속성 간 매칭 점수"""
        result = {
            'state': 0,
            'material': 0,
            'use': 0,
            'form': 0,
            'parts_mismatch': 0,
        }

        if hs4 not in self.card_attrs:
            return result

        card_attrs = self.card_attrs[hs4]

        # 각 축 매칭
        result['state'] = len(input_attrs.states & card_attrs.get('state', set()))
        result['material'] = len(input_attrs.materials & card_attrs.get('material', set()))
        result['use'] = len(input_attrs.uses_functions & card_attrs.get('use', set()))
        result['form'] = len(input_attrs.forms & card_attrs.get('form', set()))

        # 부품 불일치 (입력은 완제품인데 후보는 부품)
        if not input_attrs.is_parts and card_attrs.get('is_parts', False):
            result['parts_mismatch'] = 1

        return result

    def check_quant_rules(
        self,
        input_attrs: GlobalAttributes,
        hs4: str
    ) -> Tuple[float, int, int]:
        """정량 규칙 매칭

        Returns:
            (match_score, hard_exclude, missing_value)
        """
        if hs4 not in self.rules:
            return 0.0, 0, 0

        match_score = 0.0
        hard_exclude = 0
        missing_value = 0

        for rule in self.rules[hs4]:
            quant_rule = rule.get('quant_rule')
            if not quant_rule:
                continue

            # 정량 규칙이 있는데 입력에 정량 정보가 없으면
            if not input_attrs.has_quant:
                missing_value = 1
                continue

            # 정량 매칭 로직 (단순화)
            for qf in input_attrs.quant_facts:
                rule_prop = quant_rule.get('property', '')
                if qf.property and rule_prop and qf.property == rule_prop:
                    match_score += 1.0

                    # Hard exclude 체크
                    rule_op = quant_rule.get('op', '')
                    rule_val = quant_rule.get('value', 0)
                    if rule.get('polarity') == 'exclude' and rule.get('strength') == 'hard':
                        # 예: "50% 미만은 제외" 규칙에 해당하면
                        if self._check_quant_condition(qf.value, rule_op, rule_val):
                            hard_exclude = 1

        return match_score, hard_exclude, missing_value

    def _check_quant_condition(self, value: float, op: str, threshold: float) -> bool:
        """정량 조건 체크"""
        if op == '>=':
            return value >= threshold
        elif op == '<=':
            return value <= threshold
        elif op == '>':
            return value > threshold
        elif op == '<':
            return value < threshold
        elif op == '=':
            return abs(value - threshold) < 0.001
        return False

    def retrieve_from_kb(
        self,
        text: str,
        topk: int = 30,
        gri_signals: Optional[GRISignals] = None,
        input_attrs: Optional[GlobalAttributes] = None,
        input_attrs_8axis: Optional[GlobalAttributes8Axis] = None
    ) -> List[Candidate]:
        """KB 기반 후보 생성 (8축 속성 지원)"""
        norm_text = normalize(text)
        text_tokens = set(tokenize(text, remove_stopwords=True))

        # GRI/속성 기반 topk 조정
        actual_topk = topk
        if gri_signals:
            if gri_signals.gri2a_incomplete:
                actual_topk += 20
            if gri_signals.gri2b_mixtures:
                actual_topk += 10

        # 속성 기반 확장 (기존 호환)
        if input_attrs:
            if input_attrs.has_quant:
                actual_topk += 10
            if len(input_attrs.materials) > 1:
                actual_topk += 5

        # 8축 속성 기반 확장 (NEW)
        if input_attrs_8axis:
            if input_attrs_8axis.quantitative_rules:
                actual_topk += 10
            if len(input_attrs_8axis.material.values) > 1:
                actual_topk += 5
            if input_attrs_8axis.is_set():
                actual_topk += 5

        hs4_scores: Dict[str, float] = defaultdict(float)
        hs4_matched_kw: Dict[str, List[str]] = defaultdict(list)

        # 1. 카드 키워드 매칭
        for hs4, card in self.cards.items():
            keywords = card.get('keywords', [])
            hit_count = 0
            matched = []

            for kw in keywords:
                if kw and kw in norm_text:
                    hit_count += 1
                    matched.append(kw)

            keyword_tokens = set()
            for kw in keywords:
                keyword_tokens.update(tokenize(kw, remove_stopwords=True))

            overlap = text_tokens & keyword_tokens
            hit_count += len(overlap) * 0.5

            if hit_count > 0:
                hs4_scores[hs4] += hit_count
                hs4_matched_kw[hs4].extend(matched)

        # 2. 용어 사전 매칭
        for term, hs4_list in self.thesaurus.items():
            if term in norm_text or term in text_tokens:
                for hs4 in hs4_list:
                    hs4_scores[hs4] += 1.5

        # 3. 속성 기반 부스트 (기존 호환)
        if input_attrs:
            for hs4 in hs4_scores:
                attr_match = self.compute_attribute_match(input_attrs, hs4)
                total_match = sum(attr_match.values()) - attr_match['parts_mismatch']
                if total_match > 0:
                    hs4_scores[hs4] *= (1 + 0.1 * total_match)

        # 4. 8축 속성 기반 부스트 (NEW)
        if input_attrs_8axis:
            for hs4 in list(hs4_scores.keys()):
                axis_scores = self.compute_8axis_match_scores(input_attrs_8axis, hs4)
                total_axis_score = sum(axis_scores.values())
                if total_axis_score > 0:
                    hs4_scores[hs4] *= (1 + 0.15 * total_axis_score)

                # 충돌 패널티 적용
                conflict = self.compute_conflict_penalty(input_attrs_8axis, hs4)
                if conflict > 0:
                    hs4_scores[hs4] *= (1 - 0.3 * conflict)

            # 8축 기반 류별 부스트
            # object_nature=substance → 화학/원재료 계열 우선
            if 'substance' in input_attrs_8axis.object_nature.values:
                chemical_chapters = {'28', '29', '30', '31', '32', '33', '34', '35', '38'}
                for hs4 in hs4_scores:
                    if hs4[:2] in chemical_chapters:
                        hs4_scores[hs4] *= 1.15

            # processing_state=frozen → 식품 육류/수산 계열 가중
            if 'frozen' in input_attrs_8axis.processing_state.values:
                food_chapters = {'02', '03', '04', '07', '08', '16', '19', '20', '21'}
                for hs4 in hs4_scores:
                    if hs4[:2] in food_chapters:
                        hs4_scores[hs4] *= 1.1

            # completeness=parts → 부품 분류 HS 우선
            if input_attrs_8axis.is_parts():
                parts_chapters = {'84', '85', '87', '90'}
                for hs4 in hs4_scores:
                    if hs4[:2] in parts_chapters:
                        hs4_scores[hs4] *= 1.1

        # GRI 2(b) boost
        if gri_signals and gri_signals.gri2b_mixtures:
            for hs4, matched in hs4_matched_kw.items():
                for kw in matched:
                    if any(m in kw for m in ['재질', '성분', '소재', '합금', '혼합', 'alloy', 'mixture']):
                        hs4_scores[hs4] *= 1.2

        sorted_hs4 = sorted(hs4_scores.items(), key=lambda x: -x[1])[:actual_topk]

        candidates = []
        for hs4, score in sorted_hs4:
            candidates.append(Candidate(
                hs4=hs4,
                score_ml=0.0,
                evidence=[Evidence(
                    kind="kb_retrieval",
                    source_id=hs4,
                    text=f"KB 매칭 점수: {score:.2f}",
                    weight=score,
                    meta={'kb_score': score, 'matched_keywords': hs4_matched_kw.get(hs4, [])}
                )]
            ))

        return candidates

    def score_card(self, text: str, hs4: str) -> Tuple[float, List[Evidence], int, List[str]]:
        """카드 키워드 매칭 점수"""
        if hs4 not in self.cards:
            return 0.0, [], 0, []

        card = self.cards[hs4]
        keywords = card.get('keywords', [])
        norm_text = normalize(text)
        text_tokens = set(tokenize(text, remove_stopwords=True))

        matched = []
        for kw in keywords:
            if not kw or len(kw) < 2:
                continue

            if kw in norm_text:
                matched.append(kw)
                continue

            kw_tokens = set(tokenize(kw, remove_stopwords=True))
            if kw_tokens & text_tokens:
                matched.append(kw)

        if not matched:
            return 0.0, [], 0, []

        score = math.log(1 + len(matched))

        evidence = Evidence(
            kind="card_keyword",
            source_id=hs4,
            text=f"매칭 키워드: {', '.join(matched[:5])}",
            weight=score,
            meta={'keywords': matched, 'hit_count': len(matched)}
        )

        return score, [evidence], len(matched), matched

    def score_rule(self, text: str, hs4: str) -> Tuple[float, List[Evidence], int, int, bool, float]:
        """규칙 매칭 점수

        Returns:
            (score, evidences, inc_hits, exc_hits, has_exclude_conflict, note_support)
        """
        if hs4 not in self.rules:
            return 0.0, [], 0, 0, False, 0.0

        rules = self.rules[hs4]
        norm_text = normalize(text)
        text_tokens = set(tokenize(text, remove_stopwords=True))

        score = 0.0
        evidences = []
        inc_hits = 0
        exc_hits = 0
        has_exclude_conflict = False
        note_support = 0.0

        for rule in rules:
            chunk_type = rule['chunk_type']
            signals = rule['signals']
            chunk_id = rule['chunk_id']
            rule_text = rule['text']
            polarity = rule.get('polarity', 'neutral')
            strength = rule.get('strength', 'soft')

            matched_signals = []
            for s in signals:
                if not s or len(s) < 2:
                    continue

                if s in norm_text:
                    matched_signals.append(s)
                    continue

                s_tokens = set(tokenize(s, remove_stopwords=True))
                if s_tokens & text_tokens:
                    matched_signals.append(s)

            if not matched_signals:
                continue

            # 타입/극성별 점수
            if chunk_type == 'include_rule' or polarity == 'include':
                delta = 1.0 * len(matched_signals)
                kind = "include_rule"
                inc_hits += len(matched_signals)
                note_support += delta
            elif chunk_type == 'exclude_rule' or polarity == 'exclude':
                if strength == 'hard':
                    delta = self.hard_exclude_penalty * len(matched_signals)
                else:
                    delta = self.exclude_penalty * len(matched_signals)
                kind = "exclude_rule"
                exc_hits += len(matched_signals)
                has_exclude_conflict = True
            elif chunk_type == 'definition':
                delta = 0.5 * len(matched_signals)
                kind = "definition"
                note_support += delta * 0.5
            elif chunk_type == 'example':
                delta = 0.3 * len(matched_signals)
                kind = "example"
                note_support += delta * 0.3
            else:
                delta = 0.1 * len(matched_signals)
                kind = "general"

            score += delta

            evidences.append(Evidence(
                kind=kind,
                source_id=chunk_id,
                text=rule_text[:160],
                weight=delta,
                meta={
                    'matched_signals': matched_signals,
                    'chunk_type': chunk_type,
                    'strength': strength,
                    'rule_id': rule.get('rule_id', chunk_id),
                    'source': rule.get('source', ''),
                    'hs_version': rule.get('hs_version', '2022'),
                }
            ))

        return score, evidences, inc_hits, exc_hits, has_exclude_conflict, note_support

    def compute_features(
        self,
        text: str,
        candidate: Candidate,
        gri_signals: GRISignals,
        input_attrs: GlobalAttributes,
        model_classes: Optional[Set[str]] = None,
        input_attrs_8axis: Optional[GlobalAttributes8Axis] = None
    ) -> CandidateFeatures:
        """후보에 대한 전체 피처 벡터 계산 (8축 피처 포함)"""
        features = CandidateFeatures(hs4=candidate.hs4)

        # 기본 피처
        features.f_ml = candidate.score_ml

        # kb_score 정규화: raw score는 키워드 hit 누적값(0~50+)으로 스케일이 커서
        # LightGBM에서 lexical dominance를 유발함.
        # log1p 압축 + [0,1] clipping으로 "retrieval strength 힌트" 수준으로 축소.
        # log1p(30)≈3.43을 기준으로 정규화: score 0→0, 10→0.69, 20→0.88, 30→1.0
        for ev in candidate.evidence:
            if ev.kind == 'kb_retrieval':
                raw_kb = ev.meta.get('kb_score', 0.0)
                features.f_lexical = min(math.log1p(raw_kb) / math.log1p(30.0), 1.0)
                break

        # 카드/규칙 점수
        score_card, ev_card, card_hits, matched_kw = self.score_card(text, candidate.hs4)
        score_rule, ev_rule, inc_hits, exc_hits, has_exc, note_support = self.score_rule(text, candidate.hs4)

        features.f_card_hits = card_hits
        features.f_rule_inc_hits = inc_hits
        features.f_rule_exc_hits = exc_hits
        features.f_exclude_conflict = 1 if has_exc else 0
        features.f_note_support_sum = note_support

        # Specificity
        features.f_specificity = self.compute_specificity(matched_kw)

        # 모델 라벨 공간
        if model_classes is not None:
            features.f_not_in_model = 0 if candidate.hs4 in model_classes else 1

        # GRI 신호
        features.f_gri2a_signal = 1 if gri_signals.gri2a_incomplete else 0
        features.f_gri2b_signal = 1 if gri_signals.gri2b_mixtures else 0
        features.f_gri3_signal = 1 if gri_signals.gri3_multi_candidate else 0
        features.f_gri5_signal = 1 if gri_signals.gri5_containers else 0

        # 부품 후보 여부
        features.f_is_parts_candidate = 1 if self.is_parts_candidate(candidate.hs4) else 0

        # 기존 전역 속성 매칭 (호환성 유지)
        attr_match = self.compute_attribute_match(input_attrs, candidate.hs4)
        features.f_state_match = attr_match['state']
        features.f_material_match = attr_match['material']
        features.f_use_match = attr_match['use']
        features.f_form_match = attr_match['form']
        features.f_parts_mismatch = attr_match['parts_mismatch']

        features.f_set_signal = 1 if input_attrs.is_set else 0
        features.f_incomplete_signal = 1 if input_attrs.is_incomplete else 0

        # 정량 규칙 매칭
        quant_score, quant_hard, quant_missing = self.check_quant_rules(input_attrs, candidate.hs4)
        features.f_quant_match_score = quant_score
        features.f_quant_hard_exclude = quant_hard
        features.f_quant_missing_value = quant_missing

        # Hard exclude (note level) - exclude_conflict이면서 hard strength
        features.f_note_hard_exclude = 1 if (has_exc and any(
            r.get('strength') == 'hard' for r in self.rules.get(candidate.hs4, [])
            if r.get('polarity') == 'exclude'
        )) else 0

        # 8축 피처 계산 (NEW)
        if input_attrs_8axis is not None:
            axis_scores = self.compute_8axis_match_scores(input_attrs_8axis, candidate.hs4)

            features.f_object_match_score = axis_scores.get('object_nature', 0.0)
            features.f_material_match_score = axis_scores.get('material', 0.0)
            features.f_processing_match_score = axis_scores.get('processing_state', 0.0)
            features.f_function_match_score = axis_scores.get('function_use', 0.0)
            features.f_form_match_score = axis_scores.get('physical_form', 0.0)
            features.f_completeness_match_score = axis_scores.get('completeness', 0.0)
            features.f_quant_rule_match_score = axis_scores.get('quantitative_rules', 0.0)
            features.f_legal_scope_match_score = axis_scores.get('legal_scope', 0.0)

            # 충돌/불확실성 피처
            features.f_conflict_penalty = self.compute_conflict_penalty(
                input_attrs_8axis, candidate.hs4)
            features.f_uncertainty_penalty = self.compute_uncertainty_penalty(
                input_attrs_8axis)

        return features

    def rerank(
        self,
        text: str,
        candidates: List[Candidate],
        topk: int = 5,
        gri_signals: Optional[GRISignals] = None,
        input_attrs: Optional[GlobalAttributes] = None,
        input_attrs_8axis: Optional[GlobalAttributes8Axis] = None,
        model_classes: Optional[Set[str]] = None,
        ranker_model: Optional[Any] = None,
        legal_gate_debug: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Candidate], Dict[str, Any]]:
        """후보 재정렬 (8축 속성 지원)"""
        # GRI/속성 탐지
        if gri_signals is None:
            gri_signals = detect_gri_signals(text)

        if input_attrs is None:
            input_attrs = extract_attributes(text)

        # 8축 속성 추출 (없으면 자동 생성)
        if input_attrs_8axis is None:
            input_attrs_8axis = extract_attributes_8axis(text)

        stats = {
            'total_candidates': len(candidates),
            'card_hit_count': 0,
            'rule_hit_count': 0,
            'rule_inc_hit_count': 0,
            'rule_exc_hit_count': 0,
            'any_hit_count': 0,
            'exclude_conflict_count': 0,
            'hard_exclude_count': 0,
            'gri_signals': gri_signals.to_dict(),
            'input_attrs': input_attrs.to_dict(),
            'input_attrs_8axis': input_attrs_8axis.to_dict(),
        }

        all_features: List[CandidateFeatures] = []

        for cand in candidates:
            features = self.compute_features(
                text, cand, gri_signals, input_attrs, model_classes, input_attrs_8axis)

            score_card, ev_card, card_hits, matched_kw = self.score_card(text, cand.hs4)
            cand.score_card = score_card
            cand.evidence.extend(ev_card)

            score_rule, ev_rule, inc_hits, exc_hits, has_exc, note_support = self.score_rule(text, cand.hs4)
            cand.score_rule = score_rule
            cand.evidence.extend(ev_rule)

            # 통계
            if card_hits > 0:
                stats['card_hit_count'] += 1
            if inc_hits > 0 or exc_hits > 0:
                stats['rule_hit_count'] += 1
            stats['rule_inc_hit_count'] += inc_hits
            stats['rule_exc_hit_count'] += exc_hits
            if card_hits > 0 or inc_hits > 0:
                stats['any_hit_count'] += 1
            if has_exc:
                stats['exclude_conflict_count'] += 1
            if features.f_note_hard_exclude or features.f_quant_hard_exclude:
                stats['hard_exclude_count'] += 1

            # LegalGate 피처 주입 (학습 시 build_dataset_legal과 동일하게)
            if legal_gate_debug is not None:
                lg_results = legal_gate_debug.get('results', {})
                lg_result = lg_results.get(cand.hs4, {})
                features.f_legal_heading_term = lg_result.get('heading_term_score', 0.0)
                features.f_legal_include_support = lg_result.get('include_support_score', 0.0)
                features.f_legal_exclude_conflict = lg_result.get('exclude_conflict_score', 0.0)
                features.f_legal_redirect_penalty = lg_result.get('redirect_penalty', 0.0)

            # 점수 계산
            if ranker_model is not None:
                try:
                    feature_vec = [features.to_vector()]
                    cand.score_total = float(ranker_model.predict(feature_vec)[0])
                    features.score_total = cand.score_total
                except Exception:
                    cand.score_total = self._compute_weighted_score(
                        cand, features, gri_signals, input_attrs, input_attrs_8axis)
                    features.score_total = cand.score_total
            else:
                cand.score_total = self._compute_weighted_score(
                    cand, features, gri_signals, input_attrs, input_attrs_8axis)
                features.score_total = cand.score_total

            cand.features = features.to_dict()
            all_features.append(features)

        # 정렬
        candidates.sort(key=lambda c: c.score_total, reverse=True)

        # Hit rate 계산
        if stats['total_candidates'] > 0:
            stats['card_hit_rate'] = stats['card_hit_count'] / stats['total_candidates']
            stats['rule_hit_rate'] = stats['rule_hit_count'] / stats['total_candidates']
            stats['any_hit_rate'] = stats['any_hit_count'] / stats['total_candidates']
        else:
            stats['card_hit_rate'] = 0
            stats['rule_hit_rate'] = 0
            stats['any_hit_rate'] = 0

        stats['no_hits'] = stats['any_hit_count'] == 0
        stats['feature_breakdown'] = [f.to_dict() for f in all_features[:topk]]

        return candidates[:topk], stats

    def _compute_weighted_score(
        self,
        cand: Candidate,
        features: CandidateFeatures,
        gri_signals: GRISignals,
        input_attrs: GlobalAttributes,
        input_attrs_8axis: Optional[GlobalAttributes8Axis] = None
    ) -> float:
        """가중합 점수 계산 (8축 피처 포함)"""
        # 기본 가중합
        score = (
            self.weight_ml * cand.score_ml +
            self.weight_card * cand.score_card +
            self.weight_rule * cand.score_rule
        )

        # Specificity
        spec_weight = self.weight_specificity
        if gri_signals.gri3_multi_candidate:
            spec_weight = 0.5
        score += spec_weight * features.f_specificity

        # 기존 속성 매칭 보너스
        attr_bonus = (
            features.f_state_match +
            features.f_material_match +
            features.f_use_match +
            features.f_form_match
        )
        score += self.weight_attr_match * attr_bonus

        # 8축 매칭 보너스 (NEW)
        axis_bonus = (
            self.weight_object * features.f_object_match_score +
            self.weight_material * features.f_material_match_score +
            self.weight_processing * features.f_processing_match_score +
            self.weight_function * features.f_function_match_score +
            self.weight_form * features.f_form_match_score +
            self.weight_completeness * features.f_completeness_match_score +
            self.weight_quant * features.f_quant_rule_match_score +
            self.weight_legal * features.f_legal_scope_match_score
        )
        score += axis_bonus

        # KB retrieval 강도 힌트 (정규화된 f_lexical, LightGBM과 동일 스케일)
        score += 0.15 * features.f_lexical

        # Note support
        score += 0.1 * features.f_note_support_sum

        # 패널티
        if gri_signals.gri2a_incomplete and features.f_is_parts_candidate:
            if features.f_rule_inc_hits < 2:
                score += self.parts_penalty

        if features.f_parts_mismatch:
            score += self.parts_penalty

        if features.f_exclude_conflict:
            score += self.exclude_penalty * 0.5

        # Hard exclude
        if features.f_note_hard_exclude or features.f_quant_hard_exclude:
            score += self.hard_exclude_penalty

        # 8축 충돌/불확실성 패널티 (NEW)
        score -= features.f_conflict_penalty * 0.5
        score -= features.f_uncertainty_penalty * 0.3

        return score


# 안정성 정책 클래스 (NEW)
class StabilityPolicy:
    """분류 안정성 정책"""

    @staticmethod
    def can_hard_exclude(attrs: GlobalAttributes8Axis, axis: str) -> bool:
        """불확실한 속성으로 hard exclude 허용 여부"""
        axis_attrs = attrs.get_axis(axis)
        # 신뢰도 0.8 미만이면 hard exclude 금지
        return axis_attrs.confidence >= 0.8

    @staticmethod
    def apply_legal_priority(candidates: List[Candidate]) -> List[Candidate]:
        """법적 규칙 최우선 적용 - legal_scope 일치 후보 우선"""
        # 현재는 단순히 원본 반환 (추후 확장 가능)
        return candidates

    @staticmethod
    def should_ask_question(attrs: GlobalAttributes8Axis) -> bool:
        """데이터 부족 시 질문 전환 여부"""
        # 주요 축 3개 이상 누락시 True
        key_axes = ['material', 'processing_state', 'function_use', 'completeness']
        missing_count = sum(
            1 for axis_id in key_axes
            if not attrs.get_axis(axis_id).values
        )
        return missing_count >= 3


    def export_features(
        self,
        text: str,
        candidates: List[Candidate],
        gri_signals: Optional[GRISignals] = None,
        input_attrs: Optional[GlobalAttributes] = None,
        input_attrs_8axis: Optional[GlobalAttributes8Axis] = None,
        model_classes: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        후보별 피처 벡터 export (분석/학습용)

        Args:
            text: 입력 텍스트
            candidates: 후보 리스트
            gri_signals: GRI 신호
            input_attrs: 입력 속성 (기존)
            input_attrs_8axis: 입력 8축 속성
            model_classes: 모델 클래스 집합

        Returns:
            [{hs4, features_dict, feature_vector, feature_names}, ...]
        """
        if gri_signals is None:
            gri_signals = detect_gri_signals(text)
        if input_attrs is None:
            input_attrs = extract_attributes(text)
        if input_attrs_8axis is None:
            input_attrs_8axis = extract_attributes_8axis(text)

        results = []
        for cand in candidates:
            features = self.compute_features(
                text, cand, gri_signals, input_attrs, model_classes, input_attrs_8axis
            )

            results.append({
                'hs4': cand.hs4,
                'features_dict': features.to_dict(),
                'feature_vector': features.to_vector(),
                'feature_names': CandidateFeatures.feature_names(),
            })

        return results

    def get_feature_importance_weights(self) -> Dict[str, float]:
        """현재 피처 가중치 반환 (분석용)"""
        return {
            'weight_ml': self.weight_ml,
            'weight_card': self.weight_card,
            'weight_rule': self.weight_rule,
            'weight_specificity': self.weight_specificity,
            'weight_attr_match': self.weight_attr_match,
            'weight_object': self.weight_object,
            'weight_material': self.weight_material,
            'weight_processing': self.weight_processing,
            'weight_function': self.weight_function,
            'weight_form': self.weight_form,
            'weight_completeness': self.weight_completeness,
            'weight_quant': self.weight_quant,
            'weight_legal': self.weight_legal,
            'exclude_penalty': self.exclude_penalty,
            'hard_exclude_penalty': self.hard_exclude_penalty,
            'parts_penalty': self.parts_penalty,
        }


# 테스트
if __name__ == "__main__":
    reranker = HSReranker()
    print("KB HS4 수:", len(reranker.get_kb_hs4_set()))

    test_text = "냉동 돼지 삼겹살"
    gri = detect_gri_signals(test_text)
    attrs = extract_attributes(test_text)
    attrs8 = extract_attributes_8axis(test_text)

    print(f"\n입력: {test_text}")
    print(f"  GRI: {gri.active_signals()}")
    print(f"  속성 (7축): {attrs.summary()}")
    print(f"  속성 (8축): {attrs8.summary()}")

    kb_candidates = reranker.retrieve_from_kb(
        test_text, topk=10, gri_signals=gri,
        input_attrs=attrs, input_attrs_8axis=attrs8
    )
    print("\nKB 검색 결과:")
    for c in kb_candidates[:5]:
        print(f"  {c.hs4}: {c.evidence[0].text if c.evidence else 'N/A'}")
