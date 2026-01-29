"""
HS Clarifier - 저신뢰도 시 GRI + 8축 속성 충돌 기반 질문 생성
"""

import json
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict

from .gri_signals import GRISignals, get_gri_questions
from .attribute_extract import GlobalAttributes, GlobalAttributes8Axis, AXIS_IDS

if TYPE_CHECKING:
    from .reranker import HSReranker


# GRI 기반 질문 템플릿
GRI_QUESTIONS = {
    'gri2a': [
        "미조립/분해 상태로 수입됩니까? 완제품으로 조립이 가능한 상태입니까?",
        "완제품입니까, 부품입니까? 어떤 기계/제품의 부품입니까?",
    ],
    'gri2b': [
        "주요 성분/재질의 비율은 어떻게 됩니까? (예: 면 60%, 폴리에스터 40%)",
        "본질적 특성을 결정하는 주된 성분/재질은 무엇입니까?",
    ],
    'gri3': [
        "세트로 판매됩니까? 구성품을 각각 개별 판매합니까?",
        "세트의 본질적 특성을 부여하는 주된 물품은 무엇입니까?",
    ],
    'gri5': [
        "케이스/용기가 전용품이며 함께 제시/판매됩니까?",
        "케이스/용기가 단독으로 별도 판매되는 것입니까?",
    ],
    'gri1': [
        "해당 류/호의 주(Note)에서 특별히 규정하는 사항이 있습니까?",
    ],
    'parts': [
        "완제품입니까 부품입니까?",
        "어떤 기계/제품에 사용되는 부품입니까?",
    ],
}

# 기본 질문 (fallback)
FALLBACK_QUESTIONS = [
    "주요 재질이 무엇인가요? (금속/플라스틱/직물/가죽/목재/기타)",
    "주요 용도가 무엇인가요? (산업용/가정용/의료용/식품용/기타)",
    "가공 상태가 어떻게 되나요? (신선/냉장/냉동/건조/가공)",
]

# 속성 축별 질문 템플릿 (기존 7축)
ATTR_AXIS_QUESTIONS = {
    'state': [
        "물품의 가공/처리 상태는 무엇인가요? (생것/신선/냉장/냉동/건조/조리/가공)",
        "원재료 상태입니까, 가공된 상태입니까?",
    ],
    'material': [
        "주요 재질/성분은 무엇인가요? (금속/플라스틱/목재/직물/가죽/유리/고무/기타)",
        "복합재질인 경우, 주된 재질의 중량 비율은 얼마입니까?",
    ],
    'use': [
        "물품의 주요 용도/기능은 무엇인가요? (산업용/가정용/의료용/농업용/수송기기용)",
        "어떤 산업/분야에서 사용됩니까?",
    ],
    'form': [
        "물품의 형태는 무엇인가요? (원료/반제품/완제품/분말/액체/판/봉/관)",
        "치수나 형상에 특별한 규격이 있습니까?",
    ],
    'parts': [
        "완제품입니까, 부품/부속품입니까?",
        "어떤 기계/제품의 부품입니까? (전용/범용)",
    ],
    'quant': [
        "함량/농도/중량/크기 등 정량적 수치가 있습니까?",
        "주요 성분의 비율(%)은 얼마입니까?",
    ],
}

# 8축 질문 템플릿 (NEW)
AXIS_QUESTIONS_8 = {
    'object_nature': [
        "물품의 본질은 무엇인가요? (물질/제품/생물/혼합물/세트/기계)",
        "원료 상태의 물질입니까, 가공된 제품입니까?",
        "여러 물품의 조합(세트/키트)입니까?",
    ],
    'material': [
        "주요 재질/성분은 무엇인가요? (금속/플라스틱/목재/직물/가죽/유리/고무/화학물질)",
        "복합재질인 경우, 주된 재질의 중량 비율은?",
        "동물성 또는 식물성 재료입니까?",
    ],
    'processing_state': [
        "가공/처리 상태는? (신선/냉동/건조/정제/조립)",
        "원재료 상태입니까, 가공된 상태입니까?",
        "조립 상태인가요? (조립완료/미조립/분해상태)",
    ],
    'function_use': [
        "주요 용도/기능은 무엇인가요?",
        "어떤 산업/분야에서 사용됩니까? (식품/의료/건축/산업/가정/농업)",
        "최종 소비자용입니까, 산업용입니까?",
    ],
    'physical_form': [
        "물리적 형태는? (분말/액체/판/봉/직물/완성품)",
        "특별한 규격이나 치수가 있습니까?",
        "포장 상태는 어떻습니까? (벌크/개별포장/세트포장)",
    ],
    'completeness': [
        "완제품입니까, 부품/부속품입니까?",
        "세트/키트 형태로 제시됩니까?",
        "특정 기계/제품 전용입니까, 범용입니까?",
    ],
    'quantitative_rules': [
        "함량/농도/순도 등 정량적 수치가 있습니까?",
        "주요 성분의 비율(%)은 얼마입니까?",
        "중량/부피/크기 등 정량적 규격이 있습니까?",
    ],
    'legal_scope': [
        "특정 류/호의 주(Note)에 해당합니까?",
        "미조립/세트/케이스 관련 규정이 적용됩니까?",
        "특수 규정이 적용되는 물품입니까?",
    ],
}

# 축별 분류 분기력 우선순위 (높을수록 중요)
AXIS_PRIORITY = {
    'quantitative_rules': 10,  # 정량 규칙이 가장 결정적
    'completeness': 9,         # 부품/완제품 구분 중요
    'processing_state': 8,     # 가공상태
    'material': 7,             # 재질
    'legal_scope': 6,          # 법적 규정
    'function_use': 5,         # 용도
    'physical_form': 4,        # 형태
    'object_nature': 3,        # 물체 본질
}


class HSClarifier:
    """
    저신뢰도 판정 및 GRI 기반 질문 생성
    """

    def __init__(
        self,
        questions_path: str = "kb/structured/disambiguation_questions.jsonl",
        threshold_top1: float = 0.50,
        threshold_gap: float = 0.15
    ):
        self.questions_path = questions_path
        self.threshold_top1 = threshold_top1  # p1 < 0.50
        self.threshold_gap = threshold_gap    # (p1 - p2) < 0.15

        # hs4 -> questions
        self.questions: Dict[str, List[Dict]] = defaultdict(list)

        self._load_questions()

    def _load_questions(self):
        """질문 템플릿 로드"""
        questions_file = Path(self.questions_path)
        if not questions_file.exists():
            print(f"[Clarifier] 경고: 질문 파일 없음: {questions_file}")
            return

        count = 0
        with open(questions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    hs4 = item.get('hs4', '')
                    questions = item.get('disambiguation_questions', [])

                    if hs4 and questions:
                        self.questions[hs4] = questions
                        count += 1
                except json.JSONDecodeError:
                    continue

        print(f"[Clarifier] 질문 템플릿 로드: {count}개 HS4")

    def is_low_confidence(self, scores: List[float]) -> bool:
        """
        저신뢰도 판정

        Args:
            scores: score_total 리스트 (내림차순)

        Returns:
            True if low confidence
        """
        if not scores:
            return True

        p1 = scores[0]
        p2 = scores[1] if len(scores) > 1 else 0.0

        # 조건: p1 < 0.50 OR (p1 - p2) < 0.15
        if p1 < self.threshold_top1:
            return True
        if (p1 - p2) < self.threshold_gap:
            return True

        return False

    def get_gri_questions(
        self,
        gri_signals: GRISignals,
        max_questions: int = 2
    ) -> List[str]:
        """
        GRI 신호 기반 질문 생성

        Args:
            gri_signals: GRI 신호 객체
            max_questions: GRI 질문 최대 수

        Returns:
            GRI 관련 질문 리스트
        """
        questions = []

        # 우선순위: gri2a > gri2b > gri3 > gri5 > gri1
        if gri_signals.gri2a_incomplete:
            questions.extend(GRI_QUESTIONS['gri2a'])

        if gri_signals.gri2b_mixtures:
            questions.extend(GRI_QUESTIONS['gri2b'])

        if gri_signals.gri3_multi_candidate:
            questions.extend(GRI_QUESTIONS['gri3'])

        if gri_signals.gri5_containers:
            questions.extend(GRI_QUESTIONS['gri5'])

        if gri_signals.gri1_note_like:
            questions.extend(GRI_QUESTIONS['gri1'])

        # 중복 제거 및 제한
        seen = set()
        unique = []
        for q in questions:
            if q not in seen:
                seen.add(q)
                unique.append(q)
                if len(unique) >= max_questions:
                    break

        return unique

    def get_hs4_questions(
        self,
        hs4_list: List[str],
        max_questions: int = 2
    ) -> List[str]:
        """
        Top-K 후보의 질문 템플릿에서 질문 추출

        Args:
            hs4_list: 후보 HS4 리스트
            max_questions: 최대 질문 수

        Returns:
            질문 리스트 (중복 제거)
        """
        seen_keys: Set[str] = set()
        questions: List[str] = []

        for hs4 in hs4_list:
            if hs4 not in self.questions:
                continue

            for q_item in self.questions[hs4]:
                key = q_item.get('key', '')
                q_text = q_item.get('q', '')

                if not key or not q_text:
                    continue

                # 중복 제거 (같은 key는 한 번만)
                if key in seen_keys:
                    continue

                seen_keys.add(key)
                questions.append(q_text)

                if len(questions) >= max_questions:
                    return questions

        return questions

    def get_questions(
        self,
        hs4_list: List[str],
        gri_signals: Optional[GRISignals] = None,
        max_questions: int = 3,
        has_parts_signal: bool = False
    ) -> List[str]:
        """
        통합 질문 생성 (GRI 우선, HS4 템플릿 보조)

        Args:
            hs4_list: 후보 HS4 리스트
            gri_signals: GRI 신호 객체
            max_questions: 최대 질문 수
            has_parts_signal: 부품 신호 여부

        Returns:
            질문 리스트 (중복 제거, 최대 max_questions개)
        """
        all_questions: List[str] = []
        seen: Set[str] = set()

        # 1. GRI 기반 질문 (우선)
        if gri_signals and gri_signals.any_signal():
            gri_qs = self.get_gri_questions(gri_signals, max_questions=2)
            for q in gri_qs:
                if q not in seen:
                    seen.add(q)
                    all_questions.append(q)

        # 2. 부품 신호 시 추가 질문
        if has_parts_signal and len(all_questions) < max_questions:
            for q in GRI_QUESTIONS['parts']:
                if q not in seen:
                    seen.add(q)
                    all_questions.append(q)
                    if len(all_questions) >= max_questions:
                        break

        # 3. HS4 템플릿 질문 (보조)
        if len(all_questions) < max_questions:
            hs4_qs = self.get_hs4_questions(hs4_list, max_questions=2)
            for q in hs4_qs:
                if q not in seen:
                    seen.add(q)
                    all_questions.append(q)
                    if len(all_questions) >= max_questions:
                        break

        # 4. Fallback
        if not all_questions:
            return FALLBACK_QUESTIONS[:max_questions]

        return all_questions[:max_questions]

    def detect_attr_conflicts(
        self,
        hs4_list: List[str],
        input_attrs: Optional[GlobalAttributes],
        reranker: Optional['HSReranker']
    ) -> Dict[str, int]:
        """
        후보 간 속성 충돌 탐지

        Args:
            hs4_list: 상위 후보 HS4 리스트
            input_attrs: 입력 속성
            reranker: Reranker 객체 (카드 속성 참조용)

        Returns:
            축별 충돌 점수 (높을수록 후보간 불일치 큼)
        """
        conflicts = {
            'state': 0,
            'material': 0,
            'use': 0,
            'form': 0,
            'parts': 0,
            'quant': 0,
        }

        if not reranker or not hs4_list:
            return conflicts

        # 후보별 속성 수집
        candidate_attrs = []
        for hs4 in hs4_list[:5]:  # Top-5만
            if hs4 in reranker.card_attrs:
                candidate_attrs.append(reranker.card_attrs[hs4])

        if len(candidate_attrs) < 2:
            return conflicts

        # 축별 불일치 계산
        for axis in ['state', 'material', 'use', 'form']:
            all_values = set()
            for ca in candidate_attrs:
                all_values.update(ca.get(axis, set()))
            # 값의 다양성 = 충돌 정도
            conflicts[axis] = len(all_values)

        # 부품 여부 불일치
        parts_values = [ca.get('is_parts', False) for ca in candidate_attrs]
        if True in parts_values and False in parts_values:
            conflicts['parts'] = 3  # 부품/완제품 혼재

        # 정량 조건 필요 여부
        if input_attrs and not input_attrs.has_quant:
            # 후보 중 정량 규칙이 있는 것이 있으면
            for hs4 in hs4_list[:5]:
                if hs4 in reranker.rules:
                    for rule in reranker.rules[hs4]:
                        if rule.get('quant_rule'):
                            conflicts['quant'] += 1
                            break

        return conflicts

    def get_attr_conflict_questions(
        self,
        hs4_list: List[str],
        input_attrs: Optional[GlobalAttributes],
        reranker: Optional['HSReranker'],
        max_questions: int = 2
    ) -> List[str]:
        """
        속성 충돌이 가장 큰 축에 대한 질문 생성

        Args:
            hs4_list: 후보 HS4 리스트
            input_attrs: 입력 속성
            reranker: Reranker 객체
            max_questions: 최대 질문 수

        Returns:
            질문 리스트 (충돌 큰 순)
        """
        conflicts = self.detect_attr_conflicts(hs4_list, input_attrs, reranker)

        # 충돌 점수 순 정렬
        sorted_axes = sorted(conflicts.items(), key=lambda x: -x[1])

        questions = []
        seen = set()

        for axis, score in sorted_axes:
            if score == 0:
                continue

            if axis in ATTR_AXIS_QUESTIONS:
                for q in ATTR_AXIS_QUESTIONS[axis]:
                    if q not in seen:
                        seen.add(q)
                        questions.append(q)
                        if len(questions) >= max_questions:
                            return questions
                        break

        return questions

    def get_questions_with_context(
        self,
        hs4_list: List[str],
        gri_signals: Optional[GRISignals] = None,
        input_attrs: Optional[GlobalAttributes] = None,
        no_kb_hits: bool = False,
        has_parts_signal: bool = False,
        reranker: Optional['HSReranker'] = None,
        max_questions: int = 3
    ) -> List[str]:
        """
        컨텍스트 기반 질문 생성 (GRI + 속성 충돌 고려)

        Args:
            hs4_list: 후보 HS4 리스트
            gri_signals: GRI 신호 객체
            input_attrs: 입력 전역 속성
            no_kb_hits: KB 매칭 없음 여부
            has_parts_signal: 부품 신호 여부
            reranker: Reranker 객체 (속성 충돌 분석용)
            max_questions: 최대 질문 수

        Returns:
            질문 리스트
        """
        questions = []
        seen = set()

        # KB 매칭 없으면 더 구체적인 정보 요청
        if no_kb_hits:
            questions.append("입력 정보가 부족합니다. 더 구체적인 품명을 입력해주세요.")
            seen.add(questions[-1])

        # 1. 속성 충돌 기반 질문 (우선순위 최고)
        if reranker and len(questions) < max_questions:
            attr_qs = self.get_attr_conflict_questions(
                hs4_list,
                input_attrs,
                reranker,
                max_questions=2
            )
            for q in attr_qs:
                if q not in seen and len(questions) < max_questions:
                    seen.add(q)
                    questions.append(q)

        # 2. GRI 기반 질문
        if gri_signals and gri_signals.any_signal() and len(questions) < max_questions:
            gri_qs = self.get_gri_questions(gri_signals, max_questions=2)
            for q in gri_qs:
                if q not in seen and len(questions) < max_questions:
                    seen.add(q)
                    questions.append(q)

        # 3. 부품 신호 질문
        if has_parts_signal and len(questions) < max_questions:
            for q in GRI_QUESTIONS['parts']:
                if q not in seen and len(questions) < max_questions:
                    seen.add(q)
                    questions.append(q)

        # 4. HS4 템플릿 질문
        if len(questions) < max_questions:
            hs4_qs = self.get_hs4_questions(hs4_list, max_questions=2)
            for q in hs4_qs:
                if q not in seen and len(questions) < max_questions:
                    seen.add(q)
                    questions.append(q)

        # 5. Fallback
        if not questions:
            return FALLBACK_QUESTIONS[:max_questions]

        return questions[:max_questions]

    # ============================================================
    # 8축 기반 질문 생성 함수 (NEW)
    # ============================================================

    def detect_axis_conflicts_8(
        self,
        hs4_list: List[str],
        input_attrs: Optional[GlobalAttributes8Axis],
        reranker: Optional['HSReranker']
    ) -> Dict[str, float]:
        """
        8축 기준 후보간 충돌 점수 계산

        Args:
            hs4_list: 상위 후보 HS4 리스트
            input_attrs: 입력 8축 속성
            reranker: Reranker 객체 (카드 속성 참조용)

        Returns:
            {axis_id: conflict_score} - 높을수록 충돌 큼
        """
        conflicts = {axis_id: 0.0 for axis_id in AXIS_IDS if axis_id != 'quantitative_rules'}

        if not reranker or not hs4_list or not hasattr(reranker, 'card_attrs_8axis'):
            return conflicts

        # 후보별 8축 속성 수집
        candidate_attrs_list = []
        for hs4 in hs4_list[:5]:  # Top-5만
            if hs4 in reranker.card_attrs_8axis:
                candidate_attrs_list.append(reranker.card_attrs_8axis[hs4])

        if len(candidate_attrs_list) < 2:
            return conflicts

        # 축별 불일치 계산 (값의 다양성)
        for axis_id in conflicts.keys():
            all_values = set()
            for ca in candidate_attrs_list:
                all_values.update(ca.get(axis_id, set()))

            # 다양성 점수: 값이 다양할수록 높음
            diversity = len(all_values)
            if diversity > 1:
                conflicts[axis_id] = diversity * 1.0

        # 완성도 축 특별 처리: parts vs complete 충돌 심화
        completeness_values = []
        for ca in candidate_attrs_list:
            completeness_values.extend(ca.get('completeness', set()))
        if 'parts' in completeness_values and 'complete' in completeness_values:
            conflicts['completeness'] += 3.0

        # 정량규칙 충돌: 입력에 정량 정보 없는데 후보에 정량 규칙 있으면
        if input_attrs and not input_attrs.quantitative_rules:
            for hs4 in hs4_list[:5]:
                if hs4 in reranker.rules:
                    for rule in reranker.rules[hs4]:
                        if rule.get('quant_rule'):
                            conflicts['quantitative_rules'] = conflicts.get('quantitative_rules', 0) + 1.0
                            break

        return conflicts

    def get_targeted_questions(
        self,
        hs4_list: List[str],
        input_attrs: Optional[GlobalAttributes8Axis],
        reranker: Optional['HSReranker'],
        max_questions: int = 3
    ) -> List[str]:
        """
        분류 분기력 최대화 질문 생성

        8축 기반으로 후보간 충돌이 가장 큰 축에 대한 질문 생성

        Args:
            hs4_list: 후보 HS4 리스트
            input_attrs: 입력 8축 속성
            reranker: Reranker 객체
            max_questions: 최대 질문 수

        Returns:
            질문 리스트 (분기력 순)
        """
        conflicts = self.detect_axis_conflicts_8(hs4_list, input_attrs, reranker)

        # 우선순위 가중 적용 후 정렬
        weighted_conflicts = []
        for axis_id, score in conflicts.items():
            priority = AXIS_PRIORITY.get(axis_id, 1)
            weighted_score = score * priority
            weighted_conflicts.append((axis_id, weighted_score))

        # 가중 충돌 점수 순 정렬
        sorted_axes = sorted(weighted_conflicts, key=lambda x: -x[1])

        questions = []
        seen = set()

        for axis_id, score in sorted_axes:
            if score <= 0:
                continue

            if axis_id in AXIS_QUESTIONS_8:
                for q in AXIS_QUESTIONS_8[axis_id]:
                    if q not in seen:
                        seen.add(q)
                        questions.append(q)
                        if len(questions) >= max_questions:
                            return questions
                        break  # 축당 하나씩만

        return questions[:max_questions]

    def get_missing_axis_questions(
        self,
        input_attrs: GlobalAttributes8Axis,
        max_questions: int = 2
    ) -> List[str]:
        """
        누락된 축에 대한 질문 생성

        Args:
            input_attrs: 입력 8축 속성
            max_questions: 최대 질문 수

        Returns:
            질문 리스트
        """
        questions = []
        seen = set()

        # 우선순위 순으로 누락 축 확인
        sorted_axes = sorted(
            AXIS_PRIORITY.items(),
            key=lambda x: -x[1]
        )

        for axis_id, _ in sorted_axes:
            axis = input_attrs.get_axis(axis_id)
            # 값이 없거나 신뢰도 낮으면 질문
            if not axis.values or axis.confidence < 0.3:
                if axis_id in AXIS_QUESTIONS_8:
                    for q in AXIS_QUESTIONS_8[axis_id]:
                        if q not in seen:
                            seen.add(q)
                            questions.append(q)
                            if len(questions) >= max_questions:
                                return questions
                            break

        return questions

    def get_questions_with_8axis(
        self,
        hs4_list: List[str],
        gri_signals: Optional[GRISignals] = None,
        input_attrs: Optional[GlobalAttributes] = None,
        input_attrs_8axis: Optional[GlobalAttributes8Axis] = None,
        no_kb_hits: bool = False,
        has_parts_signal: bool = False,
        reranker: Optional['HSReranker'] = None,
        max_questions: int = 3
    ) -> List[str]:
        """
        8축 기반 컨텍스트 질문 생성 (개선된 버전)

        Args:
            hs4_list: 후보 HS4 리스트
            gri_signals: GRI 신호 객체
            input_attrs: 입력 전역 속성 (기존)
            input_attrs_8axis: 입력 8축 속성
            no_kb_hits: KB 매칭 없음 여부
            has_parts_signal: 부품 신호 여부
            reranker: Reranker 객체
            max_questions: 최대 질문 수

        Returns:
            질문 리스트
        """
        questions = []
        seen = set()

        # KB 매칭 없으면 더 구체적인 정보 요청
        if no_kb_hits:
            questions.append("입력 정보가 부족합니다. 더 구체적인 품명을 입력해주세요.")
            seen.add(questions[-1])

        # 1. 8축 기반 타겟팅 질문 (분기력 최대화)
        if reranker and input_attrs_8axis and len(questions) < max_questions:
            targeted_qs = self.get_targeted_questions(
                hs4_list, input_attrs_8axis, reranker, max_questions=2
            )
            for q in targeted_qs:
                if q not in seen and len(questions) < max_questions:
                    seen.add(q)
                    questions.append(q)

        # 2. 누락 축 질문
        if input_attrs_8axis and len(questions) < max_questions:
            missing_qs = self.get_missing_axis_questions(input_attrs_8axis, max_questions=1)
            for q in missing_qs:
                if q not in seen and len(questions) < max_questions:
                    seen.add(q)
                    questions.append(q)

        # 3. GRI 기반 질문
        if gri_signals and gri_signals.any_signal() and len(questions) < max_questions:
            gri_qs = self.get_gri_questions(gri_signals, max_questions=2)
            for q in gri_qs:
                if q not in seen and len(questions) < max_questions:
                    seen.add(q)
                    questions.append(q)

        # 4. 부품 신호 질문
        if has_parts_signal and len(questions) < max_questions:
            for q in GRI_QUESTIONS['parts']:
                if q not in seen and len(questions) < max_questions:
                    seen.add(q)
                    questions.append(q)

        # 5. HS4 템플릿 질문
        if len(questions) < max_questions:
            hs4_qs = self.get_hs4_questions(hs4_list, max_questions=2)
            for q in hs4_qs:
                if q not in seen and len(questions) < max_questions:
                    seen.add(q)
                    questions.append(q)

        # 6. Fallback
        if not questions:
            return FALLBACK_QUESTIONS[:max_questions]

        return questions[:max_questions]


# 테스트
if __name__ == "__main__":
    from .gri_signals import detect_gri_signals

    clarifier = HSClarifier()

    # 저신뢰도 테스트
    print("Low confidence test:")
    print(f"  [0.8, 0.1]: {clarifier.is_low_confidence([0.8, 0.1])}")  # False
    print(f"  [0.4, 0.3]: {clarifier.is_low_confidence([0.4, 0.3])}")  # True (p1 < 0.5)
    print(f"  [0.6, 0.5]: {clarifier.is_low_confidence([0.6, 0.5])}")  # True (gap < 0.15)

    # GRI 기반 질문 테스트
    print("\nGRI-based questions:")

    test_cases = [
        "자동차 CKD 부품",
        "면 60% 폴리에스터 40% 혼방",
        "식기 세트",
        "스마트폰 전용 케이스",
    ]

    for text in test_cases:
        gri = detect_gri_signals(text)
        questions = clarifier.get_questions(['0203', '0210'], gri_signals=gri)
        print(f"\n  {text}:")
        print(f"    GRI: {gri.active_signals()}")
        for q in questions:
            print(f"    - {q}")
