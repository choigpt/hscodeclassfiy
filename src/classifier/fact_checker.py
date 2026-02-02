"""
FactSufficiencyChecker - GRI 1 사실 충분성 검사

입력 attrs(8축) + 후보 hs4 + required_facts를 받아:
1. 후보별 missing_hard/missing_soft 계산
2. missing_hard 있으면 AUTO 금지, ASK 라우팅
3. 질문은 topN 후보를 가장 잘 구분하는 fact 우선 (분별력 기반)
4. 불확실한 attrs(낮은 confidence)로는 hard exclude 금지 (soft로 downgrade)
"""

import json
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .required_facts import RequiredFact, HS4Requirements, FactHardness, FactAxis
from .attribute_extract import GlobalAttributes8Axis, AxisAttributes
from .types import Candidate, Evidence


@dataclass
class MissingFacts:
    """후보별 부족한 사실"""
    hs4: str
    missing_hard: List[RequiredFact] = field(default_factory=list)  # 필수 부족
    missing_soft: List[RequiredFact] = field(default_factory=list)  # 선호 부족
    satisfied_facts: List[RequiredFact] = field(default_factory=list)  # 충족된 사실

    def has_hard_missing(self) -> bool:
        """필수 사실이 부족한가?"""
        return len(self.missing_hard) > 0

    def total_missing(self) -> int:
        """총 부족 개수"""
        return len(self.missing_hard) + len(self.missing_soft)

    def satisfaction_rate(self, total_facts: int) -> float:
        """충족률"""
        if total_facts == 0:
            return 1.0
        return len(self.satisfied_facts) / total_facts


@dataclass
class FactCheckResult:
    """사실 충분성 검사 결과"""
    sufficient: bool  # 충분한가?
    candidates_missing: Dict[str, MissingFacts]  # 후보별 부족 사실
    questions: List[Dict] = field(default_factory=list)  # 생성된 질문
    discriminative_facts: List[Tuple[RequiredFact, int]] = field(default_factory=list)  # 분별력 있는 fact

    def get_ask_candidates(self) -> List[str]:
        """ASK가 필요한 후보 (hard missing 있음)"""
        return [hs4 for hs4, missing in self.candidates_missing.items()
                if missing.has_hard_missing()]


class FactSufficiencyChecker:
    """사실 충분성 검사기"""

    def __init__(
        self,
        note_requirements_path: str = "kb/structured/note_requirements.jsonl",
        cards_v2_path: str = "kb/structured/hs4_cards_v2.jsonl"
    ):
        self.note_requirements_path = Path(note_requirements_path)
        self.cards_v2_path = Path(cards_v2_path)

        # HS4별 요구 사실
        self.hs4_requirements: Dict[str, HS4Requirements] = {}

        self._load_requirements()

    def _load_requirements(self):
        """요구 사실 로드"""
        # 1. 주규정 요구 사실 로드
        note_reqs_by_scope = defaultdict(list)

        if self.note_requirements_path.exists():
            with open(self.note_requirements_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    scope = data['hs_scope']
                    facts = [RequiredFact.from_dict(f) for f in data['required_facts']]
                    note_reqs_by_scope[scope].extend(facts)

            print(f"[FactChecker] Loaded note requirements: {len(note_reqs_by_scope)} scopes")

        # 2. 카드 v2 로드
        cards_with_facts = 0
        if self.cards_v2_path.exists():
            with open(self.cards_v2_path, 'r', encoding='utf-8') as f:
                for line in f:
                    card = json.loads(line)
                    hs4 = card['hs4']

                    # 카드 facts
                    card_facts = [RequiredFact.from_dict(f) for f in card.get('required_facts', [])]

                    # 해당 HS4에 적용되는 주규정 facts 수집
                    chapter = hs4[:2]
                    note_facts = []

                    # Section 주규정 (부 번호는 chapter로 추론)
                    # 간단하게 chapter_XX로 매핑
                    chapter_scope = f"chapter_{chapter}"
                    if chapter_scope in note_reqs_by_scope:
                        note_facts.extend(note_reqs_by_scope[chapter_scope])

                    # HS4Requirements 생성
                    self.hs4_requirements[hs4] = HS4Requirements(
                        hs4=hs4,
                        card_facts=card_facts,
                        note_facts=note_facts
                    )

                    if card_facts or note_facts:
                        cards_with_facts += 1

            print(f"[FactChecker] Loaded cards v2: {len(self.hs4_requirements)} HS4")
            print(f"[FactChecker] Cards with facts: {cards_with_facts}")

    def check(
        self,
        input_text: str,
        input_attrs: GlobalAttributes8Axis,
        candidates: List[Candidate]
    ) -> FactCheckResult:
        """
        사실 충분성 검사

        Args:
            input_text: 입력 텍스트
            input_attrs: 입력 8축 속성
            candidates: 후보 목록

        Returns:
            FactCheckResult
        """
        candidates_missing: Dict[str, MissingFacts] = {}

        # 각 후보에 대해 부족한 사실 계산
        for cand in candidates:
            missing = self._check_candidate(input_text, input_attrs, cand.hs4)
            candidates_missing[cand.hs4] = missing

        # 전체 충분성 판단
        # 모든 후보가 hard missing이 없으면 충분
        sufficient = all(
            not missing.has_hard_missing()
            for missing in candidates_missing.values()
        )

        # 질문 생성 (불충분할 때)
        questions = []
        discriminative_facts = []

        if not sufficient:
            questions, discriminative_facts = self._generate_questions(
                candidates, candidates_missing
            )

        return FactCheckResult(
            sufficient=sufficient,
            candidates_missing=candidates_missing,
            questions=questions,
            discriminative_facts=discriminative_facts
        )

    def _check_candidate(
        self,
        input_text: str,
        input_attrs: GlobalAttributes8Axis,
        hs4: str
    ) -> MissingFacts:
        """단일 후보에 대한 사실 부족 계산"""

        missing = MissingFacts(hs4=hs4)

        # 해당 HS4의 요구 사실 가져오기
        requirements = self.hs4_requirements.get(hs4)
        if not requirements:
            # 요구 사실이 없으면 충족으로 간주
            return missing

        all_facts = requirements.all_facts()

        for fact in all_facts:
            satisfied = self._check_single_fact(input_text, input_attrs, fact)

            if satisfied:
                missing.satisfied_facts.append(fact)
            else:
                # 부족한 사실 분류
                # 주의: confidence 낮은 attrs로는 hard exclude 금지
                hardness = fact.hardness

                if fact.hardness == FactHardness.HARD:
                    # exclude 패턴이고 입력 attrs의 confidence가 낮으면 soft로 downgrade
                    if fact.operator in ['ne', 'not_contains']:
                        if self._is_low_confidence_axis(input_attrs, fact.axis):
                            hardness = FactHardness.SOFT

                if hardness == FactHardness.HARD:
                    missing.missing_hard.append(fact)
                else:
                    missing.missing_soft.append(fact)

        return missing

    def _check_single_fact(
        self,
        input_text: str,
        input_attrs: GlobalAttributes8Axis,
        fact: RequiredFact
    ) -> bool:
        """단일 사실 충족 여부 검사"""

        axis = fact.axis
        operator = fact.operator
        value = fact.value

        # 해당 축의 attrs 가져오기
        axis_attrs = self._get_axis_attrs(input_attrs, axis)
        if not axis_attrs:
            # 입력에 해당 축 정보가 없으면 불충족
            return False

        # Operator에 따라 검사
        if operator == 'eq':
            return value.lower() in [v.lower() for v in axis_attrs.values]
        elif operator == 'ne':
            return value.lower() not in [v.lower() for v in axis_attrs.values]
        elif operator == 'contains':
            return any(value.lower() in v.lower() for v in axis_attrs.values)
        elif operator == 'not_contains':
            return not any(value.lower() in v.lower() for v in axis_attrs.values)
        elif operator in ['gt', 'gte', 'lt', 'lte']:
            # 정량 규칙 (quant axis)
            # TODO: 값 파싱 및 비교
            return False  # 일단 불충족으로

        return False

    def _get_axis_attrs(
        self,
        input_attrs: GlobalAttributes8Axis,
        axis: str
    ) -> Optional[AxisAttributes]:
        """특정 축의 속성 가져오기"""
        axis_map = {
            'object': input_attrs.object_nature,
            'material': input_attrs.material,
            'processing': input_attrs.processing_state,
            'function': input_attrs.function_use,
            'form': input_attrs.physical_form,
            'completeness': input_attrs.completeness,
            'quant': input_attrs.quant_rules,
            'legal': input_attrs.legal_scope,
        }
        return axis_map.get(axis)

    def _is_low_confidence_axis(
        self,
        input_attrs: GlobalAttributes8Axis,
        axis: str
    ) -> bool:
        """해당 축의 confidence가 낮은가?"""
        axis_attrs = self._get_axis_attrs(input_attrs, axis)
        if not axis_attrs:
            return True  # 없으면 낮은 것으로 간주

        # confidence가 0.5 미만이면 낮음
        return axis_attrs.confidence < 0.5

    def _generate_questions(
        self,
        candidates: List[Candidate],
        candidates_missing: Dict[str, MissingFacts]
    ) -> Tuple[List[Dict], List[Tuple[RequiredFact, int]]]:
        """
        질문 생성 (분별력 기반)

        topN 후보를 가장 잘 구분하는 missing fact를 우선 선택
        """
        # 1. 모든 missing_hard fact 수집
        all_hard_missing: List[Tuple[str, RequiredFact]] = []  # (hs4, fact)

        for hs4, missing in candidates_missing.items():
            for fact in missing.missing_hard:
                all_hard_missing.append((hs4, fact))

        if not all_hard_missing:
            return [], []

        # 2. fact별 분별력 계산
        # 분별력 = 해당 fact를 missing으로 가진 후보 수
        fact_discriminative: Dict[str, Set[str]] = defaultdict(set)  # fact_id → {hs4}

        for hs4, fact in all_hard_missing:
            fact_id = f"{fact.axis}_{fact.operator}_{fact.value}"
            fact_discriminative[fact_id].add(hs4)

        # 3. 분별력 높은 순으로 정렬
        discriminative_facts: List[Tuple[RequiredFact, int]] = []

        for fact_id, hs4_set in fact_discriminative.items():
            # 대표 fact 찾기
            repr_fact = next(
                (fact for _, fact in all_hard_missing
                 if f"{fact.axis}_{fact.operator}_{fact.value}" == fact_id),
                None
            )
            if repr_fact:
                discriminative_facts.append((repr_fact, len(hs4_set)))

        # 분별력 내림차순 정렬
        discriminative_facts.sort(key=lambda x: x[1], reverse=True)

        # 4. 상위 3개 fact에 대해 질문 생성
        questions = []
        for fact, disc_count in discriminative_facts[:3]:
            question = self._fact_to_question(fact, disc_count)
            questions.append(question)

        return questions, discriminative_facts

    def _fact_to_question(self, fact: RequiredFact, disc_count: int) -> Dict:
        """RequiredFact를 질문으로 변환"""

        axis_name_ko = {
            'object': '물체 종류',
            'material': '재질',
            'processing': '가공 상태',
            'function': '기능/용도',
            'form': '형태',
            'completeness': '완성도',
            'quant': '정량 규칙',
            'legal': '법적 범위',
        }

        operator_text = {
            'eq': '입니까?',
            'ne': '아닌가요?',
            'contains': '포함하나요?',
            'not_contains': '포함하지 않나요?',
            'gt': '초과하나요?',
            'gte': '이상인가요?',
            'lt': '미만인가요?',
            'lte': '이하인가요?',
        }

        axis_ko = axis_name_ko.get(fact.axis, fact.axis)
        op_text = operator_text.get(fact.operator, '?')

        question_text = f"{axis_ko}: {fact.value}{op_text}"

        return {
            'question': question_text,
            'axis': fact.axis,
            'value': fact.value,
            'operator': fact.operator,
            'source_ref': fact.source_ref,
            'discriminative_count': disc_count,
            'hardness': fact.hardness,
        }


def apply_fact_check(
    input_text: str,
    input_attrs: GlobalAttributes8Axis,
    candidates: List[Candidate]
) -> FactCheckResult:
    """FactSufficiencyChecker 편의 함수"""
    checker = FactSufficiencyChecker()
    return checker.check(input_text, input_attrs, candidates)
