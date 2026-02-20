"""
Essential Character 모듈 (GRI 3b)

4요소 가중 합산 모델로 복수 후보 HS4 간 본질적 특성 판단:
- core_function (기능 핵심): 0.35
- user_perception (인식 중심): 0.25
- area_volume (면적/부피): 0.20
- structural (구조 지배): 0.20
"""

from typing import List, Dict, Optional
from .types import Candidate, EssentialCharacterResult, ECFactor
from .attribute_extract import GlobalAttributes8Axis


class EssentialCharacterModule:
    """GRI 3(b) Essential Character 평가"""

    WEIGHTS = {
        'core_function': 0.35,
        'user_perception': 0.25,
        'area_volume': 0.20,
        'structural': 0.20,
    }

    def __init__(self, reranker=None):
        """
        Args:
            reranker: HSReranker 인스턴스 (카드/속성 데이터 접근용)
        """
        self.reranker = reranker

    def evaluate(
        self,
        input_data,
        candidates: List[Candidate],
        input_attrs_8axis: Optional[GlobalAttributes8Axis] = None
    ) -> EssentialCharacterResult:
        """
        Essential Character 평가

        Args:
            input_data: ClassificationInput 또는 str (입력 텍스트)
            candidates: GRI 3 적용 대상 후보 목록
            input_attrs_8axis: 8축 속성 (없으면 추출)

        Returns:
            EssentialCharacterResult
        """
        result = EssentialCharacterResult()

        if len(candidates) < 2:
            result.applicable = False
            result.reasoning = "후보가 2개 미만이므로 EC 불필요"
            return result

        result.applicable = True

        # 입력 텍스트 추출
        if isinstance(input_data, str):
            text = input_data
        else:
            text = getattr(input_data, 'text', '') or getattr(input_data, 'to_enriched_text', lambda: '')()

        # 8축 속성 (없으면 추출)
        if input_attrs_8axis is None:
            from .attribute_extract import extract_attributes_8axis
            input_attrs_8axis = extract_attributes_8axis(text)

        # 각 후보에 대해 4요소 점수 산출
        candidate_scores: Dict[str, float] = {}
        candidate_factors: Dict[str, Dict[str, ECFactor]] = {}

        for cand in candidates:
            hs4 = cand.hs4

            cf = self._score_core_function(text, hs4, input_attrs_8axis)
            up = self._score_user_perception(text, hs4, input_attrs_8axis)
            av = self._score_area_volume(text, hs4, input_attrs_8axis)
            st = self._score_structural(text, hs4, input_attrs_8axis)

            weighted_total = (
                self.WEIGHTS['core_function'] * cf.score +
                self.WEIGHTS['user_perception'] * up.score +
                self.WEIGHTS['area_volume'] * av.score +
                self.WEIGHTS['structural'] * st.score
            )

            candidate_scores[hs4] = weighted_total
            candidate_factors[hs4] = {
                'core_function': cf,
                'user_perception': up,
                'area_volume': av,
                'structural': st,
            }

        # Winner 결정
        if candidate_scores:
            winner_hs4 = max(candidate_scores, key=candidate_scores.get)
            result.winner_hs4 = winner_hs4
            result.candidate_scores = candidate_scores

            # Winner의 factor 상세 기록
            winner_factors = candidate_factors[winner_hs4]
            result.core_function = winner_factors['core_function']
            result.user_perception = winner_factors['user_perception']
            result.area_volume = winner_factors['area_volume']
            result.structural = winner_factors['structural']

            # Reasoning
            scores_sorted = sorted(candidate_scores.items(), key=lambda x: -x[1])
            gap = scores_sorted[0][1] - scores_sorted[1][1] if len(scores_sorted) > 1 else 0
            result.reasoning = (
                f"EC 판정: {winner_hs4} (점수 {candidate_scores[winner_hs4]:.3f}, "
                f"차이 {gap:.3f})"
            )

        return result

    def _score_core_function(
        self, text: str, hs4: str, attrs: GlobalAttributes8Axis
    ) -> ECFactor:
        """기능 핵심 점수: function_use axis + 카드 keyword 매칭"""
        factor = ECFactor(name="core_function")
        score = 0.0

        # 입력의 function_use와 후보 카드의 function 키워드 매칭
        if self.reranker and hs4 in self.reranker.card_attrs_8axis:
            card_function = self.reranker.card_attrs_8axis[hs4].get('function_use', set())
            input_function = set(attrs.function_use.values)

            if input_function and card_function:
                overlap = input_function & card_function
                if overlap:
                    score = len(overlap) / max(len(input_function), 1)
                    factor.reasoning = f"기능 매칭: {overlap}"

        # 카드 키워드로 보완
        if self.reranker and hs4 in self.reranker.cards:
            card = self.reranker.cards[hs4]
            title = card.get('title', '').lower()
            # 입력의 function_use 키워드가 title에 있는지
            for func_val in attrs.function_use.values:
                if func_val.lower() in title:
                    score = max(score, 0.5)

        factor.score = min(1.0, score)
        return factor

    def _score_user_perception(
        self, text: str, hs4: str, attrs: GlobalAttributes8Axis
    ) -> ECFactor:
        """인식 중심 점수: object_nature axis + heading title 매칭"""
        factor = ECFactor(name="user_perception")
        score = 0.0

        if self.reranker and hs4 in self.reranker.card_attrs_8axis:
            card_object = self.reranker.card_attrs_8axis[hs4].get('object_nature', set())
            input_object = set(attrs.object_nature.values)

            if input_object and card_object:
                overlap = input_object & card_object
                if overlap:
                    score = len(overlap) / max(len(input_object), 1)
                    factor.reasoning = f"객체 본질 매칭: {overlap}"

        # Heading title 단어 매칭
        if self.reranker and hs4 in self.reranker.cards:
            card = self.reranker.cards[hs4]
            title = card.get('title', '').lower()
            text_lower = text.lower()
            title_words = [w for w in title.split() if len(w) >= 2]
            if title_words:
                matched = sum(1 for w in title_words if w in text_lower)
                title_score = matched / len(title_words)
                score = max(score, title_score * 0.8)

        factor.score = min(1.0, score)
        return factor

    def _score_area_volume(
        self, text: str, hs4: str, attrs: GlobalAttributes8Axis
    ) -> ECFactor:
        """면적/부피 점수: material axis + 구성비"""
        factor = ECFactor(name="area_volume")
        score = 0.0

        if self.reranker and hs4 in self.reranker.card_attrs_8axis:
            card_material = self.reranker.card_attrs_8axis[hs4].get('material', set())
            input_material = set(attrs.material.values)

            if input_material and card_material:
                overlap = input_material & card_material
                if overlap:
                    score = len(overlap) / max(len(input_material), 1)
                    factor.reasoning = f"재질 매칭: {overlap}"

        # ClassificationInput의 materials 구성비 활용
        if hasattr(input_data := None, 'materials'):
            pass  # input_data가 ClassificationInput이면 materials 구성비 활용

        factor.score = min(1.0, score)
        return factor

    def _score_structural(
        self, text: str, hs4: str, attrs: GlobalAttributes8Axis
    ) -> ECFactor:
        """구조 지배 점수: completeness axis + card decision_attributes"""
        factor = ECFactor(name="structural")
        score = 0.0

        if self.reranker and hs4 in self.reranker.card_attrs_8axis:
            card_completeness = self.reranker.card_attrs_8axis[hs4].get('completeness', set())
            input_completeness = set(attrs.completeness.values)

            if input_completeness and card_completeness:
                overlap = input_completeness & card_completeness
                if overlap:
                    score = len(overlap) / max(len(input_completeness), 1)
                    factor.reasoning = f"완성도 매칭: {overlap}"

        # card decision_attributes 활용
        if self.reranker and hs4 in self.reranker.cards:
            card = self.reranker.cards[hs4]
            decision_attrs = card.get('decision_attributes', {})
            if decision_attrs:
                # decision_attributes가 있다는 것은 구조적 정보가 풍부
                score = max(score, 0.3)

        factor.score = min(1.0, score)
        return factor
