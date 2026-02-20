"""
리스크 평가 모듈

5개 리스크 요인으로 LOW/MED/HIGH 레벨 판정:
- 점수 차이 (Top1-Top2 gap)
- GRI 3(b) Essential Character 적용 여부
- 정보 누락 (핵심 axis 미감지)
- 판례 충돌 (동일 상품 다른 코드)
- 관할 분기 (다른 관할 가능성)

리스크 레벨: HIGH (≥5.0) / MED (2.5~5.0) / LOW (<2.5)
"""

from typing import List, Optional, Dict

from .types import (
    Candidate, RiskAssessment, GRIApplication,
    EssentialCharacterResult,
)
from .attribute_extract import GlobalAttributes8Axis


class RiskAssessor:
    """리스크 평가"""

    # 임계값
    GAP_HIGH = 0.15
    GAP_MED = 0.30
    RISK_HIGH_THRESHOLD = 5.0
    RISK_MED_THRESHOLD = 2.5

    def assess(
        self,
        candidates: List[Candidate],
        applied_gri: List[GRIApplication],
        ec_result: Optional[EssentialCharacterResult],
        input_attrs: Optional[GlobalAttributes8Axis],
        normalized_cases: Optional[List[Dict]] = None,
        input_data=None,
    ) -> RiskAssessment:
        """
        리스크 평가

        Args:
            candidates: 최종 후보 목록 (score_total 순)
            applied_gri: 적용된 GRI 통칙 목록
            ec_result: Essential Character 결과
            input_attrs: 8축 속성
            normalized_cases: 정규화된 판결 케이스 (있으면 판례 충돌 검사)
            input_data: ClassificationInput (있으면 추가 정보 활용)

        Returns:
            RiskAssessment
        """
        risk = RiskAssessment()
        total_score = 0.0
        reasons = []

        # ---- Factor 1: 점수 차이 (Top1-Top2 gap) ----
        if len(candidates) >= 2:
            gap = candidates[0].score_total - candidates[1].score_total
            risk.score_gap = gap

            if gap < self.GAP_HIGH:
                total_score += 3.0
                reasons.append(f"Top1-Top2 gap 매우 작음 ({gap:.3f} < {self.GAP_HIGH})")
            elif gap < self.GAP_MED:
                total_score += 1.5
                reasons.append(f"Top1-Top2 gap 작음 ({gap:.3f} < {self.GAP_MED})")

        # ---- Factor 2: GRI 3(b) Essential Character 적용 ----
        if ec_result and ec_result.applicable:
            total_score += 2.0
            reasons.append("GRI 3(b) Essential Character 적용됨")

        # ---- Factor 3: 정보 누락 (핵심 axis 미감지) ----
        if input_attrs is not None:
            key_axes = ['material', 'processing_state', 'function_use', 'completeness']
            missing = []
            for axis_id in key_axes:
                axis = input_attrs.get_axis(axis_id)
                if not axis.values:
                    missing.append(axis_id)

            if len(missing) >= 2:
                per_missing = 1.5
                total_score += per_missing * len(missing)
                reasons.append(f"핵심 축 {len(missing)}개 미감지: {missing}")
                risk.missing_info = missing

        # ---- Factor 4: 판례 충돌 ----
        if normalized_cases and len(candidates) >= 1:
            top_hs4 = candidates[0].hs4
            conflicting_cases = self._check_case_conflict(
                top_hs4, normalized_cases, input_data
            )
            if conflicting_cases:
                total_score += 2.0
                reasons.append(f"판례 충돌: 동일 유형 상품 다른 코드 판결 {len(conflicting_cases)}건")

        # ---- Factor 5: 관할 분기 가능성 ----
        # 현재 KR 관할만 지원하지만, 향후 관할 비교 시 사용
        jurisdiction = "KR"
        if input_data and hasattr(input_data, 'jurisdiction'):
            jurisdiction = input_data.jurisdiction

        # GRI 2/3 적용된 경우 관할별 판단 차이 가능성
        gri_ids = [g.gri_id for g in applied_gri if g.applied]
        if any(g in ['GRI2b', 'GRI3a', 'GRI3b'] for g in gri_ids):
            total_score += 1.0
            reasons.append("GRI 2b/3 적용 → 관할 간 분기 가능성")

        # ---- 레벨 결정 ----
        risk.score = total_score
        risk.reasons = reasons

        if total_score >= self.RISK_HIGH_THRESHOLD:
            risk.level = "HIGH"
        elif total_score >= self.RISK_MED_THRESHOLD:
            risk.level = "MED"
        else:
            risk.level = "LOW"

        return risk

    def _check_case_conflict(
        self,
        hs4: str,
        normalized_cases: List[Dict],
        input_data=None,
    ) -> List[Dict]:
        """동일 유형 상품이 다른 코드로 판결된 케이스 검색"""
        # 간단한 키워드 매칭으로 유사 케이스 찾기
        text = ""
        if input_data:
            if isinstance(input_data, str):
                text = input_data
            elif hasattr(input_data, 'text'):
                text = input_data.text

        if not text:
            return []

        text_lower = text.lower()
        text_tokens = set(text_lower.split())

        conflicting = []
        for case in normalized_cases:
            case_hs4 = case.get("final_code_4", "")
            if case_hs4 == hs4:
                continue  # 같은 코드는 무시

            # 간단한 유사도: case features의 materials와 입력 비교
            features = case.get("features", {})
            case_materials = [m.get("name", "") for m in features.get("materials", [])]

            # 재질이 겹치면 유사 상품 가능성
            for mat in case_materials:
                if mat and mat.lower() in text_lower:
                    conflicting.append(case)
                    break

        return conflicting[:5]  # 최대 5건
