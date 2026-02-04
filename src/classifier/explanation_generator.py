"""
Explanation Generator - 분류 근거 설명 생성

원칙:
1. Evidence를 2~3개 핵심 근거로 요약
2. 결론 변경 금지 (evidence만 정리)
3. 라이선스 고려: 원문 과다 인용 금지 (snippet 최대 50자)
4. source_ref 명시로 출처 추적 가능
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .types import Evidence, Candidate, ClassificationResult


@dataclass
class ExplanationItem:
    """설명 항목 (단일 근거)"""
    kind: str  # card_keyword, include_rule, exclude_rule, example, legal_gate, fact_check
    snippet: str  # 짧은 근거 텍스트 (최대 50자)
    source_ref: str  # 출처 (hs4, note_id, chunk_id 등)
    weight: float = 1.0  # 중요도
    detail: str = ""  # 추가 설명 (선택적)


@dataclass
class Explanation:
    """분류 설명"""
    hs4: str
    decision: str  # AUTO, ASK, REVIEW, ABSTAIN
    confidence: float

    # 핵심 근거 (2~3개)
    key_evidence: List[ExplanationItem] = field(default_factory=list)

    # 추가 정보 (선택적)
    supporting_evidence_count: int = 0
    conflicting_evidence_count: int = 0
    questions: List[str] = field(default_factory=list)  # ASK인 경우

    def to_dict(self) -> Dict[str, Any]:
        return {
            'hs4': self.hs4,
            'decision': self.decision,
            'confidence': round(self.confidence, 4),
            'key_evidence': [
                {
                    'kind': item.kind,
                    'snippet': item.snippet,
                    'source_ref': item.source_ref,
                    'weight': round(item.weight, 3),
                    'detail': item.detail,
                }
                for item in self.key_evidence
            ],
            'supporting_evidence_count': self.supporting_evidence_count,
            'conflicting_evidence_count': self.conflicting_evidence_count,
            'questions': self.questions,
        }

    def to_text(self, lang: str = 'ko') -> str:
        """
        텍스트 형식으로 설명 생성

        Args:
            lang: 언어 ('ko' 또는 'en')

        Returns:
            설명 텍스트
        """
        if lang == 'ko':
            return self._to_text_ko()
        else:
            return self._to_text_en()

    def _to_text_ko(self) -> str:
        """한국어 설명"""
        lines = []

        # 헤더
        decision_text = {
            'AUTO': '자동 분류',
            'ASK': '추가 정보 필요',
            'REVIEW': '전문가 검토 필요',
            'ABSTAIN': '분류 불가'
        }
        lines.append(f"분류: {self.hs4} ({decision_text.get(self.decision, self.decision)})")
        lines.append(f"신뢰도: {self.confidence:.2%}")

        # 핵심 근거
        if self.key_evidence:
            lines.append("\n주요 근거:")
            for i, item in enumerate(self.key_evidence, 1):
                kind_text = {
                    'card_keyword': '카드 키워드',
                    'include_rule': '포함 규칙',
                    'exclude_rule': '제외 규칙',
                    'example': '예시',
                    'legal_gate': '법적 게이트',
                    'fact_check': '사실 검증',
                }
                lines.append(f"  {i}. [{kind_text.get(item.kind, item.kind)}] {item.snippet}")
                if item.detail:
                    lines.append(f"     → {item.detail}")
                lines.append(f"     출처: {item.source_ref}")

        # 추가 정보
        if self.supporting_evidence_count > len(self.key_evidence):
            additional = self.supporting_evidence_count - len(self.key_evidence)
            lines.append(f"\n+ {additional}개의 추가 지지 근거")

        if self.conflicting_evidence_count > 0:
            lines.append(f"⚠ {self.conflicting_evidence_count}개의 상충 근거 발견")

        # 질문 (ASK인 경우)
        if self.questions:
            lines.append("\n확인이 필요한 사항:")
            for i, q in enumerate(self.questions, 1):
                lines.append(f"  {i}. {q}")

        return '\n'.join(lines)

    def _to_text_en(self) -> str:
        """영어 설명"""
        lines = []

        lines.append(f"Classification: {self.hs4} ({self.decision})")
        lines.append(f"Confidence: {self.confidence:.2%}")

        if self.key_evidence:
            lines.append("\nKey Evidence:")
            for i, item in enumerate(self.key_evidence, 1):
                lines.append(f"  {i}. [{item.kind}] {item.snippet}")
                if item.detail:
                    lines.append(f"     → {item.detail}")
                lines.append(f"     Source: {item.source_ref}")

        if self.supporting_evidence_count > len(self.key_evidence):
            additional = self.supporting_evidence_count - len(self.key_evidence)
            lines.append(f"\n+ {additional} additional supporting evidence")

        if self.conflicting_evidence_count > 0:
            lines.append(f"⚠ {self.conflicting_evidence_count} conflicting evidence found")

        if self.questions:
            lines.append("\nQuestions to clarify:")
            for i, q in enumerate(self.questions, 1):
                lines.append(f"  {i}. {q}")

        return '\n'.join(lines)


class ExplanationGenerator:
    """
    분류 결과 설명 생성기

    원칙:
    - 결론 변경 금지 (파이프라인 결과를 그대로 설명)
    - Evidence 정리만 수행
    - 라이선스 고려: snippet 최대 50자
    - source_ref 명시로 출처 추적
    """

    def __init__(self, max_key_evidence: int = 3, max_snippet_length: int = 50):
        """
        Args:
            max_key_evidence: 최대 핵심 근거 개수
            max_snippet_length: snippet 최대 길이
        """
        self.max_key_evidence = max_key_evidence
        self.max_snippet_length = max_snippet_length

    def generate(
        self,
        result: ClassificationResult,
        top1_only: bool = True
    ) -> List[Explanation]:
        """
        분류 결과에서 설명 생성

        Args:
            result: ClassificationResult
            top1_only: Top-1만 설명 (False면 Top-K 모두)

        Returns:
            Explanation 리스트
        """
        explanations = []

        # Top-1 또는 Top-K
        candidates = [result.topk[0]] if top1_only and result.topk else result.topk

        for cand in candidates:
            explanation = self._generate_for_candidate(result, cand)
            explanations.append(explanation)

        return explanations

    def _generate_for_candidate(
        self,
        result: ClassificationResult,
        candidate: Candidate
    ) -> Explanation:
        """
        단일 후보에 대한 설명 생성

        Args:
            result: ClassificationResult
            candidate: Candidate

        Returns:
            Explanation
        """
        # 1. 기본 정보
        explanation = Explanation(
            hs4=candidate.hs4,
            decision=result.decision.status,
            confidence=result.decision.confidence
        )

        # 2. Evidence 수집 및 정리
        all_evidence = candidate.evidence
        supporting = [ev for ev in all_evidence if ev.weight > 0]
        conflicting = [ev for ev in all_evidence if ev.weight < 0]

        explanation.supporting_evidence_count = len(supporting)
        explanation.conflicting_evidence_count = len(conflicting)

        # 3. 핵심 근거 선택 (중요도 기준 Top-K)
        # supporting을 weight 내림차순 정렬
        supporting_sorted = sorted(supporting, key=lambda x: -x.weight)

        for ev in supporting_sorted[:self.max_key_evidence]:
            # snippet 생성 (최대 길이 제한)
            snippet = self._truncate_text(ev.text, self.max_snippet_length)

            # detail 생성 (선택적)
            detail = ""
            if ev.kind == 'legal_gate':
                detail = f"법적 게이트 점수: {ev.weight:.2f}"
            elif ev.kind == 'fact_check':
                detail = "사실 충족 확인"

            explanation.key_evidence.append(ExplanationItem(
                kind=ev.kind,
                snippet=snippet,
                source_ref=ev.source_id,
                weight=ev.weight,
                detail=detail
            ))

        # 4. 질문 추가 (ASK인 경우)
        if result.decision.status == 'ASK' and result.questions:
            explanation.questions = [
                q['question'] if isinstance(q, dict) else str(q)
                for q in result.questions[:3]  # 최대 3개
            ]

        return explanation

    def _truncate_text(self, text: str, max_length: int) -> str:
        """
        텍스트 자르기 (라이선스 고려)

        Args:
            text: 원문
            max_length: 최대 길이

        Returns:
            잘린 텍스트
        """
        if len(text) <= max_length:
            return text

        # 단어 경계에서 자르기
        truncated = text[:max_length]

        # 마지막 공백 찾기
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.7:  # 70% 이상이면 공백에서 자름
            truncated = truncated[:last_space]

        return truncated + "..."

    def generate_batch(
        self,
        results: List[ClassificationResult],
        top1_only: bool = True
    ) -> List[List[Explanation]]:
        """
        배치 설명 생성

        Args:
            results: ClassificationResult 리스트
            top1_only: Top-1만 설명

        Returns:
            Explanation 리스트의 리스트
        """
        return [self.generate(result, top1_only) for result in results]


def explain_classification(
    result: ClassificationResult,
    max_key_evidence: int = 3,
    lang: str = 'ko'
) -> str:
    """
    분류 결과 설명 편의 함수

    Args:
        result: ClassificationResult
        max_key_evidence: 최대 핵심 근거 개수
        lang: 언어 ('ko' 또는 'en')

    Returns:
        설명 텍스트
    """
    generator = ExplanationGenerator(max_key_evidence=max_key_evidence)
    explanations = generator.generate(result, top1_only=True)

    if not explanations:
        return "설명을 생성할 수 없습니다." if lang == 'ko' else "No explanation available."

    return explanations[0].to_text(lang)


def batch_explain(
    results: List[ClassificationResult],
    output_path: str,
    lang: str = 'ko'
) -> None:
    """
    배치 설명 생성 및 저장

    Args:
        results: ClassificationResult 리스트
        output_path: 출력 파일 경로
        lang: 언어
    """
    generator = ExplanationGenerator()

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            explanations = generator.generate(result, top1_only=True)

            if explanations:
                explanation = explanations[0]

                # JSON 형식
                f.write(json.dumps({
                    'input_text': result.input_text,
                    'explanation': explanation.to_dict(),
                    'text': explanation.to_text(lang)
                }, ensure_ascii=False, indent=2))
                f.write('\n' + '='*80 + '\n')

    print(f"설명 저장: {output_path}")
