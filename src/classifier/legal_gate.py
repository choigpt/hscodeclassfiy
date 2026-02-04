"""
LegalGate (GRI 1 Gate) - 법적 게이팅 시스템

GRI 1 (호 용어 + 관련 부/류 주규정)을 기반으로 후보를 필터링:
1. heading_term_match: 호 용어가 입력과 매칭되는지
2. note_include_support: 포함 주규정이 입력을 지지하는지
3. note_exclude_conflict: 제외 주규정이 입력과 충돌하는지 (hard filter)
4. note_redirect: 다른 호로 리다이렉트하는지 (hard filter)

Hard filtering:
- exclude 주규정에 명시적으로 걸리면 해당 후보 제거 (-inf)
- redirect 주규정이 다른 호를 명시하면 해당 후보 제거하고 대상 호 추가
- 입력 정보 불충분으로 판단할 수 없으면 ASK 라우팅 (보류)
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

from .types import Candidate, Evidence
from .notes_loader import NotesLoader, get_notes_loader, TariffNote
from .utils_text import normalize, simple_contains, fuzzy_match, token_overlap


@dataclass
class LegalGateResult:
    """LegalGate 통과 결과"""
    hs4: str
    passed: bool  # True면 통과, False면 hard exclude

    # 점수
    heading_term_score: float = 0.0
    include_support_score: float = 0.0
    exclude_conflict_score: float = 0.0  # 음수: 충돌 강도
    redirect_penalty: float = 0.0  # 음수: 리다이렉트 페널티

    # 증거
    evidence: List[Evidence] = field(default_factory=list)

    # 리다이렉트 대상
    redirect_to: Optional[str] = None

    # 불확실성 플래그
    insufficient_info: bool = False
    ask_questions: List[str] = field(default_factory=list)

    def total_score(self) -> float:
        """총점 (exclude/redirect가 강하면 음수)"""
        return (
            self.heading_term_score +
            self.include_support_score +
            self.exclude_conflict_score +
            self.redirect_penalty
        )

    def should_hard_exclude(self) -> bool:
        """Hard exclude 여부 (명시적 제외 또는 강한 리다이렉트)"""
        # exclude 충돌이 강하면 제거
        if self.exclude_conflict_score < -0.7:
            return True
        # 리다이렉트가 명확하면 제거
        if self.redirect_to and self.redirect_penalty < -0.8:
            return True
        return False


class LegalGate:
    """GRI 1 기반 법적 게이팅"""

    def __init__(self, notes_loader: Optional[NotesLoader] = None):
        self.notes_loader = notes_loader or get_notes_loader()

        # KB 카드 데이터 (호 용어 매칭용)
        self.heading_terms = self._load_heading_terms()
        print(f"[LegalGate] Heading terms loaded: {len(self.heading_terms)} HS4")

    def _load_heading_terms(self) -> Dict[str, List[str]]:
        """HS4 호 용어 로드 (title_ko + scope 토큰)"""
        from pathlib import Path
        import json

        heading_terms = {}
        cards_path = Path("kb/structured/hs4_cards.jsonl")

        if not cards_path.exists():
            return heading_terms

        with open(cards_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    card = json.loads(line)
                    hs4 = card.get('hs4')
                    if not hs4:
                        continue

                    terms = []

                    # Title (heading)
                    title = card.get('title_ko', '')
                    if title:
                        # 정규화 및 토큰화
                        title_norm = normalize(title)
                        title_tokens = [t for t in title_norm.split() if len(t) >= 2]
                        terms.extend(title_tokens)

                    # Scope (keywords)
                    scope = card.get('scope', [])
                    if scope:
                        for keyword in scope:
                            kw_norm = normalize(keyword)
                            kw_tokens = [t for t in kw_norm.split() if len(t) >= 2]
                            terms.extend(kw_tokens)

                    # 중복 제거
                    if terms:
                        heading_terms[hs4] = list(set(terms))

                except Exception:
                    continue

        return heading_terms

    def apply(
        self,
        input_text: str,
        candidates: List[Candidate]
    ) -> Tuple[List[Candidate], List[str], Dict[str, any]]:
        """
        LegalGate 적용

        Returns:
            - filtered_candidates: 통과한 후보 (hard exclude 제거됨)
            - redirect_hs4s: 리다이렉트로 추가할 HS4 목록
            - debug: 디버그 정보
        """
        input_norm = normalize(input_text)

        results: Dict[str, LegalGateResult] = {}
        redirect_targets: Set[str] = set()

        # 각 후보에 대해 LegalGate 평가
        for cand in candidates:
            result = self._evaluate_candidate(input_text, input_norm, cand.hs4)
            results[cand.hs4] = result

            # 리다이렉트 대상 수집
            if result.redirect_to:
                redirect_targets.add(result.redirect_to)

        # Hard exclude 적용
        passed_candidates = []
        excluded_hs4s = []
        exclude_reasons = {}

        for cand in candidates:
            result = results[cand.hs4]

            if result.should_hard_exclude():
                excluded_hs4s.append(cand.hs4)

                # Evidence 추가 (왜 제외되었는지)
                exclude_evidence = Evidence(
                    kind="legal_gate_exclude",
                    source_id=cand.hs4,
                    text=f"GRI 1 법적 게이팅: hard exclude (점수: {result.total_score():.2f})",
                    weight=-1.0,
                    meta={
                        'exclude_conflict': result.exclude_conflict_score,
                        'redirect_penalty': result.redirect_penalty,
                        'redirect_to': result.redirect_to,
                    }
                )
                cand.evidence.insert(0, exclude_evidence)  # 최상단에 추가
                cand.evidence.extend(result.evidence)

                # 제외 이유 기록
                reasons = []
                if result.exclude_conflict_score < -0.7:
                    reasons.append(f"제외 주규정 충돌 ({result.exclude_conflict_score:.2f})")
                if result.redirect_to:
                    reasons.append(f"제{result.redirect_to}호로 리다이렉트")
                exclude_reasons[cand.hs4] = ", ".join(reasons)

                continue

            # 통과한 후보는 evidence 추가
            if result.evidence:
                # LegalGate 통과 evidence
                pass_evidence = Evidence(
                    kind="legal_gate_pass",
                    source_id=cand.hs4,
                    text=f"GRI 1 법적 게이팅 통과 (점수: {result.total_score():.2f})",
                    weight=result.total_score(),
                    meta={
                        'heading_term': result.heading_term_score,
                        'include_support': result.include_support_score,
                        'exclude_conflict': result.exclude_conflict_score,
                    }
                )
                cand.evidence.insert(0, pass_evidence)  # 최상단에 추가
                cand.evidence.extend(result.evidence)

            # LegalGate 점수를 기존 점수에 반영
            # (reranker에서 사용할 수 있도록)
            cand.features['legal_gate_score'] = result.total_score()
            cand.features['heading_term_score'] = result.heading_term_score
            cand.features['include_support'] = result.include_support_score
            cand.features['exclude_conflict'] = result.exclude_conflict_score

            passed_candidates.append(cand)

        # 리다이렉트 대상 HS4 리스트 (기존 후보에 없는 것만)
        existing_hs4s = {c.hs4 for c in passed_candidates}
        new_redirect_hs4s = [h for h in redirect_targets if h not in existing_hs4s]

        # Per-candidate results를 debug에 포함 (ranker 학습용)
        results_dict = {}
        for hs4, result in results.items():
            results_dict[hs4] = {
                'passed': result.passed,
                'heading_term_score': result.heading_term_score,
                'include_support_score': result.include_support_score,
                'exclude_conflict_score': result.exclude_conflict_score,
                'redirect_penalty': result.redirect_penalty,
                'total_score': result.total_score(),
            }

        debug = {
            'legal_gate_applied': True,
            'total_evaluated': len(candidates),
            'passed': len(passed_candidates),
            'excluded': len(excluded_hs4s),
            'excluded_hs4s': excluded_hs4s,
            'exclude_reasons': exclude_reasons,
            'redirects_added': len(new_redirect_hs4s),
            'redirect_hs4s': new_redirect_hs4s,
            'pass_rate': len(passed_candidates) / len(candidates) if candidates else 0,
            'results': results_dict,  # Per-candidate results 추가
        }

        return passed_candidates, new_redirect_hs4s, debug

    def _evaluate_candidate(
        self,
        input_text: str,
        input_norm: str,
        hs4: str
    ) -> LegalGateResult:
        """단일 후보에 대한 LegalGate 평가"""

        result = LegalGateResult(hs4=hs4, passed=True)

        # 해당 HS4의 주규정 가져오기
        note_index = self.notes_loader.get_notes_for_hs4(hs4)
        if not note_index:
            # 주규정이 없으면 중립 (통과)
            return result

        # 1. Heading term match (호 용어 매칭)
        if hs4 in self.heading_terms:
            terms = self.heading_terms[hs4]
            matched_terms = []

            for term in terms:
                if simple_contains(input_norm, term):
                    # 직접 매칭: 0.1점
                    result.heading_term_score += 0.1
                    matched_terms.append(term)
                else:
                    # Fuzzy 매칭: 0.05점
                    match_result, _ = fuzzy_match(input_norm, term)
                    if match_result:
                        result.heading_term_score += 0.05
                        matched_terms.append(f"~{term}")

            # 점수 클리핑 (최대 1.0)
            result.heading_term_score = min(result.heading_term_score, 1.0)

            # 증거 추가
            if matched_terms:
                result.evidence.append(Evidence(
                    kind='legal_heading_term',
                    source_id=hs4,
                    text=f"호 용어 매칭: {', '.join(matched_terms[:5])}",
                    weight=result.heading_term_score
                ))

        # 2. Include support (포함 주규정 지지도)
        include_notes = note_index.include_notes()
        for note in include_notes:
            score, evidence = self._match_note_content(input_text, input_norm, note, positive=True)
            if score > 0:
                result.include_support_score += score
                result.evidence.append(evidence)

        # 3. Exclude conflict (제외 주규정 충돌)
        exclude_notes = note_index.exclude_notes()
        for note in exclude_notes:
            score, evidence = self._match_note_content(input_text, input_norm, note, positive=False)
            if score > 0:
                # exclude 매칭은 음수로
                result.exclude_conflict_score -= score
                result.evidence.append(evidence)

        # 4. Redirect (리다이렉트 주규정)
        redirect_notes = note_index.redirect_notes()
        for note in redirect_notes:
            score, evidence = self._match_note_content(input_text, input_norm, note, positive=False)
            if score > 0:
                result.redirect_penalty -= score
                result.redirect_to = note.redirect_to
                result.evidence.append(evidence)

        # Passed 여부 판단
        result.passed = not result.should_hard_exclude()

        return result

    def _match_note_content(
        self,
        input_text: str,
        input_norm: str,
        note: TariffNote,
        positive: bool
    ) -> Tuple[float, Evidence]:
        """
        주규정 내용과 입력 매칭

        Returns:
            - score: 매칭 점수 (0~1)
            - evidence: 증거 객체
        """
        note_content_norm = normalize(note.note_content)

        score = 0.0
        matched_keywords = []

        # 키워드 매칭
        for kw in note.keywords:
            kw_norm = normalize(kw)
            if simple_contains(input_norm, kw_norm):
                score += 0.3
                matched_keywords.append(kw)
            else:
                # fuzzy_match returns (bool, method)
                match_result, _ = fuzzy_match(input_norm, kw_norm)
                if match_result:
                    score += 0.15
                    matched_keywords.append(f"~{kw}")

        # Token overlap (returns (bool, list))
        # 간단한 토큰 교집합 점수 계산
        input_tokens = set(input_norm.split())
        note_tokens = set(note_content_norm.split())
        if input_tokens and note_tokens:
            overlap_ratio = len(input_tokens & note_tokens) / len(input_tokens)
            score += overlap_ratio * 0.2

        # 점수 클리핑
        score = min(score, 1.0)

        # Evidence 생성
        evidence_kind = f"note_{note.note_type}"  # note_include, note_exclude, note_redirect
        evidence_text = note.note_content[:160]
        if matched_keywords:
            evidence_text = f"[{', '.join(matched_keywords[:3])}] {evidence_text}"

        weight = score if positive else -score

        evidence = Evidence(
            kind=evidence_kind,
            source_id=f"{note.level}_{note.chapter_num or 0}_{note.note_number}",
            text=evidence_text,
            weight=weight,
            meta={
                'note_level': note.level,
                'note_type': note.note_type,
                'matched_keywords': matched_keywords,
                'score': score,
            }
        )

        return score, evidence

    def _check_insufficient_info(
        self,
        input_text: str,
        note: TariffNote
    ) -> Tuple[bool, List[str]]:
        """
        입력 정보가 주규정을 판단하기에 충분한지 확인

        Returns:
            - insufficient: True면 정보 부족
            - questions: 필요한 질문 리스트
        """
        # TODO: 주규정 키워드가 입력에 전혀 없고, 판단에 필수적이면 질문 생성
        # 예: "제외: 전기로 작동하는 것" → "이 제품은 전기로 작동합니까?"

        questions = []
        insufficient = False

        # 간단한 휴리스틱: 주규정에 조건이 있는데 입력에 정보가 없으면
        conditional_patterns = [
            r'경우',
            r'때에는',
            r'한정',
            r'only if',
            r'provided that',
            r'when',
        ]

        has_condition = any(re.search(p, note.note_content, re.IGNORECASE)
                           for p in conditional_patterns)

        if has_condition:
            # 키워드가 하나도 매칭 안 되면 정보 부족 가능성
            input_norm = normalize(input_text)
            matched = sum(1 for kw in note.keywords
                         if normalize(kw) in input_norm)

            if matched == 0 and len(note.keywords) > 0:
                insufficient = True
                # 주규정 내용 기반 질문 생성
                questions.append(f"주규정 확인 필요: {note.note_content[:100]}")

        return insufficient, questions


def apply_legal_gate(
    input_text: str,
    candidates: List[Candidate],
    notes_loader: Optional[NotesLoader] = None
) -> Tuple[List[Candidate], List[str], Dict]:
    """LegalGate 편의 함수"""
    gate = LegalGate(notes_loader=notes_loader)
    return gate.apply(input_text, candidates)
