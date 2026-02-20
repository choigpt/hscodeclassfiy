"""
서브헤딩 분류 모듈 (GRI 6)

HS4 확정 후 → HS6 서브헤딩 결정:
1. 해당 HS4 산하 모든 HS6 후보 로드
2. 키워드 매칭 + 소호 주규정 적용 + 재질 구성비 + 판결 선례
3. Top-3 HS6 후보 반환
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .utils_text import normalize, tokenize


HS6_KB_PATH = "kb/structured/hs6_subheadings.jsonl"
NORMALIZED_CASES_PATH = "data/ruling_cases/normalized_cases_rule.jsonl"


@dataclass
class SubheadingCandidate:
    """HS6 서브헤딩 후보"""
    hs6: str
    hs4: str
    title_ko: str = ""
    score: float = 0.0
    matched_keywords: List[str] = field(default_factory=list)
    note_score: float = 0.0
    material_score: float = 0.0
    case_score: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "hs6": self.hs6,
            "hs4": self.hs4,
            "title_ko": self.title_ko,
            "score": round(self.score, 4),
            "matched_keywords": self.matched_keywords[:5],
            "note_score": round(self.note_score, 4),
            "material_score": round(self.material_score, 4),
            "case_score": round(self.case_score, 4),
        }


class SubheadingResolver:
    """HS4 → HS6 서브헤딩 해소"""

    def __init__(
        self,
        hs6_kb_path: str = HS6_KB_PATH,
        cases_path: str = NORMALIZED_CASES_PATH,
    ):
        # hs4 -> [hs6_entry, ...] 인덱스
        self.hs4_to_hs6: Dict[str, List[Dict]] = defaultdict(list)
        # hs6 -> entry
        self.hs6_index: Dict[str, Dict] = {}
        # 정규화된 케이스
        self.normalized_cases: List[Dict] = []

        self._load_hs6_kb(hs6_kb_path)
        self._load_cases(cases_path)

    def _load_hs6_kb(self, path: str):
        """HS6 KB 로드"""
        kb_file = Path(path)
        if not kb_file.exists():
            print(f"[SubheadingResolver] Warning: {kb_file} not found")
            return

        count = 0
        with open(kb_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    hs6 = entry.get('hs6', '')
                    hs4 = entry.get('hs4', '')
                    if hs6 and hs4:
                        self.hs4_to_hs6[hs4].append(entry)
                        self.hs6_index[hs6] = entry
                        count += 1
                except json.JSONDecodeError:
                    continue

        print(f"[SubheadingResolver] HS6 KB: {count} entries, {len(self.hs4_to_hs6)} HS4 groups")

    def _load_cases(self, path: str):
        """정규화된 케이스 로드"""
        cases_file = Path(path)
        if not cases_file.exists():
            return

        with open(cases_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    case = json.loads(line)
                    self.normalized_cases.append(case)
                except json.JSONDecodeError:
                    continue

        print(f"[SubheadingResolver] Cases: {len(self.normalized_cases)}")

    def resolve(
        self,
        input_data,
        hs4: str,
        input_attrs=None,
        topk: int = 3,
    ) -> List[SubheadingCandidate]:
        """
        HS4 확정 후 HS6 후보 반환

        Args:
            input_data: ClassificationInput 또는 str
            hs4: 확정된 HS4 코드
            input_attrs: 8축 속성 (있으면 재질 매칭에 활용)
            topk: 반환할 상위 후보 수

        Returns:
            Top-K SubheadingCandidate 리스트
        """
        hs6_entries = self.hs4_to_hs6.get(hs4, [])
        if not hs6_entries:
            return []

        # 입력 텍스트 추출
        if isinstance(input_data, str):
            text = input_data
        else:
            text = getattr(input_data, 'text', '') or getattr(input_data, 'to_enriched_text', lambda: '')()

        candidates = []
        for entry in hs6_entries:
            cand = SubheadingCandidate(
                hs6=entry['hs6'],
                hs4=hs4,
                title_ko=entry.get('title_ko', ''),
            )

            # 1. 키워드 매칭
            kw_score, matched = self._match_keywords(text, entry)
            cand.matched_keywords = matched

            # 2. 소호 주규정
            note_score = self._apply_subheading_notes(text, entry)
            cand.note_score = note_score

            # 3. 재질 구성비
            mat_score = self._check_material_composition(input_data, input_attrs, entry)
            cand.material_score = mat_score

            # 4. 판결 선례
            case_score = self._lookup_case_precedent(text, hs4, entry)
            cand.case_score = case_score

            # 총점
            cand.score = kw_score + note_score * 2.0 + mat_score * 1.5 + case_score * 1.0

            candidates.append(cand)

        # 정렬
        candidates.sort(key=lambda c: c.score, reverse=True)

        return candidates[:topk]

    def _match_keywords(self, text: str, hs6_entry: Dict) -> Tuple[float, List[str]]:
        """키워드 매칭 점수"""
        keywords = hs6_entry.get('keywords', [])
        title = hs6_entry.get('title_ko', '')

        norm_text = normalize(text).lower()
        text_tokens = set(tokenize(text, remove_stopwords=True))

        matched = []
        score = 0.0

        # 키워드 직접 매칭
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in norm_text:
                matched.append(kw)
                score += 1.0
            elif kw_lower in text_tokens:
                matched.append(kw)
                score += 0.5

        # title 토큰 매칭
        if title:
            title_tokens = set(tokenize(title, remove_stopwords=True))
            overlap = text_tokens & title_tokens
            if overlap:
                score += len(overlap) * 0.3
                matched.extend(list(overlap)[:3])

        return score, matched

    def _apply_subheading_notes(self, text: str, hs6_entry: Dict) -> float:
        """소호 주규정 적용"""
        notes = hs6_entry.get('subheading_notes', [])
        if not notes:
            return 0.0

        norm_text = normalize(text).lower()
        score = 0.0

        for note in notes:
            note_norm = normalize(note).lower()
            # 간단한 토큰 오버랩
            note_tokens = set(note_norm.split())
            text_tokens = set(norm_text.split())
            if note_tokens and text_tokens:
                overlap = note_tokens & text_tokens
                if overlap:
                    score += len(overlap) / len(note_tokens) * 0.5

        return min(1.0, score)

    def _check_material_composition(
        self, input_data, input_attrs, hs6_entry: Dict
    ) -> float:
        """재질 구성비 매칭"""
        if input_attrs is None:
            return 0.0

        # HS6 entry의 키워드에서 재질 관련 추출
        keywords = hs6_entry.get('keywords', [])
        title = hs6_entry.get('title_ko', '')
        full_text = ' '.join(keywords) + ' ' + title

        # 입력의 material axis와 비교
        input_materials = set(input_attrs.material.values) if hasattr(input_attrs, 'material') else set()
        if not input_materials:
            return 0.0

        # 간단한 매칭: 입력 재질이 HS6 키워드/title에 있는지
        full_lower = full_text.lower()
        matched = sum(1 for m in input_materials if m.lower() in full_lower)

        return min(1.0, matched / max(len(input_materials), 1))

    def _lookup_case_precedent(
        self, text: str, hs4: str, hs6_entry: Dict
    ) -> float:
        """판결 선례 조회"""
        target_hs6 = hs6_entry.get('hs6', '')
        if not self.normalized_cases:
            return 0.0

        # 해당 HS6로 판결된 케이스 수
        case_count = hs6_entry.get('case_count', 0)
        if case_count > 0:
            return min(1.0, case_count / 20.0)  # 20건 이상이면 1.0

        return 0.0
