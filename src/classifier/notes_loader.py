"""
관세율표 주규정 데이터 로더 및 인덱서

tariff_notes_clean.json을 로드하고 HS 코드별로 인덱싱:
- 부주(Section notes): 특정 부의 모든 류에 적용
- 류주(Chapter notes): 특정 류의 모든 호에 적용
- 소호주(Subheading notes): 특정 류 또는 호에 적용

주규정 타입:
- Include: "이 호에는 ... 포함한다" → 긍정적 지지
- Exclude: "이 호에서 ... 제외한다" → hard 필터링
- Redirect: "... 제XX호로 분류한다" → 다른 호로 리다이렉트
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TariffNote:
    """관세율표 주규정"""
    level: str  # 'section', 'chapter', 'subheading'
    section_num: Optional[int]
    chapter_num: Optional[int]
    note_number: str
    note_content: str

    # 파싱된 정보
    note_type: str = ""  # 'include', 'exclude', 'redirect', 'definition', 'general'
    keywords: List[str] = field(default_factory=list)
    redirect_to: Optional[str] = None  # 제XXXX호
    confidence: float = 1.0


@dataclass
class NoteIndex:
    """HS4별 주규정 인덱스"""
    hs4: str
    section_notes: List[TariffNote] = field(default_factory=list)
    chapter_notes: List[TariffNote] = field(default_factory=list)
    subheading_notes: List[TariffNote] = field(default_factory=list)

    def all_notes(self) -> List[TariffNote]:
        """모든 주규정 (부주 → 류주 → 소호주 순)"""
        return self.section_notes + self.chapter_notes + self.subheading_notes

    def include_notes(self) -> List[TariffNote]:
        """포함 주규정만"""
        return [n for n in self.all_notes() if n.note_type == 'include']

    def exclude_notes(self) -> List[TariffNote]:
        """제외 주규정만"""
        return [n for n in self.all_notes() if n.note_type == 'exclude']

    def redirect_notes(self) -> List[TariffNote]:
        """리다이렉트 주규정만"""
        return [n for n in self.all_notes() if n.note_type == 'redirect']


class NotesLoader:
    """주규정 로더 및 인덱서"""

    def __init__(self, notes_path: str = "data/tariff_notes_clean.json"):
        self.notes_path = Path(notes_path)
        self.raw_notes: List[Dict] = []
        self.parsed_notes: List[TariffNote] = []

        # HS4별 인덱스
        self.hs4_index: Dict[str, NoteIndex] = {}

        # 류별 인덱스 (HS2 → 주규정)
        self.chapter_index: Dict[int, List[TariffNote]] = defaultdict(list)

        # 부별 인덱스
        self.section_index: Dict[int, List[TariffNote]] = defaultdict(list)

        self._load()
        self._parse()
        self._build_index()

    def _load(self):
        """JSON 파일 로드"""
        if not self.notes_path.exists():
            print(f"[NotesLoader] Warning: {self.notes_path} not found")
            return

        with open(self.notes_path, 'r', encoding='utf-8') as f:
            self.raw_notes = json.load(f)

        print(f"[NotesLoader] Loaded {len(self.raw_notes)} notes from {self.notes_path}")

    def _parse(self):
        """주규정 파싱 (타입 분류 + 키워드 추출)"""
        for note_dict in self.raw_notes:
            note = TariffNote(
                level=note_dict['level'],
                section_num=note_dict.get('section_num'),
                chapter_num=note_dict.get('chapter_num'),
                note_number=note_dict['note_number'],
                note_content=note_dict['note_content']
            )

            # 주규정 타입 분류
            note.note_type = self._classify_note_type(note.note_content)

            # 키워드 추출
            note.keywords = self._extract_keywords(note.note_content)

            # 리다이렉트 대상 추출
            if note.note_type == 'redirect':
                note.redirect_to = self._extract_redirect_target(note.note_content)

            self.parsed_notes.append(note)

        print(f"[NotesLoader] Parsed {len(self.parsed_notes)} notes:")
        type_counts = defaultdict(int)
        for n in self.parsed_notes:
            type_counts[n.note_type] += 1
        for ntype, count in sorted(type_counts.items()):
            print(f"  - {ntype}: {count}")

    def _classify_note_type(self, content: str) -> str:
        """주규정 타입 분류"""
        content_lower = content.lower()

        # Exclude 패턴 (가장 강력)
        exclude_patterns = [
            r'제외한다',
            r'포함하지\s*않는다',
            r'분류하지\s*않는다',
            r'해당하지\s*않는다',
            r'제외된다',
            r'적용하지\s*않는다',
            r'is\s+excluded',
            r'does\s+not\s+include',
            r'excluding',
            r'except',
        ]
        for pattern in exclude_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return 'exclude'

        # Redirect 패턴 (다른 호로 분류 지시)
        redirect_patterns = [
            r'제\d{4}호[에로]?\s*분류',
            r'제\d{2}류[에로]?\s*분류',
            r'제\d+호\s*참조',
            r'classify\s+(?:in|under|to)\s+heading\s+\d{4}',
            r'refer\s+to\s+heading',
        ]
        for pattern in redirect_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return 'redirect'

        # Include 패턴
        include_patterns = [
            r'포함(?:한다|된다)',
            r'이\s*호에는',
            r'여기에는',
            r'(?:includes?|including|comprise)',
            r'classified\s+(?:in|under)',
        ]
        for pattern in include_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return 'include'

        # Definition 패턴
        if re.search(r'(?:이란|라\s*함은|means?|defined\s+as)', content, re.IGNORECASE):
            return 'definition'

        return 'general'

    def _extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """주규정에서 핵심 키워드 추출"""
        keywords = []

        # 따옴표로 강조된 용어
        quoted = re.findall(r'["\']([^"\']{2,30})["\']', content)
        for match in quoted:
            term = next(t for t in match if t)
            if term:
                keywords.append(term.strip())

        # 명사구 패턴 (한글)
        # "다음의 것", "...으로 제조한 것" 등
        noun_phrases = re.findall(r'([가-힣]{2,15})\s*(?:의\s*것|으로\s*한\s*것|이?나\s*것)', content)
        keywords.extend([p.strip() for p in noun_phrases if len(p) >= 2])

        # 제XXXX호 참조
        ref_headings = re.findall(r'제(\d{4})호', content)
        keywords.extend([f'제{h}호' for h in ref_headings])

        # 중복 제거 및 길이 필터링
        seen = set()
        unique_kw = []
        for kw in keywords:
            if kw not in seen and 2 <= len(kw) <= 30:
                seen.add(kw)
                unique_kw.append(kw)

        return unique_kw[:max_keywords]

    def _extract_redirect_target(self, content: str) -> Optional[str]:
        """리다이렉트 대상 호 추출"""
        # 제XXXX호로 분류
        match = re.search(r'제(\d{4})호', content)
        if match:
            return match.group(1)

        # 제XX류로 분류
        match = re.search(r'제(\d{2})류', content)
        if match:
            return match.group(1) + '00'  # HS2 → HS4 변환

        return None

    def _build_index(self):
        """HS4별 인덱스 구축"""
        # HS 코드 범위: 0101 ~ 9999
        for hs4_int in range(101, 10000):
            hs4 = f"{hs4_int:04d}"
            hs2 = int(hs4[:2])

            index = NoteIndex(hs4=hs4)

            # 부주 매핑 (부 범위는 사전 정의 필요)
            section_num = self._get_section_for_chapter(hs2)
            if section_num:
                section_notes = [n for n in self.parsed_notes
                                if n.level == 'section' and n.section_num == section_num]
                index.section_notes = section_notes

            # 류주 매핑
            chapter_notes = [n for n in self.parsed_notes
                            if n.level == 'chapter' and n.chapter_num == hs2]
            index.chapter_notes = chapter_notes

            # 소호주 매핑 (류 단위)
            subheading_notes = [n for n in self.parsed_notes
                               if n.level == 'subheading' and n.chapter_num == hs2]
            index.subheading_notes = subheading_notes

            if index.all_notes():
                self.hs4_index[hs4] = index

        print(f"[NotesLoader] Built index for {len(self.hs4_index)} HS4 codes")

    def _get_section_for_chapter(self, chapter: int) -> Optional[int]:
        """류 번호 → 부 번호 매핑 (관세율표 구조)"""
        # 제1부: 제01-05류
        if 1 <= chapter <= 5:
            return 1
        # 제2부: 제06-14류
        if 6 <= chapter <= 14:
            return 2
        # 제3부: 제15류
        if chapter == 15:
            return 3
        # 제4부: 제16-24류
        if 16 <= chapter <= 24:
            return 4
        # 제5부: 제25-27류
        if 25 <= chapter <= 27:
            return 5
        # 제6부: 제28-38류
        if 28 <= chapter <= 38:
            return 6
        # 제7부: 제39-40류
        if 39 <= chapter <= 40:
            return 7
        # 제8부: 제41-43류
        if 41 <= chapter <= 43:
            return 8
        # 제9부: 제44-46류
        if 44 <= chapter <= 46:
            return 9
        # 제10부: 제47-49류
        if 47 <= chapter <= 49:
            return 10
        # 제11부: 제50-63류
        if 50 <= chapter <= 63:
            return 11
        # 제12부: 제64-67류
        if 64 <= chapter <= 67:
            return 12
        # 제13부: 제68-70류
        if 68 <= chapter <= 70:
            return 13
        # 제14부: 제71류
        if chapter == 71:
            return 14
        # 제15부: 제72-83류
        if 72 <= chapter <= 83:
            return 15
        # 제16부: 제84-85류
        if 84 <= chapter <= 85:
            return 16
        # 제17부: 제86-89류
        if 86 <= chapter <= 89:
            return 17
        # 제18부: 제90-92류
        if 90 <= chapter <= 92:
            return 18
        # 제19부: 제93류
        if chapter == 93:
            return 19
        # 제20부: 제94-96류
        if 94 <= chapter <= 96:
            return 20
        # 제21부: 제97류
        if chapter == 97:
            return 21
        # 제22부: 제98-99류
        if 98 <= chapter <= 99:
            return 22

        return None

    def get_notes_for_hs4(self, hs4: str) -> Optional[NoteIndex]:
        """HS4 코드에 해당하는 주규정 인덱스 반환"""
        return self.hs4_index.get(hs4)

    def search_notes(self, query: str, max_results: int = 10) -> List[Tuple[TariffNote, float]]:
        """주규정 검색 (키워드 매칭)"""
        query_lower = query.lower()
        results = []

        for note in self.parsed_notes:
            score = 0.0
            content_lower = note.note_content.lower()

            # 정확한 키워드 매칭
            for kw in note.keywords:
                if kw.lower() in query_lower:
                    score += 1.0

            # 부분 매칭
            if query_lower in content_lower:
                score += 0.5

            if score > 0:
                results.append((note, score))

        # 점수순 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]


# 싱글톤 인스턴스
_notes_loader: Optional[NotesLoader] = None


def get_notes_loader() -> NotesLoader:
    """NotesLoader 싱글톤 인스턴스 반환"""
    global _notes_loader
    if _notes_loader is None:
        _notes_loader = NotesLoader()
    return _notes_loader
