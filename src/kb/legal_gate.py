"""
GRI 1 Legal Gate - tariff notes based hard filtering.

Evaluates candidates against section/chapter/subheading notes:
- heading_term_match: heading terms vs input
- note_include_support: include notes support
- note_exclude_conflict: exclude notes hard filter
- note_redirect: redirect to another heading
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

from ..text import normalize, simple_contains, fuzzy_match


NOTES_PATH = "data/tariff_notes_clean.json"
CARDS_PATH = "kb/structured/hs4_cards.jsonl"


@dataclass
class TariffNote:
    level: str  # section, chapter, subheading
    section_num: Optional[int]
    chapter_num: Optional[int]
    note_number: str
    note_content: str
    note_type: str = ""  # include, exclude, redirect, definition, general
    keywords: List[str] = field(default_factory=list)
    redirect_to: Optional[str] = None


@dataclass
class LegalGateResult:
    hs4: str
    passed: bool
    heading_term_score: float = 0.0
    include_support_score: float = 0.0
    exclude_conflict_score: float = 0.0
    redirect_penalty: float = 0.0
    redirect_to: Optional[str] = None

    def total_score(self) -> float:
        return (self.heading_term_score + self.include_support_score
                + self.exclude_conflict_score + self.redirect_penalty)

    def should_hard_exclude(self) -> bool:
        if self.exclude_conflict_score < -0.7:
            return True
        if self.redirect_to and self.redirect_penalty < -0.8:
            return True
        return False


class LegalGate:
    """GRI 1 legal gating system."""

    def __init__(self, notes_path: str = NOTES_PATH, cards_path: str = CARDS_PATH):
        self.parsed_notes: List[TariffNote] = []
        self.heading_terms: Dict[str, List[str]] = {}
        self._load_notes(notes_path)
        self._load_heading_terms(cards_path)

    def _classify_note_type(self, content: str) -> str:
        for pattern in ['제외한다', '포함하지\\s*않는다', '분류하지\\s*않는다',
                        '해당하지\\s*않는다', 'is\\s+excluded', 'does\\s+not\\s+include']:
            if re.search(pattern, content, re.IGNORECASE):
                return 'exclude'
        for pattern in ['제\\d{4}호[에로]?\\s*분류', 'classify\\s+(?:in|under)\\s+heading']:
            if re.search(pattern, content, re.IGNORECASE):
                return 'redirect'
        for pattern in ['포함(?:한다|된다)', '이\\s*호에는', 'includes?']:
            if re.search(pattern, content, re.IGNORECASE):
                return 'include'
        if re.search(r'(?:이란|라\s*함은|means?)', content, re.IGNORECASE):
            return 'definition'
        return 'general'

    def _extract_keywords(self, content: str) -> List[str]:
        keywords = []
        quoted = re.findall(r'["\']([^"\']{2,30})["\']', content)
        keywords.extend([q.strip() for q in quoted if len(q.strip()) >= 2])
        ref_headings = re.findall(r'제(\d{4})호', content)
        keywords.extend([f'제{h}호' for h in ref_headings])
        seen = set()
        return [kw for kw in keywords if kw not in seen and not seen.add(kw)][:10]

    def _extract_redirect_target(self, content: str) -> Optional[str]:
        match = re.search(r'제(\d{4})호', content)
        if match:
            return match.group(1)
        match = re.search(r'제(\d{2})류', content)
        if match:
            return match.group(1) + '00'
        return None

    def _load_notes(self, path: str):
        notes_file = Path(path)
        if not notes_file.exists():
            print(f"[LegalGate] Warning: {notes_file} not found")
            return

        with open(notes_file, 'r', encoding='utf-8') as f:
            raw_notes = json.load(f)

        for nd in raw_notes:
            note = TariffNote(
                level=nd['level'],
                section_num=nd.get('section_num'),
                chapter_num=nd.get('chapter_num'),
                note_number=nd['note_number'],
                note_content=nd['note_content'],
            )
            note.note_type = self._classify_note_type(note.note_content)
            note.keywords = self._extract_keywords(note.note_content)
            if note.note_type == 'redirect':
                note.redirect_to = self._extract_redirect_target(note.note_content)
            self.parsed_notes.append(note)

        print(f"[LegalGate] Loaded {len(self.parsed_notes)} notes")

    def _load_heading_terms(self, path: str):
        cards_file = Path(path)
        if not cards_file.exists():
            return
        with open(cards_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    card = json.loads(line)
                    hs4 = card.get('hs4')
                    if not hs4:
                        continue
                    terms = []
                    title = card.get('title_ko', '')
                    if title:
                        terms.extend([t for t in normalize(title).split() if len(t) >= 2])
                    for kw in card.get('scope', []):
                        terms.extend([t for t in normalize(kw).split() if len(t) >= 2])
                    if terms:
                        self.heading_terms[hs4] = list(set(terms))
                except Exception:
                    continue

    def _get_section_for_chapter(self, chapter: int) -> Optional[int]:
        ranges = [
            (1, 5, 1), (6, 14, 2), (15, 15, 3), (16, 24, 4), (25, 27, 5),
            (28, 38, 6), (39, 40, 7), (41, 43, 8), (44, 46, 9), (47, 49, 10),
            (50, 63, 11), (64, 67, 12), (68, 70, 13), (71, 71, 14), (72, 83, 15),
            (84, 85, 16), (86, 89, 17), (90, 92, 18), (93, 93, 19), (94, 96, 20),
            (97, 97, 21), (98, 99, 22),
        ]
        for lo, hi, sec in ranges:
            if lo <= chapter <= hi:
                return sec
        return None

    def _get_notes_for_hs4(self, hs4: str) -> List[TariffNote]:
        hs2 = int(hs4[:2])
        section_num = self._get_section_for_chapter(hs2)
        result = []
        for n in self.parsed_notes:
            if n.level == 'section' and n.section_num == section_num:
                result.append(n)
            elif n.level == 'chapter' and n.chapter_num == hs2:
                result.append(n)
            elif n.level == 'subheading' and n.chapter_num == hs2:
                result.append(n)
        return result

    def _match_note(self, input_text: str, input_norm: str, note: TariffNote) -> float:
        score = 0.0
        for kw in note.keywords:
            if simple_contains(input_norm, normalize(kw)):
                score += 0.3
            elif fuzzy_match(input_norm, normalize(kw))[0]:
                score += 0.15
        note_norm = normalize(note.note_content)
        input_tokens = set(input_norm.split())
        note_tokens = set(note_norm.split())
        if input_tokens and note_tokens:
            score += len(input_tokens & note_tokens) / len(input_tokens) * 0.2
        return min(score, 1.0)

    def _evaluate(self, input_text: str, input_norm: str, hs4: str) -> LegalGateResult:
        result = LegalGateResult(hs4=hs4, passed=True)
        notes = self._get_notes_for_hs4(hs4)
        if not notes:
            return result

        # Heading term match
        if hs4 in self.heading_terms:
            for term in self.heading_terms[hs4]:
                if simple_contains(input_norm, term):
                    result.heading_term_score += 0.1
                elif fuzzy_match(input_norm, term)[0]:
                    result.heading_term_score += 0.05
            result.heading_term_score = min(result.heading_term_score, 1.0)

        for note in notes:
            score = self._match_note(input_text, input_norm, note)
            if score > 0:
                if note.note_type == 'include':
                    result.include_support_score += score
                elif note.note_type == 'exclude':
                    result.exclude_conflict_score -= score
                elif note.note_type == 'redirect':
                    result.redirect_penalty -= score
                    result.redirect_to = note.redirect_to

        result.passed = not result.should_hard_exclude()
        return result

    def apply(
        self,
        input_text: str,
        candidate_hs4s: List[str],
    ) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        Apply legal gate filtering.

        Returns:
            (passed_hs4s, redirect_hs4s, debug_dict)
        """
        input_norm = normalize(input_text)
        results: Dict[str, LegalGateResult] = {}
        redirect_targets: Set[str] = set()

        for hs4 in candidate_hs4s:
            r = self._evaluate(input_text, input_norm, hs4)
            results[hs4] = r
            if r.redirect_to:
                redirect_targets.add(r.redirect_to)

        passed = [h for h in candidate_hs4s if not results[h].should_hard_exclude()]
        excluded = [h for h in candidate_hs4s if results[h].should_hard_exclude()]
        new_redirects = [h for h in redirect_targets if h not in set(passed)]

        debug = {
            'total_evaluated': len(candidate_hs4s),
            'passed': len(passed),
            'excluded': len(excluded),
            'excluded_hs4s': excluded,
            'redirect_hs4s': new_redirects,
            'results': {
                hs4: {
                    'passed': r.passed,
                    'heading_term_score': r.heading_term_score,
                    'include_support_score': r.include_support_score,
                    'exclude_conflict_score': r.exclude_conflict_score,
                    'redirect_penalty': r.redirect_penalty,
                    'total_score': r.total_score(),
                }
                for hs4, r in results.items()
            },
        }

        return passed, new_redirects, debug
