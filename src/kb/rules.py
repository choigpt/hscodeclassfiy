"""
Rule chunk loading and signal matching (include/exclude).
Loads hs4_rule_chunks.jsonl and thesaurus_terms.jsonl.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

from ..text import normalize, tokenize, extract_keywords

RULES_PATH = "kb/structured/hs4_rule_chunks.jsonl"
RULES_V2_PATH = "kb/structured/hs4_rule_chunks_v2.jsonl"
THESAURUS_PATH = "kb/structured/thesaurus_terms.jsonl"


class RuleIndex:
    """Rule chunk index for signal matching."""

    def __init__(
        self,
        rules_path: str = RULES_PATH,
        thesaurus_path: str = THESAURUS_PATH,
    ):
        self.rules: Dict[str, List[Dict]] = defaultdict(list)  # hs4 -> [rules]
        self.thesaurus: Dict[str, List[str]] = defaultdict(list)  # term -> [hs4]
        # v2 파일 우선 로드
        v2_path = Path(RULES_V2_PATH)
        actual_rules_path = str(v2_path) if v2_path.exists() else rules_path
        self._load_rules(actual_rules_path)
        self._load_thesaurus(thesaurus_path)

    def _infer_strength(self, text: str, chunk_type: str) -> str:
        if chunk_type != 'exclude_rule':
            return 'soft'
        hard_patterns = [
            '제외한다', '제외된다',
            '해당하지 아니한다', '해당하지 않는다',
            '분류하지 아니한다', '분류하지 않는다',
            '포함하지 아니한다', '포함하지 않는다',
            '적용하지 아니한다', '적용하지 않는다',
            '이 류에서', '이 호에서', '이 절에서',
            '다음 각 목의 것은 제외', '다음의 것은 제외',
        ]
        text_lower = text.lower()
        for p in hard_patterns:
            if p in text_lower:
                return 'hard'
        return 'soft'

    def _load_rules(self, path: str):
        rules_file = Path(path)
        if not rules_file.exists():
            print(f"[RuleIndex] Warning: {rules_file} not found")
            return

        count = 0
        for idx, line in enumerate(open(rules_file, 'r', encoding='utf-8')):
            line = line.strip()
            if not line:
                continue
            try:
                rule = json.loads(line)
                hs4 = rule.get('hs4', '')
                if not hs4:
                    continue

                chunk_type = rule.get('chunk_type', 'general')
                text = rule.get('text', '')
                signals = rule.get('signals', [])
                if not signals:
                    signals = extract_keywords(text, min_len=2, max_count=5)
                signals = [normalize(s) for s in signals if s and len(s) >= 2]

                if 'exclude' in chunk_type:
                    polarity = 'exclude'
                elif 'include' in chunk_type:
                    polarity = 'include'
                else:
                    polarity = rule.get('polarity', 'neutral')

                strength = rule.get('strength') or self._infer_strength(text, chunk_type)

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

        print(f"[RuleIndex] Loaded {count} rules for {len(self.rules)} HS4")

    def _load_thesaurus(self, path: str):
        thesaurus_file = Path(path)
        if not thesaurus_file.exists():
            print(f"[RuleIndex] Warning: {thesaurus_file} not found")
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

        print(f"[RuleIndex] Thesaurus: {count} mappings, {len(self.thesaurus)} terms")

    def get_hs4_set(self) -> Set[str]:
        return set(self.rules.keys())

    def score_rules(
        self, text: str, hs4: str,
        exclude_penalty: float = -2.0,
        hard_exclude_penalty: float = -10.0,
    ) -> Tuple[float, int, int, bool, float]:
        """
        Score rules for a given HS4.

        Returns:
            (score, inc_hits, exc_hits, has_exclude_conflict, note_support)
        """
        if hs4 not in self.rules:
            return 0.0, 0, 0, False, 0.0

        norm_text = normalize(text)
        text_tokens = set(tokenize(text, remove_stopwords=True))

        score = 0.0
        inc_hits = 0
        exc_hits = 0
        has_exc = False
        note_support = 0.0

        for rule in self.rules[hs4]:
            signals = rule['signals']
            polarity = rule.get('polarity', 'neutral')
            strength = rule.get('strength', 'soft')
            chunk_type = rule['chunk_type']

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

            n = len(matched_signals)
            if chunk_type == 'include_rule' or polarity == 'include':
                delta = 1.0 * n
                inc_hits += n
                note_support += delta
            elif chunk_type == 'exclude_rule' or polarity == 'exclude':
                delta = (hard_exclude_penalty if strength == 'hard' else exclude_penalty) * n
                exc_hits += n
                has_exc = True
            elif chunk_type == 'definition':
                delta = 0.5 * n
                note_support += delta * 0.5
            elif chunk_type == 'example':
                delta = 0.3 * n
                note_support += delta * 0.3
            else:
                delta = 0.1 * n

            score += delta

        return score, inc_hits, exc_hits, has_exc, note_support

    def match_thesaurus(self, text: str) -> Dict[str, float]:
        """Match text against thesaurus, return hs4 -> score."""
        norm_text = normalize(text)
        text_tokens = set(tokenize(text, remove_stopwords=True))
        scores: Dict[str, float] = defaultdict(float)

        for term, hs4_list in self.thesaurus.items():
            if term in norm_text or term in text_tokens:
                for hs4 in hs4_list:
                    scores[hs4] += 1.5

        return dict(scores)
