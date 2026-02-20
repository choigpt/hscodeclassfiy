"""
HS4 Card loading and keyword matching.
Loads hs4_cards.jsonl and provides card-based scoring.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

from ..text import normalize, tokenize, extract_keywords

CARDS_PATH = "kb/structured/hs4_cards.jsonl"


class CardIndex:
    """HS4 card index for keyword matching."""

    def __init__(self, cards_path: str = CARDS_PATH):
        self.cards: Dict[str, Dict] = {}  # hs4 -> card
        self.keyword_doc_freq: Dict[str, int] = defaultdict(int)
        self.total_cards: int = 0
        self._load(cards_path)
        self._compute_idf()

    def _load(self, path: str):
        cards_file = Path(path)
        if not cards_file.exists():
            print(f"[CardIndex] Warning: {cards_file} not found")
            return

        count = 0
        with open(cards_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    card = json.loads(line)
                    hs4 = card.get('hs4', '')
                    if not hs4:
                        continue

                    keywords = set()
                    for kw in card.get('includes', []):
                        if kw and len(kw) >= 2:
                            keywords.add(normalize(kw))
                    for kw in card.get('scope', []):
                        if kw and len(kw) >= 2:
                            keywords.add(normalize(kw))

                    title = card.get('title_ko', '')
                    title_tokens = extract_keywords(title, min_len=2, max_count=10)
                    keywords.update(title_tokens)

                    self.cards[hs4] = {
                        'title': title,
                        'keywords': list(keywords),
                        'excludes': card.get('excludes', []),
                        'includes': card.get('includes', []),
                        'scope': card.get('scope', []),
                        'decision_attributes': card.get('decision_attributes', {}),
                        'raw': card,
                    }
                    count += 1
                except json.JSONDecodeError:
                    continue

        self.total_cards = count
        print(f"[CardIndex] Loaded {count} cards")

    def _compute_idf(self):
        for card in self.cards.values():
            for kw in card.get('keywords', []):
                self.keyword_doc_freq[kw] += 1

    def get_hs4_set(self) -> Set[str]:
        return set(self.cards.keys())

    def score_card(self, text: str, hs4: str) -> Tuple[float, int, List[str]]:
        """
        Score a card against text.

        Returns:
            (score, hit_count, matched_keywords)
        """
        if hs4 not in self.cards:
            return 0.0, 0, []

        card = self.cards[hs4]
        keywords = card.get('keywords', [])
        norm_text = normalize(text)
        text_tokens = set(tokenize(text, remove_stopwords=True))

        matched = []
        for kw in keywords:
            if not kw or len(kw) < 2:
                continue
            if kw in norm_text:
                matched.append(kw)
                continue
            kw_tokens = set(tokenize(kw, remove_stopwords=True))
            if kw_tokens & text_tokens:
                matched.append(kw)

        if not matched:
            return 0.0, 0, []

        score = math.log(1 + len(matched))
        return score, len(matched), matched

    def compute_specificity(self, matched_keywords: List[str]) -> float:
        """IDF-based specificity score."""
        if not matched_keywords or self.total_cards == 0:
            return 0.0
        score = 0.0
        for kw in matched_keywords:
            df = self.keyword_doc_freq.get(kw, 1)
            score += math.log(self.total_cards / df) if df > 0 else 0.0
        return score

    def match_cards(self, text: str, topk: int = 30) -> List[Tuple[str, float, List[str]]]:
        """
        Match text against all cards.

        Returns:
            [(hs4, score, matched_keywords), ...] sorted by score desc
        """
        norm_text = normalize(text)
        text_tokens = set(tokenize(text, remove_stopwords=True))

        hs4_scores: Dict[str, float] = defaultdict(float)
        hs4_matched: Dict[str, List[str]] = defaultdict(list)

        for hs4, card in self.cards.items():
            keywords = card.get('keywords', [])
            hit_count = 0
            matched = []

            for kw in keywords:
                if kw and kw in norm_text:
                    hit_count += 1
                    matched.append(kw)

            keyword_tokens = set()
            for kw in keywords:
                keyword_tokens.update(tokenize(kw, remove_stopwords=True))
            overlap = text_tokens & keyword_tokens
            hit_count += len(overlap) * 0.5

            if hit_count > 0:
                hs4_scores[hs4] += hit_count
                hs4_matched[hs4].extend(matched)

        sorted_results = sorted(hs4_scores.items(), key=lambda x: -x[1])[:topk]
        return [(hs4, score, hs4_matched.get(hs4, [])) for hs4, score in sorted_results]
