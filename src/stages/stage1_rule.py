"""
Stage 1: Rule-Based (KB-only) Classifier

KB card/rule keyword matching + LegalGate filter. No ML.

Pipeline:
  Input -> normalize -> KB card keyword match (score_card)
       -> rule chunk signal match (score_rule)
       -> thesaurus match
       -> LegalGate exclude filter
       -> sort by combined score -> StageResult
"""

from typing import List, Optional
from collections import defaultdict

from ..types import BaseClassifier, StageResult, Prediction, StageID
from ..kb.cards import CardIndex
from ..kb.rules import RuleIndex
from ..kb.legal_gate import LegalGate


class RuleClassifier(BaseClassifier):
    """Rule-based classifier using KB cards + rules + thesaurus."""

    def __init__(
        self,
        card_index: Optional[CardIndex] = None,
        rule_index: Optional[RuleIndex] = None,
        legal_gate: Optional[LegalGate] = None,
        use_legal_gate: bool = True,
    ):
        self._cards = card_index or CardIndex()
        self._rules = rule_index or RuleIndex()
        self._legal_gate = legal_gate if use_legal_gate else None
        if use_legal_gate and legal_gate is None:
            try:
                self._legal_gate = LegalGate()
            except Exception:
                self._legal_gate = None

    @property
    def name(self) -> str:
        return "Rule-Based"

    @property
    def stage_id(self) -> StageID:
        return StageID.RULE

    def classify(self, text: str, topk: int = 5) -> StageResult:
        hs4_scores = defaultdict(float)

        # 1. Card keyword matching
        card_results = self._cards.match_cards(text, topk=50)
        for hs4, score, matched in card_results:
            hs4_scores[hs4] += score

        # 2. Thesaurus matching
        thesaurus_scores = self._rules.match_thesaurus(text)
        for hs4, score in thesaurus_scores.items():
            hs4_scores[hs4] += score

        # 3. Rule matching (include boosts, exclude penalizes)
        for hs4 in list(hs4_scores.keys()):
            rule_score, inc, exc, has_exc, _ = self._rules.score_rules(text, hs4)
            hs4_scores[hs4] += rule_score

        # 4. LegalGate filter
        if self._legal_gate and hs4_scores:
            all_hs4s = sorted(hs4_scores.keys(), key=lambda h: -hs4_scores[h])
            passed, redirects, _ = self._legal_gate.apply(text, all_hs4s[:100])
            excluded = set(all_hs4s) - set(passed)
            for h in excluded:
                hs4_scores.pop(h, None)
            for h in redirects:
                if h not in hs4_scores:
                    hs4_scores[h] = 0.5

        # Sort and normalize
        sorted_hs4 = sorted(hs4_scores.items(), key=lambda x: -x[1])[:topk]

        if not sorted_hs4:
            return StageResult(
                input_text=text, predictions=[], confidence=0.0,
                metadata={'stage': 'rule', 'warning': 'no_matches'},
            )

        max_score = sorted_hs4[0][1] if sorted_hs4[0][1] > 0 else 1.0
        predictions = [
            Prediction(hs4=hs4, score=min(score / max_score, 1.0), rank=i + 1)
            for i, (hs4, score) in enumerate(sorted_hs4)
        ]

        confidence = predictions[0].score if predictions else 0.0
        if len(predictions) > 1:
            gap = predictions[0].score - predictions[1].score
            confidence = min(confidence, 0.5 + gap)

        return StageResult(
            input_text=text,
            predictions=predictions,
            confidence=min(confidence, 1.0),
            metadata={'stage': 'rule'},
        )
