"""
Stage 4: Hybrid (ML + KB + LightGBM Ranker) Classifier

Full integration: ML retrieval + KB retrieval + GRI signals + 8-axis attributes
+ LegalGate filtering + LightGBM reranking + confidence routing.

Pipeline:
  Input -> GRI signal detection + 8-axis attribute extraction
       -> ML Retriever -> Top-50
       -> KB Retriever -> Top-30
       -> KB-first Merge (KB priority, ML for recall)
       -> LegalGate filter
       -> Feature extraction (44 features)
       -> LightGBM Ranker rerank
       -> Confidence routing (AUTO/ASK/REVIEW)
       -> StageResult
"""

import math
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..types import BaseClassifier, StageResult, Prediction, StageID
from ..text import normalize, simple_contains, extract_keywords, tokenize, fuzzy_match
from ..kb.cards import CardIndex
from ..kb.rules import RuleIndex
from ..kb.legal_gate import LegalGate
from ..kb.gri import GRISignals, detect_gri_signals
from ..kb.attributes import (
    Attributes8Axis, extract_attributes,
    OBJECT_NATURE_KEYWORDS, MATERIAL_KEYWORDS, PROCESSING_STATE_KEYWORDS,
    FUNCTION_USE_KEYWORDS, PHYSICAL_FORM_KEYWORDS, COMPLETENESS_KEYWORDS,
    LEGAL_SCOPE_KEYWORDS,
)


LR_PATH = "artifacts/classifier/model_lr.joblib"
LE_PATH = "artifacts/classifier/label_encoder.joblib"
RANKER_PATH = "artifacts/ranker_legal/model_legal.txt"
ST_MODEL = "jhgan/ko-sroberta-multitask"

FEATURE_NAMES = [
    'f_ml', 'f_lexical', 'f_card_hits', 'f_rule_inc_hits', 'f_rule_exc_hits',
    'f_not_in_model', 'f_gri2a_signal', 'f_gri2b_signal', 'f_gri3_signal',
    'f_gri5_signal', 'f_specificity', 'f_exclude_conflict', 'f_is_parts_candidate',
    'f_state_match', 'f_material_match', 'f_use_match', 'f_form_match',
    'f_parts_mismatch', 'f_set_signal', 'f_incomplete_signal',
    'f_quant_match_score', 'f_quant_hard_exclude', 'f_quant_missing_value',
    'f_note_hard_exclude', 'f_note_support_sum',
    'f_object_match_score', 'f_material_match_score', 'f_processing_match_score',
    'f_function_match_score', 'f_form_match_score', 'f_completeness_match_score',
    'f_quant_rule_match_score', 'f_legal_scope_match_score',
    'f_conflict_penalty', 'f_uncertainty_penalty',
    'f_legal_heading_term', 'f_legal_include_support',
    'f_legal_exclude_conflict', 'f_legal_redirect_penalty',
]


@dataclass
class CandidateState:
    """Internal candidate tracking during hybrid pipeline."""
    hs4: str
    score_ml: float = 0.0
    score_card: float = 0.0
    score_rule: float = 0.0
    source: str = ''  # 'kb', 'ml', 'kb+ml'
    card_hits: int = 0
    rule_inc_hits: int = 0
    rule_exc_hits: int = 0
    has_exclude: bool = False
    note_support: float = 0.0
    matched_keywords: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)


class HybridClassifier(BaseClassifier):
    """Hybrid classifier: ML + KB + LightGBM Ranker."""

    def __init__(
        self,
        st_model_name: str = ST_MODEL,
        lr_path: str = LR_PATH,
        le_path: str = LE_PATH,
        ranker_path: str = RANKER_PATH,
        card_index: Optional[CardIndex] = None,
        rule_index: Optional[RuleIndex] = None,
        legal_gate: Optional[LegalGate] = None,
        use_gri: bool = True,
        use_legal_gate: bool = True,
        use_8axis: bool = True,
        use_ranker: bool = True,
        ml_topk: int = 50,
        kb_topk: int = 30,
        device: Optional[str] = None,
    ):
        self._st_model_name = st_model_name
        self._device = device
        self._ml_topk = ml_topk
        self._kb_topk = kb_topk

        # Toggles
        self._use_gri = use_gri
        self._use_legal_gate = use_legal_gate
        self._use_8axis = use_8axis
        self._use_ranker = use_ranker

        # Lazy-loaded ML components
        self._st_model = None
        self._lr_model = None
        self._label_encoder = None
        self._model_classes: Optional[Set[str]] = None
        self._ranker_model = None

        # KB components
        self._cards = card_index or CardIndex()
        self._rules = rule_index or RuleIndex()
        self._legal_gate = None
        if use_legal_gate:
            self._legal_gate = legal_gate
            if self._legal_gate is None:
                try:
                    self._legal_gate = LegalGate()
                except Exception:
                    self._legal_gate = None

        # Load ML models
        self._load_ml(lr_path, le_path)
        if use_ranker:
            self._load_ranker(ranker_path)

    def _load_ml(self, lr_path: str, le_path: str):
        import joblib
        from sentence_transformers import SentenceTransformer

        self._st_model = SentenceTransformer(self._st_model_name, device=self._device)
        print(f"[Hybrid] SBERT loaded: {self._st_model_name}")

        lr_file, le_file = Path(lr_path), Path(le_path)
        if lr_file.exists() and le_file.exists():
            self._lr_model = joblib.load(lr_file)
            self._label_encoder = joblib.load(le_file)
            self._model_classes = set(self._label_encoder.classes_)
            print(f"[Hybrid] LR loaded: {len(self._label_encoder.classes_)} classes")
        else:
            raise FileNotFoundError(f"ML model not found: {lr_file} / {le_file}")

    def _load_ranker(self, path: str):
        try:
            import lightgbm as lgb
            model_file = Path(path)
            if model_file.exists():
                self._ranker_model = lgb.Booster(model_file=str(model_file))
                print(f"[Hybrid] Ranker loaded: {path}")
        except Exception as e:
            print(f"[Hybrid] Ranker load failed: {e}")

    @property
    def name(self) -> str:
        return "Hybrid (ML+KB+Ranker)"

    @property
    def stage_id(self) -> StageID:
        return StageID.HYBRID

    # ---- ML Retrieval ----

    def _ml_retrieve(self, text: str, k: int = 50) -> List[CandidateState]:
        embedding = self._st_model.encode(text, convert_to_numpy=True).reshape(1, -1)
        proba = self._lr_model.predict_proba(embedding)[0]
        top_indices = np.argsort(proba)[-k:][::-1]
        candidates = []
        for idx in top_indices:
            hs4 = self._label_encoder.inverse_transform([idx])[0]
            candidates.append(CandidateState(
                hs4=hs4, score_ml=float(proba[idx]), source='ml',
            ))
        return candidates

    # ---- KB Retrieval ----

    def _kb_retrieve(self, text: str, k: int = 30) -> List[CandidateState]:
        card_results = self._cards.match_cards(text, topk=k)
        candidates = []
        for hs4, score, matched in card_results:
            cand = CandidateState(
                hs4=hs4, score_card=score, source='kb',
                matched_keywords=matched if isinstance(matched, list) else [],
            )
            # Rule scoring
            rule_score, inc, exc, has_exc, note_sup = self._rules.score_rules(text, hs4)
            cand.score_rule = rule_score
            cand.rule_inc_hits = inc
            cand.rule_exc_hits = exc
            cand.has_exclude = has_exc
            cand.note_support = note_sup
            candidates.append(cand)
        return candidates

    # ---- Merge ----

    def _merge_candidates(
        self, ml_cands: List[CandidateState], kb_cands: List[CandidateState]
    ) -> List[CandidateState]:
        """KB-first merge: KB candidates first, then ML for recall."""
        seen: Dict[str, CandidateState] = {}

        # KB first
        for c in kb_cands:
            seen[c.hs4] = c

        # ML additions
        for c in ml_cands:
            if c.hs4 in seen:
                seen[c.hs4].score_ml = c.score_ml
                seen[c.hs4].source = 'kb+ml'
            else:
                seen[c.hs4] = c

        return list(seen.values())

    # ---- Feature Extraction ----

    def _compute_features(
        self,
        text: str,
        cand: CandidateState,
        gri: GRISignals,
        attrs: Attributes8Axis,
        input_norm: str,
    ) -> List[float]:
        """Compute 38-feature vector for a candidate (+ 4 legal gate = 42 total, padded to 38+legal=38)."""
        # f_ml
        f_ml = cand.score_ml

        # f_lexical (token overlap)
        card_text = ''
        card_data = self._cards.cards.get(cand.hs4)
        if card_data:
            card_text = ' '.join([
                card_data.get('title', ''),
                ' '.join(card_data.get('scope', [])[:5]) if isinstance(card_data.get('scope'), list) else '',
            ])
        if card_text:
            card_norm = normalize(card_text)
            card_tokens = tokenize(card_norm, remove_stopwords=True)
            input_tokens = set(tokenize(input_norm, remove_stopwords=True))
            if input_tokens and card_tokens:
                overlap_count = len(input_tokens & set(card_tokens))
                f_lexical = overlap_count / max(len(input_tokens), 1)
            else:
                f_lexical = 0.0
        else:
            f_lexical = 0.0

        # Card/rule hits
        f_card_hits = cand.card_hits if cand.card_hits else len(cand.matched_keywords)
        f_rule_inc = cand.rule_inc_hits
        f_rule_exc = cand.rule_exc_hits
        f_not_in_model = 0 if (self._model_classes and cand.hs4 in self._model_classes) else 1

        # GRI signals (input-level)
        f_gri2a = 1 if gri.gri2a_incomplete else 0
        f_gri2b = 1 if gri.gri2b_mixtures else 0
        f_gri3 = 1 if gri.gri3_multi_candidate else 0
        f_gri5 = 1 if gri.gri5_containers else 0

        # Specificity (IDF-based)
        f_specificity = self._cards.compute_specificity(cand.matched_keywords)

        # Exclude conflict
        f_exclude_conflict = 1 if cand.has_exclude else 0

        # Parts candidate
        f_parts_cand = 1 if any(
            kw in normalize(card_text) for kw in ['부품', 'part', '부분품', '부속품']
        ) else 0

        # Legacy attribute matches
        f_state = 1 if attrs.processing_state.values else 0
        f_material = 1 if attrs.material.values else 0
        f_use = 1 if attrs.function_use.values else 0
        f_form = 1 if attrs.physical_form.values else 0
        f_parts_mismatch = 0
        f_set_signal = 1 if attrs.is_set() else 0
        f_incomplete = 1 if 'unassembled' in attrs.completeness.values else 0
        f_quant_match = 0.0
        f_quant_hard_exc = 0
        f_quant_missing = 1 if (attrs.has_quant() and not card_text) else 0
        f_note_hard_exc = 0
        f_note_support = cand.note_support

        # 8-axis matching scores (card attributes vs input attributes)
        card_attrs = self._extract_card_attrs(cand.hs4)
        f_obj = self._axis_match(attrs.object_nature.values, card_attrs.get('object_nature', []))
        f_mat = self._axis_match(attrs.material.values, card_attrs.get('material', []))
        f_proc = self._axis_match(attrs.processing_state.values, card_attrs.get('processing_state', []))
        f_func = self._axis_match(attrs.function_use.values, card_attrs.get('function_use', []))
        f_form_8 = self._axis_match(attrs.physical_form.values, card_attrs.get('physical_form', []))
        f_comp = self._axis_match(attrs.completeness.values, card_attrs.get('completeness', []))
        f_quant_rule = 0.0  # Quantitative rule matching (simplified)
        f_legal_scope = self._axis_match(attrs.legal_scope.values, card_attrs.get('legal_scope', []))

        # Conflict and uncertainty
        f_conflict = 0.0
        f_uncertainty = 0.0

        # LegalGate features (filled separately if available)
        f_legal_heading = 0.0
        f_legal_inc = 0.0
        f_legal_exc = 0.0
        f_legal_redir = 0.0

        return [
            f_ml, f_lexical, float(f_card_hits), float(f_rule_inc), float(f_rule_exc),
            float(f_not_in_model), float(f_gri2a), float(f_gri2b), float(f_gri3), float(f_gri5),
            f_specificity, float(f_exclude_conflict), float(f_parts_cand),
            float(f_state), float(f_material), float(f_use), float(f_form),
            float(f_parts_mismatch), float(f_set_signal), float(f_incomplete),
            f_quant_match, float(f_quant_hard_exc), float(f_quant_missing),
            float(f_note_hard_exc), f_note_support,
            f_obj, f_mat, f_proc, f_func, f_form_8, f_comp, f_quant_rule, f_legal_scope,
            f_conflict, f_uncertainty,
            f_legal_heading, f_legal_inc, f_legal_exc, f_legal_redir,
        ]

    def _extract_card_attrs(self, hs4: str) -> Dict[str, List[str]]:
        """Extract attribute values from a card's decision_attributes."""
        card_data = self._cards.cards.get(hs4)
        if not card_data:
            return {}
        dec_attrs = card_data.get('decision_attributes', {})
        if isinstance(dec_attrs, dict) and dec_attrs:
            return dec_attrs
        # Fallback: extract from card text
        card_text = card_data.get('title', '') + ' ' + ' '.join(card_data.get('scope', [])[:10])
        extracted = extract_attributes(card_text)
        return extracted.to_dict()

    @staticmethod
    def _axis_match(input_vals: List[str], card_vals) -> float:
        if not input_vals or not card_vals:
            return 0.0
        if isinstance(card_vals, dict):
            card_vals = card_vals.get('values', [])
        if not card_vals:
            return 0.0
        input_set = set(input_vals)
        card_set = set(card_vals) if isinstance(card_vals, list) else set()
        if not card_set:
            return 0.0
        overlap = len(input_set & card_set)
        return overlap / max(len(input_set), 1)

    # ---- Scoring / Ranking ----

    def _score_heuristic(self, features: List[float]) -> float:
        """Heuristic scoring when LightGBM is unavailable."""
        f_ml = features[0]
        f_lexical = features[1]
        f_card_hits = features[2]
        f_rule_inc = features[3]
        f_rule_exc = features[4]
        f_exclude = features[11]

        score = (
            f_ml * 0.25
            + f_lexical * 0.35
            + f_card_hits * 0.02
            + f_rule_inc * 0.05
            - f_rule_exc * 0.1
            - f_exclude * 0.2
        )

        # 8-axis bonus
        for i in range(25, 33):
            score += features[i] * 0.03

        return max(score, 0.0)

    def _rank_candidates(
        self, candidates: List[CandidateState], feature_vectors: List[List[float]]
    ) -> List[Tuple[CandidateState, float]]:
        """Rank candidates using LightGBM or heuristic."""
        if self._ranker_model and self._use_ranker:
            scores = self._ranker_model.predict(np.array(feature_vectors))
            scored = list(zip(candidates, scores.tolist()))
        else:
            scored = [
                (cand, self._score_heuristic(fv))
                for cand, fv in zip(candidates, feature_vectors)
            ]
        scored.sort(key=lambda x: -x[1])
        return scored

    # ---- Classify ----

    def classify(self, text: str, topk: int = 5) -> StageResult:
        debug: Dict[str, Any] = {}
        input_norm = normalize(text)

        # Step 0: GRI + attributes
        gri = detect_gri_signals(text) if self._use_gri else GRISignals()
        attrs = extract_attributes(text) if self._use_8axis else Attributes8Axis()
        debug['gri_active'] = gri.active_signals() if self._use_gri else []
        debug['attrs_summary'] = attrs.summary()

        # Step 1: ML retrieval
        ml_cands = self._ml_retrieve(text, k=self._ml_topk)
        debug['ml_count'] = len(ml_cands)

        # Step 2: KB retrieval (adjust topk based on GRI)
        actual_kb_topk = self._kb_topk
        if gri.gri2a_incomplete:
            actual_kb_topk += 20
        if gri.gri2b_mixtures:
            actual_kb_topk += 10

        kb_cands = self._kb_retrieve(text, k=actual_kb_topk)
        debug['kb_count'] = len(kb_cands)

        # Step 3: Merge
        candidates = self._merge_candidates(ml_cands, kb_cands)
        debug['merged_count'] = len(candidates)

        # Step 3.5: LegalGate filter
        if self._use_legal_gate and self._legal_gate and candidates:
            all_hs4s = [c.hs4 for c in candidates]
            passed, redirects, lg_debug = self._legal_gate.apply(text, all_hs4s[:100])
            passed_set = set(passed)
            candidates = [c for c in candidates if c.hs4 in passed_set]
            for rhs4 in redirects:
                if rhs4 not in passed_set:
                    candidates.append(CandidateState(hs4=rhs4, source='redirect'))
            debug['legal_gate_passed'] = len(passed)
            debug['legal_gate_excluded'] = len(all_hs4s) - len(passed)
            debug['legal_gate_redirects'] = redirects

        if not candidates:
            return StageResult(
                input_text=text, predictions=[], confidence=0.0,
                metadata={'stage': 'hybrid', 'warning': 'no_candidates', **debug},
            )

        # Step 4: Feature extraction + ranking
        feature_vectors = []
        for cand in candidates:
            fv = self._compute_features(text, cand, gri, attrs, input_norm)
            feature_vectors.append(fv)

        ranked = self._rank_candidates(candidates, feature_vectors)

        # Step 5: Build predictions
        predictions = []
        max_score = ranked[0][1] if ranked else 1.0
        if max_score <= 0:
            max_score = 1.0

        for i, (cand, score) in enumerate(ranked[:topk]):
            norm_score = min(score / max_score, 1.0) if max_score > 0 else 0.0
            predictions.append(Prediction(
                hs4=cand.hs4,
                score=round(norm_score, 4),
                rank=i + 1,
            ))

        # Step 6: Confidence
        confidence = predictions[0].score if predictions else 0.0
        if len(predictions) > 1:
            gap = predictions[0].score - predictions[1].score
            confidence = min(confidence, 0.5 + gap)
        confidence = min(confidence, 1.0)

        # Decision status
        status = 'AUTO'
        if confidence < 0.3:
            status = 'REVIEW'
        elif confidence < 0.5:
            status = 'ASK'

        debug['ranker_used'] = bool(self._ranker_model and self._use_ranker)
        debug['decision_status'] = status

        return StageResult(
            input_text=text,
            predictions=predictions,
            confidence=confidence,
            metadata={'stage': 'hybrid', 'status': status, **debug},
        )
