"""
Stage 3: NL/LLM (RAG) Classifier

BM25+SBERT hybrid retrieval -> RRF fusion -> Context assembly -> Ollama LLM -> JSON parse.

Pipeline:
  Input -> BM25 search(Top-20) + SBERT search(Top-20)
       -> RRF fusion -> Top-10 context
       -> Context assembly (cards + rules + similar cases)
       -> Ollama LLM (Qwen2.5 7B) -> JSON parse
       -> StageResult
"""

import json
import pickle
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

from ..types import BaseClassifier, StageResult, Prediction, StageID
from ..text import normalize, tokenize


# ---- Defaults ----

INDEX_DIR = "artifacts/rag_index"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:7b"
ST_MODEL = "jhgan/ko-sroberta-multitask"

SYSTEM_PROMPT = """당신은 대한민국 관세청 HS 품목분류 전문가입니다.
주어진 물품 설명과 후보 HS 코드 정보를 분석하여 가장 적합한 4자리 HS 코드를 결정하세요.

## 분류 원칙 (GRI 통칙)
1. **GRI 1**: 호의 용어와 관련 부·류의 주(Note)에 따라 분류한다.
2. **GRI 2(a)**: 미완성·미조립 물품도 완성품과 동일하게 분류한다.
3. **GRI 2(b)**: 혼합물·복합물은 본질적 특성을 부여하는 재료의 호로 분류한다.
4. **GRI 3**: 둘 이상의 호에 해당하면 가장 구체적인 호 > 본질적 특성 > 최종 호.
5. **GRI 5**: 케이스·포장용기는 내용물과 함께 분류한다.

## 응답 형식
반드시 아래 JSON 형식으로만 응답하세요.

{
  "best_hs4": "4자리 HS 코드",
  "confidence": 0.0~1.0,
  "reasoning": "분류 근거 2~3문장",
  "candidates": [
    {"hs4": "코드1", "score": 0.0~1.0},
    {"hs4": "코드2", "score": 0.0~1.0},
    {"hs4": "코드3", "score": 0.0~1.0}
  ]
}"""

USER_PROMPT_TEMPLATE = """## 분류 대상 물품
{query_text}

{retrieval_context}

위 정보를 바탕으로 이 물품의 4자리 HS 코드를 결정하고, JSON 형식으로 응답하세요."""


class LLMClassifier(BaseClassifier):
    """NL/LLM classifier using RAG (BM25+SBERT retrieval + Ollama LLM)."""

    def __init__(
        self,
        index_dir: str = INDEX_DIR,
        ollama_base_url: str = OLLAMA_BASE_URL,
        ollama_model: str = OLLAMA_MODEL,
        st_model_name: str = ST_MODEL,
        top_k_retrieval: int = 20,
        top_k_context: int = 10,
        max_context_chars: int = 4000,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 120,
    ):
        self._index_dir = Path(index_dir)
        self._ollama_base_url = ollama_base_url
        self._ollama_model = ollama_model
        self._st_model_name = st_model_name
        self._top_k_retrieval = top_k_retrieval
        self._top_k_context = top_k_context
        self._max_context_chars = max_context_chars
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

        # Lazy-loaded components
        self._bm25 = None
        self._hs4_ids: List[str] = []
        self._sbert_embeddings: Optional[np.ndarray] = None
        self._thesaurus: Dict[str, List[str]] = {}
        self._cards: Dict[str, Dict] = {}
        self._rules: Dict[str, List[Dict]] = {}
        self._case_embeddings: Optional[np.ndarray] = None
        self._case_metadata: List[Dict] = []
        self._sbert_model = None
        self._llm_client = None
        self._loaded = False

    # ---- Properties ----

    @property
    def name(self) -> str:
        return "NL/LLM (RAG)"

    @property
    def stage_id(self) -> StageID:
        return StageID.LLM

    # ---- Loading ----

    def _load_index(self):
        if self._loaded:
            return

        # BM25 corpus
        bm25_path = self._index_dir / 'bm25_corpus.pkl'
        if bm25_path.exists():
            with open(bm25_path, 'rb') as f:
                bm25_data = pickle.load(f)
            self._hs4_ids = bm25_data['hs4_ids']
            tokenized_corpus = bm25_data['tokenized_corpus']
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(tokenized_corpus)
            print(f"[LLM] BM25 loaded: {len(self._hs4_ids)} docs")

        # SBERT embeddings
        sbert_path = self._index_dir / 'sbert_embeddings.npy'
        if sbert_path.exists():
            self._sbert_embeddings = np.load(str(sbert_path))
            print(f"[LLM] SBERT embeddings: {self._sbert_embeddings.shape}")

        # Thesaurus
        thesaurus_path = self._index_dir / 'thesaurus_lookup.pkl'
        if thesaurus_path.exists():
            with open(thesaurus_path, 'rb') as f:
                self._thesaurus = pickle.load(f)

        # Cards for context
        cards_path = self._index_dir / 'hs4_cards.json'
        if cards_path.exists():
            with open(cards_path, 'r', encoding='utf-8') as f:
                self._cards = json.load(f)

        # Rules for context
        rules_path = self._index_dir / 'hs4_rules.json'
        if rules_path.exists():
            with open(rules_path, 'r', encoding='utf-8') as f:
                self._rules = json.load(f)

        # Case embeddings + metadata
        case_emb_path = self._index_dir / 'case_embeddings.npy'
        case_meta_path = self._index_dir / 'case_metadata.json'
        if case_emb_path.exists() and case_meta_path.exists():
            self._case_embeddings = np.load(str(case_emb_path))
            with open(case_meta_path, 'r', encoding='utf-8') as f:
                self._case_metadata = json.load(f)

        self._loaded = True

    def _get_sbert(self):
        if self._sbert_model is None:
            from sentence_transformers import SentenceTransformer
            self._sbert_model = SentenceTransformer(self._st_model_name)
            print(f"[LLM] SBERT model loaded: {self._st_model_name}")
        return self._sbert_model

    def _get_llm_client(self):
        if self._llm_client is None:
            from openai import OpenAI
            self._llm_client = OpenAI(
                base_url=self._ollama_base_url,
                api_key="ollama",
                timeout=self._timeout,
            )
        return self._llm_client

    # ---- Retrieval ----

    def _expand_query(self, text: str) -> str:
        if not self._thesaurus:
            return text
        tokens = tokenize(text, remove_stopwords=True)
        expanded = []
        for token in tokens:
            norm = normalize(token)
            if norm in self._thesaurus:
                expanded.extend(self._thesaurus[norm][:3])
        if expanded:
            return text + ' ' + ' '.join(expanded)
        return text

    def _retrieve_bm25(self, text: str, top_k: int = 30) -> List[Tuple[str, float]]:
        tokens = tokenize(text, remove_stopwords=True)
        if not tokens or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self._hs4_ids[idx], float(scores[idx]))
            for idx in top_indices if scores[idx] > 0
        ]

    def _retrieve_dense(self, text: str, top_k: int = 30) -> List[Tuple[str, float]]:
        if self._sbert_embeddings is None:
            return []
        model = self._get_sbert()
        query_emb = model.encode([text], normalize_embeddings=True)[0]
        similarities = self._sbert_embeddings @ query_emb
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            (self._hs4_ids[idx], float(similarities[idx]))
            for idx in top_indices
        ]

    def _retrieve_hybrid(self, text: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """BM25 + SBERT -> RRF fusion."""
        self._load_index()

        expanded = self._expand_query(text)
        bm25_results = self._retrieve_bm25(expanded, top_k=30)
        dense_results = self._retrieve_dense(text, top_k=30)

        rrf_k = 60
        rrf_scores: Dict[str, float] = {}
        for rank, (hs4, _) in enumerate(bm25_results):
            rrf_scores[hs4] = rrf_scores.get(hs4, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, (hs4, _) in enumerate(dense_results):
            rrf_scores[hs4] = rrf_scores.get(hs4, 0.0) + 1.0 / (rrf_k + rank + 1)

        sorted_results = sorted(rrf_scores.items(), key=lambda x: -x[1])
        return sorted_results[:top_k]

    # ---- Context Building ----

    def _build_context(
        self, query_text: str, retrieval_results: List[Tuple[str, float]]
    ) -> str:
        parts = ["## HS 후보 정보\n"]
        current_len = 15
        max_chars = self._max_context_chars
        max_candidates = self._top_k_context

        for rank, (hs4, score) in enumerate(retrieval_results[:max_candidates], 1):
            block = self._build_candidate_block(hs4, rank, score)
            if current_len + len(block) > max_chars - 400:
                break
            parts.append(block)
            current_len += len(block)

        # Similar cases
        if self._case_embeddings is not None:
            remaining = max_chars - current_len
            if remaining > 200:
                cases = self._find_similar_cases(query_text, top_k=2)
                if cases:
                    parts.append("\n## 유사 결정사례\n")
                    for case in cases:
                        case_text = (
                            f"- 품명: {case['product_name'][:80]}\n"
                            f"  분류: 제{case['hs_heading']}호 "
                            f"(유사도: {case.get('similarity', 0):.3f})\n"
                            f"  근거: {case.get('rationale', 'N/A')[:120]}\n"
                        )
                        if current_len + len(case_text) > max_chars:
                            break
                        parts.append(case_text)
                        current_len += len(case_text)

        return '\n'.join(parts)

    def _build_candidate_block(self, hs4: str, rank: int, score: float) -> str:
        card = self._cards.get(hs4, {})
        if not card:
            return f"[후보 {rank}] 제{hs4}호 (점수: {score:.4f})\n  정보 없음\n"

        lines = [f"[후보 {rank}] 제{hs4}호 (점수: {score:.4f})"]
        lines.append(f"  품명: {card.get('title_ko', 'N/A')}")

        scope = self._to_str_list(card.get('scope', []))
        if scope:
            lines.append(f"  범위: {', '.join(scope[:8])}")

        includes = self._to_str_list(card.get('includes', []))
        if includes:
            lines.append(f"  포함: {', '.join(includes[:5])}")

        excludes = self._to_str_list(card.get('excludes', []))
        if excludes:
            lines.append(f"  제외: {', '.join(excludes[:5])}")

        # Rules (top 2)
        rule_chunks = self._rules.get(hs4, [])
        for chunk in rule_chunks[:2]:
            rule_text = chunk.get('text', '')[:150]
            chunk_type = chunk.get('chunk_type', 'rule')
            lines.append(f"  규칙({chunk_type}): {rule_text}")

        return '\n'.join(lines)

    @staticmethod
    def _to_str_list(items: list) -> List[str]:
        result = []
        for item in items:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                for key in ('reason', 'text', 'value', 'description'):
                    val = item.get(key, '')
                    if val:
                        result.append(str(val)[:100])
                        break
        return result

    def _find_similar_cases(self, query_text: str, top_k: int = 2) -> List[Dict]:
        if self._case_embeddings is None or not self._case_metadata:
            return []
        model = self._get_sbert()
        query_emb = model.encode([query_text], normalize_embeddings=True)[0]
        similarities = self._case_embeddings @ query_emb
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            case = self._case_metadata[idx].copy()
            case['similarity'] = float(similarities[idx])
            results.append(case)
        return results

    # ---- LLM ----

    def _call_llm(self, query_text: str, context: str) -> Dict[str, Any]:
        client = self._get_llm_client()

        user_prompt = USER_PROMPT_TEMPLATE.format(
            query_text=query_text,
            retrieval_context=context,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = client.chat.completions.create(
            model=self._ollama_model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        raw = response.choices[0].message.content.strip()
        parsed = self._parse_json(raw)

        if parsed is not None:
            return parsed

        # Retry once
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": "응답이 올바른 JSON 형식이 아닙니다. 순수 JSON만 출력하세요.",
        })
        response2 = client.chat.completions.create(
            model=self._ollama_model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        raw2 = response2.choices[0].message.content.strip()
        parsed2 = self._parse_json(raw2)
        if parsed2 is not None:
            return parsed2

        raise RuntimeError(f"LLM JSON parse failed: {raw2[:200]}")

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Strip markdown code blocks
        cleaned = text
        if '```json' in cleaned:
            cleaned = cleaned.split('```json', 1)[1]
        if '```' in cleaned:
            cleaned = cleaned.split('```', 1)[0]
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Extract { ... }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return None

    # ---- Main classify ----

    def classify(self, text: str, topk: int = 5) -> StageResult:
        # Step 1: Hybrid retrieval
        retrieval_results = self._retrieve_hybrid(text, top_k=self._top_k_retrieval)

        if not retrieval_results:
            return StageResult(
                input_text=text, predictions=[], confidence=0.0,
                metadata={'stage': 'llm', 'warning': 'no_retrieval'},
            )

        # Step 2: Build context
        context = self._build_context(text, retrieval_results)

        # Step 3: LLM classify
        try:
            llm_response = self._call_llm(text, context)
            predictions = self._build_predictions(llm_response, retrieval_results, topk)
            confidence = float(llm_response.get('confidence', 0.0))
            metadata = {
                'stage': 'llm',
                'reasoning': str(llm_response.get('reasoning', '')),
                'llm_success': True,
                'retrieval_count': len(retrieval_results),
            }
        except Exception as e:
            # Fallback to retrieval results
            predictions = [
                Prediction(hs4=hs4, score=score, rank=i + 1)
                for i, (hs4, score) in enumerate(retrieval_results[:topk])
            ]
            confidence = predictions[0].score * 0.3 if predictions else 0.0
            metadata = {
                'stage': 'llm',
                'llm_success': False,
                'llm_error': str(e)[:200],
                'is_fallback': True,
            }

        return StageResult(
            input_text=text,
            predictions=predictions,
            confidence=min(confidence, 1.0),
            metadata=metadata,
        )

    def _build_predictions(
        self,
        llm_response: Dict[str, Any],
        retrieval_results: List[Tuple[str, float]],
        topk: int,
    ) -> List[Prediction]:
        seen = set()
        predictions = []

        # LLM best answer
        best_hs4 = str(llm_response.get('best_hs4', ''))
        best_conf = float(llm_response.get('confidence', 0.0))
        if best_hs4:
            predictions.append(Prediction(hs4=best_hs4, score=best_conf, rank=1))
            seen.add(best_hs4)

        # LLM candidates
        for cand in llm_response.get('candidates', []):
            hs4 = str(cand.get('hs4', ''))
            if hs4 and hs4 not in seen:
                predictions.append(Prediction(
                    hs4=hs4,
                    score=float(cand.get('score', 0.0)),
                    rank=len(predictions) + 1,
                ))
                seen.add(hs4)

        # Fill from retrieval
        for hs4, score in retrieval_results:
            if len(predictions) >= topk:
                break
            if hs4 not in seen:
                predictions.append(Prediction(
                    hs4=hs4, score=score * 0.5, rank=len(predictions) + 1,
                ))
                seen.add(hs4)

        # Re-rank
        for i, p in enumerate(predictions[:topk]):
            p.rank = i + 1

        return predictions[:topk]
