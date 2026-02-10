"""
하이브리드 검색기 (BM25 + SBERT RRF)

시소러스 쿼리 확장 → BM25 희소 검색 + SBERT 밀집 검색 → RRF 결합
"""

import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.classifier.utils_text import normalize, tokenize

INDEX_DIR = "artifacts/rag_index"


class HybridRetriever:
    """BM25 + SBERT 하이브리드 검색기"""

    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = Path(index_dir)
        self._loaded = False

        # 인덱스 데이터
        self.hs4_ids: List[str] = []
        self.documents: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.sbert_embeddings: Optional[np.ndarray] = None
        self.thesaurus: Dict[str, List[str]] = {}
        self.sbert_model = None

    def load(self):
        """인덱스 로드"""
        if self._loaded:
            return

        # BM25 corpus
        bm25_path = self.index_dir / 'bm25_corpus.pkl'
        with open(bm25_path, 'rb') as f:
            bm25_data = pickle.load(f)
        self.hs4_ids = bm25_data['hs4_ids']
        self.documents = bm25_data['documents']
        self.tokenized_corpus = bm25_data['tokenized_corpus']
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[Retriever] BM25 로드: {len(self.hs4_ids)}개 문서")

        # SBERT embeddings
        sbert_path = self.index_dir / 'sbert_embeddings.npy'
        self.sbert_embeddings = np.load(str(sbert_path))
        print(f"[Retriever] SBERT 임베딩 로드: {self.sbert_embeddings.shape}")

        # Thesaurus
        thesaurus_path = self.index_dir / 'thesaurus_lookup.pkl'
        if thesaurus_path.exists():
            with open(thesaurus_path, 'rb') as f:
                self.thesaurus = pickle.load(f)
            print(f"[Retriever] 시소러스 로드: {len(self.thesaurus)}개 엔트리")

        # SBERT model (lazy load)
        self._loaded = True

    def _load_sbert_model(self):
        """SBERT 모델 로드 (lazy)"""
        if self.sbert_model is None:
            from sentence_transformers import SentenceTransformer
            self.sbert_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
            print("[Retriever] SBERT 모델 로드 완료")

    def expand_query(self, text: str) -> str:
        """
        시소러스 기반 쿼리 확장

        입력 텍스트의 토큰을 시소러스에서 찾아 동의어/약어를 추가한다.
        """
        if not self.thesaurus:
            return text

        tokens = tokenize(text, remove_stopwords=True)
        expanded_terms = []

        for token in tokens:
            norm_token = normalize(token)
            if norm_token in self.thesaurus:
                aliases = self.thesaurus[norm_token]
                # 최대 3개 동의어 추가
                for alias in aliases[:3]:
                    expanded_terms.append(alias)

        if expanded_terms:
            return text + ' ' + ' '.join(expanded_terms)
        return text

    def retrieve_bm25(self, text: str, top_k: int = 30) -> List[Tuple[str, float]]:
        """
        BM25 희소 검색

        Returns:
            [(hs4, score), ...] 상위 top_k 결과
        """
        tokens = tokenize(text, remove_stopwords=True)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # 상위 K개 인덱스
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.hs4_ids[idx], float(scores[idx])))

        return results

    def retrieve_dense(self, text: str, top_k: int = 30) -> List[Tuple[str, float]]:
        """
        SBERT 밀집 검색 (코사인 유사도)

        Returns:
            [(hs4, score), ...] 상위 top_k 결과
        """
        self._load_sbert_model()

        # 쿼리 인코딩
        query_emb = self.sbert_model.encode(
            [text],
            normalize_embeddings=True
        )[0]

        # 코사인 유사도 (정규화된 벡터의 내적)
        similarities = self.sbert_embeddings @ query_emb

        # 상위 K개
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append((self.hs4_ids[idx], float(similarities[idx])))

        return results

    def retrieve(
        self,
        text: str,
        top_k: int = 20,
        bm25_k: int = 30,
        dense_k: int = 30,
        rrf_k: int = 60,
        expand: bool = True
    ) -> List[Tuple[str, float]]:
        """
        하이브리드 검색 (BM25 + SBERT → RRF)

        Args:
            text: 입력 품명
            top_k: 최종 반환 수
            bm25_k: BM25 후보 수
            dense_k: SBERT 후보 수
            rrf_k: RRF 상수 (기본 60)
            expand: 시소러스 확장 여부

        Returns:
            [(hs4, rrf_score), ...] 상위 top_k 결과
        """
        self.load()

        # 쿼리 확장
        expanded_text = self.expand_query(text) if expand else text

        # BM25 검색
        bm25_results = self.retrieve_bm25(expanded_text, top_k=bm25_k)

        # SBERT 검색 (원본 텍스트로 - 확장된 텍스트는 의미 변경 가능)
        dense_results = self.retrieve_dense(text, top_k=dense_k)

        # RRF (Reciprocal Rank Fusion)
        rrf_scores: Dict[str, float] = {}

        for rank, (hs4, _score) in enumerate(bm25_results):
            rrf_scores[hs4] = rrf_scores.get(hs4, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, (hs4, _score) in enumerate(dense_results):
            rrf_scores[hs4] = rrf_scores.get(hs4, 0.0) + 1.0 / (rrf_k + rank + 1)

        # 정렬 후 상위 반환
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def get_retrieval_debug(
        self,
        text: str,
        bm25_k: int = 10,
        dense_k: int = 10
    ) -> Dict:
        """디버그용 상세 검색 결과"""
        self.load()

        expanded = self.expand_query(text)
        bm25_results = self.retrieve_bm25(expanded, top_k=bm25_k)
        dense_results = self.retrieve_dense(text, top_k=dense_k)

        return {
            'original_query': text,
            'expanded_query': expanded,
            'bm25_top': [(hs4, round(s, 4)) for hs4, s in bm25_results[:5]],
            'dense_top': [(hs4, round(s, 4)) for hs4, s in dense_results[:5]],
        }
