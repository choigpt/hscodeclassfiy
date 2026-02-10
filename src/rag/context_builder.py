"""
LLM 컨텍스트 빌더

검색된 후보를 LLM이 이해할 수 있는 구조화된 컨텍스트로 조립한다.
- Top-10 후보별 카드 정보 + 관련 규칙
- 유사 결정사례 2건
- 토큰 예산 관리 (~4000자)
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

INDEX_DIR = "artifacts/rag_index"


def _to_str_list(items: list) -> List[str]:
    """카드 필드를 문자열 리스트로 변환"""
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

# 토큰 예산 (한국어 기준 ~2자/토큰, 총 ~2000 토큰 ≈ 4000자)
MAX_CONTEXT_CHARS = 4000
MAX_CANDIDATES = 10
MAX_RULES_PER_CARD = 2
MAX_SIMILAR_CASES = 2


class ContextBuilder:
    """LLM 컨텍스트 조립기"""

    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.cards: Dict[str, Dict] = {}
        self.rules: Dict[str, List[Dict]] = {}
        self.case_embeddings: Optional[np.ndarray] = None
        self.case_metadata: List[Dict] = []
        self._loaded = False

    def load(self):
        """인덱스 데이터 로드"""
        if self._loaded:
            return

        # HS4 카드
        cards_path = self.index_dir / 'hs4_cards.json'
        if cards_path.exists():
            with open(cards_path, 'r', encoding='utf-8') as f:
                self.cards = json.load(f)
            print(f"[ContextBuilder] 카드 로드: {len(self.cards)}개")

        # 규칙 청크
        rules_path = self.index_dir / 'hs4_rules.json'
        if rules_path.exists():
            with open(rules_path, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)
            print(f"[ContextBuilder] 규칙 로드: {len(self.rules)} HS4")

        # 사례 임베딩 + 메타데이터
        case_emb_path = self.index_dir / 'case_embeddings.npy'
        case_meta_path = self.index_dir / 'case_metadata.json'
        if case_emb_path.exists() and case_meta_path.exists():
            self.case_embeddings = np.load(str(case_emb_path))
            with open(case_meta_path, 'r', encoding='utf-8') as f:
                self.case_metadata = json.load(f)
            print(f"[ContextBuilder] 사례 로드: {len(self.case_metadata)}건")

        self._loaded = True

    def find_similar_cases(
        self,
        query_text: str,
        top_k: int = MAX_SIMILAR_CASES
    ) -> List[Dict]:
        """SBERT 기반 유사 결정사례 검색"""
        if self.case_embeddings is None or len(self.case_metadata) == 0:
            return []

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("jhgan/ko-sroberta-multitask")
        query_emb = model.encode([query_text], normalize_embeddings=True)[0]

        similarities = self.case_embeddings @ query_emb
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            case = self.case_metadata[idx].copy()
            case['similarity'] = float(similarities[idx])
            results.append(case)

        return results

    def build_candidate_context(
        self,
        hs4: str,
        rank: int,
        retrieval_score: float
    ) -> str:
        """단일 후보의 컨텍스트 블록 생성"""
        card = self.cards.get(hs4, {})
        if not card:
            return f"[후보 {rank}] 제{hs4}호 (검색 점수: {retrieval_score:.4f})\n  정보 없음\n"

        lines = []
        lines.append(f"[후보 {rank}] 제{hs4}호 (검색 점수: {retrieval_score:.4f})")
        lines.append(f"  품명: {card.get('title_ko', 'N/A')}")

        scope = _to_str_list(card.get('scope', []))
        if scope:
            lines.append(f"  범위: {', '.join(scope[:8])}")

        includes = _to_str_list(card.get('includes', []))
        if includes:
            lines.append(f"  포함: {', '.join(includes[:5])}")

        excludes = _to_str_list(card.get('excludes', []))
        if excludes:
            lines.append(f"  제외: {', '.join(excludes[:5])}")

        # 관련 규칙 (상위 2개)
        rule_chunks = self.rules.get(hs4, [])
        for i, chunk in enumerate(rule_chunks[:MAX_RULES_PER_CARD]):
            rule_text = chunk.get('text', '')[:150]
            chunk_type = chunk.get('chunk_type', 'rule')
            lines.append(f"  규칙({chunk_type}): {rule_text}")

        return '\n'.join(lines)

    def build_context(
        self,
        query_text: str,
        retrieval_results: List[Tuple[str, float]],
        max_candidates: int = MAX_CANDIDATES,
        max_chars: int = MAX_CONTEXT_CHARS,
        include_cases: bool = True
    ) -> str:
        """
        전체 LLM 컨텍스트 조립

        Args:
            query_text: 입력 품명
            retrieval_results: [(hs4, rrf_score), ...] 검색 결과
            max_candidates: 최대 후보 수
            max_chars: 최대 글자 수
            include_cases: 유사 사례 포함 여부

        Returns:
            조립된 컨텍스트 문자열
        """
        self.load()

        parts = []
        current_len = 0

        # 후보 카드 컨텍스트
        parts.append("## HS 후보 정보\n")
        current_len += 15

        candidates_added = 0
        for rank, (hs4, score) in enumerate(retrieval_results[:max_candidates], 1):
            block = self.build_candidate_context(hs4, rank, score)
            block_len = len(block)

            # 예산 확인 (사례용 공간 400자 예약)
            reserved = 400 if include_cases else 0
            if current_len + block_len > max_chars - reserved:
                break

            parts.append(block)
            current_len += block_len
            candidates_added += 1

        # 유사 결정사례
        if include_cases and self.case_embeddings is not None:
            remaining = max_chars - current_len
            if remaining > 200:
                cases = self.find_similar_cases(query_text, top_k=MAX_SIMILAR_CASES)
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

        context = '\n'.join(parts)
        return context

    def build_context_with_metadata(
        self,
        query_text: str,
        retrieval_results: List[Tuple[str, float]],
        **kwargs
    ) -> Tuple[str, Dict]:
        """컨텍스트 + 메타데이터 반환"""
        context = self.build_context(query_text, retrieval_results, **kwargs)
        metadata = {
            'context_chars': len(context),
            'candidates_in_context': min(len(retrieval_results), kwargs.get('max_candidates', MAX_CANDIDATES)),
            'retrieval_top5': [hs4 for hs4, _ in retrieval_results[:5]],
        }
        return context, metadata
