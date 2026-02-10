"""
RAG 파이프라인 오케스트레이터

전체 흐름:
1. GRI 신호 검출
2. 시소러스 쿼리 확장
3. 하이브리드 검색 (BM25 + SBERT → RRF)
4. 컨텍스트 조립
5. LLM 분류 (Ollama Qwen2.5 7B)
6. 후처리 (LegalGate + 신뢰도 보정)
"""

import time
from typing import Optional, Dict, Any

from .types import RAGResult, RAGCandidate, RAGEvidence, Confidence
from .retriever import HybridRetriever
from .context_builder import ContextBuilder
from .llm_client import OllamaClient, LLMError
from .postprocess import PostProcessor


class RAGPipeline:
    """RAG 기반 HS 분류 파이프라인"""

    def __init__(
        self,
        index_dir: str = "artifacts/rag_index",
        ollama_base_url: str = "http://localhost:11434/v1",
        ollama_model: str = "qwen2.5:7b",
        top_k_retrieval: int = 20,
        top_k_context: int = 10,
        max_context_chars: int = 4000,
    ):
        self.retriever = HybridRetriever(index_dir=index_dir)
        self.context_builder = ContextBuilder(index_dir=index_dir)
        self.llm_client = OllamaClient(
            base_url=ollama_base_url,
            model=ollama_model,
        )
        self.postprocessor = PostProcessor()

        self.top_k_retrieval = top_k_retrieval
        self.top_k_context = top_k_context
        self.max_context_chars = max_context_chars

    def classify(self, text: str) -> RAGResult:
        """
        전체 RAG 파이프라인 실행

        Args:
            text: 입력 품명/설명

        Returns:
            RAGResult
        """
        start_time = time.time()
        debug = {}

        # [1] GRI 신호 검출
        gri_analysis = self.postprocessor.detect_gri(text)
        debug['gri_signals'] = gri_analysis.get('active_gri', [])

        # [2-3] 하이브리드 검색 (시소러스 확장 포함)
        retrieval_results = self.retriever.retrieve(
            text,
            top_k=self.top_k_retrieval,
            expand=True,
        )
        debug['retrieval_count'] = len(retrieval_results)
        debug['retrieval_top5'] = [
            (hs4, round(s, 4)) for hs4, s in retrieval_results[:5]
        ]

        if not retrieval_results:
            # 검색 결과 없음
            return RAGResult(
                input_text=text,
                best_hs4="",
                confidence=Confidence(calibrated=0.0),
                reasoning="검색 결과가 없습니다.",
                debug=debug,
            )

        # [4] 컨텍스트 조립
        context, ctx_meta = self.context_builder.build_context_with_metadata(
            query_text=text,
            retrieval_results=retrieval_results,
            max_candidates=self.top_k_context,
            max_chars=self.max_context_chars,
        )
        debug['context_chars'] = ctx_meta['context_chars']

        # [5] LLM 분류
        try:
            llm_response = self.llm_client.classify(
                query_text=text,
                retrieval_context=context,
                gri_signals=gri_analysis,
            )
            result = self._build_result_from_llm(text, llm_response, retrieval_results)
            debug['llm_success'] = True

        except (LLMError, Exception) as e:
            # LLM 실패 → 검색 결과 fallback
            debug['llm_success'] = False
            debug['llm_error'] = str(e)[:200]
            result = self._build_fallback_result(text, retrieval_results)

        # [6] 후처리
        result = self.postprocessor.apply(result, retrieval_results)

        elapsed = time.time() - start_time
        debug['elapsed_sec'] = round(elapsed, 2)
        result.debug = debug

        return result

    def _build_result_from_llm(
        self,
        text: str,
        llm_response: Dict[str, Any],
        retrieval_results: list,
    ) -> RAGResult:
        """LLM 응답을 RAGResult로 변환"""
        best_hs4 = str(llm_response.get('best_hs4', ''))
        raw_confidence = float(llm_response.get('confidence', 0.0))
        reasoning = str(llm_response.get('reasoning', ''))
        need_info = bool(llm_response.get('need_info', False))
        questions = list(llm_response.get('questions', []))

        # retrieval 점수 맵
        retrieval_map = {hs4: score for hs4, score in retrieval_results}

        # LLM이 제시한 candidates
        llm_candidates = llm_response.get('candidates', [])
        candidates = []
        for i, cand_dict in enumerate(llm_candidates):
            hs4 = str(cand_dict.get('hs4', ''))
            if not hs4:
                continue
            candidates.append(RAGCandidate(
                hs4=hs4,
                rank=i + 1,
                score_retrieval=retrieval_map.get(hs4, 0.0),
                score_llm=float(cand_dict.get('score', 0.0)),
                evidence=[RAGEvidence(
                    kind='llm_reasoning',
                    source_id=hs4,
                    text=reasoning[:160],
                    score=float(cand_dict.get('score', 0.0)),
                )],
            ))

        # best_hs4가 candidates에 없으면 추가
        if best_hs4 and not any(c.hs4 == best_hs4 for c in candidates):
            candidates.insert(0, RAGCandidate(
                hs4=best_hs4,
                rank=0,
                score_retrieval=retrieval_map.get(best_hs4, 0.0),
                score_llm=raw_confidence,
                reasoning=reasoning,
            ))

        # 검색 결과 중 LLM이 언급하지 않은 후보도 추가 (하위 순위)
        mentioned = {c.hs4 for c in candidates}
        for hs4, score in retrieval_results[:10]:
            if hs4 not in mentioned:
                candidates.append(RAGCandidate(
                    hs4=hs4,
                    rank=len(candidates) + 1,
                    score_retrieval=score,
                    score_llm=0.0,
                    evidence=[RAGEvidence(
                        kind='retrieval',
                        source_id=hs4,
                        text='검색에서만 발견 (LLM 미선택)',
                        score=score,
                    )],
                ))

        return RAGResult(
            input_text=text,
            best_hs4=best_hs4,
            candidates=candidates,
            confidence=Confidence(raw_llm=raw_confidence),
            reasoning=reasoning,
            need_info=need_info,
            questions=questions,
            is_fallback=False,
        )

    def _build_fallback_result(
        self,
        text: str,
        retrieval_results: list,
    ) -> RAGResult:
        """LLM 실패 시 검색 결과 기반 fallback"""
        candidates = []
        for rank, (hs4, score) in enumerate(retrieval_results[:10], 1):
            candidates.append(RAGCandidate(
                hs4=hs4,
                rank=rank,
                score_retrieval=score,
                score_llm=0.0,
                evidence=[RAGEvidence(
                    kind='retrieval',
                    source_id=hs4,
                    text='검색 기반 fallback (LLM 미사용)',
                    score=score,
                )],
            ))

        best_hs4 = retrieval_results[0][0] if retrieval_results else ""

        return RAGResult(
            input_text=text,
            best_hs4=best_hs4,
            candidates=candidates,
            confidence=Confidence(raw_llm=0.0),
            reasoning="LLM 호출 실패. 검색 결과 기반 fallback.",
            is_fallback=True,
        )
