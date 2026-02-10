"""
RAG 기반 HS 품목분류 파이프라인

BM25+SBERT 하이브리드 검색 + Ollama LLM (Qwen2.5 7B) 기반 RAG 시스템

Usage:
    from src.rag import RAGPipeline

    pipeline = RAGPipeline()
    result = pipeline.classify("냉동 돼지 삼겹살")
"""

from .types import RAGResult, RAGCandidate, RAGEvidence, Confidence
from .pipeline import RAGPipeline

__all__ = [
    'RAGResult',
    'RAGCandidate',
    'RAGEvidence',
    'Confidence',
    'RAGPipeline',
]
