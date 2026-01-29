"""
HS 품목분류 파이프라인 (GRI 통칙 + 전역 속성 통합)

Usage:
    from src.classifier import HSPipeline

    pipeline = HSPipeline()
    result = pipeline.classify("스마트폰")

    # GRI 신호 탐지
    from src.classifier import detect_gri_signals
    signals = detect_gri_signals("자동차 CKD 부품 세트")

    # 전역 속성 추출
    from src.classifier import extract_attributes
    attrs = extract_attributes("냉동 돼지 삼겹살")
"""

from .types import Candidate, Evidence, ClassificationResult
from .retriever import HSRetriever
from .reranker import HSReranker, CandidateFeatures
from .clarify import HSClarifier
from .pipeline import HSPipeline
from .gri_signals import GRISignals, detect_gri_signals, detect_parts_signal
from .attribute_extract import GlobalAttributes, extract_attributes, QuantFact

__all__ = [
    'Candidate',
    'Evidence',
    'ClassificationResult',
    'HSRetriever',
    'HSReranker',
    'CandidateFeatures',
    'HSClarifier',
    'HSPipeline',
    'GRISignals',
    'detect_gri_signals',
    'detect_parts_signal',
    'GlobalAttributes',
    'extract_attributes',
    'QuantFact',
]
