"""
HS Classification Benchmark Experiments Package

연구 질문:
구조화된 법적/해설서 기반 KB(규칙/카드/8축 속성)가, 짧은 품명 텍스트 HS4 분류에서
베이스라인 모델 대비 정확도와 신뢰도(calibration), 그리고 저신뢰도 라우팅 품질을
얼마나 개선하는가?
"""

from .data_split import DataSplitter, DataSample
from .baselines import TFIDFBaseline, SBertBaseline, BM25Baseline, create_baseline
from .metrics import compute_metrics, compute_topk_accuracy, compute_improvement
from .calibration import compute_ece, compute_brier_score, reliability_curve, compute_calibration
from .routing import RoutingAnalyzer, RoutingDecision, RoutingStats
from .error_analysis import ErrorAnalyzer, FailureCase, ConfusionPair
from .ablation_runner import AblationRunner, AblationConfig, AblationResult, DEFAULT_ABLATIONS

__all__ = [
    # Data
    'DataSplitter',
    'DataSample',

    # Baselines
    'TFIDFBaseline',
    'SBertBaseline',
    'BM25Baseline',
    'create_baseline',

    # Metrics
    'compute_metrics',
    'compute_topk_accuracy',
    'compute_improvement',

    # Calibration
    'compute_ece',
    'compute_brier_score',
    'reliability_curve',
    'compute_calibration',

    # Routing
    'RoutingAnalyzer',
    'RoutingDecision',
    'RoutingStats',

    # Error Analysis
    'ErrorAnalyzer',
    'FailureCase',
    'ConfusionPair',

    # Ablation
    'AblationRunner',
    'AblationConfig',
    'AblationResult',
    'DEFAULT_ABLATIONS',
]
