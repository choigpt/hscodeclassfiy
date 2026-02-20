"""
HS Classification - Unified Types

BaseClassifier interface + StageResult/Prediction dataclasses.
All 5 stages implement BaseClassifier.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any
import time


class StageID(Enum):
    RULE = 1
    ML = 2
    LLM = 3
    HYBRID = 4
    CASCADE = 5


@dataclass
class Prediction:
    hs4: str
    score: float       # [0, 1]
    rank: int = 0


@dataclass
class StageResult:
    input_text: str
    predictions: List[Prediction]
    confidence: float
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def top1(self) -> str:
        return self.predictions[0].hs4 if self.predictions else ""

    @property
    def topk_codes(self) -> List[str]:
        return [p.hs4 for p in self.predictions]


class BaseClassifier(ABC):
    @abstractmethod
    def classify(self, text: str, topk: int = 5) -> StageResult: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def stage_id(self) -> StageID: ...

    def classify_timed(self, text: str, topk: int = 5) -> StageResult:
        t0 = time.perf_counter()
        result = self.classify(text, topk)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result
