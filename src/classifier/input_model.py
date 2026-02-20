"""
구조화 입력 모델 (Structured Classification Input)

ClassificationInput: GRI 순차 파이프라인의 구조화된 입력
MaterialInfo: 재질/성분 구성 정보
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MaterialInfo:
    """재질/성분 정보"""
    name: str
    ratio: Optional[float] = None   # 0.0-1.0
    unit: str = "weight"             # weight | volume | count

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ratio": self.ratio,
            "unit": self.unit,
        }


@dataclass
class ClassificationInput:
    """구조화된 분류 입력"""
    text: str
    materials: List[MaterialInfo] = field(default_factory=list)
    function_use: Optional[str] = None
    is_set: bool = False
    is_electrical: bool = False
    image_path: Optional[str] = None
    jurisdiction: str = "KR"
    hs_version: str = "2022"

    def to_enriched_text(self) -> str:
        """기존 text-only 인터페이스와 호환되도록 통합 텍스트 생성"""
        parts = [self.text]

        if self.materials:
            mat_descs = []
            for m in self.materials:
                if m.ratio is not None:
                    mat_descs.append(f"{m.name} {m.ratio*100:.0f}%")
                else:
                    mat_descs.append(m.name)
            parts.append(f"재질: {', '.join(mat_descs)}")

        if self.function_use:
            parts.append(f"용도: {self.function_use}")

        if self.is_set:
            parts.append("세트물품")

        if self.is_electrical:
            parts.append("전기식")

        return " | ".join(parts)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "materials": [m.to_dict() for m in self.materials],
            "function_use": self.function_use,
            "is_set": self.is_set,
            "is_electrical": self.is_electrical,
            "jurisdiction": self.jurisdiction,
            "hs_version": self.hs_version,
        }
