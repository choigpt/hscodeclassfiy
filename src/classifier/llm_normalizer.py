"""
LLM Attribute Normalizer - 보조 레이어 1

입력 텍스트 → 8축 구조화 JSON (confidence + 근거 스팬 포함)

원칙:
- 법 규범 레이어(LegalGate/FactCheck)를 override할 수 없음
- confidence 낮으면 hard rule 적용 금지
- 사실 추출만 수행, 결론 도출 금지
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .attribute_extract import GlobalAttributes8Axis, AxisAttributes


@dataclass
class NormalizedAttribute:
    """LLM이 추출한 정규화된 속성"""
    axis: str  # object, material, processing, function, form, completeness, quant, legal
    values: List[str]
    confidence: float  # 0.0 ~ 1.0
    evidence_spans: List[str] = field(default_factory=list)  # 근거 텍스트 스팬
    reasoning: str = ""  # LLM의 추론 과정


class LLMAttributeNormalizer:
    """
    LLM 기반 속성 정규화기 (보조 레이어)

    법 규범을 override하지 않고, 사실 추출만 수행
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",  # 또는 "claude-3-5-sonnet"
        api_key: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # LLM 클라이언트
        self.client = None
        self._init_client()

    def _init_client(self):
        """LLM 클라이언트 초기화"""
        if not self.api_key:
            print("[LLMNormalizer] Warning: No API key found. LLM normalization disabled.")
            return

        try:
            if self.model.startswith("gpt"):
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                print(f"[LLMNormalizer] OpenAI client initialized: {self.model}")
            elif self.model.startswith("claude"):
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
                print(f"[LLMNormalizer] Anthropic client initialized: {self.model}")
        except Exception as e:
            print(f"[LLMNormalizer] Client init failed: {e}")
            self.client = None

    def normalize(
        self,
        input_text: str,
        fallback_attrs: Optional[GlobalAttributes8Axis] = None
    ) -> GlobalAttributes8Axis:
        """
        텍스트 → 8축 속성 정규화

        Args:
            input_text: 입력 텍스트
            fallback_attrs: LLM 실패시 사용할 기본 속성

        Returns:
            GlobalAttributes8Axis (confidence + evidence 포함)
        """
        if not self.client:
            # LLM 없으면 fallback 사용
            if fallback_attrs:
                return fallback_attrs
            # fallback도 없으면 빈 속성
            return self._empty_attrs()

        try:
            # LLM 호출
            normalized = self._call_llm(input_text)
            return normalized
        except Exception as e:
            print(f"[LLMNormalizer] Error: {e}")
            if fallback_attrs:
                return fallback_attrs
            return self._empty_attrs()

    def _call_llm(self, input_text: str) -> GlobalAttributes8Axis:
        """LLM 호출하여 속성 추출"""

        prompt = self._build_prompt(input_text)

        if self.model.startswith("gpt"):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            result_text = response.choices[0].message.content
        elif self.model.startswith("claude"):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=self._system_prompt(),
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            result_text = response.content[0].text
        else:
            raise ValueError(f"Unsupported model: {self.model}")

        # JSON 파싱
        result = json.loads(result_text)

        # GlobalAttributes8Axis로 변환
        return self._parse_result(result, input_text)

    def _system_prompt(self) -> str:
        """시스템 프롬프트"""
        return """You are an HS Code classification fact extractor.

Your role is to extract factual attributes from product descriptions, NOT to make classification decisions.

Extract the following 8 axes of attributes:
1. object_nature: What is the core object/item? (e.g., "chair", "plastic bag", "wheat")
2. material: What material(s) is it made of? (e.g., "plastic", "wood", "metal")
3. processing_state: Processing/preservation state (e.g., "fresh", "frozen", "dried")
4. function_use: Primary function/use (e.g., "packaging", "food", "industrial")
5. physical_form: Physical form (e.g., "powder", "liquid", "sheet")
6. completeness: Assembly/completion state (e.g., "complete", "unassembled", "parts")
7. quant_rules: Quantitative properties (e.g., "weight > 50kg", "sugar content 10%")
8. legal_scope: Legal scope keywords (e.g., "for export", "duty-free")

For each axis:
- Provide values found in the text
- Provide confidence (0.0-1.0): how certain you are about the extraction
- Provide evidence_spans: exact phrases from input that support your extraction
- If not mentioned, leave empty with confidence 0.0

Output JSON format:
{
  "object_nature": {"values": [...], "confidence": 0.0-1.0, "evidence": [...], "reasoning": "..."},
  "material": {...},
  ...
}

IMPORTANT:
- Extract facts ONLY, do not infer or guess
- If uncertain, use low confidence (< 0.5)
- Never make classification decisions
- Provide exact text spans as evidence
"""

    def _build_prompt(self, input_text: str) -> str:
        """사용자 프롬프트 생성"""
        return f"""Extract factual attributes from the following product description:

INPUT: {input_text}

Provide structured JSON output with all 8 axes.
"""

    def _parse_result(
        self,
        result: Dict[str, Any],
        input_text: str
    ) -> GlobalAttributes8Axis:
        """LLM 결과 → GlobalAttributes8Axis 변환"""

        def parse_axis(axis_data: Dict) -> AxisAttributes:
            """단일 축 파싱"""
            if not axis_data:
                return AxisAttributes(values=[], confidence=0.0)

            return AxisAttributes(
                values=axis_data.get("values", []),
                confidence=axis_data.get("confidence", 0.0),
                source="llm",
                evidence=axis_data.get("evidence", [])
            )

        return GlobalAttributes8Axis(
            input_text=input_text,
            object_nature=parse_axis(result.get("object_nature", {})),
            material=parse_axis(result.get("material", {})),
            processing_state=parse_axis(result.get("processing_state", {})),
            function_use=parse_axis(result.get("function_use", {})),
            physical_form=parse_axis(result.get("physical_form", {})),
            completeness=parse_axis(result.get("completeness", {})),
            quant_rules=parse_axis(result.get("quant_rules", {})),
            legal_scope=parse_axis(result.get("legal_scope", {}))
        )

    def _empty_attrs(self) -> GlobalAttributes8Axis:
        """빈 속성 반환"""
        empty_axis = AxisAttributes(values=[], confidence=0.0)
        return GlobalAttributes8Axis(
            input_text="",
            object_nature=empty_axis,
            material=empty_axis,
            processing_state=empty_axis,
            function_use=empty_axis,
            physical_form=empty_axis,
            completeness=empty_axis,
            quant_rules=empty_axis,
            legal_scope=empty_axis
        )


def normalize_with_llm(
    input_text: str,
    fallback_attrs: Optional[GlobalAttributes8Axis] = None,
    model: str = "gpt-4o-mini"
) -> GlobalAttributes8Axis:
    """LLM 정규화 편의 함수"""
    normalizer = LLMAttributeNormalizer(model=model)
    return normalizer.normalize(input_text, fallback_attrs)
