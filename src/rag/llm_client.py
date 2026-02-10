"""
Ollama LLM 클라이언트 (OpenAI 호환 API)

Qwen2.5 7B 모델을 Ollama를 통해 호출한다.
"""

import json
import time
from typing import Dict, Any, Optional, List

from .prompts import SYSTEM_PROMPT, build_user_prompt


# Ollama 기본 설정
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_TIMEOUT = 120  # seconds
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1024


class OllamaClient:
    """Ollama OpenAI 호환 LLM 클라이언트"""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        """OpenAI 클라이언트 (lazy init)"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key="ollama",  # Ollama는 API key 불필요
                timeout=self.timeout,
            )
        return self._client

    def is_available(self) -> bool:
        """Ollama 서버 연결 가능 여부 확인"""
        try:
            client = self._get_client()
            client.models.list()
            return True
        except Exception:
            return False

    def classify(
        self,
        query_text: str,
        retrieval_context: str,
        gri_signals: dict,
    ) -> Dict[str, Any]:
        """
        HS 분류 LLM 호출

        Args:
            query_text: 입력 품명
            retrieval_context: 검색 컨텍스트 문자열
            gri_signals: GRI 신호 딕셔너리

        Returns:
            파싱된 JSON 응답 딕셔너리

        Raises:
            LLMError: LLM 호출 또는 파싱 실패
        """
        user_prompt = build_user_prompt(query_text, retrieval_context, gri_signals)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # 1차 시도
        raw_response = self._call_llm(messages)
        parsed = self._parse_json_response(raw_response)

        if parsed is not None:
            return parsed

        # JSON 파싱 실패 시 1회 재시도 (수정 요청)
        messages.append({"role": "assistant", "content": raw_response})
        messages.append({
            "role": "user",
            "content": "응답이 올바른 JSON 형식이 아닙니다. 반드시 순수 JSON만 출력하세요. "
                       "markdown 코드블록이나 설명 없이 { 로 시작하고 } 로 끝나는 JSON만 응답하세요."
        })

        raw_response_2 = self._call_llm(messages)
        parsed_2 = self._parse_json_response(raw_response_2)

        if parsed_2 is not None:
            return parsed_2

        raise LLMError(f"JSON 파싱 실패 (2회 시도): {raw_response_2[:200]}")

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """LLM API 호출"""
        client = self._get_client()

        start = time.time()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        elapsed = time.time() - start

        content = response.choices[0].message.content.strip()
        return content

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """LLM 응답에서 JSON 추출 및 파싱"""
        # 직접 파싱 시도
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # markdown 코드블록 제거 후 시도
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

        # { ... } 추출 시도
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        return None

    def follow_up(
        self,
        conversation: List[Dict[str, str]],
        follow_up_message: str
    ) -> str:
        """멀티턴 대화 (보완 질문 응답)"""
        messages = conversation + [
            {"role": "user", "content": follow_up_message}
        ]
        return self._call_llm(messages)


class LLMError(Exception):
    """LLM 관련 에러"""
    pass
