"""
한국어 HS 분류 전문가 프롬프트 템플릿
"""

SYSTEM_PROMPT = """당신은 대한민국 관세청 HS 품목분류 전문가입니다.
주어진 물품 설명과 후보 HS 코드 정보를 분석하여 가장 적합한 4자리 HS 코드를 결정하세요.

## 분류 원칙 (GRI 통칙)
1. **GRI 1**: 호의 용어와 관련 부·류의 주(Note)에 따라 분류한다. 주의 용어가 최우선.
2. **GRI 2(a)**: 미완성·미조립 물품도 완성품과 동일하게 분류한다.
3. **GRI 2(b)**: 혼합물·복합물은 본질적 특성을 부여하는 재료/구성요소의 호로 분류한다.
4. **GRI 3**: 둘 이상의 호에 해당하면 가장 구체적인 호 > 본질적 특성 > 최종 호 순서로 결정한다.
5. **GRI 5**: 케이스·포장용기는 내용물과 함께 분류한다 (별도 거래 제외).

## 응답 형식
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트를 추가하지 마세요.

{
  "best_hs4": "4자리 HS 코드",
  "confidence": 0.0~1.0 사이의 신뢰도,
  "reasoning": "분류 근거를 한국어로 2~3문장으로 설명",
  "candidates": [
    {"hs4": "코드1", "score": 0.0~1.0},
    {"hs4": "코드2", "score": 0.0~1.0},
    {"hs4": "코드3", "score": 0.0~1.0}
  ],
  "need_info": false,
  "questions": []
}

- best_hs4: 가장 적합한 4자리 HS 코드 (반드시 후보 중에서 선택)
- confidence: 확신도 (0.9 이상이면 매우 확실, 0.5 미만이면 불확실)
- reasoning: GRI 통칙을 근거로 한 분류 이유
- candidates: 상위 3개 후보와 적합도 점수
- need_info: 추가 정보가 필요하면 true
- questions: need_info가 true일 때 필요한 질문 리스트"""

USER_PROMPT_TEMPLATE = """## 분류 대상 물품
{query_text}

{gri_context}

{retrieval_context}

위 정보를 바탕으로 이 물품의 4자리 HS 코드를 결정하고, JSON 형식으로 응답하세요."""


def build_gri_context(gri_signals: dict) -> str:
    """GRI 신호를 프롬프트 컨텍스트로 변환"""
    active = gri_signals.get('active_gri', [])
    if not active:
        return "## GRI 신호\n특별한 GRI 신호가 감지되지 않았습니다. GRI 1(호의 용어)에 따라 분류하세요."

    lines = ["## GRI 신호 (주의 필요)"]
    signals = gri_signals.get('signals', {})

    if 'gri2a' in active:
        keywords = signals.get('gri2a_incomplete', {}).get('matched_keywords', [])
        lines.append(f"- **GRI 2(a) 감지**: 미완성/미조립 관련 ({', '.join(keywords[:3])})")
        lines.append("  → 완성품으로 분류 가능 여부 판단 필요")

    if 'gri2b' in active:
        keywords = signals.get('gri2b_mixtures', {}).get('matched_keywords', [])
        lines.append(f"- **GRI 2(b) 감지**: 혼합물/복합재료 관련 ({', '.join(keywords[:3])})")
        lines.append("  → 본질적 특성을 부여하는 성분 기준으로 분류")

    if 'gri3' in active:
        keywords = signals.get('gri3_multi_candidate', {}).get('matched_keywords', [])
        lines.append(f"- **GRI 3 감지**: 세트/복수 후보 관련 ({', '.join(keywords[:3])})")
        lines.append("  → 가장 구체적인 호 또는 본질적 특성 기준 적용")

    if 'gri5' in active:
        keywords = signals.get('gri5_containers', {}).get('matched_keywords', [])
        lines.append(f"- **GRI 5 감지**: 케이스/포장 관련 ({', '.join(keywords[:3])})")
        lines.append("  → 내용물 기준 분류 (별도 거래 제외)")

    if 'gri1' in active:
        lines.append("- **GRI 1 참조**: 주규정/해설서 참조 감지")

    return '\n'.join(lines)


def build_user_prompt(
    query_text: str,
    retrieval_context: str,
    gri_signals: dict
) -> str:
    """유저 프롬프트 조립"""
    gri_context = build_gri_context(gri_signals)
    return USER_PROMPT_TEMPLATE.format(
        query_text=query_text,
        gri_context=gri_context,
        retrieval_context=retrieval_context,
    )
