"""
질문 템플릿 생성 스크립트

HS4별 불확실성 해소를 위한 질문 템플릿을 생성합니다.
- hs4: HS 코드
- disambiguation_questions: 추가 질문 리스트
"""

import json
import re
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


# HS 분류 기준별 질문 템플릿
ATTRIBUTE_QUESTIONS = {
    "material": {
        "q": "주요 재질이 무엇인가요?",
        "key": "material",
        "options": ["금속", "플라스틱", "직물", "가죽", "목재", "고무", "유리", "기타"]
    },
    "use": {
        "q": "주요 용도가 무엇인가요?",
        "key": "use",
        "options": ["산업용", "가정용", "의료용", "식품용", "운송용", "기타"]
    },
    "processing": {
        "q": "가공 상태가 어떻게 되나요?",
        "key": "processing",
        "options": ["신선", "냉장", "냉동", "건조", "가공", "조제"]
    },
    "form": {
        "q": "물품의 형태가 어떻게 되나요?",
        "key": "form",
        "options": ["분말", "액체", "고체", "기체", "펠릿", "시트", "기타"]
    },
    "size": {
        "q": "크기/중량 기준이 있나요?",
        "key": "size",
        "options": ["기준 이상", "기준 미만", "해당없음"]
    },
    "component": {
        "q": "주요 구성 성분이 무엇인가요?",
        "key": "component",
        "options": []
    }
}

# 챕터별 주요 분류 기준
CHAPTER_CRITERIA = {
    # 농산물
    "01": ["processing"],  # 살아있는 동물
    "02": ["processing", "form"],  # 육류
    "03": ["processing", "form"],  # 어류
    "04": ["processing"],  # 낙농품
    "07": ["processing"],  # 채소
    "08": ["processing"],  # 과일
    "09": ["processing", "form"],  # 커피/차

    # 화학
    "28": ["form", "component"],  # 무기화학품
    "29": ["component"],  # 유기화학품
    "30": ["use", "form"],  # 의약품
    "32": ["use"],  # 염료
    "33": ["use"],  # 화장품
    "34": ["use"],  # 비누

    # 플라스틱/고무
    "39": ["form", "use"],  # 플라스틱
    "40": ["form", "use"],  # 고무

    # 섬유
    "50": ["material"],  # 견
    "51": ["material"],  # 양모
    "52": ["material"],  # 면
    "54": ["material"],  # 인조필라멘트
    "55": ["material"],  # 인조스테이플
    "61": ["material", "use"],  # 의류(편물)
    "62": ["material", "use"],  # 의류(직물)
    "63": ["use"],  # 기타 섬유제품
    "64": ["material", "use"],  # 신발

    # 금속
    "72": ["form"],  # 철강
    "73": ["use", "form"],  # 철강제품
    "74": ["form"],  # 구리
    "76": ["form"],  # 알루미늄

    # 기계/전자
    "84": ["use", "component"],  # 기계
    "85": ["use", "component"],  # 전기기기
    "87": ["use"],  # 자동차
    "90": ["use"],  # 광학/의료기기

    # 기타
    "94": ["material", "use"],  # 가구
    "95": ["use"],  # 완구
}


def get_questions_for_hs4(hs4: str, content: str) -> List[Dict]:
    """HS4에 적합한 질문 생성"""
    chapter = hs4[:2]
    questions = []

    # 챕터별 기본 질문
    criteria = CHAPTER_CRITERIA.get(chapter, ["use", "material"])

    for criterion in criteria:
        if criterion in ATTRIBUTE_QUESTIONS:
            q_template = ATTRIBUTE_QUESTIONS[criterion].copy()
            questions.append({
                "q": q_template["q"],
                "key": q_template["key"]
            })

    # 해설서 내용에서 추가 질문 힌트
    if re.search(r'재질|재료|소재', content):
        if not any(q["key"] == "material" for q in questions):
            questions.append({
                "q": "주요 재질이 무엇인가요?",
                "key": "material"
            })

    if re.search(r'용도|목적|사용', content):
        if not any(q["key"] == "use" for q in questions):
            questions.append({
                "q": "주요 용도가 무엇인가요?",
                "key": "use"
            })

    return questions[:3]  # 최대 3개


def build_question_templates(commentary_path: str, output_path: str):
    """질문 템플릿 생성"""
    print("=" * 60)
    print("질문 템플릿 생성")
    print("=" * 60)

    with open(commentary_path, 'r', encoding='utf-8') as f:
        commentary = json.load(f)

    print(f"해설서 로드: {len(commentary)}개")

    templates = []

    for item in commentary:
        hs4 = item.get('hs_code', '')
        if not hs4 or len(hs4) != 4:
            continue

        content = item.get('content', '')
        questions = get_questions_for_hs4(hs4, content)

        if questions:
            templates.append({
                "hs4": hs4,
                "chapter": hs4[:2],
                "disambiguation_questions": questions
            })

    print(f"템플릿 생성: {len(templates)}개")

    # JSONL 형식으로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for template in templates:
            f.write(json.dumps(template, ensure_ascii=False) + '\n')

    print(f"저장: {output_path}")

    # 통계
    q_counts = defaultdict(int)
    for t in templates:
        for q in t['disambiguation_questions']:
            q_counts[q['key']] += 1

    print(f"\n질문 유형별 분포:")
    for key, count in sorted(q_counts.items(), key=lambda x: -x[1]):
        print(f"  {key}: {count}개")

    return templates


if __name__ == "__main__":
    build_question_templates(
        commentary_path="kb/raw/hs_commentary.json",
        output_path="kb/structured/disambiguation_questions.jsonl"
    )
