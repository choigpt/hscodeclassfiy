"""
HS4 카드 생성 스크립트

해설서에서 HS4별 구조화된 카드를 생성합니다.
카드 구조:
- hs4: HS 4자리 코드
- title_ko: 한글 제목
- chapter: 류 (2자리)
- scope: 범위 키워드
- includes: 포함 품목
- excludes: 제외 품목 (다른 HS로 분류)
- key_attributes: 분류 기준 속성
- legal_refs: 법적 근거
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set


def extract_scope_keywords(content: str) -> List[str]:
    """content에서 범위 키워드 추출"""
    keywords = set()

    # "이 호에는 ... 분류한다" 패턴
    patterns = [
        r'이 호에는\s+(.+?)\s*[을를이가]\s*분류',
        r'이 호에\s+(.+?)\s*[을를이가]\s*분류',
        r'여기에는\s+(.+?)\s*[을를이가]\s*포함',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            # 긴 문장 분리
            words = re.split(r'[,·\s]+', match)
            for w in words:
                w = w.strip()
                if 2 <= len(w) <= 20 and not re.match(r'^[0-9\(\)호류]+$', w):
                    keywords.add(w)

    return list(keywords)[:10]


def extract_includes(content: str, title: str) -> List[str]:
    """포함 품목 추출"""
    includes = set()

    # 제목에서 기본 품목 추출
    title_items = re.split(r'[,·\(\)]+', title)
    for item in title_items:
        item = item.strip()
        if 2 <= len(item) <= 15:
            includes.add(item)

    # "예:" 다음의 품목
    example_patterns = [
        r'예[:\s]+([^\.]+)',
        r'예를 들[면어]\s*[,:]?\s*([^\.]+)',
        r'다음[을를]?\s*포함[한하]다[:\s]*([^\.]+)',
    ]

    for pattern in example_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            items = re.split(r'[,·\s]+', match)
            for item in items:
                item = item.strip()
                if 2 <= len(item) <= 15:
                    includes.add(item)

    return list(includes)[:15]


def extract_excludes(content: str, hs_code: str) -> List[Dict]:
    """제외 품목 및 대상 HS 추출"""
    excludes = []

    # "제XXXX호" 참조 패턴
    patterns = [
        r'제(\d{4})호[로에]?\s*분류',
        r'(\d{4})호[로에]?\s*분류',
        r'제외[하한]고.+?제(\d{4})호',
        r'다만.+?제(\d{4})호',
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            target_hs = match.group(1)
            if target_hs != hs_code:  # 자기 자신 제외
                # 주변 문맥에서 이유 추출
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end]

                excludes.append({
                    "hs4": target_hs,
                    "reason": context.strip()[:100]
                })

    # 중복 제거
    seen = set()
    unique_excludes = []
    for e in excludes:
        if e["hs4"] not in seen:
            seen.add(e["hs4"])
            unique_excludes.append(e)

    return unique_excludes[:5]


def extract_key_attributes(content: str) -> List[Dict]:
    """분류 기준 속성 추출"""
    attributes = []

    # 일반적인 분류 기준
    attr_patterns = {
        "재질": r'재질[이가]?\s*(.+?)[인으로]',
        "용도": r'용도[가이]?\s*(.+?)[인인으로]',
        "가공상태": r'(신선|냉장|냉동|건조|염장|훈제|가공)',
        "형태": r'(분말|액체|고체|기체|펠릿)',
    }

    for attr_name, pattern in attr_patterns.items():
        matches = re.findall(pattern, content)
        if matches:
            values = list(set(m.strip() for m in matches if len(m.strip()) <= 10))
            if values:
                attributes.append({
                    "name": attr_name,
                    "values": values[:5]
                })

    return attributes


def build_hs4_cards(commentary_path: str, output_path: str):
    """HS4 카드 생성"""
    print("=" * 60)
    print("HS4 카드 생성")
    print("=" * 60)

    with open(commentary_path, 'r', encoding='utf-8') as f:
        commentary = json.load(f)

    print(f"해설서 로드: {len(commentary)}개")

    cards = []
    for item in commentary:
        hs_code = item.get('hs_code', '')
        if not hs_code or len(hs_code) != 4:
            continue

        title = item.get('title', '')
        content = item.get('content', '')
        chapter = item.get('chapter', hs_code[:2])

        card = {
            "hs4": hs_code,
            "title_ko": title,
            "chapter": chapter,
            "scope": extract_scope_keywords(content),
            "includes": extract_includes(content, title),
            "excludes": extract_excludes(content, hs_code),
            "key_attributes": extract_key_attributes(content),
            "legal_refs": [f"HS 해설서 {hs_code}"]
        }

        cards.append(card)

    print(f"카드 생성: {len(cards)}개")

    # JSONL 형식으로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for card in cards:
            f.write(json.dumps(card, ensure_ascii=False) + '\n')

    print(f"저장: {output_path}")

    # 통계
    has_scope = sum(1 for c in cards if c['scope'])
    has_includes = sum(1 for c in cards if c['includes'])
    has_excludes = sum(1 for c in cards if c['excludes'])

    print(f"\n통계:")
    print(f"  scope 있음: {has_scope}개 ({has_scope/len(cards)*100:.1f}%)")
    print(f"  includes 있음: {has_includes}개 ({has_includes/len(cards)*100:.1f}%)")
    print(f"  excludes 있음: {has_excludes}개 ({has_excludes/len(cards)*100:.1f}%)")

    return cards


if __name__ == "__main__":
    build_hs4_cards(
        commentary_path="kb/raw/hs_commentary.json",
        output_path="kb/structured/hs4_cards.jsonl"
    )
