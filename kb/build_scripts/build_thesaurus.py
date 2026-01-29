"""
용어 사전 생성 스크립트

결정사례에서 HS4별 표현 다양성을 추출합니다.
- term: 대표 용어
- aliases: 동의어/변형
- hs4_candidates: 분류 후보 HS4
- confidence: 신뢰도
- source: 출처
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict, Counter


def normalize_term(text: str) -> str:
    """용어 정규화"""
    # 소문자, 공백 정리
    text = text.lower().strip()
    # 특수문자 제거
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    # 다중 공백 제거
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_terms_from_case(case: Dict) -> List[str]:
    """결정사례에서 용어 추출"""
    terms = []

    product_name = case.get('product_name', '')
    description = case.get('product_description', '')

    # product_name에서 추출
    if product_name:
        # 세미콜론, 콤마로 분리
        parts = re.split(r'[;,]', product_name)
        for part in parts:
            part = part.strip()
            if 2 <= len(part) <= 50:
                terms.append(part)

    # description에서 주요 명사구 추출
    if description:
        # 괄호 안 내용 추출
        parens = re.findall(r'\(([^)]+)\)', description)
        for p in parens:
            if 2 <= len(p) <= 30:
                terms.append(p)

    return terms


def group_similar_terms(terms: List[str]) -> Dict[str, List[str]]:
    """유사 용어 그룹핑"""
    groups = defaultdict(set)

    for term in terms:
        norm = normalize_term(term)
        if len(norm) >= 2:
            # 첫 번째 등장을 대표로
            groups[norm].add(term)

    return {k: list(v) for k, v in groups.items()}


def build_thesaurus(ruling_cases_path: str, output_path: str):
    """용어 사전 생성"""
    print("=" * 60)
    print("용어 사전 생성")
    print("=" * 60)

    with open(ruling_cases_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)

    print(f"결정사례 로드: {len(cases)}개")

    # HS4별 용어 수집
    hs4_terms = defaultdict(list)
    term_to_hs4 = defaultdict(Counter)

    for case in cases:
        hs4 = case.get('hs_heading', '')
        if not hs4 or len(hs4) != 4:
            continue

        terms = extract_terms_from_case(case)
        for term in terms:
            norm = normalize_term(term)
            if norm:
                hs4_terms[hs4].append(term)
                term_to_hs4[norm][hs4] += 1

    print(f"HS4 수: {len(hs4_terms)}개")

    # 용어 사전 생성
    thesaurus = []
    processed_terms = set()

    for norm_term, hs4_counts in term_to_hs4.items():
        if norm_term in processed_terms:
            continue
        processed_terms.add(norm_term)

        # 신뢰도 계산
        total = sum(hs4_counts.values())
        if total < 1:
            continue

        # 가장 많이 등장한 HS4
        top_hs4 = hs4_counts.most_common(1)[0]
        confidence = top_hs4[1] / total

        # 원본 표현들 수집
        aliases = set()
        for hs4, terms in hs4_terms.items():
            for t in terms:
                if normalize_term(t) == norm_term:
                    aliases.add(t)

        entry = {
            "term": norm_term,
            "aliases": list(aliases)[:10],
            "hs4_candidates": [hs for hs, _ in hs4_counts.most_common(3)],
            "confidence": round(confidence, 3),
            "frequency": total,
            "source": "ruling_cases"
        }

        thesaurus.append(entry)

    # 빈도순 정렬
    thesaurus.sort(key=lambda x: -x['frequency'])

    print(f"용어 생성: {len(thesaurus)}개")

    # JSONL 형식으로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in thesaurus:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"저장: {output_path}")

    # 통계
    high_conf = sum(1 for e in thesaurus if e['confidence'] >= 0.9)
    multi_hs = sum(1 for e in thesaurus if len(e['hs4_candidates']) > 1)

    print(f"\n통계:")
    print(f"  고신뢰도 (>=0.9): {high_conf}개 ({high_conf/len(thesaurus)*100:.1f}%)")
    print(f"  다중 HS4 후보: {multi_hs}개 ({multi_hs/len(thesaurus)*100:.1f}%)")

    return thesaurus


if __name__ == "__main__":
    build_thesaurus(
        ruling_cases_path="data/ruling_cases/all_cases_full_v7.json",
        output_path="kb/structured/thesaurus_terms.jsonl"
    )
