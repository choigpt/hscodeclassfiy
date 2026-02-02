"""
HS4 카드 v2 생성 - required_facts 필드 추가

기존 hs4_cards.jsonl에 required_facts 필드를 추가:
- title_ko와 includes/excludes에서 보수적으로 추출
- v1 하위호환 유지 (기존 필드 모두 포함)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set
import sys

# 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.classifier.required_facts import (
    RequiredFact, FactAxis, FactOperator, FactHardness
)


class CardFactExtractor:
    """카드에서 요구 사실 추출"""

    def __init__(self):
        # 물체 본질 키워드 (명사)
        self.object_keywords = {
            '말', '소', '돼지', '면양', '염소', '닭', '오리', '거위',
            '의자', '책상', '침대', '가방', '신발', '의류',
            '기계', '장치', '기구', '도구', '부품',
        }

        # 재질 키워드
        self.material_keywords = {
            '플라스틱', '폴리에틸렌', '폴리프로필렌', 'PVC',
            '금속', '철', '강', '알루미늄', '구리',
            '목재', '나무', '대나무', '종이',
            '유리', '도자기', '석재',
            '고무', '가죽', '직물', '면', '양모',
        }

        # 가공 상태 키워드
        self.processing_keywords = {
            '신선한', '냉장', '냉동', '건조', '훈제',
            '살아 있는', '죽은', '가공', '조리',
        }

        # 완성도 키워드
        self.completeness_keywords = {
            '완성', '미조립', '조립', '부분품', '구성품',
        }

    def extract_from_card(self, card: Dict) -> List[RequiredFact]:
        """단일 카드에서 요구 사실 추출"""
        facts = []

        hs4 = card['hs4']
        title = card.get('title_ko', '')
        includes = card.get('includes', [])
        excludes = card.get('excludes', [])

        # 1. 제목에서 추출
        title_facts = self._extract_from_text(
            title, f"card_{hs4}_title", is_exclude=False
        )
        facts.extend(title_facts)

        # 2. includes에서 추출 (soft)
        for inc_text in includes:
            inc_facts = self._extract_from_text(
                inc_text, f"card_{hs4}_includes", is_exclude=False
            )
            facts.extend(inc_facts)

        # 3. excludes에서 추출 (hard, not_contains)
        for exc_text in excludes:
            exc_facts = self._extract_from_text(
                exc_text, f"card_{hs4}_excludes", is_exclude=True
            )
            facts.extend(exc_facts)

        # 중복 제거 (axis + value 기준)
        seen = set()
        unique_facts = []
        for fact in facts:
            key = (fact.axis, fact.value)
            if key not in seen:
                seen.add(key)
                unique_facts.append(fact)

        return unique_facts

    def _extract_from_text(
        self, text: str, source_ref: str, is_exclude: bool
    ) -> List[RequiredFact]:
        """텍스트에서 요구 사실 추출"""
        facts = []

        if not text:
            return facts

        # Operator 결정
        if is_exclude:
            operator = FactOperator.NOT_CONTAINS
            hardness = FactHardness.HARD
        else:
            operator = FactOperator.CONTAINS
            hardness = FactHardness.SOFT  # 카드에서 추출한 것은 보수적으로 soft

        # 1. 물체 본질
        for obj in self.object_keywords:
            if obj in text:
                facts.append(RequiredFact(
                    axis=FactAxis.OBJECT,
                    operator=operator,
                    value=obj,
                    hardness=hardness,
                    source_ref=source_ref,
                    confidence=0.7
                ))

        # 2. 재질
        for material in self.material_keywords:
            if material in text:
                facts.append(RequiredFact(
                    axis=FactAxis.MATERIAL,
                    operator=operator,
                    value=material,
                    hardness=hardness,
                    source_ref=source_ref,
                    confidence=0.7
                ))

        # 3. 가공 상태
        for proc in self.processing_keywords:
            if proc in text:
                facts.append(RequiredFact(
                    axis=FactAxis.PROCESSING,
                    operator=operator,
                    value=proc,
                    hardness=hardness,
                    source_ref=source_ref,
                    confidence=0.7
                ))

        # 4. 완성도
        for comp in self.completeness_keywords:
            if comp in text:
                facts.append(RequiredFact(
                    axis=FactAxis.COMPLETENESS,
                    operator=operator,
                    value=comp,
                    hardness=hardness,
                    source_ref=source_ref,
                    confidence=0.7
                ))

        return facts


def main():
    """메인 실행 함수"""
    print("="*80)
    print("HS4 Cards v2 Builder (with required_facts)")
    print("="*80)

    # 입력/출력 경로
    input_path = Path("kb/structured/hs4_cards.jsonl")
    output_path = Path("kb/structured/hs4_cards_v2.jsonl")

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    # 추출기 생성
    extractor = CardFactExtractor()

    # 카드 읽기 및 변환
    cards_v2 = []
    total_facts = 0
    cards_with_facts = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            card = json.loads(line)

            # required_facts 추출
            required_facts = extractor.extract_from_card(card)

            # v2 카드 생성 (v1 필드 모두 포함 + required_facts)
            card_v2 = card.copy()
            card_v2['required_facts'] = [f.to_dict() for f in required_facts]
            card_v2['version'] = 'v2'

            cards_v2.append(card_v2)

            if required_facts:
                cards_with_facts += 1
                total_facts += len(required_facts)

    # v2 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for card in cards_v2:
            f.write(json.dumps(card, ensure_ascii=False) + '\n')

    print(f"\nProcessed {len(cards_v2)} cards")
    print(f"Cards with facts: {cards_with_facts}")
    print(f"Total facts: {total_facts}")
    print(f"Avg facts per card: {total_facts / len(cards_v2):.2f}")
    print(f"\nSaved to {output_path}")

    # 통계
    axis_counts = {}
    hardness_counts = {'hard': 0, 'soft': 0}

    for card in cards_v2:
        for fact_dict in card['required_facts']:
            axis = fact_dict['axis']
            hardness = fact_dict['hardness']
            axis_counts[axis] = axis_counts.get(axis, 0) + 1
            hardness_counts[hardness] = hardness_counts.get(hardness, 0) + 1

    print(f"\n{'='*80}")
    print("Statistics")
    print(f"{'='*80}")
    print(f"Facts by axis:")
    for axis, count in sorted(axis_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {axis}: {count}")

    print(f"\nFacts by hardness:")
    for hardness, count in hardness_counts.items():
        print(f"  - {hardness}: {count}")

    # 샘플
    print(f"\n{'='*80}")
    print("Sample Cards (first 3 with facts)")
    print(f"{'='*80}")

    samples = [c for c in cards_v2 if c['required_facts']][:3]
    for card in samples:
        print(f"\nHS4 {card['hs4']}: {card['title_ko']}")
        print(f"  Facts: {len(card['required_facts'])}")
        for fact_dict in card['required_facts'][:3]:
            fact = RequiredFact.from_dict(fact_dict)
            print(f"    - {fact.axis}: {fact.operator} {fact.value} [{fact.hardness}]")


if __name__ == "__main__":
    main()
