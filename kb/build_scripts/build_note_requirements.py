"""
주규정에서 요구 사실(RequiredFact) 추출 빌더

tariff_notes_clean.json을 읽어서:
1. include/exclude/redirect/definition/quant 패턴 추출
2. RequiredFact로 정규화
3. kb/structured/note_requirements.jsonl로 저장
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

# 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.classifier.required_facts import (
    RequiredFact, NoteRequirement, FactAxis, FactOperator, FactHardness
)


class NoteRequirementExtractor:
    """주규정에서 요구 사실 추출기"""

    def __init__(self):
        # 재질 키워드
        self.material_keywords = {
            '플라스틱', '폴리에틸렌', '폴리프로필렌', 'PVC', '나일론',
            '금속', '철', '강', '알루미늄', '구리', '금', '은',
            '목재', '나무', '대나무', '종이', '판지',
            '유리', '도자기', '세라믹', '석재', '대리석',
            '고무', '가죽', '직물', '면', '양모', '실크', '합성섬유',
        }

        # 가공 상태 키워드
        self.processing_keywords = {
            '신선한', '냉장', '냉동', '건조', '훈제', '염장',
            '조리', '가열', '살균', '멸균',
            '미가공', '반가공', '가공',
        }

        # 완성도 키워드
        self.completeness_keywords = {
            '완성', '미완성', '미조립', '조립', '분해',
            '부분품', '구성품', '반제품',
        }

        # 기능/용도 키워드
        self.function_keywords = {
            '식용', '공업용', '의료용', '가정용', '산업용',
            '포장용', '건축용', '장식용', '운송용',
        }

    def extract_from_notes(self, notes_path: str) -> List[NoteRequirement]:
        """주규정 파일에서 요구 사실 추출"""
        with open(notes_path, 'r', encoding='utf-8') as f:
            notes_data = json.load(f)

        print(f"Loaded {len(notes_data)} notes")

        requirements = []

        for note_dict in notes_data:
            req = self._extract_from_single_note(note_dict)
            if req and req.required_facts:
                requirements.append(req)

        print(f"Extracted {len(requirements)} note requirements")
        return requirements

    def _extract_from_single_note(self, note_dict: Dict) -> Optional[NoteRequirement]:
        """단일 주규정에서 요구 사실 추출"""

        note_id = self._build_note_id(note_dict)
        note_level = note_dict['level']
        note_type = self._classify_note_type(note_dict['note_content'])
        hs_scope = self._build_hs_scope(note_dict)
        content = note_dict['note_content']

        required_facts = []

        # 1. 재질 요구 사실 추출
        material_facts = self._extract_material_facts(content, note_id, note_type)
        required_facts.extend(material_facts)

        # 2. 가공 상태 요구 사실 추출
        processing_facts = self._extract_processing_facts(content, note_id, note_type)
        required_facts.extend(processing_facts)

        # 3. 완성도 요구 사실 추출
        completeness_facts = self._extract_completeness_facts(content, note_id, note_type)
        required_facts.extend(completeness_facts)

        # 4. 기능/용도 요구 사실 추출
        function_facts = self._extract_function_facts(content, note_id, note_type)
        required_facts.extend(function_facts)

        # 5. 정량 규칙 추출
        quant_facts = self._extract_quant_facts(content, note_id, note_type)
        required_facts.extend(quant_facts)

        # 6. 제외 패턴 추출 (not_contains)
        exclude_facts = self._extract_exclude_facts(content, note_id)
        required_facts.extend(exclude_facts)

        if not required_facts:
            return None

        return NoteRequirement(
            note_id=note_id,
            note_level=note_level,
            note_type=note_type,
            hs_scope=hs_scope,
            required_facts=required_facts,
            raw_content=content
        )

    def _build_note_id(self, note_dict: Dict) -> str:
        """주규정 ID 생성"""
        level = note_dict['level']
        if level == 'section':
            return f"section_{note_dict['section_num']}_{note_dict['note_number']}"
        elif level == 'chapter':
            return f"chapter_{note_dict['chapter_num']:02d}_{note_dict['note_number']}"
        elif level == 'subheading':
            return f"subheading_{note_dict['chapter_num']:02d}_{note_dict['note_number']}"
        return f"unknown_{note_dict['note_number']}"

    def _classify_note_type(self, content: str) -> str:
        """주규정 타입 분류"""
        if re.search(r'제외한다|포함하지\s*않는다', content):
            return 'exclude'
        if re.search(r'제\d{4}호.*분류', content):
            return 'redirect'
        if re.search(r'포함한다|여기에는', content):
            return 'include'
        if re.search(r'이란|라\s*함은', content):
            return 'definition'
        return 'general'

    def _build_hs_scope(self, note_dict: Dict) -> str:
        """HS 적용 범위 생성"""
        if note_dict['level'] == 'section':
            return f"section_{note_dict['section_num']}"
        elif note_dict['chapter_num']:
            return f"chapter_{note_dict['chapter_num']:02d}"
        return "unknown"

    def _extract_material_facts(
        self, content: str, note_id: str, note_type: str
    ) -> List[RequiredFact]:
        """재질 관련 요구 사실 추출"""
        facts = []
        content_lower = content.lower()

        for material in self.material_keywords:
            material_lower = material.lower()
            if material_lower in content_lower or material in content:
                # include 주규정이면 contains, exclude면 not_contains
                if note_type == 'exclude':
                    operator = FactOperator.NOT_CONTAINS
                    hardness = FactHardness.HARD
                else:
                    operator = FactOperator.CONTAINS
                    hardness = FactHardness.SOFT

                facts.append(RequiredFact(
                    axis=FactAxis.MATERIAL,
                    operator=operator,
                    value=material,
                    hardness=hardness,
                    source_ref=note_id,
                    confidence=0.8
                ))

        return facts

    def _extract_processing_facts(
        self, content: str, note_id: str, note_type: str
    ) -> List[RequiredFact]:
        """가공 상태 관련 요구 사실 추출"""
        facts = []

        for proc in self.processing_keywords:
            if proc in content:
                if note_type == 'exclude':
                    operator = FactOperator.NOT_CONTAINS
                    hardness = FactHardness.HARD
                else:
                    operator = FactOperator.CONTAINS
                    hardness = FactHardness.SOFT

                facts.append(RequiredFact(
                    axis=FactAxis.PROCESSING,
                    operator=operator,
                    value=proc,
                    hardness=hardness,
                    source_ref=note_id,
                    confidence=0.8
                ))

        return facts

    def _extract_completeness_facts(
        self, content: str, note_id: str, note_type: str
    ) -> List[RequiredFact]:
        """완성도 관련 요구 사실 추출"""
        facts = []

        for comp in self.completeness_keywords:
            if comp in content:
                if note_type == 'exclude':
                    operator = FactOperator.NOT_CONTAINS
                    hardness = FactHardness.HARD
                else:
                    operator = FactOperator.CONTAINS
                    hardness = FactHardness.SOFT

                facts.append(RequiredFact(
                    axis=FactAxis.COMPLETENESS,
                    operator=operator,
                    value=comp,
                    hardness=hardness,
                    source_ref=note_id,
                    confidence=0.8
                ))

        return facts

    def _extract_function_facts(
        self, content: str, note_id: str, note_type: str
    ) -> List[RequiredFact]:
        """기능/용도 관련 요구 사실 추출"""
        facts = []

        for func in self.function_keywords:
            if func in content:
                if note_type == 'exclude':
                    operator = FactOperator.NOT_CONTAINS
                    hardness = FactHardness.HARD
                else:
                    operator = FactOperator.CONTAINS
                    hardness = FactHardness.SOFT

                facts.append(RequiredFact(
                    axis=FactAxis.FUNCTION,
                    operator=operator,
                    value=func,
                    hardness=hardness,
                    source_ref=note_id,
                    confidence=0.8
                ))

        return facts

    def _extract_quant_facts(
        self, content: str, note_id: str, note_type: str
    ) -> List[RequiredFact]:
        """정량 규칙 추출 (무게, 부피, 비율 등)"""
        facts = []

        # 패턴: "100분의 X", "X 킬로그램", "X%"
        quant_patterns = [
            (r'100분의\s*(\d+)', 'percentage', '%'),
            (r'(\d+)\s*%', 'percentage', '%'),
            (r'(\d+)\s*킬로그램', 'weight', 'kg'),
            (r'(\d+)\s*그램', 'weight', 'g'),
            (r'(\d+)\s*리터', 'volume', 'L'),
        ]

        for pattern, basis, unit in quant_patterns:
            matches = re.findall(pattern, content)
            for value_str in matches:
                # 비교 연산자 추론
                context_before = content[:content.find(value_str)]
                context_after = content[content.find(value_str):]

                operator = FactOperator.EQ  # 기본값
                if '초과' in context_after[:20] or '이상' in context_after[:20]:
                    operator = FactOperator.GTE
                elif '미만' in context_after[:20] or '이하' in context_after[:20]:
                    operator = FactOperator.LTE

                hardness = FactHardness.HARD if note_type == 'exclude' else FactHardness.SOFT

                facts.append(RequiredFact(
                    axis=FactAxis.QUANT,
                    operator=operator,
                    value=f"{value_str}{unit}",
                    basis=basis,
                    hardness=hardness,
                    source_ref=note_id,
                    confidence=0.9
                ))

        return facts

    def _extract_exclude_facts(self, content: str, note_id: str) -> List[RequiredFact]:
        """제외 패턴 추출 (명시적 제외 대상)"""
        facts = []

        # "다만, ... 제외한다" 패턴
        exclude_patterns = [
            r'다만[,\s]+(.+?)[은는이가를]?\s*제외한다',
            r'그러나[,\s]+(.+?)[은는이가를]?\s*포함하지\s*않는다',
        ]

        for pattern in exclude_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # 제외 대상 추출 (최대 30자)
                exclude_target = match[:30].strip()
                if len(exclude_target) > 5:
                    facts.append(RequiredFact(
                        axis=FactAxis.LEGAL,
                        operator=FactOperator.NOT_CONTAINS,
                        value=exclude_target,
                        hardness=FactHardness.HARD,
                        source_ref=note_id,
                        confidence=0.9
                    ))

        return facts


def main():
    """메인 실행 함수"""
    print("="*80)
    print("Note Requirements Builder")
    print("="*80)

    # 입력/출력 경로
    input_path = "data/tariff_notes_clean.json"
    output_dir = Path("kb/structured")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "note_requirements.jsonl"

    # 추출
    extractor = NoteRequirementExtractor()
    requirements = extractor.extract_from_notes(input_path)

    # JSONL 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for req in requirements:
            f.write(json.dumps(req.to_dict(), ensure_ascii=False) + '\n')

    print(f"\nSaved {len(requirements)} note requirements to {output_path}")

    # 통계
    total_facts = sum(len(req.required_facts) for req in requirements)
    hard_facts = sum(
        len([f for f in req.required_facts if f.hardness == 'hard'])
        for req in requirements
    )

    axis_counts = {}
    for req in requirements:
        for fact in req.required_facts:
            axis_counts[fact.axis] = axis_counts.get(fact.axis, 0) + 1

    print(f"\n{'='*80}")
    print("Statistics")
    print(f"{'='*80}")
    print(f"Total requirements: {len(requirements)}")
    print(f"Total facts: {total_facts}")
    print(f"  - Hard: {hard_facts}")
    print(f"  - Soft: {total_facts - hard_facts}")
    print(f"\nFacts by axis:")
    for axis, count in sorted(axis_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {axis}: {count}")

    # 샘플 출력 (인코딩 안전하게)
    print(f"\n{'='*80}")
    print("Sample Requirements (first 3)")
    print(f"{'='*80}")
    for req in requirements[:3]:
        print(f"\n{req.note_id} ({req.note_type}):")
        print(f"  Scope: {req.hs_scope}")
        print(f"  Facts: {len(req.required_facts)}")
        for fact in req.required_facts[:3]:
            try:
                print(f"    - {fact}")
            except UnicodeEncodeError:
                print(f"    - {fact.axis} {fact.operator} {fact.value} [{fact.hardness}]")


if __name__ == "__main__":
    main()
