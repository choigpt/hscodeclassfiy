"""
전체 KB 빌드 스크립트

모든 구조화 산출물을 순서대로 생성합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kb.build_scripts.build_cards import build_hs4_cards
from kb.build_scripts.build_chunks import build_rule_chunks
from kb.build_scripts.build_thesaurus import build_thesaurus
from kb.build_scripts.build_questions import build_question_templates


def build_all():
    """전체 KB 빌드"""
    print("=" * 70)
    print("HS 지식베이스 전체 빌드")
    print("=" * 70)
    print()

    # 출력 디렉토리 생성
    output_dir = PROJECT_ROOT / "kb" / "structured"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. HS4 카드
    print("\n[1/4] HS4 카드 생성")
    print("-" * 60)
    build_hs4_cards(
        commentary_path=str(PROJECT_ROOT / "kb/raw/hs_commentary.json"),
        output_path=str(output_dir / "hs4_cards.jsonl")
    )

    # 2. 규칙 청크
    print("\n[2/4] 규칙 청크 생성")
    print("-" * 60)
    build_rule_chunks(
        commentary_path=str(PROJECT_ROOT / "kb/raw/hs_commentary.json"),
        output_path=str(output_dir / "hs4_rule_chunks.jsonl")
    )

    # 3. 용어 사전
    print("\n[3/4] 용어 사전 생성")
    print("-" * 60)
    build_thesaurus(
        ruling_cases_path=str(PROJECT_ROOT / "data/ruling_cases/all_cases_full_v7.json"),
        output_path=str(output_dir / "thesaurus_terms.jsonl")
    )

    # 4. 질문 템플릿
    print("\n[4/4] 질문 템플릿 생성")
    print("-" * 60)
    build_question_templates(
        commentary_path=str(PROJECT_ROOT / "kb/raw/hs_commentary.json"),
        output_path=str(output_dir / "disambiguation_questions.jsonl")
    )

    print("\n" + "=" * 70)
    print("전체 빌드 완료!")
    print("=" * 70)

    # 결과 요약
    print("\n생성된 파일:")
    for f in output_dir.glob("*.jsonl"):
        lines = sum(1 for _ in open(f, 'r', encoding='utf-8'))
        print(f"  {f.name}: {lines}개 항목")


if __name__ == "__main__":
    build_all()
