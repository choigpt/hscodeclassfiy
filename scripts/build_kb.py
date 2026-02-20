"""
KB Structured Data Builder

해설서 원문 -> 카드/규칙/동의어/요건 파일 생성.
기존 kb/build_scripts/ 빌드 스크립트를 호출하는 wrapper + 통계 출력.

Usage:
    python scripts/build_kb.py                   # 전체 빌드
    python scripts/build_kb.py --cards-only      # 카드만
    python scripts/build_kb.py --stats           # 통계만 출력
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
COMMENTARY_PATH = str(PROJECT_ROOT / "kb" / "raw" / "hs_commentary.json")
CASES_PATH = str(PROJECT_ROOT / "data" / "ruling_cases" / "all_cases_full_v7.json")
OUTPUT_DIR = PROJECT_ROOT / "kb" / "structured"

# Output files
FILES = {
    "hs4_cards.jsonl": "HS4 카드 (title, scope, includes, excludes)",
    "hs4_cards_v2.jsonl": "HS4 카드 v2 (+ required_facts)",
    "hs4_rule_chunks.jsonl": "규칙 청크 (include/exclude/definition)",
    "thesaurus_terms.jsonl": "동의어 사전",
    "disambiguation_questions.jsonl": "질문 템플릿",
    "note_requirements.jsonl": "법적 요건 (주규정 추출)",
}


def print_stats():
    """현재 KB 파일 통계 출력."""
    print("=" * 70)
    print("KB Structured Data Statistics")
    print("=" * 70)

    if not OUTPUT_DIR.exists():
        print("  (출력 디렉토리 없음)")
        return

    total_entries = 0
    for filename, desc in FILES.items():
        filepath = OUTPUT_DIR / filename
        if filepath.exists():
            count = sum(1 for line in open(filepath, "r", encoding="utf-8") if line.strip())
            total_entries += count
            size_kb = filepath.stat().st_size / 1024
            print(f"  {filename:40s} {count:>6,}건  ({size_kb:>8.1f} KB) - {desc}")
        else:
            print(f"  {filename:40s}   (없음)           - {desc}")

    print(f"\n  총 항목 수: {total_entries:,}")

    # 카드 상세 통계
    cards_path = OUTPUT_DIR / "hs4_cards.jsonl"
    if cards_path.exists():
        chapters = set()
        has_includes = 0
        has_excludes = 0
        has_scope = 0
        total = 0
        with open(cards_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                card = json.loads(line)
                total += 1
                hs4 = card.get("hs4", "")
                if len(hs4) >= 2:
                    chapters.add(hs4[:2])
                if card.get("includes"):
                    has_includes += 1
                if card.get("excludes"):
                    has_excludes += 1
                if card.get("scope"):
                    has_scope += 1

        print(f"\n  카드 상세:")
        print(f"    챕터 수: {len(chapters)}")
        print(f"    includes 보유: {has_includes}/{total} ({has_includes/total*100:.1f}%)")
        print(f"    excludes 보유: {has_excludes}/{total} ({has_excludes/total*100:.1f}%)")
        print(f"    scope 보유: {has_scope}/{total} ({has_scope/total*100:.1f}%)")

    # 시소러스 상세 통계
    thesaurus_path = OUTPUT_DIR / "thesaurus_terms.jsonl"
    if thesaurus_path.exists():
        hs4_set = set()
        total_aliases = 0
        total_terms = 0
        with open(thesaurus_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                total_terms += 1
                aliases = entry.get("aliases", [])
                total_aliases += len(aliases)
                for h in entry.get("hs4_candidates", []):
                    hs4_set.add(h)
                if entry.get("hs4"):
                    hs4_set.add(entry["hs4"])

        print(f"\n  시소러스 상세:")
        print(f"    용어 수: {total_terms}")
        print(f"    총 동의어 수: {total_aliases}")
        print(f"    커버 HS4 수: {len(hs4_set)}")


def build_cards():
    """HS4 카드 생성."""
    from kb.build_scripts.build_cards import build_hs4_cards

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(OUTPUT_DIR / "hs4_cards.jsonl")
    build_hs4_cards(commentary_path=COMMENTARY_PATH, output_path=output_path)
    return output_path


def build_cards_v2():
    """HS4 카드 v2 생성 (required_facts 추가)."""
    from kb.build_scripts.build_cards_v2 import main as build_v2_main

    build_v2_main()


def build_rules():
    """규칙 청크 생성."""
    from kb.build_scripts.build_chunks import build_rule_chunks

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(OUTPUT_DIR / "hs4_rule_chunks.jsonl")
    build_rule_chunks(commentary_path=COMMENTARY_PATH, output_path=output_path)
    return output_path


def build_thesaurus():
    """동의어 사전 생성."""
    from kb.build_scripts.build_thesaurus import build_thesaurus as _build_thesaurus

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(OUTPUT_DIR / "thesaurus_terms.jsonl")
    _build_thesaurus(ruling_cases_path=CASES_PATH, output_path=output_path)
    return output_path


def build_questions():
    """질문 템플릿 생성."""
    from kb.build_scripts.build_questions import build_question_templates

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(OUTPUT_DIR / "disambiguation_questions.jsonl")
    build_question_templates(commentary_path=COMMENTARY_PATH, output_path=output_path)
    return output_path


def build_note_requirements():
    """법적 요건 추출."""
    from kb.build_scripts.build_note_requirements import main as build_notes_main

    build_notes_main()


def build_all():
    """전체 KB 빌드 (6단계)."""
    print("=" * 70)
    print("HS Knowledge Base Full Build")
    print("=" * 70)

    t0 = time.time()
    steps = [
        ("1/6", "HS4 카드 생성", build_cards),
        ("2/6", "규칙 청크 생성", build_rules),
        ("3/6", "동의어 사전 생성", build_thesaurus),
        ("4/6", "질문 템플릿 생성", build_questions),
        ("5/6", "카드 v2 (required_facts)", build_cards_v2),
        ("6/6", "법적 요건 추출", build_note_requirements),
    ]

    for step_num, desc, fn in steps:
        print(f"\n[{step_num}] {desc}")
        print("-" * 60)
        t_step = time.time()
        try:
            fn()
            elapsed = time.time() - t_step
            print(f"  -> 완료 ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  -> 실패: {e}")

    total_time = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"전체 빌드 완료! ({total_time:.1f}s)")
    print("=" * 70)

    print_stats()


def main():
    parser = argparse.ArgumentParser(description="KB Structured Data Builder")
    parser.add_argument("--stats", action="store_true", help="통계만 출력")
    parser.add_argument("--cards-only", action="store_true", help="카드만 빌드")
    parser.add_argument("--rules-only", action="store_true", help="규칙만 빌드")
    parser.add_argument("--thesaurus-only", action="store_true", help="시소러스만 빌드")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    if args.cards_only:
        print("[카드 빌드]")
        build_cards()
        build_cards_v2()
        print_stats()
        return

    if args.rules_only:
        print("[규칙 빌드]")
        build_rules()
        print_stats()
        return

    if args.thesaurus_only:
        print("[시소러스 빌드]")
        build_thesaurus()
        print_stats()
        return

    build_all()


if __name__ == "__main__":
    main()
