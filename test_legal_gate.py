"""
LegalGate 통합 테스트
"""
import sys
from pathlib import Path

# 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.classifier.notes_loader import NotesLoader
from src.classifier.legal_gate import LegalGate
from src.classifier.types import Candidate


def test_notes_loader():
    """NotesLoader 테스트"""
    print("="*80)
    print("Test 1: NotesLoader")
    print("="*80)

    loader = NotesLoader()

    # HS4 0401 (밀크) 주규정 확인
    idx = loader.get_notes_for_hs4('0401')
    if idx:
        print(f"\nHS4 0401 (밀크):")
        print(f"  Section notes: {len(idx.section_notes)}")
        print(f"  Chapter notes: {len(idx.chapter_notes)}")
        print(f"  Subheading notes: {len(idx.subheading_notes)}")
        print(f"  Include notes: {len(idx.include_notes())}")
        print(f"  Exclude notes: {len(idx.exclude_notes())}")
        print(f"  Redirect notes: {len(idx.redirect_notes())}")

        # 제외 주규정 샘플
        if idx.exclude_notes():
            print(f"\n  제외 주규정 샘플:")
            for note in idx.exclude_notes()[:2]:
                print(f"    - {note.note_content[:100]}...")

    # 주규정 검색 테스트
    results = loader.search_notes("플라스틱", max_results=3)
    print(f"\n'플라스틱' 검색 결과: {len(results)}개")
    for note, score in results[:3]:
        ch_str = f"제{note.chapter_num:02d}류" if note.chapter_num else f"제{note.section_num}부"
        print(f"  - {ch_str}, {note.note_type}, 점수: {score:.2f}")
        print(f"    {note.note_content[:80]}...")

    return loader


def test_legal_gate(loader: NotesLoader):
    """LegalGate 테스트"""
    print("\n" + "="*80)
    print("Test 2: LegalGate")
    print("="*80)

    gate = LegalGate(notes_loader=loader)

    # 테스트 케이스 1: 밀크 (제외 주규정 있음)
    input_text = "신선한 우유 (지방 2%)"
    candidates = [
        Candidate(hs4='0401', score_ml=0.9),  # 밀크
        Candidate(hs4='0402', score_ml=0.7),  # 농축 밀크
        Candidate(hs4='0403', score_ml=0.5),  # 요구르트
    ]

    print(f"\nInput: {input_text}")
    print(f"Candidates before LegalGate: {[c.hs4 for c in candidates]}")

    passed, redirects, debug = gate.apply(input_text, candidates)

    print(f"\nLegalGate Results:")
    print(f"  Passed: {len(passed)} ({[c.hs4 for c in passed]})")
    print(f"  Excluded: {debug['excluded']} ({debug['excluded_hs4s']})")
    if debug.get('exclude_reasons'):
        print(f"  Exclude reasons:")
        for hs4, reason in debug['exclude_reasons'].items():
            print(f"    {hs4}: {reason}")
    print(f"  Redirects: {redirects}")
    print(f"  Pass rate: {debug['pass_rate']:.1%}")

    # Evidence 확인
    if passed:
        print(f"\nFirst passed candidate ({passed[0].hs4}) evidence:")
        for ev in passed[0].evidence[:3]:
            print(f"  - [{ev.kind}] {ev.text[:80]}")

    # 테스트 케이스 2: 플라스틱 제품
    print("\n" + "-"*80)
    input_text2 = "폴리에틸렌으로 만든 포장용 봉투"
    candidates2 = [
        Candidate(hs4='3923', score_ml=0.85),  # 플라스틱 제품
        Candidate(hs4='3920', score_ml=0.70),  # 플라스틱 판
        Candidate(hs4='4202', score_ml=0.45),  # 가방류
    ]

    print(f"\nInput: {input_text2}")
    print(f"Candidates before LegalGate: {[c.hs4 for c in candidates2]}")

    passed2, redirects2, debug2 = gate.apply(input_text2, candidates2)

    print(f"\nLegalGate Results:")
    print(f"  Passed: {len(passed2)} ({[c.hs4 for c in passed2]})")
    print(f"  Excluded: {debug2['excluded']} ({debug2['excluded_hs4s']})")
    print(f"  Pass rate: {debug2['pass_rate']:.1%}")


def test_pipeline_integration():
    """Pipeline 통합 테스트"""
    print("\n" + "="*80)
    print("Test 3: Pipeline Integration")
    print("="*80)

    try:
        from src.classifier.pipeline import HSPipeline

        # LegalGate 활성화된 파이프라인
        pipeline = HSPipeline(
            use_legal_gate=True,
            use_gri=True,
            use_8axis=False,  # 8축은 비활성화 (빠른 테스트)
            use_ranker=False  # ranker 없이 테스트
        )

        # 테스트 입력
        test_inputs = [
            "신선한 우유",
            "폴리에틸렌 봉투",
            "목재로 만든 의자",
        ]

        for text in test_inputs:
            print(f"\n{'-'*80}")
            print(f"Input: {text}")

            try:
                result = pipeline.classify(text, topk=3)

                print(f"\nResults:")
                for i, cand in enumerate(result.topk, 1):
                    print(f"  {i}. HS4 {cand.hs4}: {cand.score_total:.4f}")
                    if cand.features.get('gri1_definitive'):
                        print(f"     [GRI 1 확정]")

                # Debug 정보
                if 'legal_gate' in result.debug:
                    lg = result.debug['legal_gate']
                    print(f"\n  LegalGate:")
                    print(f"    Evaluated: {lg['total_evaluated']}")
                    print(f"    Passed: {lg['passed']}")
                    print(f"    Excluded: {lg['excluded']}")

                if 'gri_decision' in result.debug:
                    print(f"\n  GRI Decision: {result.debug['gri_decision']}")

            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"Pipeline import failed: {e}")
        print("(Retriever나 다른 모듈이 없을 수 있음 - 정상)")


if __name__ == "__main__":
    print("LegalGate System Integration Test\n")

    # Test 1: NotesLoader
    loader = test_notes_loader()

    # Test 2: LegalGate
    test_legal_gate(loader)

    # Test 3: Pipeline (optional)
    test_pipeline_integration()

    print("\n" + "="*80)
    print("Test Completed!")
    print("="*80)
