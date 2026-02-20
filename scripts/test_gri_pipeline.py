"""
GRI 순차 파이프라인 통합 테스트

10개 대표 케이스로 end-to-end 테스트:
1. GRI 순차 적용 로그 출력
2. 8개 필수 출력 항목 확인
3. Essential Character 적용 케이스 확인
4. 리스크 레벨 출력 확인
5. HS6 분류 결과 확인

Usage:
    python scripts/test_gri_pipeline.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# 대표 테스트 케이스 10개
# ============================================================
TEST_CASES = [
    {
        "text": "냉동 돼지 삼겹살",
        "expected_chapter": "02",
        "description": "GRI 1 단순 분류",
    },
    {
        "text": "면 60% 폴리에스터 40% 혼방 직물",
        "expected_chapter": "52",
        "description": "GRI 2b 혼합물",
    },
    {
        "text": "자동차 CKD 부품 세트",
        "expected_chapter": "87",
        "description": "GRI 2a 미조립 + GRI 3 세트",
    },
    {
        "text": "세트 구성: 칼, 포크, 숟가락 (스테인레스 스틸)",
        "expected_chapter": "82",
        "description": "GRI 3 세트 (Essential Character)",
    },
    {
        "text": "스마트폰 전용 가죽 케이스",
        "expected_chapter": "42",
        "description": "GRI 5 용기/케이스",
    },
    {
        "text": "LED TV 55인치",
        "expected_chapter": "85",
        "description": "전자 제품 (GRI 1)",
    },
    {
        "text": "농도 70% 이상 에탄올 수용액",
        "expected_chapter": "22",
        "description": "정량 규칙",
    },
    {
        "text": "의료용 실리콘 튜브",
        "expected_chapter": "39",
        "description": "용도 특정 (의료)",
    },
    {
        "text": "플라스틱 필름 두께 0.5mm 인쇄된",
        "expected_chapter": "39",
        "description": "재질 + 형태",
    },
    {
        "text": "미조립 가구 키트 (원목 프레임, 금속 다리)",
        "expected_chapter": "94",
        "description": "GRI 2a + 복합 재질",
    },
]

# 8개 필수 출력 항목
REQUIRED_OUTPUT_FIELDS = [
    'input_text',
    'topk',
    'decision',
    'applied_gri',
    'essential_character',
    'risk',
    'rule_references',
    'case_evidence',
]


def run_test():
    print("=" * 70)
    print("GRI 순차 파이프라인 통합 테스트")
    print("=" * 70)

    # 파이프라인 초기화
    print("\n[초기화]")
    try:
        from src.classifier.pipeline import HSPipeline
        pipeline = HSPipeline(
            use_gri=True,
            use_legal_gate=True,
            use_8axis=True,
        )
        print("  Pipeline 초기화 완료")
    except Exception as e:
        print(f"  Pipeline 초기화 실패: {e}")
        print("  KB-only 모드로 테스트 진행...")
        from src.classifier.pipeline import HSPipeline
        pipeline = HSPipeline(
            retriever=None,
            use_gri=True,
            use_legal_gate=True,
            use_8axis=True,
        )

    # 테스트 결과 집계
    total = len(TEST_CASES)
    passed = 0
    failed = 0
    errors = []

    gri_applied_count = {f"GRI{i}": 0 for i in [1, 2, 3, 5, 6]}
    risk_levels = {"LOW": 0, "MED": 0, "HIGH": 0}
    ec_applied = 0
    hs6_resolved = 0

    print(f"\n[테스트 실행] {total}건")
    print("-" * 70)

    for idx, tc in enumerate(TEST_CASES, 1):
        text = tc["text"]
        expected_ch = tc["expected_chapter"]
        desc = tc["description"]

        print(f"\n[{idx}/{total}] {text}")
        print(f"  설명: {desc}")
        print(f"  기대 류: {expected_ch}")

        try:
            start_time = time.time()
            result = pipeline.classify(text, topk=5)
            elapsed = (time.time() - start_time) * 1000

            result_dict = result.to_dict()

            # 1. 필수 출력 항목 확인
            missing_fields = []
            for field in REQUIRED_OUTPUT_FIELDS:
                if field not in result_dict:
                    missing_fields.append(field)

            if missing_fields:
                errors.append(f"[{idx}] 누락 필드: {missing_fields}")
                print(f"  [FAIL] 필수 필드 누락: {missing_fields}")
                failed += 1
                continue

            # 2. Top-1 결과
            top1 = result.topk[0] if result.topk else None
            top1_hs4 = top1.hs4 if top1 else "없음"
            top1_hs6 = top1.hs6 if top1 else ""
            top1_score = top1.score_total if top1 else 0

            print(f"  Top-1: HS4={top1_hs4}, HS6={top1_hs6}, 점수={top1_score:.4f}")
            print(f"  결정: {result.decision.status} ({result.decision.reason})")
            print(f"  소요: {elapsed:.0f}ms")

            # 3. GRI 적용 순서 확인
            if result.applied_gri:
                gri_steps = []
                for gri in result.applied_gri:
                    gri_steps.append(f"{gri.gri_id}({'O' if gri.applied else 'X'})")
                    if gri.applied and gri.gri_id in gri_applied_count:
                        gri_applied_count[gri.gri_id] += 1
                print(f"  GRI 순차: {' → '.join(gri_steps)}")

                # GRI 순서 검증: GRI1은 반드시 첫 번째
                gri_ids = [g.gri_id for g in result.applied_gri]
                if gri_ids and gri_ids[0] != "GRI1":
                    errors.append(f"[{idx}] GRI 순서 위반: {gri_ids}")
                    print(f"  [WARN] GRI 1이 첫 번째가 아님!")

            # 4. Essential Character
            if result.essential_character and result.essential_character.applicable:
                ec_applied += 1
                print(f"  EC: winner={result.essential_character.winner_hs4}")
                print(f"       {result.essential_character.reasoning}")

            # 5. 리스크
            if result.risk:
                risk_levels[result.risk.level] = risk_levels.get(result.risk.level, 0) + 1
                print(f"  리스크: {result.risk.level} (점수={result.risk.score:.1f})")
                if result.risk.reasons:
                    for r in result.risk.reasons[:2]:
                        print(f"    - {r}")

            # 6. HS6 결과
            if top1_hs6:
                hs6_resolved += 1
                print(f"  HS6: {top1_hs6}")

            # 7. Rule references
            if result.rule_references:
                print(f"  규칙 참조: {len(result.rule_references)}건")
                for ref in result.rule_references[:2]:
                    print(f"    - [{ref.source}] {ref.text_snippet[:60]}")

            passed += 1

        except Exception as e:
            errors.append(f"[{idx}] 예외: {e}")
            print(f"  [ERROR] {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    # ============================================================
    # 결과 요약
    # ============================================================
    print("\n" + "=" * 70)
    print("테스트 결과 요약")
    print("=" * 70)

    print(f"\n  총 테스트: {total}")
    print(f"  성공: {passed}")
    print(f"  실패: {failed}")

    print(f"\n  GRI 적용 분포:")
    for gri_id, count in sorted(gri_applied_count.items()):
        print(f"    {gri_id}: {count}/{total}")

    print(f"\n  리스크 레벨 분포:")
    for level in ["LOW", "MED", "HIGH"]:
        count = risk_levels.get(level, 0)
        print(f"    {level}: {count}")

    print(f"\n  Essential Character 적용: {ec_applied}/{total}")
    print(f"  HS6 해소: {hs6_resolved}/{total}")

    if errors:
        print(f"\n  오류 목록:")
        for err in errors:
            print(f"    {err}")

    # 최종 판정
    print(f"\n{'=' * 70}")
    if failed == 0:
        print("  [PASS] 모든 테스트 통과!")
    else:
        print(f"  [FAIL] {failed}건 실패")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
