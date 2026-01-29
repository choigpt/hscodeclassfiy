"""
HS 분류 파이프라인 진단 및 스모크 테스트 (GRI + 8축 전역 속성 통합)
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

from .gri_signals import detect_gri_signals, detect_parts_signal
from .attribute_extract import extract_attributes, extract_attributes_8axis, AXIS_IDS


# 8축 ID 목록
AXIS_NAMES_KO = {
    'object_nature': '물체본질',
    'material': '재질',
    'processing_state': '가공상태',
    'function_use': '기능용도',
    'physical_form': '물리형태',
    'completeness': '완성도',
    'quantitative_rules': '정량규칙',
    'legal_scope': '법적범위',
}


# 대표 품목 리스트 (GRI 테스트 포함)
SMOKE_TEST_ITEMS = [
    # 식품류
    "냉동 돼지 삼겹살",
    "냉동 소고기 안심",
    "생닭 통닭",
    "냉동 새우",
    "참치 통조림",
    "볶은 커피 원두",
    "인스턴트 커피",
    "녹차 티백",
    "밀크 초콜릿",
    "과일 주스 농축액",
    "올리브 오일",
    "라면 스프",
    "냉동 피자",
    "아이스크림",
    "맥주",
    "위스키",
    # 전자제품
    "스마트폰",
    "노트북 컴퓨터",
    "태블릿 PC",
    "무선 이어폰",
    "블루투스 스피커",
    "LED TV 55인치",
    "전자레인지",
    "세탁기 드럼형",
    "에어컨 실외기",
    "냉장고",
    "전기밥솥",
    "헤어드라이어",
    # 의류/잡화
    "남성용 면 티셔츠",
    "여성용 청바지",
    "운동화 러닝화",
    "가죽 핸드백",
    "선글라스",
    "손목시계 쿼츠",
    # 기계/부품
    "자동차 엔진 부품",
    "베어링 볼베어링",
    "전동공구 드릴",
    "펌프 원심펌프",
    # 화학/의약
    "아스피린 정제",
    "비타민C 보충제",
    "샴푸",
    "립스틱 화장품",
    "플라스틱 필름",
    "고무 타이어",
    # 기타
    "종이 박스",
    "유리병",
    "철강 파이프",
    "알루미늄 호일",
    # GRI 테스트용 추가
    "자동차 CKD 부품 세트",
    "면 60% 폴리에스터 40% 혼방 직물",
    "스마트폰 전용 케이스",
    "미조립 가구 키트",
    "식기 세트 (칼, 포크, 숟가락)",
]


def run_smoke_test(
    output_dir: str = "artifacts/diagnostics",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    스모크 테스트 실행 (GRI 플래그 포함)

    Args:
        output_dir: 결과 저장 디렉토리
        verbose: 상세 출력 여부

    Returns:
        테스트 결과 요약
    """
    from .pipeline import HSPipeline

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("스모크 테스트 시작 (GRI + 전역 속성 통합)")
    print("=" * 60)
    print(f"테스트 항목: {len(SMOKE_TEST_ITEMS)}개")
    print()

    # 파이프라인 초기화
    print("[파이프라인 초기화 중...]")
    try:
        pipeline = HSPipeline()
    except RuntimeError as e:
        print(f"오류: {e}")
        print("모델이 없습니다. 먼저 학습을 실행하세요.")
        return {"error": str(e)}

    # Label space 진단
    print("\n[Label Space 진단]")
    diagnosis = pipeline.diagnose_label_space()
    print(f"  모델 클래스: {diagnosis['model_classes_count']}개")
    print(f"  KB 클래스: {diagnosis['kb_classes_count']}개")
    print(f"  공통: {diagnosis['common_count']}개")
    print(f"  모델 커버리지: {diagnosis['coverage_model']*100:.1f}%")

    # 테스트 실행
    print("\n[분류 테스트 실행]")
    results = []
    results_8axis = []  # 8축 리포트용 결과
    stats = {
        'total': len(SMOKE_TEST_ITEMS),
        'kb_hit_count': 0,
        'low_confidence_count': 0,
        'avg_evidence_count': 0,
        'avg_top1_score': 0,
        # GRI 통계
        'gri_signal_count': 0,
        'gri2a_count': 0,
        'gri2b_count': 0,
        'gri3_count': 0,
        'gri5_count': 0,
        'parts_signal_count': 0,
        # 전역 속성 통계
        'state_present_count': 0,
        'material_present_count': 0,
        'use_present_count': 0,
        'form_present_count': 0,
        'quant_present_count': 0,
        'attr_match_count': 0,
        'hard_exclude_count': 0,
        # 질문 축 분포
        'question_axis_dist': {},
        # 8축 통계
        'axis_8_extraction_count': {axis: 0 for axis in AXIS_IDS},
    }

    total_evidence = 0
    total_score = 0

    for i, item in enumerate(SMOKE_TEST_ITEMS, 1):
        result = pipeline.classify(item, topk=5)

        # GRI 신호
        gri_signals = result.debug.get('gri_signals', {})
        active_gri = result.debug.get('active_gri', [])
        parts_signal = result.debug.get('parts_signal', {})

        # 전역 속성
        input_attrs = result.debug.get('input_attrs', {})
        attrs_summary = result.debug.get('attrs_summary', '')

        # 통계 계산
        top1 = result.topk[0] if result.topk else None
        has_kb_hit = result.debug.get('rerank_stats', {}).get('any_hit_count', 0) > 0
        evidence_count = sum(len(c.evidence) for c in result.topk)

        if has_kb_hit:
            stats['kb_hit_count'] += 1
        if result.low_confidence:
            stats['low_confidence_count'] += 1

        total_evidence += evidence_count
        if top1:
            total_score += top1.score_total

        # GRI 통계
        if active_gri:
            stats['gri_signal_count'] += 1
        if gri_signals.get('gri2a_incomplete'):
            stats['gri2a_count'] += 1
        if gri_signals.get('gri2b_mixtures'):
            stats['gri2b_count'] += 1
        if gri_signals.get('gri3_multi_candidate'):
            stats['gri3_count'] += 1
        if gri_signals.get('gri5_containers'):
            stats['gri5_count'] += 1
        if parts_signal.get('is_parts'):
            stats['parts_signal_count'] += 1

        # 전역 속성 통계
        if input_attrs.get('states'):
            stats['state_present_count'] += 1
        if input_attrs.get('materials'):
            stats['material_present_count'] += 1
        if input_attrs.get('uses_functions'):
            stats['use_present_count'] += 1
        if input_attrs.get('forms'):
            stats['form_present_count'] += 1
        if input_attrs.get('has_quant'):
            stats['quant_present_count'] += 1

        # 속성 매칭/제외 통계
        rerank_stats = result.debug.get('rerank_stats', {})
        if rerank_stats.get('hard_exclude_count', 0) > 0:
            stats['hard_exclude_count'] += 1

        # Top-1 속성 매칭 점수
        if top1 and top1.features:
            f = top1.features
            attr_match_sum = (
                f.get('f_state_match', 0) +
                f.get('f_material_match', 0) +
                f.get('f_use_match', 0) +
                f.get('f_form_match', 0)
            )
            if attr_match_sum > 0:
                stats['attr_match_count'] += 1

        # 질문 축 분포
        for q in result.questions:
            # 질문 내용에서 축 추정
            if any(kw in q for kw in ['상태', '가공', '신선', '냉동', '건조', '조리']):
                stats['question_axis_dist']['state'] = stats['question_axis_dist'].get('state', 0) + 1
            elif any(kw in q for kw in ['재질', '성분', '소재', '금속', '플라스틱', '목재']):
                stats['question_axis_dist']['material'] = stats['question_axis_dist'].get('material', 0) + 1
            elif any(kw in q for kw in ['용도', '기능', '산업', '가정', '의료']):
                stats['question_axis_dist']['use'] = stats['question_axis_dist'].get('use', 0) + 1
            elif any(kw in q for kw in ['형태', '형상', '완제품', '반제품', '원료']):
                stats['question_axis_dist']['form'] = stats['question_axis_dist'].get('form', 0) + 1
            elif any(kw in q for kw in ['부품', '부속', '완제품입니까']):
                stats['question_axis_dist']['parts'] = stats['question_axis_dist'].get('parts', 0) + 1
            elif any(kw in q for kw in ['함량', '농도', '비율', '%', '중량']):
                stats['question_axis_dist']['quant'] = stats['question_axis_dist'].get('quant', 0) + 1
            elif any(kw in q for kw in ['미조립', 'CKD', 'SKD']):
                stats['question_axis_dist']['gri2a'] = stats['question_axis_dist'].get('gri2a', 0) + 1
            elif any(kw in q for kw in ['혼합', '혼방', '본질적']):
                stats['question_axis_dist']['gri2b'] = stats['question_axis_dist'].get('gri2b', 0) + 1
            elif any(kw in q for kw in ['세트', '구성']):
                stats['question_axis_dist']['gri3'] = stats['question_axis_dist'].get('gri3', 0) + 1
            elif any(kw in q for kw in ['케이스', '용기', '포장']):
                stats['question_axis_dist']['gri5'] = stats['question_axis_dist'].get('gri5', 0) + 1

        # 8축 속성 추출 및 통계
        input_attrs_8axis = result.debug.get('input_attrs_8axis', {})
        input_attrs_8axis_summary = result.debug.get('input_attrs_8axis_summary', '')

        # 8축 추출 통계
        for axis in AXIS_IDS:
            if axis == 'quantitative_rules':
                has_value = bool(input_attrs_8axis.get('quantitative_rules', []))
            else:
                axis_data = input_attrs_8axis.get(axis, {})
                has_value = bool(axis_data.get('values', []))
            if has_value:
                stats['axis_8_extraction_count'][axis] += 1

        # 8축 리포트용 데이터 수집
        top1_features_dict = top1.features if top1 and top1.features else {}
        results_8axis.append({
            'item': item,
            'top1_hs4': top1.hs4 if top1 else '',
            'top1_score': top1.score_total if top1 else 0,
            'low_confidence': result.low_confidence,
            'input_attrs_8axis': input_attrs_8axis,
            'top1_features': top1_features_dict,
            'axis_conflicts': result.debug.get('axis_conflicts', {}),
        })

        # 결과 저장
        row = {
            'item': item,
            'top1_hs4': top1.hs4 if top1 else '',
            'top1_score': round(top1.score_total, 4) if top1 else 0,
            'top1_ml': round(top1.score_ml, 4) if top1 else 0,
            'top1_card': round(top1.score_card, 4) if top1 else 0,
            'top1_rule': round(top1.score_rule, 4) if top1 else 0,
            'top2_hs4': result.topk[1].hs4 if len(result.topk) > 1 else '',
            'top3_hs4': result.topk[2].hs4 if len(result.topk) > 2 else '',
            'top4_hs4': result.topk[3].hs4 if len(result.topk) > 3 else '',
            'top5_hs4': result.topk[4].hs4 if len(result.topk) > 4 else '',
            'has_kb_hit': has_kb_hit,
            'evidence_count': evidence_count,
            'low_confidence': result.low_confidence,
            'ml_candidates': result.debug.get('ml_candidates_count', 0),
            'kb_candidates': result.debug.get('kb_candidates_count', 0),
            'merged_candidates': result.debug.get('merged_candidates_count', 0),
            'card_hit_rate': round(result.debug.get('rerank_stats', {}).get('card_hit_rate', 0), 4),
            'rule_hit_rate': round(result.debug.get('rerank_stats', {}).get('rule_hit_rate', 0), 4),
            # GRI 플래그
            'gri_active': ','.join(active_gri) if active_gri else '',
            'gri2a': 1 if gri_signals.get('gri2a_incomplete') else 0,
            'gri2b': 1 if gri_signals.get('gri2b_mixtures') else 0,
            'gri3': 1 if gri_signals.get('gri3_multi_candidate') else 0,
            'gri5': 1 if gri_signals.get('gri5_containers') else 0,
            'is_parts': 1 if parts_signal.get('is_parts') else 0,
            # 전역 속성 (기존 7축)
            'attrs_summary': attrs_summary[:100] if attrs_summary else '',
            'has_state': 1 if input_attrs.get('states') else 0,
            'has_material': 1 if input_attrs.get('materials') else 0,
            'has_use': 1 if input_attrs.get('uses_functions') else 0,
            'has_form': 1 if input_attrs.get('forms') else 0,
            'has_quant': 1 if input_attrs.get('has_quant') else 0,
            # 8축 속성 요약
            'attrs_8axis_summary': input_attrs_8axis_summary[:100] if input_attrs_8axis_summary else '',
            # 피처 (Top-1)
            'f_specificity': round(top1.features.get('f_specificity', 0), 4) if top1 and top1.features else 0,
            'f_exclude_conflict': top1.features.get('f_exclude_conflict', 0) if top1 and top1.features else 0,
            'f_state_match': top1.features.get('f_state_match', 0) if top1 and top1.features else 0,
            'f_material_match': top1.features.get('f_material_match', 0) if top1 and top1.features else 0,
            'f_use_match': top1.features.get('f_use_match', 0) if top1 and top1.features else 0,
            'f_form_match': top1.features.get('f_form_match', 0) if top1 and top1.features else 0,
            'f_hard_exclude': (
                (top1.features.get('f_note_hard_exclude', 0) or top1.features.get('f_quant_hard_exclude', 0))
                if top1 and top1.features else 0
            ),
            # 8축 피처 (Top-1)
            'f_object_match': round(top1.features.get('f_object_match_score', 0), 4) if top1 and top1.features else 0,
            'f_processing_match': round(top1.features.get('f_processing_match_score', 0), 4) if top1 and top1.features else 0,
            'f_function_match': round(top1.features.get('f_function_match_score', 0), 4) if top1 and top1.features else 0,
            'f_completeness_match': round(top1.features.get('f_completeness_match_score', 0), 4) if top1 and top1.features else 0,
            'f_legal_match': round(top1.features.get('f_legal_scope_match_score', 0), 4) if top1 and top1.features else 0,
            'f_conflict_penalty': round(top1.features.get('f_conflict_penalty', 0), 4) if top1 and top1.features else 0,
            'f_uncertainty_penalty': round(top1.features.get('f_uncertainty_penalty', 0), 4) if top1 and top1.features else 0,
            # 질문
            'questions': ' | '.join(result.questions) if result.questions else '',
        }
        results.append(row)

        if verbose:
            kb_tag = "[KB]" if has_kb_hit else "[NO_KB]"
            conf_tag = "[LOW]" if result.low_confidence else "[OK]"
            gri_tag = f"[GRI:{','.join(active_gri)}]" if active_gri else ""
            attr_tag = f"[ATTR:{attrs_summary[:30]}]" if attrs_summary else ""
            print(f"  [{i:2d}] {item[:20]:20s} -> {row['top1_hs4']} ({row['top1_score']:.3f}) {kb_tag} {conf_tag} {gri_tag}")

    # 평균 계산
    stats['avg_evidence_count'] = total_evidence / stats['total']
    stats['avg_top1_score'] = total_score / stats['total']
    stats['kb_hit_rate'] = stats['kb_hit_count'] / stats['total']
    stats['low_confidence_rate'] = stats['low_confidence_count'] / stats['total']
    stats['gri_signal_rate'] = stats['gri_signal_count'] / stats['total']
    # 속성 비율
    stats['state_present_rate'] = stats['state_present_count'] / stats['total']
    stats['material_present_rate'] = stats['material_present_count'] / stats['total']
    stats['use_present_rate'] = stats['use_present_count'] / stats['total']
    stats['form_present_rate'] = stats['form_present_count'] / stats['total']
    stats['quant_present_rate'] = stats['quant_present_count'] / stats['total']
    stats['attr_match_rate'] = stats['attr_match_count'] / stats['total']

    # CSV 저장
    csv_file = output_path / "smoke_test.csv"
    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n결과 저장: {csv_file}")

    # JSON 저장 (상세)
    json_file = output_path / "smoke_test.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'stats': stats,
            'diagnosis': diagnosis,
            'results': results
        }, f, ensure_ascii=False, indent=2)

    print(f"상세 저장: {json_file}")

    # 속성 보고서 저장
    attr_report = {
        'attribute_presence': {
            'state': {'count': stats['state_present_count'], 'rate': stats['state_present_rate']},
            'material': {'count': stats['material_present_count'], 'rate': stats['material_present_rate']},
            'use': {'count': stats['use_present_count'], 'rate': stats['use_present_rate']},
            'form': {'count': stats['form_present_count'], 'rate': stats['form_present_rate']},
            'quant': {'count': stats['quant_present_count'], 'rate': stats['quant_present_rate']},
        },
        'attribute_matching': {
            'match_count': stats['attr_match_count'],
            'match_rate': stats['attr_match_rate'],
            'hard_exclude_count': stats['hard_exclude_count'],
        },
        'question_axis_distribution': stats['question_axis_dist'],
    }
    attr_report_file = output_path / "attributes_report.json"
    with open(attr_report_file, 'w', encoding='utf-8') as f:
        json.dump(attr_report, f, ensure_ascii=False, indent=2)

    print(f"속성 보고서: {attr_report_file}")

    # 8축 리포트 생성
    print("\n[8축 진단 리포트 생성 중...]")
    report_8axis = generate_8axis_report(results_8axis, output_dir)
    report_8axis_file = report_8axis.get('json_file', '')

    # 요약 출력
    print("\n" + "=" * 60)
    print("스모크 테스트 결과 요약")
    print("=" * 60)
    print(f"  테스트 항목: {stats['total']}개")
    print(f"  KB 히트율: {stats['kb_hit_rate']*100:.1f}% ({stats['kb_hit_count']}/{stats['total']})")
    print(f"  저신뢰도 비율: {stats['low_confidence_rate']*100:.1f}% ({stats['low_confidence_count']}/{stats['total']})")
    print(f"  평균 근거 수: {stats['avg_evidence_count']:.1f}")
    print(f"  평균 Top-1 점수: {stats['avg_top1_score']:.4f}")

    print("\n[GRI 신호 통계]")
    print(f"  GRI 신호 탐지: {stats['gri_signal_count']}개 ({stats['gri_signal_rate']*100:.1f}%)")
    print(f"  GRI 2(a) 미조립: {stats['gri2a_count']}개")
    print(f"  GRI 2(b) 혼합물: {stats['gri2b_count']}개")
    print(f"  GRI 3 세트: {stats['gri3_count']}개")
    print(f"  GRI 5 케이스: {stats['gri5_count']}개")
    print(f"  부품 신호: {stats['parts_signal_count']}개")

    print("\n[전역 속성 통계]")
    print(f"  상태(state) 추출: {stats['state_present_count']}개 ({stats['state_present_rate']*100:.1f}%)")
    print(f"  재질(material) 추출: {stats['material_present_count']}개 ({stats['material_present_rate']*100:.1f}%)")
    print(f"  용도(use) 추출: {stats['use_present_count']}개 ({stats['use_present_rate']*100:.1f}%)")
    print(f"  형태(form) 추출: {stats['form_present_count']}개 ({stats['form_present_rate']*100:.1f}%)")
    print(f"  정량(quant) 추출: {stats['quant_present_count']}개 ({stats['quant_present_rate']*100:.1f}%)")
    print(f"  속성 매칭 성공: {stats['attr_match_count']}개 ({stats['attr_match_rate']*100:.1f}%)")
    print(f"  Hard Exclude 적용: {stats['hard_exclude_count']}개")

    # 8축 추출 통계 출력
    print("\n[8축 전역 속성 추출 통계]")
    for axis in AXIS_IDS:
        count = stats['axis_8_extraction_count'].get(axis, 0)
        rate = count / stats['total'] * 100
        bar = "█" * int(rate / 5) + "░" * (20 - int(rate / 5))
        print(f"  {AXIS_NAMES_KO.get(axis, axis):10s} {bar} {rate:5.1f}% ({count}/{stats['total']})")

    if stats['question_axis_dist']:
        print("\n[질문 축 분포]")
        for axis, count in sorted(stats['question_axis_dist'].items(), key=lambda x: -x[1]):
            print(f"  {axis}: {count}개")

    # 경고
    if stats['kb_hit_rate'] < 0.5:
        print(f"\n[!] 경고: KB 히트율이 50% 미만입니다. KB 데이터 확인 필요.")
    if stats['low_confidence_rate'] > 0.7:
        print(f"\n[!] 경고: 저신뢰도 비율이 70%를 초과합니다.")

    return {
        'stats': stats,
        'diagnosis': diagnosis,
        'csv_file': str(csv_file),
        'json_file': str(json_file),
        'attr_report_file': str(attr_report_file),
        'report_8axis_file': report_8axis_file,
    }


def compute_8axis_stats(results: List[Dict]) -> Dict[str, Any]:
    """
    8축 기반 통계 계산

    Args:
        results: 분류 결과 리스트

    Returns:
        8축 통계
    """
    total = len(results)
    if total == 0:
        return {}

    stats = {
        # 축별 속성 추출 성공률
        'axis_extraction_rate': {axis: 0.0 for axis in AXIS_IDS},
        'axis_extraction_count': {axis: 0 for axis in AXIS_IDS},

        # 축별 후보 일치율 (top1 기준)
        'axis_match_rate': {axis: 0.0 for axis in AXIS_IDS},
        'axis_match_count': {axis: 0 for axis in AXIS_IDS},

        # 축별 충돌 빈도
        'axis_conflict_frequency': {axis: 0 for axis in AXIS_IDS},
        'axis_avg_conflict_score': {axis: 0.0 for axis in AXIS_IDS},

        # 저신뢰도 감소 기여도 (어떤 축 정보가 신뢰도 향상에 기여했는지)
        'confidence_contribution': {axis: 0.0 for axis in AXIS_IDS},

        # HS 류별 성능 분포
        'chapter_performance': defaultdict(lambda: {'count': 0, 'avg_score': 0.0, 'low_conf_count': 0}),

        # 전체 통계
        'total_items': total,
        'avg_axes_extracted': 0.0,
        'items_with_all_axes': 0,
        'items_with_no_axes': 0,
    }

    conflict_sums = {axis: 0.0 for axis in AXIS_IDS}
    conflict_counts = {axis: 0 for axis in AXIS_IDS}
    contribution_sums = {axis: 0.0 for axis in AXIS_IDS}
    total_axes_count = 0

    for r in results:
        input_attrs_8axis = r.get('input_attrs_8axis', {})
        top1_features = r.get('top1_features', {})
        top1_hs4 = r.get('top1_hs4', '')
        top1_score = r.get('top1_score', 0)
        low_confidence = r.get('low_confidence', False)

        # 축별 추출 성공 체크
        axes_extracted = 0
        for axis in AXIS_IDS:
            if axis == 'quantitative_rules':
                has_value = bool(input_attrs_8axis.get('quantitative_rules', []))
            else:
                axis_data = input_attrs_8axis.get(axis, {})
                has_value = bool(axis_data.get('values', []))

            if has_value:
                stats['axis_extraction_count'][axis] += 1
                axes_extracted += 1

        total_axes_count += axes_extracted

        if axes_extracted == len(AXIS_IDS):
            stats['items_with_all_axes'] += 1
        elif axes_extracted == 0:
            stats['items_with_no_axes'] += 1

        # 축별 매칭 점수 (top1 기준)
        for axis in AXIS_IDS:
            feature_key = f'f_{axis}_match_score'
            if axis == 'quantitative_rules':
                feature_key = 'f_quant_rule_match_score'
            match_score = top1_features.get(feature_key, 0)
            if match_score > 0:
                stats['axis_match_count'][axis] += 1

        # 충돌 점수
        conflict_penalty = top1_features.get('f_conflict_penalty', 0)
        axis_conflicts = r.get('axis_conflicts', {})
        for axis, score in axis_conflicts.items():
            if score > 0:
                stats['axis_conflict_frequency'][axis] += 1
                conflict_sums[axis] += score
                conflict_counts[axis] += 1

        # 류별 성능
        if top1_hs4 and len(top1_hs4) >= 2:
            chapter = top1_hs4[:2]
            stats['chapter_performance'][chapter]['count'] += 1
            stats['chapter_performance'][chapter]['avg_score'] += top1_score
            if low_confidence:
                stats['chapter_performance'][chapter]['low_conf_count'] += 1

        # 신뢰도 기여도 (축 매칭 점수가 높을수록 신뢰도 향상에 기여)
        if not low_confidence:
            for axis in AXIS_IDS:
                feature_key = f'f_{axis}_match_score'
                if axis == 'quantitative_rules':
                    feature_key = 'f_quant_rule_match_score'
                match_score = top1_features.get(feature_key, 0)
                contribution_sums[axis] += match_score

    # 비율 계산
    for axis in AXIS_IDS:
        stats['axis_extraction_rate'][axis] = stats['axis_extraction_count'][axis] / total
        stats['axis_match_rate'][axis] = stats['axis_match_count'][axis] / total

        if conflict_counts[axis] > 0:
            stats['axis_avg_conflict_score'][axis] = conflict_sums[axis] / conflict_counts[axis]

        if total > 0:
            stats['confidence_contribution'][axis] = contribution_sums[axis] / total

    stats['avg_axes_extracted'] = total_axes_count / total if total > 0 else 0

    # 류별 평균 점수 계산
    for chapter, data in stats['chapter_performance'].items():
        if data['count'] > 0:
            data['avg_score'] = data['avg_score'] / data['count']
            data['low_conf_rate'] = data['low_conf_count'] / data['count']

    # defaultdict를 일반 dict로 변환
    stats['chapter_performance'] = dict(stats['chapter_performance'])

    return stats


def generate_8axis_report(
    results: List[Dict],
    output_dir: str = "artifacts/diagnostics"
) -> Dict[str, Any]:
    """
    8축 기반 진단 리포트 생성

    Args:
        results: 분류 결과 리스트 (각 결과에 input_attrs_8axis, top1_features 포함)
        output_dir: 출력 디렉토리

    Returns:
        리포트 요약
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 8축 통계 계산
    stats_8axis = compute_8axis_stats(results)

    if not stats_8axis:
        print("결과 데이터가 없습니다.")
        return {}

    # JSON 리포트 저장
    report = {
        'summary': {
            'total_items': stats_8axis['total_items'],
            'avg_axes_extracted': round(stats_8axis['avg_axes_extracted'], 2),
            'items_with_all_axes': stats_8axis['items_with_all_axes'],
            'items_with_no_axes': stats_8axis['items_with_no_axes'],
        },
        'axis_extraction_rate': {
            axis: round(rate, 4)
            for axis, rate in stats_8axis['axis_extraction_rate'].items()
        },
        'axis_match_rate': {
            axis: round(rate, 4)
            for axis, rate in stats_8axis['axis_match_rate'].items()
        },
        'axis_conflict_frequency': stats_8axis['axis_conflict_frequency'],
        'axis_avg_conflict_score': {
            axis: round(score, 4)
            for axis, score in stats_8axis['axis_avg_conflict_score'].items()
        },
        'confidence_contribution': {
            axis: round(contrib, 4)
            for axis, contrib in stats_8axis['confidence_contribution'].items()
        },
        'chapter_performance': {
            ch: {
                'count': data['count'],
                'avg_score': round(data['avg_score'], 4),
                'low_conf_rate': round(data.get('low_conf_rate', 0), 4)
            }
            for ch, data in stats_8axis['chapter_performance'].items()
        },
    }

    json_file = output_path / "8axis_report.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # CSV 축별 통계
    axis_csv_file = output_path / "8axis_stats.csv"
    with open(axis_csv_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['axis_id', 'axis_name_ko', 'extraction_rate', 'match_rate',
                        'conflict_frequency', 'avg_conflict_score', 'confidence_contribution'])
        for axis in AXIS_IDS:
            writer.writerow([
                axis,
                AXIS_NAMES_KO.get(axis, axis),
                round(stats_8axis['axis_extraction_rate'][axis], 4),
                round(stats_8axis['axis_match_rate'][axis], 4),
                stats_8axis['axis_conflict_frequency'][axis],
                round(stats_8axis['axis_avg_conflict_score'][axis], 4),
                round(stats_8axis['confidence_contribution'][axis], 4),
            ])

    # CSV 류별 성능
    chapter_csv_file = output_path / "chapter_performance.csv"
    with open(chapter_csv_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['chapter', 'count', 'avg_score', 'low_conf_rate'])
        for ch in sorted(stats_8axis['chapter_performance'].keys()):
            data = stats_8axis['chapter_performance'][ch]
            writer.writerow([
                ch,
                data['count'],
                round(data['avg_score'], 4),
                round(data.get('low_conf_rate', 0), 4),
            ])

    # 콘솔 출력
    print("\n" + "=" * 60)
    print("8축 전역 속성 진단 리포트")
    print("=" * 60)

    print(f"\n[요약]")
    print(f"  총 항목: {stats_8axis['total_items']}개")
    print(f"  평균 추출 축 수: {stats_8axis['avg_axes_extracted']:.2f}/8")
    print(f"  전체 축 추출 성공: {stats_8axis['items_with_all_axes']}개")
    print(f"  축 추출 실패 (0개): {stats_8axis['items_with_no_axes']}개")

    print(f"\n[축별 추출률]")
    for axis in AXIS_IDS:
        rate = stats_8axis['axis_extraction_rate'][axis]
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"  {AXIS_NAMES_KO.get(axis, axis):10s} {bar} {rate*100:5.1f}%")

    print(f"\n[축별 매칭률 (Top-1)]")
    for axis in AXIS_IDS:
        rate = stats_8axis['axis_match_rate'][axis]
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"  {AXIS_NAMES_KO.get(axis, axis):10s} {bar} {rate*100:5.1f}%")

    print(f"\n[축별 충돌 빈도]")
    sorted_conflicts = sorted(stats_8axis['axis_conflict_frequency'].items(), key=lambda x: -x[1])
    for axis, freq in sorted_conflicts[:5]:
        if freq > 0:
            avg_score = stats_8axis['axis_avg_conflict_score'][axis]
            print(f"  {AXIS_NAMES_KO.get(axis, axis):10s} {freq}회 (평균 충돌 점수: {avg_score:.3f})")

    print(f"\n[신뢰도 기여도 (높을수록 분류 정확도에 기여)]")
    sorted_contrib = sorted(stats_8axis['confidence_contribution'].items(), key=lambda x: -x[1])
    for axis, contrib in sorted_contrib:
        if contrib > 0:
            bar = "█" * int(contrib * 40) + "░" * max(0, 10 - int(contrib * 40))
            print(f"  {AXIS_NAMES_KO.get(axis, axis):10s} {bar} {contrib:.3f}")

    print(f"\n저장된 파일:")
    print(f"  - {json_file}")
    print(f"  - {axis_csv_file}")
    print(f"  - {chapter_csv_file}")

    return {
        'report': report,
        'json_file': str(json_file),
        'axis_csv_file': str(axis_csv_file),
        'chapter_csv_file': str(chapter_csv_file),
    }


def run_single_test(text: str, verbose: bool = True) -> Dict[str, Any]:
    """
    단일 품목 테스트 (디버깅용)

    Args:
        text: 테스트할 품명
        verbose: 상세 출력 여부

    Returns:
        분류 결과
    """
    from .pipeline import HSPipeline

    pipeline = HSPipeline()
    result = pipeline.classify(text, topk=5)

    if verbose:
        print(f"입력: {text}")

        # GRI 신호
        active_gri = result.debug.get('active_gri', [])
        if active_gri:
            print(f"GRI 신호: {', '.join(active_gri)}")

        parts = result.debug.get('parts_signal', {})
        if parts.get('is_parts'):
            print(f"부품 신호: {parts.get('matched', [])}")

        # 전역 속성 (기존)
        attrs_summary = result.debug.get('attrs_summary', '')
        if attrs_summary:
            print(f"전역 속성 (7축): {attrs_summary}")

        # 8축 전역 속성
        attrs_8axis_summary = result.debug.get('input_attrs_8axis_summary', '')
        if attrs_8axis_summary:
            print(f"전역 속성 (8축): {attrs_8axis_summary}")

        print("-" * 40)
        for i, c in enumerate(result.topk, 1):
            in_model = c.hs4 in pipeline.get_model_classes()
            model_tag = "" if in_model else " [NOT_IN_MODEL]"
            print(f"[{i}] HS {c.hs4}{model_tag}")
            print(f"    총점: {c.score_total:.4f} (ML: {c.score_ml:.4f}, Card: {c.score_card:.4f}, Rule: {c.score_rule:.4f})")
            if c.features:
                f = c.features
                print(f"    피처: spec={f.get('f_specificity', 0):.2f}, exc={f.get('f_exclude_conflict', 0)}")
                # 속성 매칭 피처 (기존 7축)
                attr_feats = []
                if f.get('f_state_match', 0) > 0:
                    attr_feats.append(f"state={f.get('f_state_match')}")
                if f.get('f_material_match', 0) > 0:
                    attr_feats.append(f"material={f.get('f_material_match')}")
                if f.get('f_use_match', 0) > 0:
                    attr_feats.append(f"use={f.get('f_use_match')}")
                if f.get('f_form_match', 0) > 0:
                    attr_feats.append(f"form={f.get('f_form_match')}")
                if attr_feats:
                    print(f"    속성매칭(7축): {', '.join(attr_feats)}")

                # 8축 매칭 피처
                axis8_feats = []
                for axis in AXIS_IDS:
                    key = f'f_{axis}_match_score'
                    if axis == 'quantitative_rules':
                        key = 'f_quant_rule_match_score'
                    score = f.get(key, 0)
                    if score > 0:
                        axis8_feats.append(f"{AXIS_NAMES_KO.get(axis, axis)[:4]}={score:.2f}")
                if axis8_feats:
                    print(f"    속성매칭(8축): {', '.join(axis8_feats)}")

                # 패널티
                conflict = f.get('f_conflict_penalty', 0)
                uncertainty = f.get('f_uncertainty_penalty', 0)
                if conflict > 0 or uncertainty > 0:
                    print(f"    패널티: conflict={conflict:.3f}, uncertainty={uncertainty:.3f}")

            if c.evidence:
                for ev in c.evidence[:2]:
                    print(f"    - [{ev.kind}] {ev.text[:60]}...")
        print("-" * 40)
        print(f"저신뢰도: {result.low_confidence}")
        if result.questions:
            print("질문:")
            for q in result.questions[:3]:
                print(f"  - {q}")

    return result.to_dict()


def build_gri_tags(
    cases_path: str = "data/ruling_cases/all_cases_full_v7.json",
    output_dir: str = "artifacts/diagnostics"
) -> Dict[str, Any]:
    """
    결정사례에서 GRI 태그 약지도 부여

    Args:
        cases_path: 결정사례 JSON 경로
        output_dir: 출력 디렉토리

    Returns:
        태그 통계
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 결정사례 로드
    cases_file = Path(cases_path)
    if not cases_file.exists():
        print(f"오류: 결정사례 파일 없음: {cases_file}")
        return {"error": "file not found"}

    with open(cases_file, 'r', encoding='utf-8') as f:
        cases = json.load(f)

    print(f"결정사례 로드: {len(cases)}개")

    # GRI 태그 부여
    stats = {
        'total': len(cases),
        'gri1': 0,
        'gri2a': 0,
        'gri2b': 0,
        'gri3': 0,
        'gri5': 0,
        'parts': 0,
        'any_gri': 0,
    }

    results = []
    for case in cases:
        product_name = case.get('product_name', '')
        rationale = case.get('rationale', '')
        description = case.get('description', '')

        # 전체 텍스트에서 GRI 탐지
        full_text = f"{product_name} {rationale} {description}"
        gri = detect_gri_signals(full_text)
        parts = detect_parts_signal(full_text)

        row = {
            'case_id': case.get('case_id', ''),
            'hs_heading': case.get('hs_heading', ''),
            'product_name': product_name[:100],
            'gri1': 1 if gri.gri1_note_like else 0,
            'gri2a': 1 if gri.gri2a_incomplete else 0,
            'gri2b': 1 if gri.gri2b_mixtures else 0,
            'gri3': 1 if gri.gri3_multi_candidate else 0,
            'gri5': 1 if gri.gri5_containers else 0,
            'parts': 1 if parts['is_parts'] else 0,
            'any_gri': 1 if gri.any_signal() else 0,
        }
        results.append(row)

        # 통계
        if gri.gri1_note_like:
            stats['gri1'] += 1
        if gri.gri2a_incomplete:
            stats['gri2a'] += 1
        if gri.gri2b_mixtures:
            stats['gri2b'] += 1
        if gri.gri3_multi_candidate:
            stats['gri3'] += 1
        if gri.gri5_containers:
            stats['gri5'] += 1
        if parts['is_parts']:
            stats['parts'] += 1
        if gri.any_signal():
            stats['any_gri'] += 1

    # CSV 저장
    csv_file = output_path / "gri_tag_stats.csv"
    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n태그 결과 저장: {csv_file}")

    # 통계 출력
    print("\n[GRI 태그 통계]")
    print(f"  총 사례: {stats['total']}")
    print(f"  GRI 신호 있음: {stats['any_gri']} ({stats['any_gri']/stats['total']*100:.1f}%)")
    print(f"  GRI 1 (주): {stats['gri1']}")
    print(f"  GRI 2(a) (미조립): {stats['gri2a']}")
    print(f"  GRI 2(b) (혼합물): {stats['gri2b']}")
    print(f"  GRI 3 (세트): {stats['gri3']}")
    print(f"  GRI 5 (케이스): {stats['gri5']}")
    print(f"  부품: {stats['parts']}")

    return {
        'stats': stats,
        'csv_file': str(csv_file)
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--gri-tags":
            build_gri_tags()
        else:
            # 단일 테스트
            run_single_test(sys.argv[1])
    else:
        # 스모크 테스트
        run_smoke_test()
