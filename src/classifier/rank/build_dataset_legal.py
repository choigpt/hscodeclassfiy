"""
Ranking Dataset Builder with LegalGate Integration

결정사례에서 LightGBM LambdaMART 학습용 데이터셋 생성 (법 규범 통합)

아키텍처 원칙:
- Layer 1 (Law): LegalGate가 GRI 1 기반으로 hard exclude 수행
- Layer 2 (Auxiliary): Ranker는 LegalGate 통과 후보만 재정렬
- Ranker는 Layer 1 결정을 override할 수 없음

학습 데이터:
- LegalGate.passed=True인 후보만 포함
- LegalGate features 추가: heading_term_score, include_support, exclude_conflict, redirect_penalty
- 정답이 LegalGate에서 제외되면 해당 query 전체 스킵 (법이 틀렸을 가능성 낮음)
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from ..gri_signals import detect_gri_signals
from ..reranker import HSReranker, CandidateFeatures
from ..types import Candidate
from ..legal_gate import LegalGate
from ..attribute_extract import extract_attributes, extract_attributes_8axis


def build_rank_dataset_with_legal(
    cases_path: str = "data/ruling_cases/all_cases_full_v7.json",
    output_dir: str = "artifacts/ranker_legal",
    retriever_topk: int = 50,
    kb_topk: int = 30,
    max_cases: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    결정사례에서 랭킹 학습 데이터셋 생성 (LegalGate 통합)

    각 결정사례(query)에 대해:
    1. ML + KB 후보 생성
    2. LegalGate 적용 → passed=False 제거
    3. 후보별 feature vector 생성 (LegalGate features 포함)
    4. 정답 후보 = 1, 나머지 = 0

    Args:
        cases_path: 결정사례 JSON 경로
        output_dir: 출력 디렉토리
        retriever_topk: ML retriever top-K
        kb_topk: KB retriever top-K
        max_cases: 최대 처리 사례 수 (디버깅용)
        verbose: 상세 출력

    Returns:
        데이터셋 통계
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

    if max_cases:
        cases = cases[:max_cases]

    print(f"결정사례 로드: {len(cases)}개")

    # Reranker + LegalGate 초기화
    print("Reranker 및 LegalGate 초기화 중...")
    reranker = HSReranker()
    legal_gate = LegalGate()

    # 데이터셋 생성
    # LegalGate features 추가
    # CandidateFeatures.feature_names()에 LegalGate 피처 포함됨
    dataset = {
        'queries': [],       # query_id -> text
        'features': [],      # (query_id, doc_id, feature_vector, label)
        'feature_names': CandidateFeatures.feature_names(),
    }

    stats = {
        'total_cases': len(cases),
        'processed': 0,
        'skipped_no_candidates': 0,
        'skipped_no_label': 0,
        'skipped_legal_gate_exclude_answer': 0,  # LegalGate가 정답 제외 (법이 맞다고 가정)
        'total_pairs': 0,
        'positive_pairs': 0,
        'avg_candidates_before_legal': 0.0,
        'avg_candidates_after_legal': 0.0,
    }

    print("\n데이터셋 생성 중 (LegalGate 적용)...")

    cand_before_total = 0
    cand_after_total = 0
    processed_count = 0

    for i, case in enumerate(cases):
        if verbose and (i + 1) % 500 == 0:
            print(f"  진행: {i+1}/{len(cases)}")

        product_name = case.get('product_name', '').strip()
        hs_heading = case.get('hs_heading', '')

        if not product_name or not hs_heading or len(hs_heading) != 4:
            stats['skipped_no_label'] += 1
            continue

        query_id = i

        # GRI 신호 탐지
        gri_signals = detect_gri_signals(product_name)

        # 전역 속성 추출
        input_attrs = extract_attributes(product_name)
        input_attrs_8axis = extract_attributes_8axis(product_name)

        # KB 후보 생성
        kb_candidates = reranker.retrieve_from_kb(
            product_name,
            topk=kb_topk,
            gri_signals=gri_signals
        )

        if not kb_candidates:
            stats['skipped_no_candidates'] += 1
            continue

        cand_before_total += len(kb_candidates)

        # 정답 후보가 KB에 있는지 확인
        has_positive = any(c.hs4 == hs_heading for c in kb_candidates)

        # 정답이 없으면 추가
        if not has_positive:
            kb_candidates.append(Candidate(hs4=hs_heading, score_ml=0.0))

        # LegalGate 적용
        legal_candidates, redirect_hs4s, legal_debug = legal_gate.apply(
            product_name, kb_candidates
        )

        # 리다이렉트 HS4 추가
        if redirect_hs4s:
            for rhs4 in redirect_hs4s:
                legal_candidates.append(Candidate(hs4=rhs4, score_ml=0.0))

        cand_after_total += len(legal_candidates)

        # LegalGate 결과에서 각 후보의 LegalGateResult 추출
        legal_results = legal_debug.get('results', {})

        # 정답이 LegalGate에서 제외되었는지 확인
        answer_passed = any(c.hs4 == hs_heading for c in legal_candidates)

        if not answer_passed:
            # 정답이 LegalGate에서 제외됨 → 법이 맞다고 가정, 이 query 스킵
            stats['skipped_legal_gate_exclude_answer'] += 1
            continue

        if len(legal_candidates) == 0:
            stats['skipped_no_candidates'] += 1
            continue

        # 피처 계산
        dataset['queries'].append({
            'query_id': query_id,
            'text': product_name,
            'label_hs4': hs_heading,
        })

        for doc_id, cand in enumerate(legal_candidates):
            # 기본 features
            features_base = reranker.compute_features(
                product_name,
                cand,
                gri_signals,
                input_attrs=input_attrs,
                model_classes=None,
                input_attrs_8axis=input_attrs_8axis
            )

            # LegalGate features 추가 (CandidateFeatures에 직접 설정)
            legal_result = legal_results.get(cand.hs4)
            if legal_result:
                features_base.f_legal_heading_term = legal_result.get('heading_term_score', 0.0)
                features_base.f_legal_include_support = legal_result.get('include_support_score', 0.0)
                features_base.f_legal_exclude_conflict = legal_result.get('exclude_conflict_score', 0.0)
                features_base.f_legal_redirect_penalty = legal_result.get('redirect_penalty', 0.0)

            # 벡터 (to_vector()에 LegalGate 피처 포함)
            feature_vector = features_base.to_vector()

            label = 1 if cand.hs4 == hs_heading else 0

            dataset['features'].append({
                'query_id': query_id,
                'doc_id': doc_id,
                'hs4': cand.hs4,
                'features': feature_vector,
                'label': label,
            })

            stats['total_pairs'] += 1
            if label == 1:
                stats['positive_pairs'] += 1

        stats['processed'] += 1
        processed_count += 1

    # 평균 계산
    if processed_count > 0:
        stats['avg_candidates_before_legal'] = cand_before_total / processed_count
        stats['avg_candidates_after_legal'] = cand_after_total / processed_count

    # 저장
    print("\n데이터셋 저장 중...")

    # 1. Feature CSV (LightGBM 입력용)
    feature_csv = output_path / "rank_features_legal.csv"
    with open(feature_csv, 'w', encoding='utf-8', newline='') as f:
        header = ['query_id', 'doc_id', 'hs4', 'label'] + dataset['feature_names']
        writer = csv.writer(f)
        writer.writerow(header)

        for item in dataset['features']:
            row = [
                item['query_id'],
                item['doc_id'],
                item['hs4'],
                item['label'],
            ] + item['features']
            writer.writerow(row)

    print(f"  피처 CSV: {feature_csv}")

    # 2. Query 정보
    query_json = output_path / "rank_queries_legal.json"
    with open(query_json, 'w', encoding='utf-8') as f:
        json.dump(dataset['queries'], f, ensure_ascii=False, indent=2)

    print(f"  쿼리 JSON: {query_json}")

    # 3. 통계
    stats_json = output_path / "rank_stats_legal.json"
    with open(stats_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"  통계 JSON: {stats_json}")

    # 4. 아키텍처 문서
    arch_doc = output_path / "ARCHITECTURE.md"
    with open(arch_doc, 'w', encoding='utf-8') as f:
        f.write("""# Ranking Architecture - Law + Auxiliary Layer

## 아키텍처 원칙

### Layer 1: Law (Hard Policy)
- **LegalGate (GRI 1)**: 호 용어 + 주규정 기반 법적 필터링
- Hard exclude: exclude 주규정에 명시적으로 걸리면 제거
- Redirect: 다른 호로 리다이렉트하는 경우 대상 호 추가
- **Layer 1 결정은 절대 override 불가**

### Layer 2: Auxiliary (Learned Ranking)
- **LightGBM LambdaMART Ranker**: LegalGate 통과 후보를 재정렬
- LegalGate features를 포함한 rich feature set 사용
- Layer 1이 제거한 후보는 절대 복원하지 않음
- 정확도 향상을 위한 보조 레이어

## 학습 데이터 원칙

1. **LegalGate Filtering**: passed=True인 후보만 포함
2. **Answer Filtering**: 정답이 LegalGate에서 제외되면 해당 query 스킵
   - 이유: 법이 맞다고 가정 (법 > 결정사례)
3. **Feature Integration**: LegalGate features를 ranker feature set에 추가

## Features

### Base Features (from CandidateFeatures)
- ML retrieval: f_ml, f_lexical
- Card/Rule hits: f_card_hits, f_rule_inc_hits, f_rule_exc_hits
- GRI signals: f_gri2a_signal, f_gri2b_signal, f_gri3_signal, f_gri5_signal
- 8-axis attributes: f_object_match_score, f_material_match_score, ...
- Specificity: f_specificity

### Legal Features (from LegalGate)
- **f_legal_heading_term**: 호 용어 매칭 점수
- **f_legal_include_support**: 포함 주규정 지지 점수
- **f_legal_exclude_conflict**: 제외 주규정 충돌 점수 (음수)
- **f_legal_redirect_penalty**: 리다이렉트 페널티 (음수)

## 평가 지표

- NDCG@1, @3, @5
- Top-1 Accuracy
- Macro F1 (chapter-level)
""")

    print(f"  아키텍처 문서: {arch_doc}")

    # 요약
    print("\n" + "=" * 60)
    print("데이터셋 생성 완료 (LegalGate 통합)")
    print("=" * 60)
    print(f"  총 사례: {stats['total_cases']}")
    print(f"  처리됨: {stats['processed']}")
    print(f"  스킵 (후보 없음): {stats['skipped_no_candidates']}")
    print(f"  스킵 (라벨 없음): {stats['skipped_no_label']}")
    print(f"  스킵 (LegalGate가 정답 제외): {stats['skipped_legal_gate_exclude_answer']}")
    print(f"  총 쌍 수: {stats['total_pairs']}")
    print(f"  양성 쌍 수: {stats['positive_pairs']}")
    print(f"  양성 비율: {stats['positive_pairs']/stats['total_pairs']*100:.2f}%")
    print(f"\n  평균 후보 수 (LegalGate 전): {stats['avg_candidates_before_legal']:.1f}")
    print(f"  평균 후보 수 (LegalGate 후): {stats['avg_candidates_after_legal']:.1f}")
    print(f"  LegalGate 필터링 비율: {(1 - stats['avg_candidates_after_legal']/stats['avg_candidates_before_legal'])*100:.1f}%")

    return {
        'stats': stats,
        'feature_csv': str(feature_csv),
        'query_json': str(query_json),
    }


if __name__ == "__main__":
    import sys

    max_cases = None
    if len(sys.argv) > 1:
        max_cases = int(sys.argv[1])

    build_rank_dataset_with_legal(max_cases=max_cases)
