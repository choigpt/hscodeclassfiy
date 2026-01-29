"""
Ranking Dataset Builder

결정사례에서 LightGBM LambdaMART 학습용 데이터셋 생성
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from ..gri_signals import detect_gri_signals
from ..reranker import HSReranker, CandidateFeatures
from ..types import Candidate


def build_rank_dataset(
    cases_path: str = "data/ruling_cases/all_cases_full_v7.json",
    output_dir: str = "artifacts/ranker",
    retriever_topk: int = 50,
    kb_topk: int = 30,
    max_cases: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    결정사례에서 랭킹 학습 데이터셋 생성

    각 결정사례(query)에 대해:
    1. ML + KB 후보 생성
    2. 후보별 feature vector 생성
    3. 정답 후보 = 1, 나머지 = 0

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

    # Reranker만 로드 (Retriever는 무거우므로 KB만 사용)
    print("Reranker 초기화 중...")
    reranker = HSReranker()

    # 데이터셋 생성
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
        'total_pairs': 0,
        'positive_pairs': 0,
    }

    print("\n데이터셋 생성 중...")

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

        # KB 후보 생성
        kb_candidates = reranker.retrieve_from_kb(
            product_name,
            topk=kb_topk,
            gri_signals=gri_signals
        )

        if not kb_candidates:
            stats['skipped_no_candidates'] += 1
            continue

        # 정답 후보가 KB에 있는지 확인
        has_positive = any(c.hs4 == hs_heading for c in kb_candidates)

        # 정답이 없으면 추가
        if not has_positive:
            kb_candidates.append(Candidate(hs4=hs_heading, score_ml=0.0))

        # 피처 계산
        dataset['queries'].append({
            'query_id': query_id,
            'text': product_name,
            'label_hs4': hs_heading,
        })

        for doc_id, cand in enumerate(kb_candidates):
            features = reranker.compute_features(
                product_name,
                cand,
                gri_signals,
                model_classes=None  # 모델 클래스는 나중에 추가 가능
            )

            label = 1 if cand.hs4 == hs_heading else 0

            dataset['features'].append({
                'query_id': query_id,
                'doc_id': doc_id,
                'hs4': cand.hs4,
                'features': features.to_vector(),
                'label': label,
            })

            stats['total_pairs'] += 1
            if label == 1:
                stats['positive_pairs'] += 1

        stats['processed'] += 1

    # 저장
    print("\n데이터셋 저장 중...")

    # 1. Feature CSV (LightGBM 입력용)
    feature_csv = output_path / "rank_features.csv"
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
    query_json = output_path / "rank_queries.json"
    with open(query_json, 'w', encoding='utf-8') as f:
        json.dump(dataset['queries'], f, ensure_ascii=False, indent=2)

    print(f"  쿼리 JSON: {query_json}")

    # 3. 통계
    stats_json = output_path / "rank_stats.json"
    with open(stats_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"  통계 JSON: {stats_json}")

    # 요약
    print("\n" + "=" * 60)
    print("데이터셋 생성 완료")
    print("=" * 60)
    print(f"  총 사례: {stats['total_cases']}")
    print(f"  처리됨: {stats['processed']}")
    print(f"  스킵 (후보 없음): {stats['skipped_no_candidates']}")
    print(f"  스킵 (라벨 없음): {stats['skipped_no_label']}")
    print(f"  총 쌍 수: {stats['total_pairs']}")
    print(f"  양성 쌍 수: {stats['positive_pairs']}")
    print(f"  양성 비율: {stats['positive_pairs']/stats['total_pairs']*100:.2f}%")

    return {
        'stats': stats,
        'feature_csv': str(feature_csv),
        'query_json': str(query_json),
    }


def load_rank_dataset(
    feature_csv: str = "artifacts/ranker/rank_features.csv"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    랭킹 데이터셋 로드

    Args:
        feature_csv: 피처 CSV 경로

    Returns:
        (X, y, groups)
        - X: 피처 행렬 (n_samples, n_features)
        - y: 라벨 배열 (n_samples,)
        - groups: 그룹 크기 배열 (LightGBM group용)
    """
    import pandas as pd

    df = pd.read_csv(feature_csv)

    # 피처 컬럼 추출
    feature_cols = [c for c in df.columns if c.startswith('f_')]
    X = df[feature_cols].values
    y = df['label'].values

    # 그룹 (query별 문서 수)
    groups = df.groupby('query_id').size().values

    return X, y, groups


if __name__ == "__main__":
    import sys

    max_cases = None
    if len(sys.argv) > 1:
        max_cases = int(sys.argv[1])

    build_rank_dataset(max_cases=max_cases)
