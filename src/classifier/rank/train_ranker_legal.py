"""
LightGBM Ranker Training with LegalGate Features

LegalGate 통합 랭킹 모델 학습:
- Layer 1 (Law): LegalGate가 필터링한 후보만 학습
- Layer 2 (Auxiliary): Ranker가 통과 후보를 재정렬
- LegalGate features 포함
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: lightgbm not installed. Install with: pip install lightgbm")


def load_dataset_with_legal(
    feature_csv: str = "artifacts/ranker_legal/rank_features_legal.csv"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    LegalGate 통합 랭킹 데이터셋 로드

    Args:
        feature_csv: 피처 CSV 경로

    Returns:
        (X, y, groups, feature_names)
        - X: 피처 행렬 (n_samples, n_features)
        - y: 라벨 배열 (n_samples,)
        - groups: 그룹 크기 배열 (LightGBM group용)
        - feature_names: 피처 이름 리스트
    """
    df = pd.read_csv(feature_csv)

    # 피처 컬럼 추출 (f_로 시작)
    feature_cols = [c for c in df.columns if c.startswith('f_')]
    X = df[feature_cols].values
    y = df['label'].values

    # 그룹 (query별 문서 수)
    groups = df.groupby('query_id').size().values

    return X, y, groups, feature_cols


def train_ranker_with_legal(
    feature_csv: str = "artifacts/ranker_legal/rank_features_legal.csv",
    output_path: str = "artifacts/ranker_legal",
    test_split: float = 0.2,
    random_seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    LegalGate 통합 랭킹 모델 학습

    Args:
        feature_csv: 피처 CSV 경로
        output_path: 출력 경로
        test_split: 테스트 분할 비율
        random_seed: 랜덤 시드
        verbose: 상세 출력

    Returns:
        학습 결과 딕셔너리
    """
    if not LIGHTGBM_AVAILABLE:
        print("Error: lightgbm이 설치되지 않았습니다.")
        return {"error": "lightgbm not installed"}

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    print("데이터 로드 중...")
    X, y, groups, feature_cols = load_dataset_with_legal(feature_csv)

    print(f"  총 샘플: {len(X)}")
    print(f"  총 쿼리: {len(groups)}")
    print(f"  피처 수: {len(feature_cols)}")
    print(f"  양성 샘플: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")

    # LegalGate features 확인
    legal_features = [f for f in feature_cols if 'legal' in f]
    print(f"\n  LegalGate features: {len(legal_features)}")
    for lf in legal_features:
        print(f"    - {lf}")

    # Train/Test 분할 (query 단위)
    print("\nTrain/Test 분할 중 (query 단위)...")

    np.random.seed(random_seed)
    n_queries = len(groups)
    n_test_queries = int(n_queries * test_split)

    # 쿼리 인덱스 셔플
    query_indices = np.arange(n_queries)
    np.random.shuffle(query_indices)

    train_query_indices = set(query_indices[:-n_test_queries])
    test_query_indices = set(query_indices[-n_test_queries:])

    # 샘플 분할
    train_mask = []
    test_mask = []
    train_groups = []
    test_groups = []

    current_query = 0
    start_idx = 0

    for group_size in groups:
        end_idx = start_idx + group_size

        if current_query in train_query_indices:
            train_mask.extend(range(start_idx, end_idx))
            train_groups.append(group_size)
        else:
            test_mask.extend(range(start_idx, end_idx))
            test_groups.append(group_size)

        start_idx = end_idx
        current_query += 1

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"  Train: {len(train_groups)} queries, {len(X_train)} samples")
    print(f"  Test: {len(test_groups)} queries, {len(X_test)} samples")

    # LightGBM Dataset 생성
    print("\nLightGBM Dataset 생성 중...")
    train_data = lgb.Dataset(
        X_train, label=y_train, group=train_groups,
        feature_name=feature_cols
    )
    test_data = lgb.Dataset(
        X_test, label=y_test, group=test_groups,
        feature_name=feature_cols,
        reference=train_data
    )

    # 하이퍼파라미터
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': random_seed,
    }

    # 학습
    print("\n학습 시작...")

    callbacks = []
    if verbose:
        callbacks.append(lgb.log_evaluation(period=100))
        callbacks.append(lgb.early_stopping(stopping_rounds=50))

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=callbacks,
    )

    # 평가
    print("\n평가 중...")

    from sklearn.metrics import ndcg_score

    def compute_ndcg(y_true, y_pred, groups, k=5):
        """Query별 NDCG@K 평균 계산"""
        ndcgs = []
        start = 0
        for group_size in groups:
            end = start + group_size
            if group_size > 1:
                true = y_true[start:end].reshape(1, -1)
                pred = y_pred[start:end].reshape(1, -1)
                try:
                    ndcg = ndcg_score(true, pred, k=min(k, group_size))
                    ndcgs.append(ndcg)
                except:
                    pass
            start = end
        return np.mean(ndcgs) if ndcgs else 0.0

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    ndcg1_train = compute_ndcg(y_train, y_pred_train, train_groups, k=1)
    ndcg3_train = compute_ndcg(y_train, y_pred_train, train_groups, k=3)
    ndcg5_train = compute_ndcg(y_train, y_pred_train, train_groups, k=5)

    ndcg1_test = compute_ndcg(y_test, y_pred_test, test_groups, k=1)
    ndcg3_test = compute_ndcg(y_test, y_pred_test, test_groups, k=3)
    ndcg5_test = compute_ndcg(y_test, y_pred_test, test_groups, k=5)

    print("\n결과:")
    print(f"  Train NDCG@1: {ndcg1_train:.4f}")
    print(f"  Train NDCG@3: {ndcg3_train:.4f}")
    print(f"  Train NDCG@5: {ndcg5_train:.4f}")
    print(f"  Test NDCG@1: {ndcg1_test:.4f}")
    print(f"  Test NDCG@3: {ndcg3_test:.4f}")
    print(f"  Test NDCG@5: {ndcg5_test:.4f}")

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    importance_dict = dict(zip(feature_cols, importance))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: -x[1])

    print("\nFeature Importance (top 20):")
    for feat, imp in sorted_importance[:20]:
        print(f"  {feat}: {imp:.2f}")

    # LegalGate features importance
    print("\nLegalGate Features Importance:")
    legal_importance = [(f, imp) for f, imp in sorted_importance if 'legal' in f]
    for feat, imp in legal_importance:
        rank = next(i for i, (f, _) in enumerate(sorted_importance, 1) if f == feat)
        print(f"  {feat}: {imp:.2f} (rank #{rank})")

    # 모델 저장
    model_file = output_path / "model_legal.txt"
    model.save_model(str(model_file))
    print(f"\n모델 저장: {model_file}")

    # 결과 저장
    results = {
        'params': params,
        'train_queries': len(train_groups),
        'test_queries': len(test_groups),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'metrics': {
            'train_ndcg1': ndcg1_train,
            'train_ndcg3': ndcg3_train,
            'train_ndcg5': ndcg5_train,
            'test_ndcg1': ndcg1_test,
            'test_ndcg3': ndcg3_test,
            'test_ndcg5': ndcg5_test,
        },
        'feature_importance': sorted_importance[:50],  # Top 50
        'legal_feature_importance': legal_importance,
    }

    results_file = output_path / "train_results_legal.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"결과 저장: {results_file}")

    return results


def load_ranker_legal(
    model_path: str = "artifacts/ranker_legal/model_legal.txt"
) -> Optional[Any]:
    """
    학습된 LegalGate 통합 ranker 모델 로드

    Args:
        model_path: 모델 파일 경로

    Returns:
        LightGBM Booster 또는 None
    """
    if not LIGHTGBM_AVAILABLE:
        print("lightgbm이 설치되지 않았습니다.")
        return None

    try:
        model_file = Path(model_path)
        if model_file.exists():
            return lgb.Booster(model_file=str(model_file))
        else:
            print(f"모델 파일 없음: {model_path}")
            return None
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        from .build_dataset_legal import build_rank_dataset_with_legal
        print("=" * 80)
        print("Step 1: Building Dataset with LegalGate")
        print("=" * 80)
        build_rank_dataset_with_legal()
        print("\n" + "=" * 80)
        print("Step 2: Training Ranker with LegalGate Features")
        print("=" * 80)
        train_ranker_with_legal()
    else:
        train_ranker_with_legal()
