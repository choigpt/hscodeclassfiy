"""
LightGBM LambdaMART Ranker Training

LightGBM을 사용한 Learning-to-Rank 모델 학습
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

from .build_dataset import load_rank_dataset


def train_ranker(
    feature_csv: str = "artifacts/ranker/rank_features.csv",
    output_dir: str = "artifacts/ranker",
    test_size: float = 0.1,
    params: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    LightGBM LambdaMART 모델 학습

    Args:
        feature_csv: 피처 CSV 경로
        output_dir: 출력 디렉토리
        test_size: 테스트셋 비율 (query 단위로 분할)
        params: LightGBM 파라미터 (없으면 기본값)
        verbose: 상세 출력

    Returns:
        학습 결과 통계
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("오류: lightgbm이 설치되지 않았습니다.")
        print("  pip install lightgbm")
        return {"error": "lightgbm not installed"}

    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    print("데이터 로드 중...")
    df = pd.read_csv(feature_csv)

    feature_cols = [c for c in df.columns if c.startswith('f_')]
    X = df[feature_cols].values
    y = df['label'].values
    query_ids = df['query_id'].values

    print(f"  샘플 수: {len(X)}")
    print(f"  피처 수: {len(feature_cols)}")
    print(f"  쿼리 수: {len(df['query_id'].unique())}")

    # Query 단위 train/test 분할
    unique_queries = df['query_id'].unique()
    n_test = int(len(unique_queries) * test_size)

    np.random.seed(42)
    test_queries = set(np.random.choice(unique_queries, n_test, replace=False))
    train_queries = set(unique_queries) - test_queries

    train_mask = df['query_id'].isin(train_queries)
    test_mask = df['query_id'].isin(test_queries)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # 그룹 크기 계산
    train_groups = df[train_mask].groupby('query_id').size().values
    test_groups = df[test_mask].groupby('query_id').size().values

    print(f"  Train: {len(X_train)} ({len(train_queries)} queries)")
    print(f"  Test: {len(X_test)} ({len(test_queries)} queries)")

    # LightGBM 데이터셋
    train_data = lgb.Dataset(
        X_train, y_train,
        group=train_groups,
        feature_name=feature_cols
    )
    test_data = lgb.Dataset(
        X_test, y_test,
        group=test_groups,
        feature_name=feature_cols,
        reference=train_data
    )

    # 기본 파라미터
    if params is None:
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
            'seed': 42,
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

    # NDCG@K
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

    print("\nFeature Importance (top 10):")
    for feat, imp in sorted_importance[:10]:
        print(f"  {feat}: {imp:.2f}")

    # 모델 저장
    model_file = output_path / "model.txt"
    model.save_model(str(model_file))
    print(f"\n모델 저장: {model_file}")

    # 결과 저장
    results = {
        'params': params,
        'train_queries': len(train_queries),
        'test_queries': len(test_queries),
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
        'feature_importance': sorted_importance,
    }

    results_file = output_path / "train_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"결과 저장: {results_file}")

    return results


def load_ranker(
    model_path: str = "artifacts/ranker/model.txt"
) -> Optional[Any]:
    """
    학습된 ranker 모델 로드

    Args:
        model_path: 모델 파일 경로

    Returns:
        LightGBM Booster 또는 None
    """
    try:
        import lightgbm as lgb
        model_file = Path(model_path)
        if model_file.exists():
            return lgb.Booster(model_file=str(model_file))
        else:
            print(f"모델 파일 없음: {model_path}")
            return None
    except ImportError:
        print("lightgbm이 설치되지 않았습니다.")
        return None
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        from .build_dataset import build_rank_dataset
        build_rank_dataset()

    train_ranker()
