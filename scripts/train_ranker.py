"""
LightGBM Ranker Trainer (Stage 4) - LambdaMART

새 구조(src/stages/, src/kb/) 기반으로 LightGBM LambdaMART 학습.

핵심 흐름:
  1. 결정사례 순회 -> 케이스별 ML+KB 후보 생성
  2. GRI 신호 + 8축 속성 추출
  3. 후보별 feature 벡터 계산
  4. LegalGate 통과 후보만 학습 데이터에 포함
  5. 정답 HS4 = label 1, 나머지 = label 0
  6. LightGBM LambdaMART 학습 (query-level grouping)
  7. artifacts/ranker_legal/model_legal.txt 저장

Usage:
    python scripts/train_ranker.py
    python scripts/train_ranker.py --max-cases 1000 --test-ratio 0.2
    python scripts/train_ranker.py --output-dir artifacts/ranker_v2
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm not installed. pip install lightgbm")
    sys.exit(1)

from sklearn.metrics import ndcg_score

from src.data import load_cases, stratified_split
from src.text import normalize, tokenize
from src.kb.cards import CardIndex
from src.kb.rules import RuleIndex
from src.kb.legal_gate import LegalGate
from src.kb.gri import detect_gri_signals
from src.kb.attributes import extract_attributes
from src.stages.stage4_hybrid import (
    HybridClassifier, CandidateState, FEATURE_NAMES,
)

# Defaults
DEFAULT_OUTPUT_DIR = "artifacts/ranker_legal"
RANDOM_SEED = 42
NUM_BOOST_ROUND = 1000
EARLY_STOPPING = 50


def build_dataset(
    max_cases: Optional[int] = None,
    test_ratio: float = 0.2,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Build ranking dataset using new src/stages/ and src/kb/ modules.

    For each ruling case:
      1. Generate ML + KB candidates via HybridClassifier internals
      2. Apply LegalGate -> keep only passed candidates
      3. Compute 38-feature vector
      4. Label: 1 if candidate == answer HS4, 0 otherwise

    Returns dict with features, labels, groups, and stats.
    """
    print("=" * 70)
    print("Ranking Dataset Builder (new structure)")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data and models...")
    all_samples = load_cases()
    print(f"  Total samples: {len(all_samples)}")

    if max_cases and max_cases < len(all_samples):
        all_samples = all_samples[:max_cases]
        print(f"  Limited to: {max_cases}")

    # Initialize components
    print("  Initializing KB components...")
    card_index = CardIndex()
    rule_index = RuleIndex()

    try:
        legal_gate = LegalGate()
    except Exception as e:
        print(f"  Warning: LegalGate init failed ({e}), continuing without")
        legal_gate = None

    # Initialize HybridClassifier for ML + feature computation
    print("  Initializing HybridClassifier...")
    try:
        hybrid = HybridClassifier(
            card_index=card_index,
            rule_index=rule_index,
            legal_gate=legal_gate,
            use_ranker=False,  # Don't need existing ranker for dataset building
        )
    except FileNotFoundError as e:
        print(f"  Error: ML model not found. Run train_ml.py first.")
        print(f"  Detail: {e}")
        return {"error": "ml_model_not_found"}

    # Build dataset
    print(f"\n[2/4] Building dataset...")

    all_features = []   # list of feature vectors
    all_labels = []     # list of labels (0 or 1)
    all_groups = []     # group sizes for each query
    all_queries = []    # query metadata

    stats = {
        "total_samples": len(all_samples),
        "processed": 0,
        "skipped_no_candidates": 0,
        "skipped_legal_gate_exclude_answer": 0,
        "total_pairs": 0,
        "positive_pairs": 0,
    }

    cand_before_total = 0
    cand_after_total = 0
    processed_count = 0

    for i, sample in enumerate(all_samples):
        if verbose and (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{len(all_samples)} "
                  f"(queries: {processed_count}, pairs: {stats['total_pairs']})")

        text = sample.text
        answer_hs4 = sample.hs4
        input_norm = normalize(text)

        # GRI + attributes
        gri = detect_gri_signals(text)
        attrs = extract_attributes(text)

        # ML retrieval
        ml_cands = hybrid._ml_retrieve(text, k=50)

        # KB retrieval (adjust topk based on GRI)
        kb_topk = 30
        if gri.gri2a_incomplete:
            kb_topk += 20
        if gri.gri2b_mixtures:
            kb_topk += 10
        kb_cands = hybrid._kb_retrieve(text, k=kb_topk)

        # Merge
        candidates = hybrid._merge_candidates(ml_cands, kb_cands)

        if not candidates:
            stats["skipped_no_candidates"] += 1
            continue

        # Ensure answer is in candidates
        has_answer = any(c.hs4 == answer_hs4 for c in candidates)
        if not has_answer:
            candidates.append(CandidateState(hs4=answer_hs4, source="injected"))

        cand_before_total += len(candidates)

        # LegalGate filtering
        if legal_gate:
            all_hs4s = [c.hs4 for c in candidates]
            passed, redirects, lg_debug = legal_gate.apply(text, all_hs4s[:100])
            passed_set = set(passed)

            # Check if answer passes
            if answer_hs4 not in passed_set:
                stats["skipped_legal_gate_exclude_answer"] += 1
                continue

            candidates = [c for c in candidates if c.hs4 in passed_set]

            # Add redirect targets
            for rhs4 in redirects:
                if rhs4 not in passed_set:
                    candidates.append(CandidateState(hs4=rhs4, source="redirect"))
        else:
            lg_debug = {}

        cand_after_total += len(candidates)

        if len(candidates) < 2:
            # Need at least 2 candidates for ranking
            stats["skipped_no_candidates"] += 1
            continue

        # Compute features for each candidate
        query_features = []
        query_labels = []
        legal_results = lg_debug.get("results", {})

        for cand in candidates:
            fv = hybrid._compute_features(text, cand, gri, attrs, input_norm)

            # Inject LegalGate features if available
            if legal_results and cand.hs4 in legal_results:
                lr = legal_results[cand.hs4]
                # Last 4 features are legal gate features
                fv[-4] = lr.get("heading_term_score", 0.0)
                fv[-3] = lr.get("include_support_score", 0.0)
                fv[-2] = lr.get("exclude_conflict_score", 0.0)
                fv[-1] = lr.get("redirect_penalty", 0.0)

            label = 1 if cand.hs4 == answer_hs4 else 0
            query_features.append(fv)
            query_labels.append(label)

        all_features.extend(query_features)
        all_labels.extend(query_labels)
        all_groups.append(len(query_features))
        all_queries.append({
            "query_id": i,
            "text": text,
            "label_hs4": answer_hs4,
            "n_candidates": len(query_features),
        })

        stats["total_pairs"] += len(query_features)
        stats["positive_pairs"] += sum(query_labels)
        stats["processed"] += 1
        processed_count += 1

    # Averages
    if processed_count > 0:
        stats["avg_candidates_before_legal"] = round(cand_before_total / processed_count, 1)
        stats["avg_candidates_after_legal"] = round(cand_after_total / processed_count, 1)

    print(f"\n  Dataset built:")
    print(f"    Queries: {processed_count}")
    print(f"    Total pairs: {stats['total_pairs']}")
    print(f"    Positive pairs: {stats['positive_pairs']}")
    print(f"    Skipped (no cands): {stats['skipped_no_candidates']}")
    print(f"    Skipped (LG exclude answer): {stats['skipped_legal_gate_exclude_answer']}")

    return {
        "features": np.array(all_features, dtype=np.float32),
        "labels": np.array(all_labels, dtype=np.float32),
        "groups": all_groups,
        "queries": all_queries,
        "stats": stats,
        "feature_names": FEATURE_NAMES,
    }


def split_by_query(
    dataset: Dict, test_ratio: float = 0.2, seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    """Split dataset by query (not by pair) into train/test."""
    rng = np.random.RandomState(seed)

    n_queries = len(dataset["groups"])
    indices = np.arange(n_queries)
    rng.shuffle(indices)

    n_test = max(1, int(n_queries * test_ratio))
    test_qi = set(indices[:n_test].tolist())
    train_qi = set(indices[n_test:].tolist())

    X = dataset["features"]
    y = dataset["labels"]
    groups = dataset["groups"]

    train_X, train_y, train_groups = [], [], []
    test_X, test_y, test_groups = [], [], []

    offset = 0
    for qi, gs in enumerate(groups):
        chunk_X = X[offset:offset + gs]
        chunk_y = y[offset:offset + gs]
        if qi in test_qi:
            test_X.append(chunk_X)
            test_y.append(chunk_y)
            test_groups.append(gs)
        else:
            train_X.append(chunk_X)
            train_y.append(chunk_y)
            train_groups.append(gs)
        offset += gs

    return {
        "X_train": np.vstack(train_X) if train_X else np.array([]),
        "y_train": np.concatenate(train_y) if train_y else np.array([]),
        "train_groups": train_groups,
        "X_test": np.vstack(test_X) if test_X else np.array([]),
        "y_test": np.concatenate(test_y) if test_y else np.array([]),
        "test_groups": test_groups,
    }


def eval_topk(pred: np.ndarray, y: np.ndarray, groups: List[int]) -> Dict[str, float]:
    """Evaluate Top-K accuracy and NDCG."""
    t1, t3, t5, total = 0, 0, 0, 0
    ndcg1s, ndcg3s, ndcg5s = [], [], []

    start = 0
    for gs in groups:
        end = start + gs
        if gs <= 1 or y[start:end].sum() == 0:
            start = end
            continue
        total += 1
        y_q = y[start:end]
        p_q = pred[start:end]
        ri = np.argsort(-p_q)

        if y_q[ri[0]] == 1:
            t1 += 1
        if any(y_q[ri[i]] == 1 for i in range(min(3, gs))):
            t3 += 1
        if any(y_q[ri[i]] == 1 for i in range(min(5, gs))):
            t5 += 1

        try:
            ndcg1s.append(ndcg_score(y_q.reshape(1, -1), p_q.reshape(1, -1), k=1))
            ndcg3s.append(ndcg_score(y_q.reshape(1, -1), p_q.reshape(1, -1), k=min(3, gs)))
            ndcg5s.append(ndcg_score(y_q.reshape(1, -1), p_q.reshape(1, -1), k=min(5, gs)))
        except Exception:
            pass
        start = end

    return {
        "queries": total,
        "top1_acc": t1 / total if total else 0,
        "top3_acc": t3 / total if total else 0,
        "top5_acc": t5 / total if total else 0,
        "ndcg@1": float(np.mean(ndcg1s)) if ndcg1s else 0,
        "ndcg@3": float(np.mean(ndcg3s)) if ndcg3s else 0,
        "ndcg@5": float(np.mean(ndcg5s)) if ndcg5s else 0,
    }


def train_ranker(
    max_cases: Optional[int] = None,
    test_ratio: float = 0.2,
    seed: int = RANDOM_SEED,
    output_dir: str = DEFAULT_OUTPUT_DIR,
):
    """Full training pipeline: build dataset -> train LightGBM -> evaluate."""
    t0 = time.time()

    # 1. Build dataset
    dataset = build_dataset(max_cases=max_cases, seed=seed)
    if "error" in dataset:
        return

    if dataset["stats"]["processed"] == 0:
        print("ERROR: No queries processed. Check data and models.")
        return

    # 2. Split
    print(f"\n[3/4] Train/test split ({1-test_ratio:.0%}/{test_ratio:.0%})...")
    data = split_by_query(dataset, test_ratio=test_ratio, seed=seed)
    print(f"  Train: {len(data['train_groups'])} queries, {len(data['y_train'])} pairs")
    print(f"  Test:  {len(data['test_groups'])} queries, {len(data['y_test'])} pairs")

    # 3. Train LightGBM
    print(f"\n[4/4] Training LightGBM LambdaMART...")

    train_data = lgb.Dataset(
        data["X_train"], label=data["y_train"],
        group=data["train_groups"],
        feature_name=dataset["feature_names"],
    )
    valid_data = lgb.Dataset(
        data["X_test"], label=data["y_test"],
        group=data["test_groups"],
        feature_name=dataset["feature_names"],
        reference=train_data,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [1, 3, 5],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "min_child_samples": 10,
        "seed": seed,
        "verbose": -1,
    }

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING, verbose=True),
        lgb.log_evaluation(100),
    ]

    t_train = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[valid_data],
        valid_names=["test"],
        callbacks=callbacks,
    )
    train_time = time.time() - t_train
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Best iteration: {model.best_iteration}")

    # 4. Evaluate
    print("\nEvaluation:")
    test_pred = model.predict(data["X_test"])
    metrics = eval_topk(test_pred, data["y_test"], data["test_groups"])
    print(f"  Queries: {metrics['queries']}")
    print(f"  Top-1 Acc: {metrics['top1_acc']:.4f} ({metrics['top1_acc']*100:.2f}%)")
    print(f"  Top-3 Acc: {metrics['top3_acc']:.4f} ({metrics['top3_acc']*100:.2f}%)")
    print(f"  Top-5 Acc: {metrics['top5_acc']:.4f} ({metrics['top5_acc']*100:.2f}%)")
    print(f"  NDCG@1: {metrics['ndcg@1']:.4f}")
    print(f"  NDCG@3: {metrics['ndcg@3']:.4f}")
    print(f"  NDCG@5: {metrics['ndcg@5']:.4f}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(
        zip(dataset["feature_names"], importance),
        key=lambda x: x[1], reverse=True,
    )
    print("\nTop-10 Feature Importance (gain):")
    for fname, gain in feat_imp[:10]:
        print(f"  {fname:35s} {gain:>10.1f}")

    # 5. Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_file = out_path / "model_legal.txt"
    model.save_model(str(model_file))
    print(f"\nModel saved: {model_file}")

    # Save CSV dataset
    csv_file = out_path / "rank_features_legal.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        header = ["query_id", "label"] + dataset["feature_names"]
        writer = csv.writer(f)
        writer.writerow(header)

        offset = 0
        for qi, gs in enumerate(dataset["groups"]):
            for j in range(gs):
                row = [qi, int(dataset["labels"][offset + j])]
                row.extend(dataset["features"][offset + j].tolist())
                writer.writerow(row)
            offset += gs
    print(f"Features CSV saved: {csv_file}")

    # Save queries
    queries_file = out_path / "rank_queries_legal.json"
    with open(queries_file, "w", encoding="utf-8") as f:
        json.dump(dataset["queries"], f, ensure_ascii=False, indent=2)

    # Save stats + metrics
    results = {
        "dataset_stats": dataset["stats"],
        "metrics": metrics,
        "feature_importance_top10": {k: round(v, 1) for k, v in feat_imp[:10]},
        "params": params,
        "best_iteration": model.best_iteration,
        "train_time_s": round(train_time, 1),
        "total_time_s": round(time.time() - t0, 1),
    }
    results_file = out_path / "train_results_legal.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved: {results_file}")

    # Final summary
    total_time = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Ranker Training Complete ({total_time:.1f}s)")
    print(f"{'=' * 70}")
    print(f"  Queries: {dataset['stats']['processed']}")
    print(f"  Pairs: {dataset['stats']['total_pairs']}")
    print(f"  Top-1: {metrics['top1_acc']*100:.2f}%  Top-5: {metrics['top5_acc']*100:.2f}%")
    print(f"  NDCG@5: {metrics['ndcg@5']:.4f}")
    print(f"  Model: {model_file}")


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM Ranker")
    parser.add_argument("--max-cases", type=int, default=None, help="Max cases to process")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    train_ranker(
        max_cases=args.max_cases,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
