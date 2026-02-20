"""
ML Model Trainer (Stage 2) - SBERT + Logistic Regression

결정사례 데이터로 SBERT 임베딩 + Logistic Regression 모델 학습.
기존 retriever.train_model() 대비 개선:
  1. product_description 추가 (--use-desc)
  2. 동의어 데이터 증강 (--augment)
  3. Stratified split
  4. 학습 결과 리포트 자동 출력

Usage:
    python scripts/train_ml.py                          # 기본 (product_name만)
    python scripts/train_ml.py --use-desc               # +description
    python scripts/train_ml.py --use-desc --augment     # +증강
    python scripts/train_ml.py --test-ratio 0.2 --seed 42
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import Sample, load_cases, stratified_split, get_split_stats

# Defaults
SBERT_MODEL = "jhgan/ko-sroberta-multitask"
OUTPUT_DIR = "artifacts/classifier"
THESAURUS_PATH = "kb/structured/thesaurus_terms.jsonl"
MIN_SAMPLES_PER_CLASS = 3


def build_text(sample: Sample, use_desc: bool = False) -> str:
    """Build input text from sample."""
    text = sample.text
    if use_desc and sample.description:
        text = f"{text} {sample.description}"
    return text.strip()


def load_thesaurus() -> Dict[str, List[str]]:
    """Load thesaurus: term -> aliases mapping."""
    path = Path(THESAURUS_PATH)
    if not path.exists():
        print(f"[Warning] Thesaurus not found: {path}")
        return {}

    lookup = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            term = entry.get("term", "").strip()
            aliases = entry.get("aliases", [])
            if term and aliases:
                lookup[term.lower()] = [a for a in aliases if a.strip()]
    print(f"[Thesaurus] {len(lookup)} terms loaded")
    return lookup


def augment_samples(
    samples: List[Sample],
    thesaurus: Dict[str, List[str]],
    use_desc: bool,
    max_augment_per_sample: int = 2,
    sparse_threshold: int = 10,
    seed: int = 42,
) -> List[Sample]:
    """Augment sparse classes with synonym substitution."""
    rng = random.Random(seed)

    # Find sparse classes
    hs4_counts = Counter(s.hs4 for s in samples)
    sparse_hs4 = {h for h, c in hs4_counts.items() if c < sparse_threshold}

    if not sparse_hs4:
        print("[Augment] No sparse classes found")
        return samples

    print(f"[Augment] Sparse classes (< {sparse_threshold} samples): {len(sparse_hs4)}")

    augmented = []
    for sample in samples:
        if sample.hs4 not in sparse_hs4:
            continue

        text = build_text(sample, use_desc)
        text_lower = text.lower()

        # Find matching thesaurus terms
        matched_terms = []
        for term, aliases in thesaurus.items():
            if term in text_lower and aliases:
                matched_terms.append((term, aliases))

        if not matched_terms:
            continue

        # Generate augmented samples
        for _ in range(min(max_augment_per_sample, 3)):
            new_text = text
            # Pick a random term to substitute
            term, aliases = rng.choice(matched_terms)
            replacement = rng.choice(aliases)

            # Case-insensitive substitution (first occurrence)
            import re
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            new_text = pattern.sub(replacement, new_text, count=1)

            if new_text != text:
                augmented.append(Sample(
                    id=f"{sample.id}_aug{len(augmented)}",
                    text=new_text,
                    hs4=sample.hs4,
                    hs6=sample.hs6,
                    hs10=sample.hs10,
                    description="",
                    meta={"augmented": True, "original_id": sample.id},
                ))

    print(f"[Augment] Generated {len(augmented)} augmented samples")
    return samples + augmented


def train(
    use_desc: bool = False,
    augment: bool = False,
    test_ratio: float = 0.15,
    seed: int = 42,
    output_dir: str = OUTPUT_DIR,
):
    """Train SBERT + LR model."""
    import joblib
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    t0 = time.time()

    print("=" * 70)
    print("ML Model Training (SBERT + Logistic Regression)")
    print("=" * 70)
    print(f"  use_desc: {use_desc}")
    print(f"  augment: {augment}")
    print(f"  test_ratio: {test_ratio}")
    print(f"  seed: {seed}")
    print(f"  output_dir: {output_dir}")

    # 1. Load data
    print(f"\n[1/6] Loading data...")
    all_samples = load_cases()
    print(f"  Total samples: {len(all_samples)}")

    # 2. Stratified split
    print(f"\n[2/6] Stratified split (train={1-test_ratio:.0%}, test={test_ratio:.0%})...")
    train_samples, test_samples = stratified_split(
        all_samples,
        train_ratio=1.0 - test_ratio,
        test_ratio=test_ratio,
        min_per_class=MIN_SAMPLES_PER_CLASS,
        seed=seed,
    )

    train_stats = get_split_stats(train_samples)
    test_stats = get_split_stats(test_samples)
    print(f"  Train: {train_stats['n_samples']} samples, {train_stats['n_classes']} classes")
    print(f"  Test:  {test_stats['n_samples']} samples, {test_stats['n_classes']} classes")

    # 3. Augmentation (optional)
    if augment:
        print(f"\n[3/6] Data augmentation...")
        thesaurus = load_thesaurus()
        if thesaurus:
            train_samples = augment_samples(
                train_samples, thesaurus, use_desc, seed=seed,
            )
            aug_stats = get_split_stats(train_samples)
            print(f"  Train after augment: {aug_stats['n_samples']} samples, {aug_stats['n_classes']} classes")
        else:
            print("  Skipped (no thesaurus)")
    else:
        print(f"\n[3/6] Augmentation: skipped")

    # 4. Build texts + encode
    print(f"\n[4/6] Building texts and encoding...")
    train_texts = [build_text(s, use_desc) for s in train_samples]
    test_texts = [build_text(s, use_desc) for s in test_samples]
    train_labels = [s.hs4 for s in train_samples]
    test_labels = [s.hs4 for s in test_samples]

    # Label encoding
    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)

    # Filter test to known classes
    known_classes = set(le.classes_)
    valid_test_mask = [l in known_classes for l in test_labels]
    test_texts_f = [t for t, v in zip(test_texts, valid_test_mask) if v]
    test_labels_f = [l for l, v in zip(test_labels, valid_test_mask) if v]
    y_test = le.transform(test_labels_f)

    print(f"  Train texts: {len(train_texts)}")
    print(f"  Test texts (known classes): {len(test_texts_f)}/{len(test_texts)}")
    print(f"  Classes: {len(le.classes_)}")

    # SBERT embedding
    print(f"\n  Loading SBERT: {SBERT_MODEL}")
    st_model = SentenceTransformer(SBERT_MODEL)

    print(f"  Encoding train ({len(train_texts)} texts)...")
    X_train = st_model.encode(train_texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"  Encoding test ({len(test_texts_f)} texts)...")
    X_test = st_model.encode(test_texts_f, show_progress_bar=True, convert_to_numpy=True)
    print(f"  Embedding shape: {X_train.shape}")

    # 5. Train LR
    print(f"\n[5/6] Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        C=1.0,
        solver="lbfgs",
    )
    t_lr = time.time()
    lr.fit(X_train, y_train)
    lr_time = time.time() - t_lr
    print(f"  LR training time: {lr_time:.1f}s")

    # 6. Evaluate
    print(f"\n[6/6] Evaluation...")

    # Top-1 accuracy
    top1_acc = lr.score(X_test, y_test)

    # Top-K accuracy
    proba = lr.predict_proba(X_test)
    top3_correct = 0
    top5_correct = 0
    for i in range(len(y_test)):
        top_indices = np.argsort(proba[i])[::-1]
        if y_test[i] in top_indices[:3]:
            top3_correct += 1
        if y_test[i] in top_indices[:5]:
            top5_correct += 1

    top3_acc = top3_correct / len(y_test)
    top5_acc = top5_correct / len(y_test)

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    lr_file = out_path / "model_lr.joblib"
    le_file = out_path / "label_encoder.joblib"
    joblib.dump(lr, lr_file)
    joblib.dump(le, le_file)

    total_time = time.time() - t0

    # Report
    print(f"\n{'=' * 70}")
    print("Training Report")
    print(f"{'=' * 70}")
    print(f"  Model: SBERT ({SBERT_MODEL}) + Logistic Regression")
    print(f"  Use description: {use_desc}")
    print(f"  Augmentation: {augment}")
    print(f"  Train samples: {len(train_texts)}")
    print(f"  Test samples: {len(test_texts_f)}")
    print(f"  Classes: {len(le.classes_)}")
    print(f"\n  Performance (on test set):")
    print(f"    Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"    Top-3 Accuracy: {top3_acc:.4f} ({top3_acc*100:.2f}%)")
    print(f"    Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    print(f"\n  Timing:")
    print(f"    LR training: {lr_time:.1f}s")
    print(f"    Total: {total_time:.1f}s")
    print(f"\n  Saved:")
    print(f"    {lr_file}")
    print(f"    {le_file}")

    # Save training metadata
    meta = {
        "use_desc": use_desc,
        "augment": augment,
        "test_ratio": test_ratio,
        "seed": seed,
        "sbert_model": SBERT_MODEL,
        "n_train": len(train_texts),
        "n_test": len(test_texts_f),
        "n_classes": len(le.classes_),
        "top1_accuracy": round(top1_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "lr_time_s": round(lr_time, 1),
        "total_time_s": round(total_time, 1),
    }
    meta_file = out_path / "train_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def main():
    parser = argparse.ArgumentParser(description="Train ML model (SBERT + LR)")
    parser.add_argument("--use-desc", action="store_true", help="Include product_description")
    parser.add_argument("--augment", action="store_true", help="Enable synonym augmentation")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    train(
        use_desc=args.use_desc,
        augment=args.augment,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
