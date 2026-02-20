"""
Data loading and train/test split for HS classification.
Loads ruling cases from all_cases_full_v7.json.
"""

import json
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict


CASES_PATH = "data/ruling_cases/all_cases_full_v7.json"


@dataclass
class Sample:
    """Single evaluation sample."""
    id: str
    text: str          # product_name
    hs4: str           # ground truth HS4
    hs6: str = ""
    hs10: str = ""
    description: str = ""  # product_description (optional)
    meta: Dict[str, Any] = field(default_factory=dict)


def load_cases(path: str = CASES_PATH) -> List[Sample]:
    """Load ruling cases JSON and return as Sample list."""
    with open(path, 'r', encoding='utf-8') as f:
        cases = json.load(f)

    samples = []
    for i, case in enumerate(cases):
        hs_code = case.get('hs_code', '') or ''
        hs_heading = case.get('hs_heading', '') or ''
        product_name = case.get('product_name', '').strip()

        if hs_heading and len(hs_heading) == 4:
            hs4 = hs_heading
        elif hs_code and len(hs_code) >= 4:
            hs4 = hs_code[:4]
        else:
            continue

        if not product_name or len(product_name) < 2:
            continue

        hs6 = hs_code[:6] if hs_code and len(hs_code) >= 6 else ""
        hs10 = hs_code if hs_code and len(hs_code) >= 10 else ""
        description = case.get('product_description', '').strip()

        content = f"{product_name}_{hs4}_{i}"
        sample_id = hashlib.md5(content.encode()).hexdigest()[:12]

        samples.append(Sample(
            id=sample_id,
            text=product_name,
            hs4=hs4,
            hs6=hs6,
            hs10=hs10,
            description=description,
            meta={
                'original_index': i,
                'case_number': case.get('case_number', ''),
                'decision_date': case.get('decision_date', ''),
            }
        ))

    return samples


def stratified_split(
    samples: List[Sample],
    train_ratio: float = 0.70,
    test_ratio: float = 0.30,
    min_per_class: int = 3,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample]]:
    """
    Stratified train/test split.

    Returns:
        (train_samples, test_samples)
    """
    rng = random.Random(seed)

    # Group by HS4
    class_samples: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        class_samples[s.hs4].append(s)

    train, test = [], []

    for hs4, group in class_samples.items():
        if len(group) < min_per_class:
            train.extend(group)
            continue

        rng.shuffle(group)
        n_train = max(1, int(len(group) * train_ratio))
        train.extend(group[:n_train])
        test.extend(group[n_train:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def load_test_samples(
    path: str = CASES_PATH,
    n: Optional[int] = None,
    seed: int = 42,
) -> List[Sample]:
    """Load and return test split samples. Optionally limit to n."""
    all_samples = load_cases(path)
    _, test = stratified_split(all_samples, seed=seed)

    if n is not None and n < len(test):
        rng = random.Random(seed)
        test = rng.sample(test, n)

    return test


def get_split_stats(samples: List[Sample]) -> Dict[str, Any]:
    """Return basic stats about a sample set."""
    hs4_counts = Counter(s.hs4 for s in samples)
    return {
        'n_samples': len(samples),
        'n_classes': len(hs4_counts),
        'avg_per_class': len(samples) / len(hs4_counts) if hs4_counts else 0,
        'min_per_class': min(hs4_counts.values()) if hs4_counts else 0,
        'max_per_class': max(hs4_counts.values()) if hs4_counts else 0,
    }
