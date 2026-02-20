"""
Single-stage evaluator.

Runs a BaseClassifier against a list of Samples, computes StageMetrics.
Extended with GRI pipeline metrics: HS6 accuracy, GRI distribution, risk levels.
"""

import time
import sys
from typing import List, Optional, Dict, Any
from collections import Counter

from ..types import BaseClassifier, StageResult
from ..data import Sample
from ..metrics import (
    StageMetrics, compute_topk_accuracy, compute_f1,
    compute_latency_stats, compute_chapter_accuracy,
)


class StageEvaluator:
    """Evaluate a single classifier stage."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def evaluate(
        self,
        clf: BaseClassifier,
        samples: List[Sample],
        topk: int = 5,
    ) -> StageMetrics:
        """
        Run classifier on all samples and compute metrics.

        Args:
            clf: Classifier implementing BaseClassifier.
            samples: Test samples with ground truth hs4.
            topk: Number of top predictions to consider.

        Returns:
            StageMetrics with top-1/3/5, F1, latency stats.
        """
        true_labels = []
        pred_labels = []      # top-1 predictions
        pred_topk = []        # top-k prediction lists
        latencies = []
        details = []

        # GRI pipeline extended metrics
        hs6_correct_top1 = 0
        hs6_correct_top3 = 0
        hs6_correct_top5 = 0
        hs6_total = 0
        gri_counter = Counter()
        risk_counter = Counter()
        ec_count = 0

        n = len(samples)
        for i, sample in enumerate(samples):
            if self.verbose and (i + 1) % 10 == 0:
                pct = (i + 1) / n * 100
                print(f"\r  [{clf.name}] {i+1}/{n} ({pct:.0f}%)", end='', flush=True)

            try:
                result = clf.classify_timed(sample.text, topk=topk)
                codes = result.topk_codes
                top1 = codes[0] if codes else ""
                latency = result.latency_ms

                # Extract GRI pipeline info from raw result if available
                raw_result = getattr(result, 'raw_result', None)
                if raw_result:
                    # GRI distribution
                    for gri_app in getattr(raw_result, 'applied_gri', []):
                        if hasattr(gri_app, 'applied') and gri_app.applied:
                            gri_counter[gri_app.gri_id] += 1

                    # Risk level
                    risk = getattr(raw_result, 'risk', None)
                    if risk:
                        risk_counter[risk.level] += 1

                    # EC usage
                    ec = getattr(raw_result, 'essential_character', None)
                    if ec and ec.applicable:
                        ec_count += 1

                    # HS6 accuracy
                    if hasattr(sample, 'hs6') and sample.hs6:
                        hs6_total += 1
                        topk_cands = getattr(raw_result, 'topk', [])
                        hs6_preds = [c.hs6 for c in topk_cands if hasattr(c, 'hs6') and c.hs6]
                        if hs6_preds:
                            if hs6_preds[0] == sample.hs6:
                                hs6_correct_top1 += 1
                            if sample.hs6 in hs6_preds[:3]:
                                hs6_correct_top3 += 1
                            if sample.hs6 in hs6_preds[:5]:
                                hs6_correct_top5 += 1

            except Exception as e:
                codes = []
                top1 = ""
                latency = 0.0
                if self.verbose:
                    print(f"\n  [ERROR] sample {sample.id}: {e}")

            true_labels.append(sample.hs4)
            pred_labels.append(top1)
            pred_topk.append(codes[:topk])
            latencies.append(latency)

            details.append({
                'sample_id': sample.id,
                'text': sample.text[:80],
                'true_hs4': sample.hs4,
                'pred_top1': top1,
                'pred_topk': codes[:topk],
                'correct_top1': top1 == sample.hs4,
                'correct_topk': sample.hs4 in codes[:topk],
                'latency_ms': round(latency, 1),
            })

        if self.verbose:
            print()  # newline after progress

        # Compute metrics
        topk_acc = compute_topk_accuracy(true_labels, pred_topk)
        f1_scores = compute_f1(true_labels, pred_labels)
        lat_stats = compute_latency_stats(latencies)

        unique_classes = len(set(true_labels))

        metrics = StageMetrics(
            stage_name=clf.name,
            n_samples=n,
            n_classes=unique_classes,
            top1=topk_acc.get('top1', 0.0),
            top3=topk_acc.get('top3', 0.0),
            top5=topk_acc.get('top5', 0.0),
            f1_macro=f1_scores.get('f1_macro', 0.0),
            f1_weighted=f1_scores.get('f1_weighted', 0.0),
            latency_mean=lat_stats.get('mean', 0.0),
            latency_median=lat_stats.get('median', 0.0),
            latency_p95=lat_stats.get('p95', 0.0),
            details=details,
        )

        # Attach GRI extended metrics as extra details
        gri_extended = {
            'hs6_top1': hs6_correct_top1 / hs6_total if hs6_total > 0 else 0.0,
            'hs6_top3': hs6_correct_top3 / hs6_total if hs6_total > 0 else 0.0,
            'hs6_top5': hs6_correct_top5 / hs6_total if hs6_total > 0 else 0.0,
            'hs6_total': hs6_total,
            'gri_distribution': dict(gri_counter),
            'risk_distribution': dict(risk_counter),
            'ec_applied_count': ec_count,
            'ec_applied_ratio': ec_count / n if n > 0 else 0.0,
        }

        # Store in details as a summary entry
        if metrics.details:
            metrics.details.append({'_gri_extended_metrics': gri_extended})

        return metrics
