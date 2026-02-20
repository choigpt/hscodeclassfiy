"""
Multi-stage comparison engine.

Runs multiple classifiers on the same sample set, collects metrics, returns comparison.
"""

from typing import List, Dict, Optional
from collections import OrderedDict

from ..types import BaseClassifier
from ..data import Sample
from ..metrics import StageMetrics
from .evaluator import StageEvaluator


class StagedComparison:
    """Compare multiple classifier stages side-by-side."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._evaluator = StageEvaluator(verbose=verbose)

    def run(
        self,
        classifiers: List[BaseClassifier],
        samples: List[Sample],
        topk: int = 5,
    ) -> OrderedDict:
        """
        Run all classifiers on the same samples.

        Args:
            classifiers: List of classifiers to evaluate.
            samples: Test samples.
            topk: Number of top predictions.

        Returns:
            OrderedDict mapping stage_name -> StageMetrics.
        """
        results = OrderedDict()

        for clf in classifiers:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Stage {clf.stage_id.value}: {clf.name}")
                print(f"{'='*60}")

            metrics = self._evaluator.evaluate(clf, samples, topk=topk)
            results[clf.name] = metrics

            if self.verbose:
                print(f"  Top-1: {metrics.top1*100:.1f}%  "
                      f"Top-3: {metrics.top3*100:.1f}%  "
                      f"Top-5: {metrics.top5*100:.1f}%  "
                      f"F1: {metrics.f1_macro:.3f}  "
                      f"Avg: {metrics.latency_mean:.0f}ms")

        return results
