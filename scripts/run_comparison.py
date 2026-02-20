"""
Main comparison script: run multiple classification stages and compare results.

Usage:
    python scripts/run_comparison.py --stages all --samples 200
    python scripts/run_comparison.py --stages 1,2,4 --samples 50
    python scripts/run_comparison.py --stages rule,ml,hybrid --samples 100
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data import load_test_samples, get_split_stats
from src.eval.comparison import StagedComparison
from src.eval.report import format_table, export_json, export_csv


STAGE_ALIASES = {
    '1': 'rule', 'rule': 'rule',
    '2': 'ml', 'ml': 'ml',
    '3': 'llm', 'llm': 'llm', 'rag': 'llm',
    '4': 'hybrid', 'hybrid': 'hybrid',
    '5': 'cascade', 'cascade': 'cascade',
}


def parse_stages(stages_str: str):
    """Parse stages argument into list of stage names."""
    if stages_str.lower() == 'all':
        return ['rule', 'ml', 'llm', 'hybrid', 'cascade']

    stages = []
    for s in stages_str.split(','):
        s = s.strip().lower()
        if s in STAGE_ALIASES:
            name = STAGE_ALIASES[s]
            if name not in stages:
                stages.append(name)
        else:
            print(f"Warning: unknown stage '{s}', skipping")
    return stages


def build_classifier(stage_name: str):
    """Build and return a classifier by stage name."""
    if stage_name == 'rule':
        from src.stages.stage1_rule import RuleClassifier
        return RuleClassifier()

    elif stage_name == 'ml':
        from src.stages.stage2_ml import MLClassifier
        return MLClassifier()

    elif stage_name == 'llm':
        from src.stages.stage3_llm import LLMClassifier
        return LLMClassifier()

    elif stage_name == 'hybrid':
        from src.stages.stage4_hybrid import HybridClassifier
        return HybridClassifier()

    elif stage_name == 'cascade':
        from src.stages.stage5_cascade import CascadeClassifier
        return CascadeClassifier()

    else:
        raise ValueError(f"Unknown stage: {stage_name}")


def main():
    parser = argparse.ArgumentParser(description='HS Classification Stage Comparison')
    parser.add_argument('--stages', type=str, default='1,2',
                        help='Stages to compare: all, or comma-separated (1,2,4 or rule,ml,hybrid)')
    parser.add_argument('--samples', type=int, default=50,
                        help='Number of test samples (default: 50)')
    parser.add_argument('--topk', type=int, default=5,
                        help='Top-K predictions (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
    parser.add_argument('--output-dir', type=str, default='artifacts/results',
                        help='Output directory for results')
    parser.add_argument('--no-details', action='store_true',
                        help='Exclude per-sample details from JSON output')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    args = parser.parse_args()

    # Parse stages
    stage_names = parse_stages(args.stages)
    if not stage_names:
        print("No valid stages specified. Use --stages all or --stages 1,2,4")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"HS Classification - Stage Comparison")
    print(f"{'='*60}")
    print(f"Stages: {', '.join(stage_names)}")
    print(f"Samples: {args.samples}")
    print(f"Top-K: {args.topk}")
    print(f"Seed: {args.seed}")

    # Load test data
    print(f"\n[Loading test data...]")
    samples = load_test_samples(n=args.samples, seed=args.seed)
    stats = get_split_stats(samples)
    print(f"  Loaded {stats['n_samples']} samples, {stats['n_classes']} classes")

    # Build classifiers
    print(f"\n[Initializing classifiers...]")
    classifiers = []
    for name in stage_names:
        try:
            clf = build_classifier(name)
            classifiers.append(clf)
            print(f"  [{clf.stage_id.value}] {clf.name} - OK")
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")

    if not classifiers:
        print("No classifiers loaded. Check model artifacts.")
        sys.exit(1)

    # Run comparison
    comparison = StagedComparison(verbose=not args.quiet)
    t0 = time.time()
    results = comparison.run(classifiers, samples, topk=args.topk)
    elapsed = time.time() - t0

    # Print table
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(format_table(results))
    print(f"\nTotal time: {elapsed:.1f}s")

    # Export
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    stages_tag = '_'.join(stage_names)

    json_path = output_dir / f"comparison_{stages_tag}_{timestamp}.json"
    csv_path = output_dir / f"comparison_{stages_tag}_{timestamp}.csv"

    export_json(results, str(json_path), include_details=not args.no_details)
    export_csv(results, str(csv_path))

    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()
