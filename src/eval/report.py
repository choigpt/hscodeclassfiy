"""
Report generation: ASCII table, JSON export, CSV export.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Optional
from collections import OrderedDict

from ..metrics import StageMetrics


def format_table(results: OrderedDict) -> str:
    """Format comparison results as ASCII table."""
    header = (
        f"{'Stage':>5} | {'Name':<22} | {'Top-1':>6} | {'Top-3':>6} | "
        f"{'Top-5':>6} | {'F1':>6} | {'Avg ms':>7}"
    )
    sep = '-' * len(header)

    lines = [sep, header, sep]

    for i, (name, m) in enumerate(results.items(), 1):
        line = (
            f"{i:>5} | {name:<22} | {m.top1*100:>5.1f}% | {m.top3*100:>5.1f}% | "
            f"{m.top5*100:>5.1f}% | {m.f1_macro:>5.3f} | {m.latency_mean:>6.0f}"
        )
        lines.append(line)

    lines.append(sep)

    # Summary
    if results:
        best_top1 = max(results.values(), key=lambda m: m.top1)
        best_f1 = max(results.values(), key=lambda m: m.f1_macro)
        lines.append(f"\nBest Top-1: {best_top1.stage_name} ({best_top1.top1*100:.1f}%)")
        lines.append(f"Best F1:    {best_f1.stage_name} ({best_f1.f1_macro:.3f})")

        first = list(results.values())[0]
        lines.append(f"Samples: {first.n_samples}, Classes: {first.n_classes}")

    return '\n'.join(lines)


def export_json(results: OrderedDict, path: str, include_details: bool = True):
    """Export full results as JSON."""
    output = {
        'summary': {},
        'stages': {},
    }

    for name, m in results.items():
        output['summary'][name] = m.to_dict()
        if include_details:
            output['stages'][name] = {
                **m.to_dict(),
                'details': m.details,
            }
        else:
            output['stages'][name] = m.to_dict()

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"JSON exported: {out_path}")


def format_gri_extended(results: OrderedDict) -> str:
    """Format GRI extended metrics (HS6, GRI distribution, Risk, EC)."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("GRI Pipeline Extended Metrics")
    lines.append("=" * 60)

    for name, m in results.items():
        # Find gri_extended in details
        gri_ext = None
        if m.details:
            for d in m.details:
                if isinstance(d, dict) and '_gri_extended_metrics' in d:
                    gri_ext = d['_gri_extended_metrics']
                    break

        if not gri_ext:
            continue

        lines.append(f"\n[{name}]")

        # HS6 accuracy
        if gri_ext.get('hs6_total', 0) > 0:
            lines.append(f"  HS6 Accuracy:")
            lines.append(f"    Top-1: {gri_ext['hs6_top1']*100:.1f}%")
            lines.append(f"    Top-3: {gri_ext['hs6_top3']*100:.1f}%")
            lines.append(f"    Top-5: {gri_ext['hs6_top5']*100:.1f}%")
            lines.append(f"    Samples with HS6: {gri_ext['hs6_total']}")

        # GRI distribution
        gri_dist = gri_ext.get('gri_distribution', {})
        if gri_dist:
            lines.append(f"  GRI Application Distribution:")
            for gri_id, count in sorted(gri_dist.items()):
                lines.append(f"    {gri_id}: {count}")

        # Risk distribution
        risk_dist = gri_ext.get('risk_distribution', {})
        if risk_dist:
            lines.append(f"  Risk Level Distribution:")
            for level in ['LOW', 'MED', 'HIGH']:
                count = risk_dist.get(level, 0)
                lines.append(f"    {level}: {count}")

        # EC usage
        lines.append(f"  Essential Character:")
        lines.append(f"    Applied: {gri_ext.get('ec_applied_count', 0)}")
        lines.append(f"    Ratio: {gri_ext.get('ec_applied_ratio', 0)*100:.1f}%")

    return '\n'.join(lines)


def export_csv(results: OrderedDict, path: str):
    """Export summary as CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'stage', 'name', 'n_samples', 'n_classes',
        'top1', 'top3', 'top5',
        'f1_macro', 'f1_weighted',
        'latency_mean', 'latency_median', 'latency_p95',
    ]

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (name, m) in enumerate(results.items(), 1):
            writer.writerow({
                'stage': i,
                'name': name,
                'n_samples': m.n_samples,
                'n_classes': m.n_classes,
                'top1': round(m.top1, 4),
                'top3': round(m.top3, 4),
                'top5': round(m.top5, 4),
                'f1_macro': round(m.f1_macro, 4),
                'f1_weighted': round(m.f1_weighted, 4),
                'latency_mean': round(m.latency_mean, 1),
                'latency_median': round(m.latency_median, 1),
                'latency_p95': round(m.latency_p95, 1),
            })

    print(f"CSV exported: {out_path}")
