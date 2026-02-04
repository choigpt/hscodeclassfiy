#!/usr/bin/env python3
"""
Hybrid vs KB-only 회귀 분석 스크립트

KB-only에서 맞았지만 Hybrid에서 틀린 샘플(regressions)과
KB-only에서 틀렸지만 Hybrid에서 맞은 샘플(improvements)을 추출.

Usage:
    python scripts/analyze_hybrid_regressions.py \
        artifacts/eval/kb_only_20260203_203155 \
        artifacts/eval/hybrid_20260203_203321
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_predictions(run_dir: Path) -> List[Dict[str, Any]]:
    """predictions_test.jsonl 로드"""
    pred_file = run_dir / 'predictions_test.jsonl'
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")

    predictions = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    return predictions


def extract_top_candidates(pred: Dict[str, Any], topk: int = 5) -> List[Dict[str, Any]]:
    """Top-K 후보의 핵심 정보 추출"""
    topk_list = pred.get('topk', [])[:topk]

    extracted = []
    for cand in topk_list:
        extracted.append({
            'hs4': cand.get('hs4'),
            'score_total': round(cand.get('score_total', 0), 4),
            'score_ml': round(cand.get('score_ml', 0), 4),
            'score_card': round(cand.get('score_card', 0), 4),
            'score_rule': round(cand.get('score_rule', 0), 4),
        })

    return extracted


def analyze_regressions(kb_dir: Path, hy_dir: Path):
    """
    KB-only vs Hybrid 회귀 분석

    Args:
        kb_dir: KB-only evaluation directory
        hy_dir: Hybrid evaluation directory
    """
    print(f"Loading predictions...")
    print(f"  KB-only: {kb_dir}")
    print(f"  Hybrid:  {hy_dir}")

    kb_preds = load_predictions(kb_dir)
    hy_preds = load_predictions(hy_dir)

    if len(kb_preds) != len(hy_preds):
        print(f"WARNING: Sample count mismatch (KB={len(kb_preds)}, HY={len(hy_preds)})")
        min_len = min(len(kb_preds), len(hy_preds))
        kb_preds = kb_preds[:min_len]
        hy_preds = hy_preds[:min_len]

    print(f"  Total samples: {len(kb_preds)}")
    print()

    # 카테고리 분류
    regressions = []  # KB correct, HY wrong
    improvements = []  # KB wrong, HY correct
    both_correct = []
    both_wrong = []

    for kb, hy in zip(kb_preds, hy_preds):
        sample_id = kb.get('sample_id', 'unknown')
        text = kb.get('text', '')
        true_hs4 = kb.get('true_hs4', '')

        kb_top1 = kb['topk'][0]['hs4'] if kb.get('topk') else None
        hy_top1 = hy['topk'][0]['hs4'] if hy.get('topk') else None

        kb_correct = (kb_top1 == true_hs4)
        hy_correct = (hy_top1 == true_hs4)

        # 공통 레코드 구조
        record = {
            'sample_id': sample_id,
            'text': text[:200],  # 처음 200자만
            'text_len': len(text),
            'true_hs4': true_hs4,
            'kb_top1': kb_top1,
            'hy_top1': hy_top1,
            'kb_topk': extract_top_candidates(kb, topk=5),
            'hy_topk': extract_top_candidates(hy, topk=5),
            'hy_ml_top5': hy.get('debug', {}).get('ml_top5', []),
            'hy_debug': {
                'kb_margin': hy.get('debug', {}).get('kb_margin'),
                'kb_locked': hy.get('debug', {}).get('kb_locked'),
                'w_ml': hy.get('debug', {}).get('w_ml'),
                'use_strong_ml': hy.get('debug', {}).get('use_strong_ml'),
                'top1_source': hy.get('debug', {}).get('top1_source'),
                'merge_stats': hy.get('debug', {}).get('merge_stats'),
                'retriever_used': hy.get('debug', {}).get('retriever_used'),
                'ranker_applied': hy.get('debug', {}).get('ranker_applied'),
            }
        }

        if kb_correct and not hy_correct:
            regressions.append(record)
        elif not kb_correct and hy_correct:
            improvements.append(record)
        elif kb_correct and hy_correct:
            both_correct.append(record)
        else:
            both_wrong.append(record)

    # 통계 출력
    print("="*80)
    print("Analysis Results")
    print("="*80)
    print(f"KB correct, HY wrong (regressions):  {len(regressions):3d}")
    print(f"KB wrong, HY correct (improvements): {len(improvements):3d}")
    print(f"Both correct:                         {len(both_correct):3d}")
    print(f"Both wrong:                           {len(both_wrong):3d}")
    print(f"Total:                                {len(kb_preds):3d}")
    print()

    # Top-1 overlap
    same_top1 = sum(1 for kb, hy in zip(kb_preds, hy_preds)
                    if kb['topk'][0]['hs4'] == hy['topk'][0]['hs4'])
    print(f"Top-1 same:                           {same_top1:3d} ({same_top1/len(kb_preds):.1%})")
    print()

    # 파일 저장
    output_dir = Path(hy_dir)

    # Regressions
    regression_file = output_dir / 'hybrid_regressions.jsonl'
    with open(regression_file, 'w', encoding='utf-8') as f:
        for rec in regressions:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"Regressions saved: {regression_file} ({len(regressions)} samples)")

    # Improvements
    improvement_file = output_dir / 'hybrid_improvements.jsonl'
    with open(improvement_file, 'w', encoding='utf-8') as f:
        for rec in improvements:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"Improvements saved: {improvement_file} ({len(improvements)} samples)")

    # Summary
    summary = {
        'kb_dir': str(kb_dir),
        'hy_dir': str(hy_dir),
        'total_samples': len(kb_preds),
        'regressions_count': len(regressions),
        'improvements_count': len(improvements),
        'both_correct_count': len(both_correct),
        'both_wrong_count': len(both_wrong),
        'top1_same_count': same_top1,
        'top1_same_rate': round(same_top1 / len(kb_preds), 4),
        'net_gain': len(improvements) - len(regressions),
        'kb_accuracy': round((len(regressions) + len(both_correct)) / len(kb_preds), 4),
        'hy_accuracy': round((len(improvements) + len(both_correct)) / len(kb_preds), 4),
    }

    summary_file = output_dir / 'hybrid_diff_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved: {summary_file}")
    print()

    # Summary 출력
    print("="*80)
    print("Summary")
    print("="*80)
    print(f"KB accuracy:  {summary['kb_accuracy']:.1%}")
    print(f"HY accuracy:  {summary['hy_accuracy']:.1%}")
    print(f"Net gain:     {summary['net_gain']:+d} ({summary['improvements_count']} - {summary['regressions_count']})")
    print()

    if summary['net_gain'] > 0:
        print(f"[PASS] Hybrid improves over KB-only by {summary['net_gain']} samples")
    elif summary['net_gain'] < 0:
        print(f"[FAIL] Hybrid regresses from KB-only by {-summary['net_gain']} samples")
    else:
        print(f"[EQUAL] Hybrid and KB-only have same accuracy")

    return summary


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        print("\nError: Expected 2 arguments (kb_dir, hy_dir)")
        sys.exit(1)

    kb_dir = Path(sys.argv[1])
    hy_dir = Path(sys.argv[2])

    if not kb_dir.exists():
        print(f"Error: KB-only directory not found: {kb_dir}")
        sys.exit(1)

    if not hy_dir.exists():
        print(f"Error: Hybrid directory not found: {hy_dir}")
        sys.exit(1)

    analyze_regressions(kb_dir, hy_dir)


if __name__ == '__main__':
    main()
