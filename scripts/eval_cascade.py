"""
Cascade Pipeline Evaluation

Hybrid-First + RAG Escalation 전략 평가

Usage:
    python scripts/eval_cascade.py [n_samples]
    python scripts/eval_cascade.py 50
"""
import json
import time
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))


def load_test_samples(n=50, seed=42):
    with open("data/ruling_cases/all_cases_full_v7.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    random.seed(seed)
    random.shuffle(data)
    train_end = int(len(data) * 0.8)
    val_end = train_end + int(len(data) * 0.1)
    test_data = data[val_end:]
    valid = [s for s in test_data if s.get("product_name") and len(s.get("hs_heading", "")) == 4]
    return valid[:n]


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    print(f"=== Cascade Pipeline Evaluation ({N} samples) ===\n")

    from src.cascade.pipeline import CascadePipeline

    pipe = CascadePipeline(
        hybrid_mode="hybrid",
        ollama_model="qwen2.5:7b",
        escalation_threshold_top1=0.50,
        escalation_threshold_gap=0.15,
    )

    samples = load_test_samples(n=N)
    print(f"Test samples loaded: {len(samples)}\n")

    results = []
    stats = {
        "total": len(samples),
        "hybrid_direct": 0,    # 에스컬레이션 없이 Hybrid로 처리
        "escalated": 0,        # RAG 에스컬레이션
        "source_hybrid": 0,    # 최종 Hybrid 결과 채택
        "source_rag": 0,       # 최종 RAG 결과 채택
        "source_rag_confirmed": 0,  # 교차검증 확인
        "hit1": 0,
        "hit5": 0,
        # 에스컬레이션 세부
        "esc_hit1": 0,         # 에스컬레이션 케이스 중 Top-1 정답
        "esc_total": 0,
        "no_esc_hit1": 0,      # 비에스컬레이션 케이스 중 Top-1 정답
        "no_esc_total": 0,
    }

    for i, s in enumerate(samples):
        text = s["product_name"]
        true_hs4 = s["hs_heading"]

        t0 = time.time()
        r = pipe.classify(text)
        elapsed = time.time() - t0

        pred = r.best_hs4
        top5 = [c.hs4 for c in r.final_result.topk[:5]]
        hit1 = pred == true_hs4
        hit5 = true_hs4 in top5

        if hit1:
            stats["hit1"] += 1
        if hit5:
            stats["hit5"] += 1

        if r.escalated:
            stats["escalated"] += 1
            stats["esc_total"] += 1
            if hit1:
                stats["esc_hit1"] += 1
        else:
            stats["hybrid_direct"] += 1
            stats["no_esc_total"] += 1
            if hit1:
                stats["no_esc_hit1"] += 1

        stats[f"source_{r.source}"] += 1

        # Debug info
        esc_tag = f"ESC→{r.source}" if r.escalated else "DIRECT"
        status = "OK" if hit1 else ""
        rag_info = ""
        if r.rag_result:
            rag_info = f" rag={r.rag_result.best_hs4}"

        print(
            f"  [{i+1}/{N}] {elapsed:.1f}s | [{esc_tag:>16}] "
            f"pred={pred} true={true_hs4}{rag_info} {status}"
        )
        sys.stdout.flush()

        results.append({
            "text": text[:40],
            "true": true_hs4,
            "pred": pred,
            "top5": top5,
            "hit1": hit1,
            "hit5": hit5,
            "source": r.source,
            "escalated": r.escalated,
            "escalation_reason": r.escalation_reason,
            "confidence": round(r.confidence, 4),
            "elapsed": round(elapsed, 1),
            "debug": {
                "hybrid_top1": r.debug.get("hybrid_top1", ""),
                "hybrid_conf": r.debug.get("hybrid_confidence", 0),
                "rag_top1": r.debug.get("rag_top1", ""),
                "rag_conf": r.debug.get("rag_confidence", 0),
                "merge_reason": r.debug.get("merge", {}).get("merge_reason", ""),
            },
        })

    # Summary
    n = stats["total"]
    top1_acc = stats["hit1"] / n * 100 if n else 0
    top5_acc = stats["hit5"] / n * 100 if n else 0
    esc_rate = stats["escalated"] / n * 100 if n else 0
    avg_sec = sum(r["elapsed"] for r in results) / n if n else 0

    esc_acc = stats["esc_hit1"] / stats["esc_total"] * 100 if stats["esc_total"] else 0
    no_esc_acc = stats["no_esc_hit1"] / stats["no_esc_total"] * 100 if stats["no_esc_total"] else 0

    print(f"\n{'='*70}")
    print(f"Cascade Pipeline Results ({N} samples)")
    print(f"{'='*70}")
    print(f"  Overall Top-1 Accuracy:   {top1_acc:.1f}% ({stats['hit1']}/{n})")
    print(f"  Overall Top-5 Accuracy:   {top5_acc:.1f}% ({stats['hit5']}/{n})")
    print(f"  Avg sec/sample:           {avg_sec:.1f}s")
    print()
    print(f"  Escalation Rate:          {esc_rate:.1f}% ({stats['escalated']}/{n})")
    print(f"  Direct (Hybrid) Count:    {stats['hybrid_direct']}")
    print(f"  Escalated Count:          {stats['escalated']}")
    print()
    print(f"  Direct Top-1 Acc:         {no_esc_acc:.1f}% ({stats['no_esc_hit1']}/{stats['no_esc_total']})")
    print(f"  Escalated Top-1 Acc:      {esc_acc:.1f}% ({stats['esc_hit1']}/{stats['esc_total']})")
    print()
    print(f"  Final Source Breakdown:")
    print(f"    hybrid:                 {stats['source_hybrid']}")
    print(f"    rag:                    {stats['source_rag']}")
    print(f"    rag_confirmed:          {stats['source_rag_confirmed']}")
    print(f"{'='*70}")

    # Comparison
    print(f"\n  Comparison (same {N} samples):")
    print(f"    Cascade:   {top1_acc:.1f}% Top-1")
    print(f"    (vs Hybrid-only 63.0%, RAG-only 63.5% on 200 samples)")

    # Save
    output = {
        "n_samples": N,
        "config": {
            "escalation_threshold_top1": pipe.escalation_threshold_top1,
            "escalation_threshold_gap": pipe.escalation_threshold_gap,
            "rag_min_confidence": pipe.rag_min_confidence,
            "cross_validation_boost": pipe.cross_validation_boost,
        },
        "stats": stats,
        "summary": {
            "top1_acc": round(top1_acc, 1),
            "top5_acc": round(top5_acc, 1),
            "escalation_rate": round(esc_rate, 1),
            "avg_sec": round(avg_sec, 1),
            "direct_acc": round(no_esc_acc, 1),
            "escalated_acc": round(esc_acc, 1),
        },
        "details": results,
    }
    os.makedirs("artifacts/eval", exist_ok=True)
    outpath = f"artifacts/eval/cascade_{N}samples.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {outpath}")
