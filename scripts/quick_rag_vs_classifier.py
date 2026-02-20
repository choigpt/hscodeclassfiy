"""
RAG vs Classifier 빠른 비교 (소규모 샘플)
CPU Ollama 환경에서 실행 가능하도록 소수 샘플만 사용
"""
import json
import time
import random
import sys
import os

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

def load_test_samples(n=20, seed=42):
    """동일한 seed로 test split에서 n개 추출"""
    with open("data/ruling_cases/all_cases_full_v7.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    random.seed(seed)
    random.shuffle(data)
    # 80/10/10 split → test는 마지막 10%
    train_end = int(len(data) * 0.8)
    val_end = train_end + int(len(data) * 0.1)
    test_data = data[val_end:]
    # 유효 샘플만
    valid = [s for s in test_data if s.get("product_name") and len(s.get("hs_heading", "")) == 4]
    return valid[:n]


def run_rag(samples):
    """RAG 파이프라인 평가"""
    from src.rag.pipeline import RAGPipeline
    pipe = RAGPipeline()
    results = []
    for i, s in enumerate(samples):
        text = s["product_name"]
        true_hs4 = s["hs_heading"]
        t0 = time.time()
        try:
            r = pipe.classify(text)
            pred_hs4 = r.best_hs4
            candidates_hs4 = [c.hs4 for c in r.candidates[:5]]
            elapsed = time.time() - t0
            results.append({
                "text": text[:40],
                "true": true_hs4,
                "pred": pred_hs4,
                "top5": candidates_hs4,
                "hit1": pred_hs4 == true_hs4,
                "hit5": true_hs4 in candidates_hs4,
                "confidence": round(r.confidence.calibrated, 3),
                "fallback": r.is_fallback,
                "elapsed": round(elapsed, 1),
            })
        except Exception as e:
            elapsed = time.time() - t0
            results.append({
                "text": text[:40],
                "true": true_hs4,
                "pred": "ERR",
                "top5": [],
                "hit1": False,
                "hit5": False,
                "confidence": 0,
                "fallback": True,
                "elapsed": round(elapsed, 1),
                "error": str(e)[:80],
            })
        print(f"  RAG [{i+1}/{len(samples)}] {elapsed:.1f}s | pred={results[-1]['pred']} true={true_hs4} {'OK' if results[-1]['hit1'] else ''}")
        sys.stdout.flush()
    return results


def run_classifier_kb(samples):
    """Classifier KB-only 파이프라인 평가"""
    from src.classifier.pipeline import HSPipeline
    from src.classifier.reranker import HSReranker
    from src.classifier.clarify import HSClarifier
    pipe = HSPipeline(
        retriever=None, reranker=HSReranker(), clarifier=HSClarifier(),
        use_gri=True, use_legal_gate=True, use_8axis=True,
        use_rules=True, use_ranker=False, use_questions=True
    )
    results = []
    for i, s in enumerate(samples):
        text = s["product_name"]
        true_hs4 = s["hs_heading"]
        t0 = time.time()
        r = pipe.classify(text)
        elapsed = time.time() - t0
        topk_hs4 = [c.hs4 for c in r.topk[:5]]
        pred_hs4 = topk_hs4[0] if topk_hs4 else ""
        results.append({
            "text": text[:40],
            "true": true_hs4,
            "pred": pred_hs4,
            "top5": topk_hs4,
            "hit1": pred_hs4 == true_hs4,
            "hit5": true_hs4 in topk_hs4,
            "confidence": round(r.topk[0].score_total, 3) if r.topk else 0,
            "elapsed": round(elapsed, 1),
        })
        print(f"  KB  [{i+1}/{len(samples)}] {elapsed:.1f}s | pred={pred_hs4} true={true_hs4} {'OK' if results[-1]['hit1'] else ''}")
        sys.stdout.flush()
    return results


def run_classifier_hybrid(samples):
    """Classifier Hybrid 파이프라인 평가"""
    from src.classifier.pipeline import HSPipeline
    from src.classifier.retriever import HSRetriever
    from src.classifier.reranker import HSReranker
    from src.classifier.clarify import HSClarifier
    pipe = HSPipeline(
        retriever=HSRetriever(), reranker=HSReranker(), clarifier=HSClarifier(),
        ranker_model_path="artifacts/ranker_legal/model_legal.txt",
        use_gri=True, use_legal_gate=True, use_8axis=True,
        use_rules=True, use_ranker=True, use_questions=True
    )
    results = []
    for i, s in enumerate(samples):
        text = s["product_name"]
        true_hs4 = s["hs_heading"]
        t0 = time.time()
        r = pipe.classify(text)
        elapsed = time.time() - t0
        topk_hs4 = [c.hs4 for c in r.topk[:5]]
        pred_hs4 = topk_hs4[0] if topk_hs4 else ""
        results.append({
            "text": text[:40],
            "true": true_hs4,
            "pred": pred_hs4,
            "top5": topk_hs4,
            "hit1": pred_hs4 == true_hs4,
            "hit5": true_hs4 in topk_hs4,
            "confidence": round(r.topk[0].score_total, 3) if r.topk else 0,
            "elapsed": round(elapsed, 1),
        })
        print(f"  HYB [{i+1}/{len(samples)}] {elapsed:.1f}s | pred={pred_hs4} true={true_hs4} {'OK' if results[-1]['hit1'] else ''}")
        sys.stdout.flush()
    return results


def summarize(name, results):
    n = len(results)
    hit1 = sum(1 for r in results if r["hit1"])
    hit5 = sum(1 for r in results if r["hit5"])
    avg_time = sum(r["elapsed"] for r in results) / n if n else 0
    fallback_count = sum(1 for r in results if r.get("fallback", False))
    return {
        "name": name,
        "n": n,
        "top1_acc": round(hit1 / n * 100, 1) if n else 0,
        "top5_acc": round(hit5 / n * 100, 1) if n else 0,
        "hit1": hit1,
        "hit5": hit5,
        "avg_sec": round(avg_time, 1),
        "total_sec": round(sum(r["elapsed"] for r in results), 0),
        "fallback": fallback_count,
    }


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print(f"=== RAG vs Classifier 비교 ({N} samples) ===\n")

    samples = load_test_samples(n=N)
    print(f"Test samples loaded: {len(samples)}\n")

    # 1. KB-only
    print("[1/3] Classifier KB-only")
    kb_results = run_classifier_kb(samples)
    kb_summary = summarize("KB-only", kb_results)
    print()

    # 2. Hybrid
    print("[2/3] Classifier Hybrid")
    hybrid_results = run_classifier_hybrid(samples)
    hybrid_summary = summarize("Hybrid", hybrid_results)
    print()

    # 3. RAG
    print("[3/3] RAG (BM25+SBERT+Qwen2.5 7B)")
    rag_results = run_rag(samples)
    rag_summary = summarize("RAG", rag_results)
    print()

    # 결과 테이블
    print("=" * 70)
    print(f"{'Metric':<20} {'KB-only':>10} {'Hybrid':>10} {'RAG':>10}")
    print("-" * 70)
    for key, label in [
        ("top1_acc", "Top-1 Acc (%)"),
        ("top5_acc", "Top-5 Acc (%)"),
        ("hit1", "Top-1 Hits"),
        ("hit5", "Top-5 Hits"),
        ("avg_sec", "Avg sec/sample"),
        ("total_sec", "Total sec"),
        ("fallback", "Fallback count"),
    ]:
        print(f"{label:<20} {kb_summary[key]:>10} {hybrid_summary[key]:>10} {rag_summary[key]:>10}")
    print("=" * 70)

    # 샘플별 비교
    print("\n=== 샘플별 비교 ===")
    print(f"{'#':<3} {'Text':<30} {'True':<6} {'KB':<6} {'HYB':<6} {'RAG':<6} {'Note'}")
    print("-" * 90)
    for i in range(len(samples)):
        kb = kb_results[i]
        hy = hybrid_results[i]
        rg = rag_results[i]
        note = ""
        if rg["hit1"] and not kb["hit1"] and not hy["hit1"]:
            note = "RAG-only-win"
        elif kb["hit1"] and not rg["hit1"]:
            note = "KB>RAG"
        elif hy["hit1"] and not rg["hit1"]:
            note = "HYB>RAG"
        elif rg["hit1"]:
            note = "all-hit" if kb["hit1"] and hy["hit1"] else "RAG-hit"
        print(f"{i+1:<3} {kb['text']:<30} {kb['true']:<6} {kb['pred']:<6} {hy['pred']:<6} {rg['pred']:<6} {note}")

    # JSON 저장
    output = {
        "n_samples": N,
        "summaries": [kb_summary, hybrid_summary, rag_summary],
        "details": {
            "kb_only": kb_results,
            "hybrid": hybrid_results,
            "rag": rag_results,
        }
    }
    outpath = f"artifacts/eval/quick_compare_{N}samples.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {outpath}")
