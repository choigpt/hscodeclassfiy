"""
RAG LLM Ablation Study
동일한 RAG 프레임워크(BM25+SBERT retrieval)에서 LLM backbone만 교체하여 성능 비교

Usage:
    python scripts/rag_ablation_llm.py [n_samples]
    python scripts/rag_ablation_llm.py 50
"""
import json
import time
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))


# 비교할 모델 목록 (Ollama에 설치된 것만 실행)
MODELS_TO_TEST = [
    "qwen2.5:7b",
    "gemma2:9b",
    "llama3.2:3b",
]


def get_available_models():
    """Ollama에 설치된 모델 확인"""
    import subprocess
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    installed = []
    for line in result.stdout.strip().split("\n")[1:]:  # skip header
        if line.strip():
            name = line.split()[0]
            installed.append(name)
    return installed


def load_test_samples(n=50, seed=42):
    """동일한 seed로 test split에서 n개 추출"""
    with open("data/ruling_cases/all_cases_full_v7.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    random.seed(seed)
    random.shuffle(data)
    train_end = int(len(data) * 0.8)
    val_end = train_end + int(len(data) * 0.1)
    test_data = data[val_end:]
    valid = [s for s in test_data if s.get("product_name") and len(s.get("hs_heading", "")) == 4]
    return valid[:n]


def run_rag_with_model(model_name, samples):
    """특정 LLM 모델로 RAG 파이프라인 실행"""
    from src.rag.pipeline import RAGPipeline
    pipe = RAGPipeline(ollama_model=model_name)

    results = []
    errors = 0
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
                "elapsed": round(elapsed, 1),
            })
        except Exception as e:
            elapsed = time.time() - t0
            errors += 1
            results.append({
                "text": text[:40],
                "true": true_hs4,
                "pred": "ERR",
                "top5": [],
                "hit1": False,
                "hit5": False,
                "confidence": 0,
                "elapsed": round(elapsed, 1),
                "error": str(e)[:80],
            })
        status = "OK" if results[-1]["hit1"] else ("ERR" if results[-1]["pred"] == "ERR" else "")
        print(f"  [{model_name}] [{i+1}/{len(samples)}] {elapsed:.1f}s | pred={results[-1]['pred']} true={true_hs4} {status}")
        sys.stdout.flush()

    return results, errors


def summarize(name, results):
    n = len(results)
    hit1 = sum(1 for r in results if r["hit1"])
    hit5 = sum(1 for r in results if r["hit5"])
    avg_time = sum(r["elapsed"] for r in results) / n if n else 0
    error_count = sum(1 for r in results if r.get("error"))
    return {
        "model": name,
        "n": n,
        "top1_acc": round(hit1 / n * 100, 1) if n else 0,
        "top5_acc": round(hit5 / n * 100, 1) if n else 0,
        "hit1": hit1,
        "hit5": hit5,
        "avg_sec": round(avg_time, 1),
        "total_sec": round(sum(r["elapsed"] for r in results), 0),
        "errors": error_count,
    }


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    print(f"=== RAG LLM Ablation Study ({N} samples) ===\n")

    # 설치된 모델 확인
    installed = get_available_models()
    print(f"Installed models: {installed}")
    models = [m for m in MODELS_TO_TEST if m in installed]
    print(f"Models to test: {models}\n")

    if not models:
        print("No testable models found!")
        sys.exit(1)

    samples = load_test_samples(n=N)
    print(f"Test samples loaded: {len(samples)}\n")

    all_results = {}
    all_summaries = []

    for idx, model in enumerate(models):
        print(f"\n[{idx+1}/{len(models)}] Testing: {model}")
        print("=" * 60)
        results, errors = run_rag_with_model(model, samples)
        summary = summarize(model, results)
        all_results[model] = results
        all_summaries.append(summary)
        print(f"\n  >> {model}: Top-1={summary['top1_acc']}%, Top-5={summary['top5_acc']}%, "
              f"Avg={summary['avg_sec']}s, Errors={summary['errors']}")

    # 결과 테이블
    print("\n" + "=" * 80)
    print(f"{'Metric':<20}", end="")
    for s in all_summaries:
        print(f" {s['model']:>15}", end="")
    print()
    print("-" * 80)
    for key, label in [
        ("top1_acc", "Top-1 Acc (%)"),
        ("top5_acc", "Top-5 Acc (%)"),
        ("hit1", "Top-1 Hits"),
        ("hit5", "Top-5 Hits"),
        ("avg_sec", "Avg sec/sample"),
        ("total_sec", "Total sec"),
        ("errors", "Errors"),
    ]:
        print(f"{label:<20}", end="")
        for s in all_summaries:
            print(f" {s[key]:>15}", end="")
        print()
    print("=" * 80)

    # JSON 저장
    output = {
        "n_samples": N,
        "models": models,
        "summaries": all_summaries,
        "details": all_results,
    }
    os.makedirs("artifacts/eval", exist_ok=True)
    outpath = f"artifacts/eval/rag_ablation_{N}samples.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {outpath}")
