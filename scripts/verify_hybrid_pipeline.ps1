# HS Classification System - Hybrid Pipeline Verification Script
# Purpose: Verify ranker is applied and metrics are correct

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Hybrid Pipeline Verification" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Step 1: Run hybrid eval with 200 samples
Write-Host "[1/4] Running hybrid evaluation (200 samples)..." -ForegroundColor Yellow
python -m src.classifier.eval.run_eval --mode hybrid --limit 200 --seed 42

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Hybrid eval failed" -ForegroundColor Red
    exit 1
}

# Find the latest hybrid run directory
$evalDir = Get-ChildItem -Path "artifacts/eval" -Directory |
    Where-Object { $_.Name -like "hybrid_*" } |
    Sort-Object CreationTime -Descending |
    Select-Object -First 1

if (-not $evalDir) {
    Write-Host "ERROR: No hybrid run directory found" -ForegroundColor Red
    exit 1
}

$runPath = $evalDir.FullName
Write-Host "  Run directory: $runPath" -ForegroundColor Green
Write-Host ""

# Step 2: Check retriever/ranker usage rates
Write-Host "[2/4] Checking retriever/ranker usage..." -ForegroundColor Yellow

$usageCheckScript = @"
import json
import sys

with open('$runPath/predictions_test.jsonl', 'r', encoding='utf-8') as f:
    lines = [line for line in f]

total = len(lines)
retriever_count = 0
ranker_count = 0
nonzero_ml_count = 0

for line in lines:
    d = json.loads(line)
    debug = d['debug']

    if debug.get('retriever_used'):
        retriever_count += 1

    if debug.get('ranker_applied'):
        ranker_count += 1

    # Check score_ml nonzero
    topk = d.get('topk', [])
    if topk and topk[0].get('score_ml', 0.0) > 0:
        nonzero_ml_count += 1

retriever_rate = retriever_count / total if total > 0 else 0
ranker_rate = ranker_count / total if total > 0 else 0
ml_nonzero_rate = nonzero_ml_count / total if total > 0 else 0

print(f'  Total samples: {total}')
print(f'  Retriever used: {retriever_count}/{total} ({retriever_rate:.2%})')
print(f'  Ranker applied: {ranker_count}/{total} ({ranker_rate:.2%})')
print(f'  score_ml nonzero: {nonzero_ml_count}/{total} ({ml_nonzero_rate:.2%})')

# Hybrid should have retriever and ranker
if retriever_rate < 0.95:
    print(f'  WARNING: Retriever usage rate is low! Expected >95%, got {retriever_rate:.2%}')
if ranker_rate < 0.95:
    print(f'  WARNING: Ranker applied rate is low! Expected >95%, got {ranker_rate:.2%}')
"@

python -c $usageCheckScript
Write-Host ""

# Step 3: Display key metrics
Write-Host "[3/4] Displaying key metrics..." -ForegroundColor Yellow

$metricsScript = @"
import json

with open('$runPath/metrics_summary.json', 'r', encoding='utf-8') as f:
    metrics = json.load(f)

print('  Core Metrics:')
print(f'    Total Samples: {metrics["total_samples"]}')
print(f'    Top-1 Accuracy: {metrics["top1_accuracy"]:.4f}')
print(f'    Top-3 Accuracy: {metrics["top3_accuracy"]:.4f}')
print(f'    Top-5 Accuracy: {metrics["top5_accuracy"]:.4f}')
print(f'    Macro F1: {metrics["macro_f1"]:.4f}')
print(f'    Weighted F1: {metrics["weighted_f1"]:.4f}')
print(f'    Candidate Recall@5: {metrics["candidate_recall_5"]:.4f}')
print(f'    ECE: {metrics["ece"]:.4f}')
print(f'    Brier Score: {metrics["brier_score"]:.4f}')
print(f'    Auto Rate: {metrics["auto_rate"]:.4f}')
print(f'    Legal Conflict Rate: {metrics["legal_conflict_rate"]:.4f}')
"@

python -c $metricsScript
Write-Host ""

# Step 4: Check LegalGate feature usage
Write-Host "[4/4] Checking usage audit..." -ForegroundColor Yellow

$usageScript = @"
import json

with open('$runPath/usage_summary.json', 'r', encoding='utf-8') as f:
    usage = json.load(f)

print('  Usage Statistics:')
print(f'    Retriever Usage Rate: {usage.get("retriever_usage_rate", 0.0):.2%}')
print(f'    Ranker Usage Rate: {usage["ranker_usage_rate"]:.2%}')
print(f'    LegalGate Usage Rate: {usage["legal_gate_usage_rate"]:.2%}')
print(f'    Avg Legal Excluded: {usage["avg_legal_excluded"]:.2f}')
print(f'    Avg Notes Support: {usage["avg_notes_support"]:.2f}')
print(f'    Avg Cards Hits: {usage["avg_cards_hits"]:.2f}')
print(f'    Avg Rule Hits: {usage["avg_rule_hits"]:.2f}')

# Check for retriever_usage_rate key
if 'retriever_usage_rate' not in usage:
    print('  WARNING: retriever_usage_rate not found in usage_summary.json')

if usage.get('retriever_usage_rate', 0.0) < 0.95:
    print(f'  WARNING: Low retriever usage rate: {usage.get("retriever_usage_rate", 0.0):.2%}')
if usage['ranker_usage_rate'] < 0.95:
    print(f'  WARNING: Low ranker usage rate: {usage["ranker_usage_rate"]:.2%}')
"@

python -c $usageScript
Write-Host ""

# Success summary
Write-Host "=" * 80 -ForegroundColor Green
Write-Host "Verification Complete!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Green
Write-Host ""
Write-Host "Run Directory: $runPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Compare with KB-only: python -m src.classifier.eval.run_eval --mode kb_only --limit 200 --seed 42"
Write-Host "  2. Check feature importance: Review docs/LEGAL_FEATURE_AUDIT.md"
Write-Host "  3. Rebuild ranker if needed: python -m src.classifier.rank.train_ranker_legal --build"
Write-Host ""
