# KB-only vs Hybrid Mode Comparison

**Date**: 2026-02-03
**Evaluation**: 200 samples (test split, seed=42)
**Model**: Full dataset (7,198 samples) with f_legal_heading_term active

---

## Executive Summary

**결론**: **KB-only 모드가 Hybrid 모드보다 37.5% 더 우수**

| Metric | KB-only | Hybrid | 차이 | 승자 |
|--------|---------|--------|------|------|
| **Top-1 Accuracy** | **12.0%** | **7.5%** | **-4.5%** | **KB-only** |
| Top-3 Accuracy | 23.5% | 14.0% | -9.5% | KB-only |
| Top-5 Accuracy | 35.5% | 19.0% | -16.5% | KB-only |

**권장 사항**: 현재는 KB-only 모드를 기본으로 사용하고, ML Retriever 개선 후 재평가

---

## 모드 개요

### KB-only Mode
- ❌ ML Retriever, ML Ranker (비활성화)
- ✅ KB Retrieval (Lexical + Cards + Rules)
- ✅ LegalGate (GRI 1)
- 빠름 (~0.27초/샘플)

### Hybrid Mode
- ✅ ML Retriever (ko-sroberta + LR)
- ✅ ML Ranker (LightGBM, 39 features)
- ✅ KB Retrieval + LegalGate
- 느림 (~0.62초/샘플, 2.3배)

---

## 핵심 문제

### 1. ML Retriever 품질 문제
```
KB-only recall@5: 35.5%
Hybrid recall@5: 19.0%
```
ML retriever가 정답을 포함하는 후보를 적게 생성

### 2. Ranker 효과 부족
```
Ranker NDCG@1: 0.78 (validation)
Hybrid accuracy: 0.075 (test)
```
Ranker가 bad retrieval을 복구하지 못함

### 3. Feature Dominance
```
f_lexical: 251,890 (90% 기여)
f_legal_heading_term: 281 (0.1% 기여)
```
f_lexical이 너무 강해서 다른 피처 무시

---

## 권장 조치

**즉시 (1일)**:
- KB-only를 기본 모드로 사용

**단기 (1주)**:
- Calibration layer 구현
- 전체 test split 평가

**중기 (1개월)**:
- ML retriever fine-tuning
- Feature scaling 조정

**장기 (3개월)**:
- Ensemble approach
- Meta-ranker 개발

---

**상세 내용**: `docs/HEADING_TERMS_INTEGRATION_REPORT.md` 참조
