# HS Code Classification System

한국 관세청 HS Code (품목분류번호) 자동 분류 시스템

**법적 근거 기반** + **설명 가능한** + **능동적 질의**를 특징으로 하는 하이브리드 ML/KB 시스템

---

## 주요 특징

- ✅ **GRI 준수**: 5개 General Rules of Interpretation (GRI 1/2a/2b/3/5) 지원
- ✅ **8-Axis 속성**: 재질, 용도, 가공상태 등 다차원 의미 분석
- ✅ **법적 필터링**: Tariff Notes 기반 LegalGate (GRI 1)
- ✅ **하이브리드 접근**: ML Retriever + KB Reranker + LightGBM Ranker
- ✅ **설명 가능성**: 모든 예측에 Evidence 기반 근거 제공
- ✅ **질문 생성**: Low confidence 시 자동 질문 (최대 3개)
- ✅ **2가지 모드**: KB-only (순수 규칙) / Hybrid (ML + KB)

---

## 성능

### End-to-End (Test 200 samples)

| 모드 | Top-1 Acc | Top-5 Acc | 특징 |
|------|-----------|-----------|------|
| **KB-only** | 12.0% | 35.5% | ML 불사용, 설명 가능 |
| **Hybrid** | **13.5%** | 19.0% | KB-first + ML recall |

### LightGBM Ranker (Test 1,428 queries)

| Metric | Train | Test |
|--------|-------|------|
| Top-1 Acc | 0.7907 | **0.7661** |
| Top-5 Acc | 0.9466 | **0.9426** |
| NDCG@5 | 0.8856 | **0.8716** |

**최신 성과** (2026-02-08):
- f_lexical 정규화: fallback weighted-score 경로 수정 완료
- KB lock 안정화: internal rerank pool 5 → 20 확장
- Fact-insufficient 로직: 8-axis 기반 판정으로 교체
- Feature dominance 분석: f_lexical gain 86.8% (tree invariance로 정규화 무효, 구조적 접근 필요)

---

## 빠른 시작

### 1. 설치

```bash
pip install -r requirements.txt
```

### 2. 평가 실행

**KB-only 모드** (순수 지식 베이스):
```bash
python -m src.classifier.eval.run_eval --mode kb_only --limit 200 --seed 42
```

**Hybrid 모드** (ML + KB):
```bash
python -m src.classifier.eval.run_eval --mode hybrid --limit 200 --seed 42
```

### 3. 회귀 분석

```bash
python scripts/analyze_hybrid_regressions.py \
  artifacts/eval/kb_only_20260203_214958 \
  artifacts/eval/hybrid_20260203_220018
```

### 4. 단일 예측

```python
from src.classifier.pipeline import HSPipeline
from src.classifier.retriever import HSRetriever
from src.classifier.reranker import HSReranker

# Hybrid 모드
pipeline = HSPipeline(
    retriever=HSRetriever(),
    reranker=HSReranker(),
    use_ranker=True
)

result = pipeline.classify("냉동 돼지 삼겹살")
print(f"Top1: {result.topk[0].hs4}")
print(f"Decision: {result.decision.status}")
```

---

## 프로젝트 구조

```
HS/
├── src/
│   ├── classifier/              # 핵심 파이프라인 (7,642 lines)
│   │   ├── pipeline.py          # Orchestration
│   │   ├── retriever.py         # ML Retriever (SBERT + LR)
│   │   ├── reranker.py          # KB Reranker (Card/Rule)
│   │   ├── gri_signals.py       # GRI 탐지
│   │   ├── attribute_extract.py # 8-Axis 속성 추출
│   │   ├── legal_gate.py        # Tariff Notes 필터링
│   │   ├── fact_checker.py      # 정보 충분성 검증
│   │   ├── clarify.py           # 질문 생성
│   │   └── explanation_generator.py  # 설명 생성
│   └── experiments/             # 평가 프레임워크 (5,003 lines)
│       ├── run_benchmark.py     # 4개 모델 비교
│       ├── ablation_runner.py   # 컴포넌트 ablation
│       └── bucket_analyzer.py   # 난이도별 분석
│
├── data/
│   ├── ruling_cases/
│   │   └── all_cases_full_v7.json  # 7,198 cases
│   ├── benchmarks/              # Train/Val/Test splits
│   ├── tariff_notes_clean.json  # Tariff notes
│   └── tariff_notes_clean.txt
│
├── kb/
│   ├── raw/
│   │   └── hs_commentary.json   # WCO HS Explanatory Notes
│   ├── structured/
│   │   ├── hs4_cards_v2.jsonl   # 1,240 HS4 cards
│   │   ├── hs4_rule_chunks.jsonl # 11,912 rule chunks
│   │   ├── thesaurus_terms.jsonl # 7,098 terms
│   │   └── note_requirements.jsonl # Legal notes
│   └── build_scripts/           # KB 빌드 도구
│
├── artifacts/
│   ├── classifier/              # ML 모델
│   │   ├── model_lr.joblib      # Logistic Regression
│   │   └── label_encoder.joblib
│   ├── ranker_legal/            # LightGBM 모델
│   │   └── model_legal.txt
│   └── eval/                    # 평가 결과 (4개 최신 runs)
│
├── docs/                        # 문서 (10개)
│   ├── PROJECT_SUMMARY_20260204.md  # 프로젝트 종합 정리 ⭐
│   ├── MODE_SEPARATION_FIX_REPORT.md # KB-first 전략
│   ├── METHODOLOGY.md           # 방법론
│   └── ...
│
├── scripts/                     # 유틸리티
│   ├── analyze_hybrid_regressions.py
│   ├── train_ranker_and_sanity_check.py  # Ranker 학습 + sanity check
│   ├── run_full_evaluation.py
│   └── parse_tariff_notes_v2.py
│
├── CHANGELOG.md                 # 변경 이력 ⭐
├── README.md                    # 이 파일
└── requirements.txt
```

---

## 파이프라인 흐름

```
Input Text (품목 설명)
    ↓
[Step 0] GRI Signals + 8-Axis Attributes 추출
    ↓
[Step 1] ML Retriever → Top-50 candidates
    ↓
[Step 2] KB Retrieval → Top-30 candidates (GRI 조정)
    ↓
[Step 3] Merge (KB-first + ML recall)
    ↓
[Step 3.5] LegalGate (GRI 1) → Hard filtering
    ↓
[Step 4] Reranking (Card/Rule + 8-Axis + LightGBM)
    ↓
[Step 5] Confidence Check
    ↓
[Step 6] Question Generation (if needed)
    ↓
Output: Top-5 + Decision (AUTO/ASK) + Questions + Evidence
```

---

## 주요 컴포넌트

### 1. GRI Signals Detector
- **GRI 1**: Tariff notes → LegalGate 활성화
- **GRI 2a**: 미조립 → 완성품 후보 확대 (+20)
- **GRI 2b**: 혼합물 → 재질 후보 확대 (+10)
- **GRI 3**: 세트 → Set 분석
- **GRI 5**: 포장 → Container 로직

### 2. 8-Axis Attributes
1. object_nature (물체 본질)
2. material (재질/성분)
3. processing_state (가공상태)
4. function_use (기능/용도)
5. physical_form (물리적 형태)
6. completeness (완성도)
7. quantitative_rules (정량규칙)
8. legal_scope (법적범위)

### 3. LegalGate (GRI 1 Filter)
- Heading term 매칭 (1,239개 HS4)
- Include rules: 긍정 증거
- Exclude rules: 하드 필터 (위반 시 후보 제거)
- Redirect rules: 올바른 HS4로 안내

### 4. KB-first Merge Strategy
- KB 후보 우선 배치
- ML은 KB에 없는 것만 추가 (recall 보강)
- KB confidence gate: 고신뢰 KB 예측 보호
- Conditional ML weight: 상황별 0.05~0.5 동적 조정

### 5. Confidence Routing
- **AUTO**: High confidence (Top1-Top2 margin > threshold)
- **ASK**: Low confidence → 2-3개 질문 생성
- **REVIEW**: Legal conflict / Fact insufficient
- **ABSTAIN**: 분류 불가

---

## 평가 프레임워크

### KB-only vs Hybrid 비교

```bash
# 1. KB-only 평가
python -m src.classifier.eval.run_eval --mode kb_only --limit 200 --seed 42

# 2. Hybrid 평가
python -m src.classifier.eval.run_eval --mode hybrid --limit 200 --seed 42

# 3. 회귀 분석
python scripts/analyze_hybrid_regressions.py [kb_dir] [hybrid_dir]
```

### 출력 파일
- `predictions_test.jsonl`: 샘플별 예측 결과
- `metrics_summary.json`: Top-K accuracy, F1, ECE, etc.
- `usage_audit.jsonl`: 컴포넌트 사용 추적
- `hybrid_regressions.jsonl`: KB✓ → HY✗ 샘플
- `hybrid_improvements.jsonl`: KB✗ → HY✓ 샘플
- `hybrid_diff_summary.json`: Net gain 요약

---

## 벤치마크

### 4개 모델 비교

```bash
python -m src.experiments.run_benchmark
```

**모델**:
1. **B0_TFIDF_LR**: TF-IDF + Logistic Regression
2. **B1_SBERT_LR**: Sentence BERT + LR
3. **KB-only**: 순수 KB (Card + Rule + LegalGate)
4. **Hybrid**: Full pipeline (ML + KB + Ranker)

### Ablation Study

```bash
python -m src.experiments.ablation_runner
```

**Progression**:
- P1: ML-only
- P2: + GRI signals
- P3: + 8-axis attributes
- P4: + KB rules
- P5: + LightGBM ranker
- P6: + Question generation

---

## KB 빌드

```bash
# 전체 빌드
python kb/build_scripts/build_all.py

# 개별 빌드
python kb/build_scripts/build_cards_v2.py       # HS4 cards
python kb/build_scripts/build_chunks.py         # Rule chunks
python kb/build_scripts/build_thesaurus.py      # Thesaurus
python kb/build_scripts/build_note_requirements.py  # Tariff notes
```

---

## 모델 학습

### ML Retriever 학습

```python
from src.classifier.retriever import HSRetriever

retriever = HSRetriever()
retriever.train_model(
    train_texts=[...],
    train_labels=[...],
    save_path="artifacts/classifier/"
)
```

### LightGBM Ranker 학습 + Sanity Check

```bash
# 기존 CSV 재사용 (f_lexical 정규화 적용, ~4분)
python scripts/train_ranker_and_sanity_check.py

# 데이터셋 처음부터 재구축 (~30분)
python scripts/train_ranker_and_sanity_check.py --rebuild
```

**출력물** (`artifacts/<timestamp>_ranker_sanity/`):
- `sanity_report.md`: Feature importance + 실험 비교표
- `metrics.json`: 전체 메트릭 JSON
- `model.txt`: LightGBM 모델
- `rank_features_normalized.csv`: 정규화된 학습 데이터

---

## 문서

| 문서 | 설명 |
|------|------|
| **PROJECT_SUMMARY_20260204.md** | 프로젝트 종합 정리 (기능, 모델, 성능, 향후 방향) ⭐ |
| **CHANGELOG.md** | 개발 이력 및 주요 변경사항 ⭐ |
| **METHODOLOGY.md** | 전체 방법론 및 접근법 |
| **MODE_SEPARATION_FIX_REPORT.md** | KB-first 전략 및 모드 검증 |
| **COMPARE_KB_ONLY_VS_HYBRID.md** | 성능 비교 분석 |
| **HEADING_TERMS_INTEGRATION_REPORT.md** | GRI 1 구현 상세 |
| **LEGAL_FEATURE_AUDIT.md** | Legal features 검증 |
| **CALIBRATION_DIAG.md** | Confidence calibration 분석 |

---

## 향후 계획

### 단기 (1-2개월)
1. **ML Retriever Fine-tuning**: HS domain 특화
   - 목표: Top-5 recall 19% → 30%+
2. **Confidence Calibration**: Temperature scaling
   - 목표: ECE 0.77 → 0.3 이하
3. **f_lexical Dominance 해소**: tree invariance 대응
   - feature_interaction_constraints / 2-stage ranker / max_bin 축소
   - 현재: 정규화/파라미터 튜닝 실험 완료 (효과 없음 확인)

### 중기 (3-6개월)
1. HS6 (6-digit) 분류 확장
2. Multi-lingual support (영어, 중국어)
3. Active learning loop
4. LLM integration PoC

### 장기 (6-12개월)
1. Production deployment (FastAPI + React)
2. Regulatory compliance layer (관세율, 수입요건)
3. 1,000 MAU 목표

---

## 기술 스택

- **Python**: 3.9+
- **ML/NLP**: Sentence Transformers, Scikit-learn, LightGBM
- **Data**: Pandas, NumPy, JSON/JSONL
- **Embedding Model**: `jhgan/ko-sroberta-multitask`

---

## 라이선스

이 프로젝트는 연구/교육 목적입니다.

**데이터 출처**:
- 관세청 품목분류 결정사례 (공개 데이터)
- WCO HS Explanatory Notes (공식 해설서)

---

## 기여

Issues 및 Pull Requests 환영합니다.

---

**마지막 업데이트**: 2026-02-08
**프로젝트 상태**: 성능 최적화 단계 (연구 → 프로토타입 → **최적화**)
