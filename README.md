# HS Code Classification System

한국 관세청 HS Code (품목분류번호) 자동 분류 시스템

**GRI 순차 적용** + **법적 근거 기반** + **설명 가능한** + **능동적 질의**를 특징으로 하는 하이브리드 ML/KB 시스템

---

## 주요 특징

- ✅ **GRI 순차 파이프라인**: GRI 1→2→3→5→6 순차 적용 오케스트레이터
- ✅ **Essential Character (GRI 3b)**: 4요소 가중 합산 모델 (기능핵심/인식중심/면적부피/구조지배)
- ✅ **HS6 서브헤딩 분류 (GRI 6)**: HS4 확정 후 HS6 자동 해소 (2,822개 서브헤딩)
- ✅ **리스크 평가**: 5개 요인 기반 LOW/MED/HIGH 등급 (점수차/EC적용/정보누락/판례충돌/관할분기)
- ✅ **판결 정규화**: 7,198건 관세청 결정사례 구조화 (GRI/HS6/rejected codes/decisive reasoning)
- ✅ **8-Axis 속성**: 재질, 용도, 가공상태 등 다차원 의미 분석
- ✅ **법적 필터링**: Tariff Notes 기반 LegalGate (GRI 1)
- ✅ **하이브리드 접근**: ML Retriever + KB Reranker + LightGBM Ranker
- ✅ **설명 가능성**: 모든 예측에 Evidence + Rule Reference + Case Evidence 제공
- ✅ **질문 생성**: Low confidence 시 자동 질문 (최대 3개)
- ✅ **구조화 입력**: ClassificationInput (재질 구성비, 세트 여부, 전기 여부 등)

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

### 2. KB 데이터 준비

```bash
# Rule ID 마이그레이션 (v1→v2, rule_id/source/hs_version 부여)
python scripts/migrate_rule_ids.py

# 판결 케이스 정규화
python scripts/normalize_cases.py --mode rule

# HS6 서브헤딩 KB 구축
python scripts/build_subheading_kb.py
```

### 3. 단일 예측 (text-only)

```python
from src.classifier.pipeline import HSPipeline

pipeline = HSPipeline(use_gri=True, use_legal_gate=True, use_8axis=True)
result = pipeline.classify("냉동 돼지 삼겹살")

print(f"Top1: {result.topk[0].hs4} (HS6: {result.topk[0].hs6})")
print(f"Decision: {result.decision.status}")
print(f"GRI 적용: {[g.gri_id for g in result.applied_gri if g.applied]}")
print(f"Risk: {result.risk.level}")
```

### 4. 구조화 입력 예측

```python
from src.classifier.pipeline import HSPipeline
from src.classifier.input_model import ClassificationInput, MaterialInfo

pipeline = HSPipeline(use_gri=True, use_legal_gate=True, use_8axis=True)
input_data = ClassificationInput(
    text="면 60% 폴리에스터 40% 혼방 직물",
    materials=[
        MaterialInfo(name="면", ratio=0.6),
        MaterialInfo(name="폴리에스터", ratio=0.4),
    ],
)
result = pipeline.classify_structured(input_data)
```

### 5. GRI 파이프라인 통합 테스트

```bash
python scripts/test_gri_pipeline.py
```

### 6. 평가 실행

```bash
python scripts/eval_cascade.py
```

---

## 프로젝트 구조

```
HS/
├── src/
│   ├── classifier/                    # 핵심 파이프라인
│   │   ├── pipeline.py                # 전체 Orchestration
│   │   ├── gri_orchestrator.py        # GRI 순차 오케스트레이터 (GRI 1→2→3→5→6)
│   │   ├── essential_character.py     # GRI 3(b) Essential Character 4요소 모델
│   │   ├── subheading_resolver.py     # GRI 6 HS4→HS6 서브헤딩 해소
│   │   ├── risk_assessor.py           # 리스크 평가 (LOW/MED/HIGH)
│   │   ├── input_model.py             # 구조화 입력 (ClassificationInput)
│   │   ├── retriever.py               # ML Retriever (SBERT + LR)
│   │   ├── reranker.py                # KB Reranker (Card/Rule + rule_id 추적)
│   │   ├── gri_signals.py             # GRI 신호 탐지
│   │   ├── attribute_extract.py       # 8-Axis 속성 추출
│   │   ├── legal_gate.py              # Tariff Notes 필터링 (GRI 1)
│   │   ├── fact_checker.py            # 정보 충분성 검증
│   │   ├── clarify.py                 # 질문 생성
│   │   └── explanation_generator.py   # 설명 생성
│   │
│   ├── eval/                          # 평가 프레임워크 (Stage 비교)
│   │   ├── evaluator.py               # GRI 확장 메트릭 포함
│   │   └── report.py                  # ASCII + GRI 통계 리포트
│   │
│   ├── cascade/                       # Cascade (Hybrid+RAG)
│   ├── stages/                        # Stage 어댑터
│   ├── types.py                       # 공통 타입 (BaseClassifier, StageResult)
│   ├── data.py                        # Sample 데이터 로더
│   ├── metrics.py                     # 평가 지표 계산
│   └── text.py                        # 텍스트 전처리
│
├── data/
│   ├── ruling_cases/
│   │   ├── all_cases_full_v7.json     # 7,198 cases (원본)
│   │   ├── normalized_cases_rule.jsonl # 정규화된 케이스 (규칙 기반)
│   │   └── normalization_comparison.json # 정규화 품질 비교
│   ├── benchmarks/                    # Train/Val/Test splits
│   └── tariff_notes_clean.json        # Tariff notes
│
├── kb/
│   ├── structured/
│   │   ├── hs4_cards_v2.jsonl         # 1,240 HS4 cards
│   │   ├── hs4_rule_chunks.jsonl      # 11,912 rule chunks (v1)
│   │   ├── hs4_rule_chunks_v2.jsonl   # 11,912 rule chunks (rule_id 포함)
│   │   ├── hs6_subheadings.jsonl      # 2,822 HS6 서브헤딩 KB
│   │   ├── thesaurus_terms.jsonl      # 7,098 terms
│   │   └── note_requirements.jsonl    # Legal notes
│   └── build_scripts/                 # KB 빌드 도구
│
├── artifacts/
│   ├── classifier/                    # ML 모델
│   ├── ranker_legal/                  # LightGBM 모델
│   └── eval/                          # 평가 결과
│
├── scripts/
│   ├── migrate_rule_ids.py            # Rule ID 마이그레이션 (v1→v2)
│   ├── normalize_cases.py             # 판결 케이스 정규화 (규칙/LLM)
│   ├── build_subheading_kb.py         # HS6 서브헤딩 KB 구축
│   ├── test_gri_pipeline.py           # GRI 파이프라인 통합 테스트
│   ├── eval_cascade.py                # Cascade 평가
│   ├── train_ml.py                    # ML 학습
│   ├── train_ranker.py                # Ranker 학습
│   └── ...
│
├── configs/
│   └── stages.yaml                    # Stage + GRI 오케스트레이터 설정
│
├── CHANGELOG.md                       # 변경 이력
├── README.md                          # 이 파일
└── requirements.txt
```

---

## 파이프라인 흐름

```
Input (품목 설명 or ClassificationInput)
    ↓
[Step 0] GRI Signals + 8-Axis Attributes 추출
    ↓
[Step 1] ML Retriever → Top-50 candidates
    ↓
[Step 2] KB Retrieval → Top-30 candidates
    ↓
[Step 3] Merge (KB-first + ML recall)
    ↓
╔══════════════════════════════════════════╗
║   GRI 순차 오케스트레이터               ║
╠══════════════════════════════════════════╣
║ [GRI 1] LegalGate → Hard exclude/redirect ║
║    ↓                                     ║
║ [GRI 2] 미완성(2a) + 혼합물(2b) 처리     ║
║    ↓                                     ║
║ [GRI 3] 복수 후보 해소                    ║
║    3(a) 구체적 호 우선                    ║
║    3(b) Essential Character (4요소)       ║
║    3(c) 수 순서 최말위 (fallback)         ║
║    ↓                                     ║
║ [GRI 5] 용기/포장 처리                    ║
║    ↓                                     ║
║ [GRI 6] HS4→HS6 서브헤딩 분류            ║
║    ↓                                     ║
║ [Risk] 리스크 평가 (LOW/MED/HIGH)        ║
╚══════════════════════════════════════════╝
    ↓
[Step 4] Confidence Check + Question Generation
    ↓
Output: Top-5 + HS6 + Decision + GRI 적용 이력
        + Essential Character + Risk + Rule References
        + Case Evidence + Questions (if needed)
```

---

## 주요 컴포넌트

### 1. GRI 순차 오케스트레이터 (`gri_orchestrator.py`)
ML+KB 후보 생성 후 GRI 통칙을 순차적으로 적용:
- **GRI 1**: LegalGate를 통한 호 용어/주규정 기반 hard exclude + redirect
- **GRI 2a**: 미완성/미조립 → 완성품 호에 분류 가능 확장
- **GRI 2b**: 혼합물 → 구성 재질별 후보 확장
- **GRI 3(a)**: 가장 구체적 호 우선 (specificity scoring)
- **GRI 3(b)**: Essential Character 4요소 모델
- **GRI 3(c)**: 수 순서 최말위 (fallback)
- **GRI 5**: 용기/포장 → 내용물 기준 분류
- **GRI 6**: HS4→HS6 서브헤딩 세분화

### 2. Essential Character (`essential_character.py`)
GRI 3(b) 적용 시 4요소 가중 합산:

| 요소 | 가중치 | 판단 기준 |
|------|--------|----------|
| core_function | 0.35 | 기능/용도 축 + 카드 키워드 매칭 |
| user_perception | 0.25 | 물체본질 축 + heading title 매칭 |
| area_volume | 0.20 | 재질 축 + 구성비 |
| structural | 0.20 | 완성도 축 + 카드 속성 |

### 3. HS6 서브헤딩 분류 (`subheading_resolver.py`)
HS4 확정 후 2,822개 HS6 후보에서 최적 서브헤딩 결정:
- 키워드 매칭 + 소호 주규정 적용
- 재질 구성비 기반 매칭
- 판결 선례 참조

### 4. 리스크 평가 (`risk_assessor.py`)

| 요인 | 조건 | 점수 |
|------|------|------|
| 점수 차이 | Top1-Top2 gap < 0.15 | +3.0 |
| GRI 3(b) | Essential Character 적용됨 | +2.0 |
| 정보 누락 | 핵심 axis 2개+ 미감지 | +1.5/개 |
| 판례 충돌 | 동일 상품 다른 코드 판결 존재 | +2.0 |
| 관할 분기 | 다른 관할 다른 분류 가능성 | +1.0 |

레벨: **HIGH** (≥5.0) / **MED** (2.5~5.0) / **LOW** (<2.5)

### 5. 8-Axis Attributes
1. object_nature (물체 본질)
2. material (재질/성분)
3. processing_state (가공상태)
4. function_use (기능/용도)
5. physical_form (물리적 형태)
6. completeness (완성도)
7. quantitative_rules (정량규칙)
8. legal_scope (법적범위)

### 6. LegalGate (GRI 1 Filter)
- Heading term 매칭 (1,239개 HS4)
- Include rules: 긍정 증거
- Exclude rules: 하드 필터 (위반 시 후보 제거)
- Redirect rules: 올바른 HS4로 안내

### 7. Confidence Routing
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
3. **GRI 파이프라인 정확도 최적화**: EC 가중치 튜닝, HS6 매칭 정교화

### 중기 (3-6개월)
1. Multi-lingual support (영어, 중국어)
2. Active learning loop
3. LLM 기반 판결 정규화 (Mode B) 품질 개선
4. HS10 (10-digit) 분류 확장

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

**마지막 업데이트**: 2026-02-20
**프로젝트 상태**: GRI 순차 파이프라인 구현 완료 (연구 → 프로토타입 → 최적화 → **GRI 통합**)
