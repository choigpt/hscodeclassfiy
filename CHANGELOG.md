# HS Code Classification System - 변경 이력

프로젝트의 주요 개발 이력 및 기능 추가 내역

---

## 2026-02-20: GRI 순차 파이프라인 전면 구현

### 개요
기존 ML+KB 하이브리드 파이프라인을 **GRI 통칙 순차 적용 오케스트레이터**로 전면 교체.
GRI 1→2→3→5→6 순차 적용, Essential Character 모듈, HS6 서브헤딩 분류, 리스크 평가,
판결 정규화를 포함하는 6-Phase 구현.

### Phase 1: 기반 타입 및 데이터 모델

#### 구조화 입력 모델
- `src/classifier/input_model.py` 신규: `ClassificationInput`, `MaterialInfo` 데이터클래스
- 재질 구성비, 세트 여부, 전기 여부, 관할, HS 버전 등 구조화 입력 지원
- `to_enriched_text()` 메서드로 기존 text-only 인터페이스와 호환

#### 타입 확장
- `src/classifier/types.py`: 6개 새 데이터클래스 추가
  - `GRIApplication`: GRI 적용 이력 (gri_id, applied, candidates_before/after)
  - `ECFactor`, `EssentialCharacterResult`: EC 4요소 판정 결과
  - `RiskAssessment`: 리스크 등급 (LOW/MED/HIGH)
  - `RuleReference`: 규칙 참조 (rule_id, source, hs_version)
  - `CaseEvidence`: 판결 증거 (case_id, jurisdiction, decisive_reasoning)
- `Candidate`에 `hs6: str` 필드 추가
- `ClassificationResult`에 `applied_gri`, `essential_character`, `risk`, `rule_references`, `case_evidence` 추가

#### Rule ID 마이그레이션
- `scripts/migrate_rule_ids.py` 신규: 11,912개 규칙에 결정론적 rule_id 부여
  - 포맷: `{hs4}_{chunk_type}_{sha256(text[:100])[:8]}`
  - source 추론 (explanatory_note, include_rule, exclude_rule 등)
  - hs_version: "2022"
- `kb/structured/hs4_rule_chunks_v2.jsonl` 생성
- `src/kb/rules.py`: v2 파일 우선 로드, rule_id/source/hs_version 저장
- `src/classifier/reranker.py`: Evidence.meta에 rule_id 메타데이터 노출

#### 판결 케이스 정규화
- `scripts/normalize_cases.py` 신규: 7,198건 관세청 결정사례 구조화
  - Mode A (규칙 기반): regex로 HS4/6/10, GRI, rejected codes, decisive reasoning 파싱
  - Mode B (하이브리드): Mode A + Ollama LLM 보완 (저확신 케이스)
  - 평균 확신도 0.842, GRI 정보 96.2% 추출 성공
- `data/ruling_cases/normalized_cases_rule.jsonl` 생성
- `data/ruling_cases/normalization_comparison.json` 생성

### Phase 2: GRI 순차 오케스트레이터 (핵심)
- `src/classifier/gri_orchestrator.py` 신규: `GRIOrchestrator` 클래스
  - `classify()`: ML+KB 후보에 GRI 1→2→3→5→6 순차 적용
  - `_apply_gri1()`: LegalGate 기반 hard exclude + redirect
  - `_apply_gri2()`: 미완성(2a) + 혼합물(2b) 처리
  - `_apply_gri3()`: 3(a) specificity → 3(b) Essential Character → 3(c) fallback
  - `_apply_gri5()`: 용기/포장 → 내용물 기준 분류
  - `_apply_gri6()`: HS4→HS6 서브헤딩 해소
  - Rule reference 수집, Case evidence 연결
- `src/classifier/pipeline.py`: 기존 post-merge 로직을 오케스트레이터 호출로 교체
  - `classify_structured(input_data)` 메서드 추가

### Phase 3: Essential Character (GRI 3b)
- `src/classifier/essential_character.py` 신규: `EssentialCharacterModule`
  - 4요소 가중 합산: core_function(0.35) + user_perception(0.25) + area_volume(0.20) + structural(0.20)
  - 후보별 요소 점수 산출 → 가중 합산 → winner 결정

### Phase 4: HS6 서브헤딩 분류 (GRI 6)
- `scripts/build_subheading_kb.py` 신규: HS6 KB 구축
  - 관세청 HS부호 Excel + 판결사례 + 소호주규정 통합
  - 2,822개 HS6 서브헤딩 생성
- `kb/structured/hs6_subheadings.jsonl` 생성
- `src/classifier/subheading_resolver.py` 신규: `SubheadingResolver`
  - 키워드 매칭 + 소호 주규정 + 재질 구성비 + 판결 선례

### Phase 5: 리스크 평가
- `src/classifier/risk_assessor.py` 신규: `RiskAssessor`
  - 5개 요인: 점수차(+3.0), EC적용(+2.0), 정보누락(+1.5/개), 판례충돌(+2.0), 관할분기(+1.0)
  - HIGH (≥5.0) / MED (2.5~5.0) / LOW (<2.5)

### Phase 6: 통합 및 검증
- `configs/stages.yaml`: `gri_orchestrator` 설정 섹션 추가 (EC 가중치, 리스크 임계값, 서브헤딩 경로)
- `src/eval/evaluator.py`: HS6 정확도, GRI 분포, Risk/EC 메트릭 확장
- `src/eval/report.py`: `format_gri_extended()` GRI 확장 통계 리포트 추가
- `scripts/test_gri_pipeline.py` 신규: 10건 end-to-end 통합 테스트
  - 8개 필수 출력 항목 검증, GRI 순차 순서 확인, EC/Risk/HS6 출력 확인
  - **10/10 전건 PASS**

### 파일 추가
- `src/classifier/input_model.py`
- `src/classifier/gri_orchestrator.py`
- `src/classifier/essential_character.py`
- `src/classifier/subheading_resolver.py`
- `src/classifier/risk_assessor.py`
- `scripts/migrate_rule_ids.py`
- `scripts/normalize_cases.py`
- `scripts/build_subheading_kb.py`
- `scripts/test_gri_pipeline.py`

### 파일 수정
- `src/classifier/types.py`: 6개 새 데이터클래스, Candidate.hs6, ClassificationResult 확장
- `src/classifier/pipeline.py`: GRIOrchestrator 통합, classify_structured() 추가
- `src/kb/rules.py`: v2 rule 파일 우선 로드
- `src/classifier/reranker.py`: v2 rule 파일 + Evidence.meta rule_id 노출
- `configs/stages.yaml`: gri_orchestrator 설정 추가
- `src/eval/evaluator.py`: GRI 확장 메트릭
- `src/eval/report.py`: GRI 확장 리포트

### 생성 데이터
- `kb/structured/hs4_rule_chunks_v2.jsonl`: 11,912 rules (rule_id 포함)
- `data/ruling_cases/normalized_cases_rule.jsonl`: 7,198건 정규화
- `data/ruling_cases/normalization_comparison.json`: 정규화 품질 비교
- `kb/structured/hs6_subheadings.jsonl`: 2,822 HS6 서브헤딩

---

## 2026-02-08: Pipeline 성능 개선 + LightGBM Ranker Sanity Check

### 문제
1. **f_lexical dominance**: kb_score(0~39)가 정규화 없이 LightGBM/fallback에 사용 → lexical feature가 gain 86.8% 독점
2. **KB lock 불안정**: rerank pool=5로 좁아 KB top-1이 rerank 후 누락
3. **Fact-insufficient 판정 미작동**: `questions_generated_count` 조건이 항상 0 (dead code)

### 해결

#### [1] f_lexical 정규화 (`reranker.py`)
- `log1p(x)/log1p(30)` + clamp(1.0)으로 [0,1] 정규화
- Fallback weighted-score: max 기여 5.85 → 0.15로 제한
- **LightGBM path**: tree invariance로 gain 불변 (expected) → 구조적 접근 필요

#### [2] KB lock 안정화 (`pipeline.py`)
- Internal rerank topk: 5 → 20 확장
- KB top-1을 넓은 pool에서 탐색 후 lock → 최종 trim to user topk

#### [3] Fact-insufficient 로직 수정 (`pipeline.py`)
- Dead code 제거 (`questions_generated_count`)
- 8-axis 속성 추출을 w_ml 계산 전으로 이동
- 핵심 4축 (material, processing_state, function_use, completeness) 중 2개+ 누락시 fact_insufficient

### Ranker Sanity Check 결과

#### Baseline (LightGBM, 39 features)
| Metric | Train | Test |
|--------|-------|------|
| Top-1 Acc | 0.7907 | 0.7661 |
| Top-5 Acc | 0.9466 | 0.9426 |
| NDCG@5 | 0.8856 | 0.8716 |

#### Dominance 완화 실험
| | Baseline | Exp A (no f_lexical) | Exp B (regularized) |
|---|----------|---------------------|---------------------|
| Test Top-1 | 0.7661 | 0.3894 (-0.38) | 0.7703 (+0.004) |
| Test Top-5 | 0.9426 | 0.7017 (-0.24) | 0.9356 (-0.007) |
| NDCG@5 | 0.8716 | 0.3079 (-0.56) | 0.8691 (-0.003) |
| f_lexical ratio | 86.8% | N/A | 86.3% |

#### 핵심 발견
- f_lexical 제거시 catastrophic drop → 진짜 핵심 정보원
- 정규화/파라미터 튜닝으로 dominance 해소 불가 (tree invariance)
- Train/test leakage: 48건 (3.4%), query_id 분리로 심각도 낮음

### 파일 수정
- `src/classifier/reranker.py`: f_lexical 정규화, fallback score 추가
- `src/classifier/pipeline.py`: internal topk 확장, 8axis 이동, fact-insufficient 수정

### 파일 추가
- `scripts/train_ranker_and_sanity_check.py`: 학습 + sanity check + 실험 스크립트
- `artifacts/20260208_*/sanity_report.md`: Feature importance 리포트 (v2)
- `artifacts/20260208_*/metrics.json`: 실험 메트릭 JSON

### 문서
- `CHANGELOG.md`: 이 항목 추가
- `README.md`: LightGBM Ranker 성능, 프로젝트 구조, 향후 계획 업데이트
- `docs/METHODOLOGY.md`: Ranker 학습 검증 섹션 추가
- `docs/PROJECT_SUMMARY_20260204.md`: 성능, 진행상태, Feature Rebalancing 업데이트
- `artifacts/ranker_legal/ARCHITECTURE.md`: 정규화 및 실험 결과 반영

---

## 2026-02-04: 프로젝트 종합 정리

### 추가
- **PROJECT_SUMMARY_20260204.md**: 프로젝트 전체 종합 문서 (기능, 모델, 성능, 향후 방향)
- **파일 정리**: 불필요한 파일 28개 삭제, 평가 결과 4개만 유지

### 개선
- 프로젝트 구조 최적화
- 문서 통합 (CHANGELOG.md 생성)

---

## 2026-02-03: Hybrid 모드 성능 개선 (KB-first 전략)

### 문제
- Hybrid 모드가 KB-only보다 성능 저하 (7.5% vs 12%)
- KB 정답을 ML이 방해하는 현상 (regressions 11개)

### 해결
- **KB-first merge 전략**: KB 후보 우선, ML은 recall 보강용
- **KB confidence gate**: KB score 기반 lock (threshold: 10.0, margin: 3.0)
- **Conditional ML weight**: 상황별 ML 가중치 동적 조정 (0.05~0.5)

### 성과
- Top-1 Accuracy: 7.5% → **13.5%** (+6.0pp)
- Regressions: 11 → 9 (-18%)
- Improvements: 2 → 12 (+500%)
- **Net gain: -9 → +3** (Hybrid가 KB-only 능가)

### 파일 수정
- `src/classifier/pipeline.py`: KB-first merge, confidence gate, conditional ML weight
- `scripts/analyze_hybrid_regressions.py`: 회귀 분석 스크립트 신규 작성

### 문서
- `docs/MODE_SEPARATION_FIX_REPORT.md`: 모드 분리 검증 및 KB-first 전략

---

## 2026-02-03: 모드 분리 검증 강화

### 문제
- KB-only 모드에서 ML retriever 사용 의심 (오래된 평가 결과 확인)
- config.json에 실제 사용 구성 미기록

### 해결
- `_validate_mode_separation()` 메서드 추가: RuntimeError로 위반 차단
- config.json 개선: `retriever_present`, `ranker_model_loaded`, `heading_terms_len` 추가
- y_true_hs4 alias 추가 (외부 도구 호환성)

### 검증 결과
- KB-only: retriever_used=0%, score_ml=0% ✅
- Hybrid: retriever_used=100%, ranker_applied=100% ✅
- Mode separation 완벽 작동 확인

### 파일 수정
- `src/classifier/eval/run_eval.py`: 강제 검증, config 로깅 개선

---

## 2026-02-03: Evaluation + Usage Audit 패키지 완성

### 추가
- **run_eval.py**: KB-only vs Hybrid 평가 프레임워크
  - 2가지 모드 지원 (kb_only, hybrid)
  - Random seed, limit, split ratio 설정 가능
  - Timestamped output 디렉토리
- **usage_audit.py**: 컴포넌트 사용 추적
  - retriever_used, ranker_applied 기록
  - 3-level ranker 검증 (config, model, actual)
- **metrics.py**: 평가 지표 계산
  - Top-K accuracy, F1, ECE, Brier score
- **report.py**: 결과 리포트 생성

### 파일 추가
- `src/classifier/eval/__init__.py`
- `src/classifier/eval/run_eval.py`
- `src/classifier/eval/usage_audit.py`
- `src/classifier/eval/metrics.py`
- `src/classifier/eval/report.py`

### 문서
- `docs/WORK_LOG_20250203_eval_package.md`

---

## 2026-02-03: Enhanced Diagnostics (Bucket + Confusion + Explanation)

### 추가
- **Bucket Analyzer**: 난이도별 샘플 분류
  - fact_insufficient, legal_conflict, short_text, ambiguous, rare_class
  - Bucket별 성능 분석
- **Confusion Pair Analysis**: Top-20 혼동 쌍 추출
- **Explanation Generator**: Evidence 기반 설명 생성
  - 2-3개 핵심 증거 선택
  - 50자 이내 snippet (저작권 준수)
  - Source reference 제공

### 파일 추가
- `src/classifier/explanation_generator.py`
- `src/experiments/bucket_analyzer.py`
- Enhanced diagnostics in `enhanced_diagnostics.py`

### 문서
- `docs/WORK_LOG_20250203_evaluation_diagnostics_explanation.md`

---

## 2026-02-02: LightGBM Ranker 학습 파이프라인

### 추가
- **Learned Ranker**: LightGBM을 이용한 학습 기반 재순위화
  - Pairwise ranking (LambdaRank)
  - 39개 features (lexical, card, rule, 8-axis, GRI)
  - NDCG@5 최적화
- **Feature Importance**: f_lexical이 dominant (251,890 vs 281)

### 파일 추가
- `src/experiments/train_ranker.py`
- `artifacts/ranker_legal/model_legal.txt`

### 문서
- `docs/WORK_LOG_20250202_learned_reranker.md`

---

## 2026-02-03: Heading Terms Integration (GRI 1 강화)

### 추가
- **LegalGate Heading Term Matching**: 1,239개 HS4 호 용어 매칭
  - Fuzzy matching (SequenceMatcher)
  - Token overlap 기반
  - GRI 1 "호의 용어" 규칙 준수

### 개선
- LegalGate 필터링 정확도 향상
- Heading term 불일치 시 후보 제외 또는 점수 하향

### 파일 수정
- `src/classifier/legal_gate.py`: heading_terms 로드 및 매칭
- `kb/structured/note_requirements.jsonl`: heading 필드 추가

### 문서
- `docs/HEADING_TERMS_INTEGRATION_REPORT.md`

---

## 2026-01-29: Benchmark Framework 구축

### 추가
- **Benchmark Runner**: 4개 모델 자동 평가
  - B0_TFIDF_LR, B1_SBERT_LR, KB-only, Hybrid
  - 8개 metrics 카테고리
- **Ablation Runner**: Pipeline 컴포넌트 단계별 평가
  - P1~P6 progression (ML-only → Full)

### 파일 추가
- `src/experiments/run_benchmark.py`
- `src/experiments/ablation_runner.py`
- `src/experiments/enhanced_evaluator.py`
- `configs/benchmark.yaml`

### 문서
- `docs/WORK_LOG_20250129_benchmark_framework.md`

---

## 초기 구현 (날짜 미상)

### Core Components
- **Pipeline**: 전체 orchestration (`pipeline.py`)
- **ML Retriever**: SBERT + Logistic Regression (`retriever.py`)
- **KB Reranker**: Card/Rule matching (`reranker.py`)
- **GRI Signals**: 5개 법적 해석 규칙 탐지 (`gri_signals.py`)
- **8-Axis Attributes**: 다차원 속성 추출 (`attribute_extract.py`)
- **LegalGate**: Tariff notes 기반 필터링 (`legal_gate.py`)
- **FactChecker**: 정보 충분성 검증 (`fact_checker.py`)
- **Clarifier**: 질문 생성 (`clarify.py`)

### Knowledge Base
- **hs4_cards.jsonl**: 1,240개 HS4 카드
- **hs4_rule_chunks.jsonl**: 11,912개 규칙 청크
- **thesaurus_terms.jsonl**: 7,098개 용어
- **note_requirements.jsonl**: Tariff notes 구조화

### Training Data
- **all_cases_full_v7.json**: 7,198개 관세청 품목분류 결정사례

### 문서
- `docs/METHODOLOGY.md`: 전체 방법론
- `docs/FINAL_PORTFOLIO_REPORT.md`: 기능 요약
- `README.md`: 프로젝트 개요

---

## 문서 히스토리

| 문서 | 작성일 | 설명 |
|------|--------|------|
| sanity_report.md (v2) | 2026-02-08 | LightGBM Ranker feature importance + 실험 비교표 |
| PROJECT_SUMMARY_20260204.md | 2026-02-04 | 프로젝트 종합 정리 (기능, 모델, 성능, 방향) |
| MODE_SEPARATION_FIX_REPORT.md | 2026-02-03 | 모드 분리 검증 및 KB-first 전략 |
| COMPARE_KB_ONLY_VS_HYBRID.md | 2026-02-03 | KB-only vs Hybrid 성능 비교 |
| EVAL_VALIDATION_REPORT.md | 2026-02-03 | 평가 프레임워크 검증 |
| HEADING_TERMS_INTEGRATION_REPORT.md | 2026-02-03 | Heading terms 통합 |
| LEGAL_FEATURE_AUDIT.md | 2026-02-03 | Legal feature 추출 검증 |
| CALIBRATION_DIAG.md | 2026-02-03 | Confidence calibration 분석 |
| WORK_LOG_20250203_eval_package.md | 2026-02-03 | 평가 패키지 구현 |
| WORK_LOG_20250203_evaluation_diagnostics_explanation.md | 2026-02-03 | Diagnostics 시스템 |
| WORK_LOG_20250202_learned_reranker.md | 2026-02-02 | LightGBM ranker 학습 |
| WORK_LOG_20250129_benchmark_framework.md | 2026-01-29 | 벤치마크 프레임워크 |
| METHODOLOGY.md | - | 전체 방법론 |
| FINAL_PORTFOLIO_REPORT.md | - | 기능 요약 |

---

**마지막 업데이트**: 2026-02-20
