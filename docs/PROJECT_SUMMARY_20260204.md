# HS Code Classification System - 종합 프로젝트 정리

**작성일**: 2026-02-04
**프로젝트**: 한국 관세청 HS Code 자동 분류 시스템
**코드베이스**: 15,889 lines (Python)

---

## 1. 프로젝트 개요

### 1.1 목적
- 한국 수출입 품목의 HS Code (품목분류번호) 자동 분류
- 법적 근거 기반 투명한 분류 (설명 가능성)
- 불확실한 경우 질문 생성으로 정확도 향상
- 관세사/통관사의 업무 효율화

### 1.2 핵심 가치
1. **법적 정합성**: 관세법 GRI (General Rules of Interpretation) 준수
2. **설명 가능성**: 모든 분류 결과에 근거(evidence) 제공
3. **능동적 질의**: Low confidence 시 질문 생성으로 정보 보완
4. **하이브리드 접근**: ML + KB (Knowledge Base) 결합

### 1.3 데이터
- **학습 데이터**: 7,198개 관세청 품목분류 결정사례
- **HS4 커버리지**: 1,240개 호(4단위 코드)
- **KB 소스**: WCO HS Explanatory Notes (한글판)

---

## 2. 시스템 아키텍처

### 2.1 파이프라인 구조

```
Input Text (품목 설명)
    ↓
[Step 0] GRI Signals + 8-Axis Attributes 추출
    ↓
[Step 1] ML Retriever → Top-50 candidates
    ↓
[Step 2] KB Retrieval → Top-30 candidates (GRI 기반 조정)
    ↓
[Step 3] Merge (KB-first + ML recall) → Union
    ↓
[Step 3.5] LegalGate (GRI 1 적용) → Hard filtering
    ↓
[Step 4] Reranking (Card/Rule + 8-Axis + LightGBM)
    ↓
[Step 5] Confidence Check
    ↓
[Step 6] Question Generation (if needed)
    ↓
Output: Top-5 + Decision (AUTO/ASK) + Questions
```

### 2.2 핵심 컴포넌트

| 컴포넌트 | 역할 | 입력 | 출력 |
|---------|------|------|------|
| **GRI Detector** | 법적 해석 규칙 신호 탐지 | Text | GRI 1/2a/2b/3/5 flags |
| **8-Axis Extractor** | 속성 추출 (재질, 용도 등) | Text | 8개 축별 속성 리스트 |
| **ML Retriever** | 의미 기반 후보 생성 | Text embedding | Top-50 candidates + ML scores |
| **KB Reranker** | KB 규칙 기반 재순위화 | Text + Candidates | Card/Rule scores |
| **LegalGate** | 법적 제약 필터링 (GRI 1) | Candidates + Notes | Filtered candidates |
| **LightGBM Ranker** | 학습 기반 최종 순위화 | Features → Ranking | Reranked Top-K |
| **Clarifier** | 질문 생성 | Low confidence result | 2-3 clarification questions |
| **FactChecker** | 정보 충분성 검증 | Attributes + HS4 | Missing facts |
| **Explainer** | 결과 설명 생성 | Evidence list | User-facing explanation |

### 2.3 레이어 구조

```
┌─────────────────────────────────────────────┐
│  Application Layer (API/UI)                 │
├─────────────────────────────────────────────┤
│  Pipeline Orchestration (pipeline.py)       │
├─────────────────────────────────────────────┤
│  Decision Layer                             │
│  - Confidence Evaluation                    │
│  - AUTO/ASK Routing                         │
│  - Question Generation                      │
├─────────────────────────────────────────────┤
│  Scoring Layer                              │
│  - ML Retriever (Semantic)                  │
│  - KB Reranker (Card/Rule)                  │
│  - LightGBM Ranker (Learned)                │
├─────────────────────────────────────────────┤
│  Feature Extraction Layer                   │
│  - GRI Signals                              │
│  - 8-Axis Attributes                        │
│  - Text Normalization                       │
├─────────────────────────────────────────────┤
│  Legal Constraint Layer                     │
│  - LegalGate (Tariff Notes)                 │
│  - FactChecker (Required Facts)             │
├─────────────────────────────────────────────┤
│  Knowledge Base Layer                       │
│  - HS4 Cards (1,240)                        │
│  - Rule Chunks (11,912)                     │
│  - Thesaurus (7,098 terms)                  │
│  - Tariff Notes                             │
└─────────────────────────────────────────────┘
```

---

## 3. 주요 기능

### 3.1 GRI (General Rules of Interpretation) 지원

| GRI | 의미 | 구현 |
|-----|------|------|
| **GRI 1** | 호의 용어, 주 규정 우선 | LegalGate: heading term 매칭 + note 기반 hard filter |
| **GRI 2a** | 미조립/분해 물품 | 완성품 후보 확대 (+20 candidates) |
| **GRI 2b** | 혼합물/합금 | 재질 후보 확대 (+10 candidates) |
| **GRI 3** | 세트/복합 물품 | Set 분석 신호 전달 |
| **GRI 5** | 포장 용기 | Container 로직 활성화 |

**효과**: GRI 신호 기반으로 후보 생성 전략을 동적 조정 → 법적 해석 규칙 준수

### 3.2 8-Axis 속성 프레임워크

기존 단순 키워드 매칭을 넘어 다차원 의미 속성 추출:

| 축 | 예시 속성 | 활용 |
|---|---------|------|
| 1. object_nature | substance, product, machine, food | 물체 본질 파악 |
| 2. material | metal, plastic, wood, textile | 재질 기반 분류 |
| 3. processing_state | fresh, frozen, dried, cooked | 가공 상태 구분 |
| 4. function_use | industrial, household, medical | 용도 기반 분류 |
| 5. physical_form | powder, liquid, solid, sheet | 형태 구분 |
| 6. completeness | finished, incomplete, parts | 완성도 (GRI 2a 연계) |
| 7. quantitative_rules | 50% 이상, 순도 95% | 정량 조건 |
| 8. legal_scope | GRI notes, includes, excludes | 법적 범위 |

**효과**: 다차원 속성으로 candidate 신뢰도 향상 + 질문 생성 정교화

### 3.3 LegalGate (법적 제약 필터)

**동작**:
1. Tariff Notes에서 include/exclude/redirect 규칙 로드
2. Input text에서 heading term 매칭 검증
3. Exclude 규칙 위반 시 후보 제거 (hard filter)
4. Include 규칙 매칭 시 증거 추가 (positive evidence)

**예시**:
- Input: "플라스틱 장난감 자동차"
- HS 9503 note: "장난감이어야 함"
- HS 8703 note: "실제 작동 차량만 해당"
- → LegalGate가 8703 제거, 9503 유지

**효과**: 법적 모순 방지, 분류 정확도 향상

### 3.4 Fact Sufficiency Checker

**역할**: 분류에 필요한 정보가 충분한지 검증

**프로세스**:
1. HS4별 required_facts 로드 (hs4_cards_v2.jsonl)
2. Input에서 추출된 8-axis 속성과 비교
3. Hard missing facts 발견 시 → ASK 결정
4. 질문 생성기에 missing fact 전달

**예시**:
- HS 0201 (냉장 소고기) requires: `processing_state=chilled`
- Input: "소고기" (processing_state 없음)
- → Missing fact: "냉장? 냉동?"
- → Question: "이 소고기는 냉장 상태인가요, 냉동 상태인가요?"

**효과**: 불충분한 정보로 인한 오분류 방지

### 3.5 Adaptive Confidence Routing

**Decision Status**:
- **AUTO**: High confidence (Top1 vs Top2 margin > threshold)
- **ASK**: Low confidence → 2-3개 질문 생성
- **REVIEW**: Legal conflict 또는 fact insufficient
- **ABSTAIN**: 분류 불가 (KB 범위 밖)

**Threshold**:
- Default: Top1 - Top2 score > 0.3
- LegalGate 단독 후보: AUTO (confidence=0.9)
- Fact missing: ASK (confidence=0.0)

**효과**: 오분류 리스크 감소, 사용자 신뢰도 향상

### 3.6 Context-Aware Question Generation

**질문 전략**:
1. **GRI 기반**: GRI 2a 활성화 → "완성품인가요?"
2. **Attribute 기반**: 재질 불명 → "주 재질이 무엇인가요?"
3. **Top candidate 기반**: Top2가 비슷 → "용도는 A인가요 B인가요?"
4. **Missing fact 기반**: 필수 정보 누락 → 직접 질문

**제약**:
- 최대 3개 질문 (사용자 피로도 고려)
- 중복 제거 (같은 내용 다른 표현 제거)

**효과**: 사용자 부담 최소화하며 정보 보완

### 3.7 Evidence-Based Explanation

**원칙**:
1. 파이프라인 결과 변경 금지 (read-only)
2. 핵심 증거 2-3개만 선택 (사용자 친화)
3. Snippet 50자 이내 (저작권 준수)
4. Source reference 제공 (추적 가능)

**Evidence 종류**:
- `kb_retrieval`: KB 매칭 점수
- `card_keyword`: 카드 키워드 매칭
- `rule_include`: Include 규칙 매칭
- `rule_exclude`: Exclude 규칙 저촉
- `legal_gate_pass`: LegalGate 통과
- `legal_heading_term`: Heading term 매칭
- `8axis_match`: 8축 속성 일치

**효과**: 투명성, 신뢰성, 법적 근거 제공

---

## 4. 모델 성능

### 4.1 평가 데이터셋

- **Total**: 7,198 ruling cases
- **Split**: Train 70% (5,758) / Val 15% (719) / Test 15% (200)
- **Random seed**: 42 (재현성 보장)
- **Evaluation**: Test set (200 samples)

### 4.2 모델 비교 (Latest Results)

#### **KB-only 모드** (artifacts/eval/kb_only_20260203_214958)

| Metric | Value | 설명 |
|--------|-------|------|
| Top-1 Accuracy | **12.0%** | 정답이 1순위 |
| Top-3 Accuracy | 23.5% | 정답이 3순위 이내 |
| Top-5 Accuracy | 35.5% | 정답이 5순위 이내 |
| Candidate Recall@5 | 35.5% | 후보 생성 recall |
| ECE | 0.78 | Calibration error (높을수록 나쁨) |
| AUTO Rate | 49.5% | 자동 분류 비율 |
| ASK Rate | 50.5% | 질문 생성 비율 |

**특징**:
- ML 모델 불사용 (retriever=None, ranker=None)
- 순수 KB 기반 (Card + Rule + LegalGate)
- 설명 가능성 100%
- 추론 속도 빠름 (평균 0.44초/sample)

#### **Hybrid 모드 (Before KB-first)** (artifacts/eval/hybrid_20260203_214335)

| Metric | Value | 변화 |
|--------|-------|------|
| Top-1 Accuracy | **7.5%** | ↓ 4.5pp (KB-only 대비) |
| Top-3 Accuracy | 14.0% | ↓ 9.5pp |
| Top-5 Accuracy | 19.0% | ↓ 16.5pp |
| ECE | 0.83 | ↑ 0.05 (악화) |

**문제**:
- ML retriever가 KB 정답을 **방해**
- Regressions (KB✓ → HY✗): **11개**
- Improvements (KB✗ → HY✓): **2개**
- **Net gain: -9 samples** ❌

#### **Hybrid 모드 (After KB-first)** (artifacts/eval/hybrid_20260203_220018)

| Metric | Value | 변화 (vs KB-only) | 변화 (vs Before) |
|--------|-------|------------------|------------------|
| Top-1 Accuracy | **13.5%** | ↑ 1.5pp ✅ | ↑ 6.0pp ✅ |
| Top-3 Accuracy | 16.5% | ↓ 7.0pp | ↑ 2.5pp |
| Top-5 Accuracy | 19.0% | ↓ 16.5pp | → |
| ECE | 0.77 | ↓ 0.01 (개선) | ↓ 0.06 (개선) |
| AUTO Rate | 46.5% | ↓ 3.0pp | - |

**개선**:
- KB-first merge 전략 적용
- KB confidence gate (KB score 기반 lock)
- Conditional ML weight (0.05~0.5 동적 조정)
- Regressions: **9개** (11→9, -18%)
- Improvements: **12개** (2→12, +500%)
- **Net gain: +3 samples** ✅

**KB Lock 통계**:
- KB locked 조건 충족: 54/200 (27%)
- KB lock 실제 적용: 18/200 (9%)
- Top1 source: Ranker 182, KB locked 18

### 4.3 성능 분석

#### **강점**
1. **Top-1 정확도 향상**: KB-only 12% → Hybrid 13.5%
2. **KB 정답 보호**: KB confidence gate가 고신뢰 예측 보호
3. **ML recall 보완**: KB에 없는 후보를 ML이 추가 (12 improvements)

#### **약점**
1. **Top-3/5 recall 저하**: Hybrid의 Top-5가 KB-only보다 낮음 (19% vs 35.5%)
   - 원인: ML retriever 품질 부족, KB 후보 희석
2. **ECE 높음** (0.77~0.83): Confidence calibration 부족
   - 모델이 자신감을 잘못 측정
3. **절대 정확도 낮음**: 13.5%는 여전히 개선 필요

#### **주요 오류 패턴** (Confusion Pairs)
| True HS4 | Pred HS4 | 빈도 | 원인 추정 |
|----------|----------|------|----------|
| 1704 (설탕과자) | 2822 (강철 제품) | 2 | 재질 혼동 |
| 4803 (화장지) | 4818 (식탁용품) | 2 | 용도 유사 |
| 3909 (아미노수지) | 3908 (폴리아미드) | 1 | 화학 물질 세분화 |

---

## 5. 현재 구현 상태

### 5.1 완료된 컴포넌트 ✅

| 컴포넌트 | 상태 | 파일 | 특징 |
|---------|------|------|------|
| GRI Detector | ✅ 완료 | `gri_signals.py` | 5개 GRI 신호 탐지 |
| 8-Axis Extractor | ✅ 완료 | `attribute_extract.py` | 8축 속성 프레임워크 |
| ML Retriever | ✅ 완료 | `retriever.py` | SBERT + LR |
| KB Reranker | ✅ 완료 | `reranker.py` | Card/Rule 매칭 |
| LegalGate | ✅ 완료 | `legal_gate.py` | GRI 1 필터링 |
| FactChecker | ✅ 완료 | `fact_checker.py` | Required facts 검증 |
| Clarifier | ✅ 완료 | `clarify.py` | 질문 생성 |
| Explainer | ✅ 완료 | `explanation_generator.py` | Evidence 기반 설명 |
| LightGBM Ranker | ✅ 완료 | Training pipeline | 학습 기반 재순위화 |
| Evaluation Framework | ✅ 완료 | `src/classifier/eval/` | KB-only vs Hybrid |
| Regression Analyzer | ✅ 완료 | `analyze_hybrid_regressions.py` | 성능 변화 추적 |
| Mode Separation Validator | ✅ 완료 | `run_eval.py` | KB-only 순수성 검증 |
| Enhanced Diagnostics | ✅ 완료 | `src/experiments/` | Bucket/Confusion 분석 |

### 5.2 진행 중 / 개선 필요 🔧

| 항목 | 상태 | 우선순위 |
|------|------|----------|
| ML Retriever Fine-tuning | 🔧 필요 | **High** |
| Confidence Calibration | 🔧 필요 | **High** |
| f_lexical Dominance 구조적 해소 | 🔧 실험 완료, 구조적 접근 필요 | **High** |
| Top-5 Recall 개선 | 🔧 필요 | Medium |
| ~~Feature Scaling 조정~~ | ✅ 완료 (2026-02-08) | ~~Medium~~ |
| API 서버 구현 | ⏸️ 대기 | Low |
| UI/UX 개발 | ⏸️ 대기 | Low |

---

## 6. 향후 개선 방향

### 6.1 단기 목표 (1-2개월)

#### **Priority 1: ML Retriever 품질 개선** 🔥

**현재 문제**:
- Top-5 recall 19% (KB-only 35.5% 대비 저조)
- ML candidates가 KB candidates를 희석

**해결 방안**:
1. **Domain-specific Fine-tuning**
   - 현재: Generic Korean SBERT (`jhgan/ko-sroberta-multitask`)
   - 개선: HS code 도메인 데이터로 fine-tuning
   - 방법: Contrastive learning (positive: 같은 HS4, negative: 다른 HS4)

2. **Hard Negative Mining**
   - Confusion pairs를 hard negatives로 사용
   - 예: 1704 vs 2822 (자주 혼동) → 구분 학습 강화

3. **Better Embedding Model**
   - 최신 한국어 모델 시도:
     - `klue/roberta-large`
     - `team-lucid/deberta-v3-large-korean`
   - Multilingual 모델: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

4. **Ensemble Retrieval**
   - BM25 (lexical) + SBERT (semantic) 결합
   - Reciprocal Rank Fusion (RRF)

**예상 효과**: Top-5 recall 19% → 30%+

#### **Priority 2: Confidence Calibration** 🔥

**현재 문제**:
- ECE 0.77~0.83 (높을수록 나쁨, 0이 이상적)
- 모델이 자신감을 과대/과소평가
- AUTO/ASK routing 신뢰도 저하

**해결 방안**:
1. **Temperature Scaling**
   - Logits에 temperature 파라미터 적용
   - Validation set에서 optimal temperature 탐색
   - 구현 간단, 효과적

2. **Isotonic Regression**
   - 예측 확률을 실제 정확도로 매핑
   - Sklearn 내장 함수 사용 가능

3. **Platt Scaling**
   - Logistic regression으로 calibration
   - Binary → multi-class 확장

4. **Ensemble Calibration**
   - Multiple models 평균으로 uncertainty 감소

**예상 효과**: ECE 0.77 → 0.3 이하

#### **Priority 3: Feature Importance Re-balancing** (2026-02-08 실험 완료)

**현재 문제**:
- `f_lexical` gain: 251,890 (86.8%) — 다른 38개 피처 합계보다 6.5배
- Tree-based 모델은 monotonic transform에 불변 → log1p 정규화 무효

**실험 결과** (2026-02-08):
| 실험 | Test Top-1 | NDCG@5 | f_lexical ratio |
|------|-----------|--------|-----------------|
| Baseline | 0.7661 | 0.8716 | 86.8% |
| Exp A: f_lexical 제거 | 0.3894 (-0.38) | 0.3079 (-0.56) | N/A |
| Exp B: regularized (ff=0.7, md=6, mgs=0.5) | 0.7703 (+0.004) | 0.8691 (-0.003) | 86.3% |

**결론**:
- f_lexical은 핵심 정보원 (제거시 catastrophic drop)
- 정규화/파라미터 튜닝으로 dominance 해소 불가 (tree invariance)
- Fallback weighted-score 경로는 정규화로 정상 수정됨 (max 기여 5.85 → 0.15)

**남은 해결 방안** (구조적 접근):
1. **feature_interaction_constraints**: f_lexical 독립 그룹 분리
2. **max_bin 축소** (f_lexical 전용): 분할 해상도 제한
3. **2-stage ranker**: f_lexical 없이 1차 랭킹 → f_lexical로 보정
4. **feature_fraction_bynode**: 노드 단위 피처 샘플링

**예상 효과**: LegalGate 효과 증대, 법적 정합성 향상

### 6.2 중기 목표 (3-6개월)

#### **1. 6-Digit (HS6) 분류 확장**

**현재**: HS4 (4-digit) 분류만 지원
**목표**: HS6 (6-digit) 세분류까지 지원

**방법**:
- Hierarchical classification: HS4 → HS6
- Two-stage pipeline:
  1. HS4 분류 (현재 시스템)
  2. HS6 세분화 (같은 HS4 내 후보만 비교)

**데이터 요구사항**: HS6 레벨 ruling cases 필요

#### **2. Multi-lingual Support**

**현재**: 한국어 전용
**목표**: 영어, 중국어 지원

**방법**:
- Multilingual embedding model
- Language detection → 언어별 KB 매핑
- GRI 용어 다국어 사전

#### **3. Active Learning Loop**

**목표**: 사용자 피드백으로 지속 개선

**프로세스**:
1. 사용자가 예측 결과에 대해 정답 제공
2. Incorrect predictions를 training data에 추가
3. 주기적 모델 재학습 (monthly)
4. A/B testing으로 성능 검증

#### **4. Explainability 강화**

**현재**: Evidence 리스트 제공
**목표**: Natural language 설명 생성

**방법**:
- Template-based: "이 물품은 {재질}로 만들어졌고, {용도}로 사용되므로 HS {code}로 분류됩니다."
- LLM-based: GPT-4 등으로 설명문 생성 (evidence를 context로 제공)

### 6.3 장기 목표 (6-12개월)

#### **1. End-to-End LLM Integration**

**접근법 A: LLM as Retriever**
- GPT-4, Claude 등을 zero-shot/few-shot retriever로 사용
- Prompt: "다음 물품의 HS code는 무엇입니까? {text}"
- 장점: 별도 학습 불필요
- 단점: API 비용, 속도, 법적 근거 부족

**접근법 B: LLM as Reranker**
- KB/ML candidates를 LLM이 재순위화
- Prompt에 tariff notes, GRI rules 제공
- 장점: 법적 추론 능력 활용
- 단점: Latency, cost

**접근법 C: Hybrid (현재 시스템 + LLM)**
- 현재 pipeline으로 Top-10 생성
- LLM이 최종 순위화 + 설명 생성
- 최적 균형점

#### **2. Regulatory Compliance Layer**

**목표**: 분류뿐 아니라 관세율, 수입요건까지 제공

**확장**:
- HS code → 관세율 조회 (관세청 DB 연동)
- HS code → 수입요건 (검역, 허가 등)
- Total landed cost 계산기

#### **3. Production Deployment**

**Infrastructure**:
- FastAPI backend
- React/Vue frontend
- PostgreSQL (사용자 데이터, 피드백 저장)
- Redis (caching)
- Docker + Kubernetes

**Scale**:
- 1,000 req/sec 처리 목표
- 평균 응답 시간 < 1초
- 99.9% uptime

---

## 7. 기술 스택

### 7.1 Core ML/NLP

| 기술 | 용도 | 버전 |
|------|------|------|
| **Python** | 언어 | 3.9+ |
| **Sentence Transformers** | Embedding | 2.2+ |
| **Scikit-learn** | LR, metrics | 1.3+ |
| **LightGBM** | Ranking | 4.0+ |
| **spaCy** | (Optional) NLP | 3.5+ |

### 7.2 Data Processing

| 기술 | 용도 |
|------|------|
| **Pandas** | DataFrame |
| **NumPy** | Numerical |
| **JSON/JSONL** | Structured data |

### 7.3 Evaluation & Experiment

| 기술 | 용도 |
|------|------|
| **Custom eval framework** | Pipeline 평가 |
| **Ablation runner** | Component 비교 |
| **Bucket analyzer** | Error analysis |

### 7.4 Future (Production)

| 기술 | 용도 |
|------|------|
| **FastAPI** | REST API |
| **React** | Frontend |
| **PostgreSQL** | Database |
| **Redis** | Caching |
| **Docker** | Containerization |

---

## 8. 주요 도전 과제

### 8.1 기술적 도전

1. **Class Imbalance**
   - 일부 HS4는 수백 개 샘플, 일부는 3개 이하
   - Rare class에서 정확도 저조

2. **Semantic Ambiguity**
   - "플라스틱 장난감 자동차" vs "플라스틱 모형 자동차"
   - 미묘한 용어 차이로 HS code 달라짐

3. **Legal Complexity**
   - GRI 규칙이 복잡하고 상호의존적
   - Note 해석이 애매한 경우 많음

4. **Data Scarcity**
   - 7,198개 샘플은 1,240개 class 대비 부족
   - Class당 평균 5.8개 샘플

### 8.2 비즈니스 도전

1. **Legal Liability**
   - AI 분류 오류 시 책임 소재
   - 관세사 최종 검토 필수

2. **User Trust**
   - 사용자가 AI 결과를 신뢰하도록 설득
   - 설명 가능성이 핵심

3. **Continuous Update**
   - HS code 체계는 매년 변경
   - KB 업데이트 프로세스 필요

---

## 9. 성공 지표 (KPI)

### 9.1 기술 지표

| 지표 | 현재 | 목표 (3개월) | 목표 (6개월) |
|------|------|-------------|-------------|
| **Top-1 Accuracy** | 13.5% | 20% | 30% |
| **Top-5 Accuracy** | 19.0% | 35% | 50% |
| **ECE** | 0.77 | < 0.5 | < 0.3 |
| **AUTO Rate (High Conf)** | 46.5% | 60% | 70% |
| **ASK Rate** | 53.5% | 35% | 25% |
| **Avg Response Time** | 0.6s | < 1s | < 0.5s |

### 9.2 비즈니스 지표 (향후)

| 지표 | 정의 | 목표 |
|------|------|------|
| **User Adoption** | 주간 활성 사용자 | 1,000+ |
| **Time Saving** | 사용자당 시간 절감 | 30분/일 |
| **Correction Rate** | 사용자가 AI 결과 수정 비율 | < 30% |
| **NPS** | Net Promoter Score | > 50 |

---

## 10. 결론

### 10.1 프로젝트 성과

✅ **구현 완료**:
- 법적 근거 기반 HS code 분류 시스템
- GRI 준수 + 8-axis 속성 + KB 규칙
- KB-only vs Hybrid 모드 지원
- 설명 가능한 결과 + 질문 생성

✅ **핵심 성과**:
- Top-1 Accuracy: 13.5% (Hybrid with KB-first)
- Hybrid가 KB-only 능가 (+1.5pp)
- 법적 정합성 검증 완료 (LegalGate)
- 17개 상세 작업 로그 문서화

### 10.2 차별화 요소

1. **법적 근거 기반**: GRI 규칙 준수, tariff notes 활용
2. **하이브리드 접근**: ML + KB 최적 결합
3. **능동적 질의**: 불확실성 인지 → 질문 생성
4. **다차원 속성**: 8-axis semantic framework
5. **완전한 추적성**: 모든 예측에 evidence chain

### 10.3 다음 단계

**즉시 실행** (1-2주):
1. ML retriever fine-tuning 실험 시작
2. Temperature scaling으로 calibration 개선
3. Feature weight 조정 실험

**단기** (1-2개월):
1. Top-5 recall 30% 달성
2. ECE < 0.5 달성
3. Hard negative mining 적용

**중기** (3-6개월):
1. HS6 분류 확장
2. Active learning loop 구축
3. LLM integration PoC

**장기** (6-12개월):
1. Production deployment
2. 1,000 MAU 달성
3. Regulatory compliance layer 추가

---

## 부록

### A. 주요 파일 위치

| 파일 | 경로 | 설명 |
|------|------|------|
| Main Pipeline | `src/classifier/pipeline.py` | 전체 orchestration |
| ML Retriever | `src/classifier/retriever.py` | SBERT + LR |
| KB Reranker | `src/classifier/reranker.py` | Card/Rule scoring |
| Evaluation | `src/classifier/eval/run_eval.py` | KB-only vs Hybrid |
| Regression Analysis | `scripts/analyze_hybrid_regressions.py` | 성능 변화 추적 |
| KB Cards | `data/hs4_cards_v2.jsonl` | 1,240 HS4 정보 |
| Training Data | `data/ruling_cases/all_cases_full_v7.json` | 7,198 cases |

### B. 평가 결과 위치

| Run | 경로 | 설명 |
|-----|------|------|
| Latest KB-only | `artifacts/eval/kb_only_20260203_214958/` | 12.0% accuracy |
| Latest Hybrid | `artifacts/eval/hybrid_20260203_220018/` | 13.5% accuracy |
| Regression Analysis | `artifacts/eval/hybrid_20260203_220018/hybrid_diff_summary.json` | Net gain: +3 |

### C. 문서 위치

| 문서 | 경로 | 내용 |
|------|------|------|
| Methodology | `docs/METHODOLOGY.md` | 전체 접근법 |
| Portfolio Report | `docs/FINAL_PORTFOLIO_REPORT.md` | 기능 요약 |
| Mode Separation | `docs/MODE_SEPARATION_FIX_REPORT.md` | KB-only 검증 |
| Evaluation Package | `docs/WORK_LOG_20250203_eval_package.md` | 평가 프레임워크 |

---

**문서 끝**
