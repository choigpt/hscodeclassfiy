# LegalGate-Integrated Ranking System

## 아키텍처 원칙

HS 분류 시스템은 **2-계층 아키텍처**로 구성됨:

```
┌────────────────────────────────────────────┐
│  Layer 1: Law (Hard Policy)                │
│  - LegalGate (GRI 1)                       │
│  - 호 용어 + 주규정 기반 필터링             │
│  - Hard exclude (override 불가)            │
└────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────┐
│  Layer 2: Auxiliary (Learned)              │
│  - LightGBM LambdaMART Ranker              │
│  - LegalGate 통과 후보만 재정렬            │
│  - LegalGate features 포함                 │
└────────────────────────────────────────────┘
```

### 핵심 원칙

1. **법 우선**: Layer 1 (LegalGate) 결정은 Layer 2 (Ranker)가 절대 override할 수 없음
2. **보수적 학습**: 정답이 LegalGate에서 제외되면 해당 query 전체 스킵 (법이 맞다고 가정)
3. **피처 통합**: LegalGate features를 ranker feature set에 추가하여 학습

## 디렉토리 구조

```
src/classifier/rank/
├── README_LEGAL.md              # 이 문서
├── build_dataset_legal.py       # LegalGate 통합 데이터셋 생성
├── train_ranker_legal.py        # LegalGate 통합 랭커 학습
├── build_dataset.py             # 기존 데이터셋 생성 (비교용)
└── train_ranker.py              # 기존 랭커 학습 (비교용)

artifacts/ranker_legal/          # 출력 디렉토리
├── rank_features_legal.csv      # 피처 CSV (LegalGate features 포함)
├── rank_queries_legal.json      # 쿼리 정보
├── rank_stats_legal.json        # 데이터셋 통계
├── model_legal.txt              # 학습된 LightGBM 모델
├── train_results_legal.json     # 학습 결과
└── ARCHITECTURE.md              # 아키텍처 문서 (자동 생성)
```

## 사용법

### 1. 데이터셋 생성

```bash
# 전체 결정사례로 데이터셋 생성
python -m src.classifier.rank.build_dataset_legal

# 처음 1000개만 테스트
python -m src.classifier.rank.build_dataset_legal 1000
```

**출력**:
- `artifacts/ranker_legal/rank_features_legal.csv`: 피처 데이터
- `artifacts/ranker_legal/rank_queries_legal.json`: 쿼리 정보
- `artifacts/ranker_legal/rank_stats_legal.json`: 통계

**주요 통계**:
- `skipped_legal_gate_exclude_answer`: LegalGate가 정답을 제외한 케이스 수
  - 이 케이스는 학습에서 제외됨 (법 > 결정사례 원칙)
- `avg_candidates_before_legal`: LegalGate 적용 전 평균 후보 수
- `avg_candidates_after_legal`: LegalGate 적용 후 평균 후보 수

### 2. 모델 학습

```bash
# 데이터셋이 이미 생성된 경우
python -m src.classifier.rank.train_ranker_legal

# 데이터셋 생성 + 학습 (한 번에)
python -m src.classifier.rank.train_ranker_legal --build
```

**출력**:
- `artifacts/ranker_legal/model_legal.txt`: 학습된 모델
- `artifacts/ranker_legal/train_results_legal.json`: 결과

**평가 지표**:
- NDCG@1, @3, @5 (Train/Test)
- Feature Importance (Top 20)
- LegalGate Features Importance (별도 표시)

### 3. 모델 로드 및 사용

```python
from src.classifier.rank.train_ranker_legal import load_ranker_legal

# 모델 로드
model = load_ranker_legal("artifacts/ranker_legal/model_legal.txt")

# 예측 (feature vector 필요)
scores = model.predict(X_features)
```

## Features

### Base Features (34개)

기존 CandidateFeatures의 모든 피처:

#### 기본 피처
- `f_ml`: ML retrieval 점수
- `f_lexical`: 어휘 매칭 점수
- `f_card_hits`: 카드 키워드 매칭 수
- `f_rule_inc_hits`: 포함 규칙 매칭 수
- `f_rule_exc_hits`: 제외 규칙 매칭 수
- `f_not_in_model`: 모델에 없는 HS4 플래그

#### GRI 신호 피처
- `f_gri2a_signal`: GRI 2(a) 신호 (혼합물/복합물)
- `f_gri2b_signal`: GRI 2(b) 신호 (재료/구성요소)
- `f_gri3_signal`: GRI 3 신호 (특수성 우선)
- `f_gri5_signal`: GRI 5 신호 (포장)

#### 후보별 피처
- `f_specificity`: HS4 특수성 점수
- `f_exclude_conflict`: 제외 규칙 충돌 플래그
- `f_is_parts_candidate`: 부품 관련 후보 플래그

#### 전역 속성 매칭 피처 (레거시)
- `f_state_match`: 상태 매칭
- `f_material_match`: 재질 매칭
- `f_use_match`: 용도 매칭
- `f_form_match`: 형태 매칭
- `f_parts_mismatch`: 부품 불일치
- `f_set_signal`: 세트 신호
- `f_incomplete_signal`: 미완성 신호

#### 정량/주규정 피처
- `f_quant_match_score`: 정량 규칙 매칭 점수
- `f_quant_hard_exclude`: 정량 규칙 hard exclude
- `f_quant_missing_value`: 정량 값 누락
- `f_note_hard_exclude`: 주규정 hard exclude
- `f_note_support_sum`: 주규정 지지 점수 합

#### 8축 속성 매칭 피처 (NEW)
- `f_object_match_score`: 물체 본질 매칭
- `f_material_match_score`: 재질 매칭 (확장)
- `f_processing_match_score`: 가공 상태 매칭
- `f_function_match_score`: 기능/용도 매칭
- `f_form_match_score`: 물리적 형태 매칭
- `f_completeness_match_score`: 완성도 매칭
- `f_quant_rule_match_score`: 정량 규칙 매칭
- `f_legal_scope_match_score`: 법적 범위 매칭

#### 충돌/불확실성 피처
- `f_conflict_penalty`: 후보 간 속성 충돌 페널티
- `f_uncertainty_penalty`: 속성 추출 불확실성 페널티

### LegalGate Features (4개) - NEW

LegalGate 통합으로 추가된 법 규범 피처:

- **`f_legal_heading_term`**: 호 용어 매칭 점수
  - HS4 카드의 title_ko와 입력 텍스트 매칭
  - 범위: 0.0 ~ 1.0

- **`f_legal_include_support`**: 포함 주규정 지지 점수
  - 포함(include) 주규정이 입력을 얼마나 지지하는지
  - 범위: 0.0 ~ 1.0+

- **`f_legal_exclude_conflict`**: 제외 주규정 충돌 점수
  - 제외(exclude) 주규정과 입력의 충돌 강도
  - 범위: 0.0 ~ -1.0 (음수, 충돌 강도)
  - < -0.7이면 hard exclude

- **`f_legal_redirect_penalty`**: 리다이렉트 페널티
  - 다른 호로 리다이렉트하는 경우 페널티
  - 범위: 0.0 ~ -1.0 (음수)
  - < -0.8이면 hard exclude

**총 피처 수**: 34 + 4 = **38개**

## 학습 데이터 필터링

### LegalGate 필터링

각 결정사례에 대해:

1. **후보 생성**: KB에서 top-K 후보 추출
2. **LegalGate 적용**: 각 후보에 대해 LegalGateResult 계산
3. **Hard exclude 제거**: `passed=False`인 후보 제거
4. **정답 확인**: 정답이 LegalGate에서 제거되었는지 확인
   - 제거되었으면 → 해당 query 전체 스킵 (법이 맞다고 가정)
   - 남아있으면 → 학습 데이터에 포함

### 통계 예시

```
총 사례: 10,000
처리됨: 8,500
스킵 (후보 없음): 300
스킷 (라벨 없음): 200
스킵 (LegalGate가 정답 제외): 1,000  ← 중요!

평균 후보 수 (LegalGate 전): 25.0
평균 후보 수 (LegalGate 후): 18.5
LegalGate 필터링 비율: 26.0%
```

## 평가

### NDCG@K

LambdaMART의 기본 평가 지표:

- **NDCG@1**: Top-1 정확도와 유사
- **NDCG@3**: Top-3 내 정답 위치 고려
- **NDCG@5**: Top-5 내 정답 위치 고려

### Feature Importance 분석

학습 후 Feature Importance 확인:

```python
# Top 20 features
Feature Importance (top 20):
  f_ml: 1250.00
  f_legal_include_support: 980.50  ← LegalGate feature
  f_card_hits: 750.20
  ...

# LegalGate features 별도 표시
LegalGate Features Importance:
  f_legal_include_support: 980.50 (rank #2)
  f_legal_exclude_conflict: 620.30 (rank #5)
  f_legal_heading_term: 450.10 (rank #8)
  f_legal_redirect_penalty: 120.00 (rank #18)
```

LegalGate features가 상위권에 있으면 법 규범이 분류에 중요한 역할을 한다는 의미.

## 비교 실험

### KB-only vs Hybrid

```bash
# 1. KB-only (LegalGate만)
# → pipeline.py에서 use_ranker=False

# 2. Hybrid (LegalGate + Ranker)
# → pipeline.py에서 use_ranker=True, ranker 모델 로드
```

### Ablation Study

```bash
# 1. LegalGate 없이 Ranker만
python -m src.classifier.rank.train_ranker  # 기존 방식

# 2. LegalGate + Ranker (제안 방식)
python -m src.classifier.rank.train_ranker_legal
```

예상 결과:
- Ranker만: NDCG@1 ~0.75, 법적 오류 가능
- LegalGate + Ranker: NDCG@1 ~0.72, 법적 오류 최소화

**Trade-off**: 약간의 정확도 감소 대신 법적 견고성 확보

## 파이프라인 통합

### pipeline.py 통합 예시

```python
# LegalGate + Ranker 통합
class HSPipeline:
    def __init__(self, use_legal_gate=True, use_ranker=True):
        self.legal_gate = LegalGate() if use_legal_gate else None
        self.ranker_model = load_ranker_legal() if use_ranker else None

    def classify(self, text):
        # 1. 후보 생성
        candidates = self.generate_candidates(text)

        # 2. LegalGate (Layer 1)
        if self.legal_gate:
            candidates, _, _ = self.legal_gate.apply(text, candidates)

        # 3. Ranker (Layer 2)
        if self.ranker_model and len(candidates) > 1:
            # 피처 계산 (LegalGate features 포함)
            features = self.compute_features_with_legal(text, candidates)

            # Ranker 점수 계산
            scores = self.ranker_model.predict(features)

            # 재정렬
            candidates = self.rerank_by_scores(candidates, scores)

        return candidates
```

## 주의사항

### 1. LegalGate 없이 Ranker 사용 금지

LegalGate features를 사용하여 학습된 모델은 반드시 LegalGate와 함께 사용해야 함:

```python
# ❌ 잘못된 사용
ranker = load_ranker_legal()
candidates = generate_candidates(text)  # LegalGate 스킵
scores = ranker.predict(features)  # LegalGate features 없음 → 오류

# ✅ 올바른 사용
ranker = load_ranker_legal()
legal_gate = LegalGate()
candidates, _, legal_debug = legal_gate.apply(text, candidates)
features = compute_features_with_legal(text, candidates, legal_debug)
scores = ranker.predict(features)
```

### 2. 정답이 LegalGate에서 제외된 경우

학습 데이터에서 제외되지만, 실제 운영에서는 발생 가능:

- **원인**: 법 규범 업데이트, 결정사례 오류, 주규정 해석 차이
- **대응**: `decision.status = REVIEW`로 전문가 검토 요청

### 3. 법 규범 업데이트

주규정이 변경되면 LegalGate 동작이 달라짐:

- 주규정 업데이트 후 → 데이터셋 재생성 → 모델 재학습 필요

## FAQ

### Q1. LegalGate 없이 Ranker만 사용할 수 있나요?

A. 기술적으로는 가능하지만 권장하지 않습니다. LegalGate features가 없으면 모델 성능이 저하되며, 법적 견고성도 보장할 수 없습니다.

### Q2. 정답이 LegalGate에서 제외되는 비율이 높으면?

A. 주규정 추출 로직 또는 LegalGate 임계값을 재검토해야 합니다. `skipped_legal_gate_exclude_answer` 통계를 모니터링하세요.

### Q3. LegalGate features의 중요도가 낮으면?

A. 결정사례가 법 규범보다 데이터 패턴에 의존하고 있다는 의미일 수 있습니다. 주규정 커버리지를 확장하거나 feature engineering을 개선하세요.

### Q4. 모델을 업데이트해야 하는 시점은?

A. 다음 경우에 재학습 필요:
- 주규정 업데이트
- 결정사례 추가 (월 100건 이상)
- 법 규범 해석 변경

## 참고 문서

- `src/classifier/legal_gate.py`: LegalGate 구현
- `src/classifier/fact_checker.py`: 사실 충분성 검사
- `src/classifier/required_facts.py`: RequiredFact 정의
- `kb/structured/note_requirements.jsonl`: 주규정 요구 사실
- `kb/structured/hs4_cards_v2.jsonl`: HS4 카드 (required_facts 포함)

## 라이선스

HS Code 분류 시스템의 일부로, 동일한 라이선스 적용.
