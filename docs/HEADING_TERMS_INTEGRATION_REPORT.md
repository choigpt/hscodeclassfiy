# Heading Terms Integration Report

**Date**: 2026-02-03
**Author**: Claude Code
**Objective**: heading_terms 로드 → dataset 재빌드 → ranker 재학습 → 평가 → 피처 활성화 확인

---

## Executive Summary

### ✅ 달성된 목표

1. **heading_terms 로드 완료**: 1,239개 HS4 코드의 호 용어 (title_ko + scope)
2. **피처 계산 구현**: LegalGate에 heading_term_score 매칭 로직 추가
3. **Dataset 재빌드**: 전체 7,198샘플 → 168,677 후보 쌍
4. **Ranker 재학습**: f_legal_heading_term importance = 280.99 (rank #24)
5. **모드 분리 검증**: KB-only retriever_used=0.0, Hybrid=1.0

### ⚠️ 발견된 문제

1. **Hybrid 성능 저하**: KB-only (12%) > Hybrid (7.5%) - 37.5% 하락
2. **피처 영향력 제한적**: f_legal_heading_term이 활성화되었으나 예측 변경 없음
3. **f_lexical 압도적 우세**: 251,890 vs 281 (900배 차이)

---

## Step 1: Heading Terms 로드 구현

### 1.1 구현 내용

**파일**: `src/classifier/legal_gate.py`

```python
def _load_heading_terms(self) -> Dict[str, List[str]]:
    """HS4 호 용어 로드 (title_ko + scope 토큰)"""
    heading_terms = {}
    cards_path = Path("kb/structured/hs4_cards.jsonl")

    with open(cards_path, 'r', encoding='utf-8') as f:
        for line in f:
            card = json.loads(line)
            hs4 = card.get('hs4')
            terms = []

            # Title (heading)
            title = card.get('title_ko', '')
            if title:
                title_norm = normalize(title)
                title_tokens = [t for t in title_norm.split() if len(t) >= 2]
                terms.extend(title_tokens)

            # Scope (keywords)
            scope = card.get('scope', [])
            for keyword in scope:
                kw_norm = normalize(keyword)
                kw_tokens = [t for t in kw_norm.split() if len(t) >= 2]
                terms.extend(kw_tokens)

            if terms:
                heading_terms[hs4] = list(set(terms))

    return heading_terms
```

### 1.2 Heading Term 매칭 로직

**파일**: `src/classifier/legal_gate.py:262-290`

```python
# 1. Heading term match (호 용어 매칭)
if hs4 in self.heading_terms:
    terms = self.heading_terms[hs4]
    matched_terms = []

    for term in terms:
        if simple_contains(input_norm, term):
            # 직접 매칭: 0.1점
            result.heading_term_score += 0.1
            matched_terms.append(term)
        else:
            # Fuzzy 매칭: 0.05점
            match_result, _ = fuzzy_match(input_norm, term)
            if match_result:
                result.heading_term_score += 0.05
                matched_terms.append(f"~{term}")

    # 점수 클리핑 (최대 1.0)
    result.heading_term_score = min(result.heading_term_score, 1.0)

    # 증거 추가
    if matched_terms:
        result.evidence.append(Evidence(
            kind='legal_heading_term',
            source_id=hs4,
            text=f"호 용어 매칭: {', '.join(matched_terms[:5])}",
            weight=result.heading_term_score
        ))
```

### 1.3 로드 결과

```
[LegalGate] Heading terms loaded: 1239 HS4
```

**통계**:
- 총 HS4 코드: 1,239개
- 평균 term 수/HS4: ~15개 (추정)
- 정규화: normalize() 함수 적용 (소문자, 공백 정리)
- 최소 토큰 길이: 2자

---

## Step 2: Dataset 재빌드

### 2.1 소량 검증 (200샘플)

**명령**: `python -m src.classifier.rank.build_dataset_legal 200`

**결과**:
- 처리 샘플: 200
- 총 후보 쌍: 4,831
- f_legal_heading_term nonzero_rate: **7.33%** ✓

### 2.2 전체 Dataset 재빌드

**명령**: `python -m src.classifier.rank.train_ranker_legal --build`

**결과**:
```
전체 샘플: 7198
처리 완료: 7196
총 후보 쌍: 168,677
정답 후보 쌍: 7,196 (4.27%)

평균 후보 수 (LegalGate 전): 23.3
평균 후보 수 (LegalGate 후): 23.4
LegalGate 필터링 효과: -0.5%
```

### 2.3 Feature 분포

| Feature | Nonzero Rate | Mean | Range |
|---------|--------------|------|-------|
| f_legal_scope_match_score | 0.43% | 0.0015 | [0, 0.5] |
| **f_legal_heading_term** | **8.27%** | **0.0087** | **[0, 0.5]** |
| f_legal_include_support | 1.17% | 0.0003 | [0, 0.18] |
| f_legal_exclude_conflict | 3.78% | -0.0016 | [-0.36, 0] |
| f_legal_redirect_penalty | 0.94% | -0.0004 | [-0.3, 0] |

**분석**:
- heading_term이 168,677개 후보 중 **13,946개 (8.27%)** 에서 활성화
- 평균 점수는 낮음 (0.0087) → 대부분 0.1~0.2 범위

---

## Step 3: Ranker 재학습

### 3.1 학습 구성

**Train/Test Split**:
- Train: 5,757 queries (134,968 samples)
- Test: 1,439 queries (33,709 samples)

**Early Stopping**:
- Best iteration: 189
- Validation rounds: 50

### 3.2 성능 지표

| Metric | Train | Test |
|--------|-------|------|
| NDCG@1 | 0.8064 | 0.7802 |
| NDCG@3 | 0.8649 | 0.8458 |
| NDCG@5 | 0.8856 | 0.8716 |

**분석**:
- Test NDCG@1=0.78은 양호한 수준
- Train/Test 차이가 작아 overfitting 없음

### 3.3 Feature Importance (Top 20)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | f_lexical | 251,890.46 |
| 2 | f_specificity | 5,435.30 |
| 3 | f_form_match_score | 5,122.08 |
| 4 | f_material_match_score | 4,974.64 |
| 5 | f_card_hits | 3,306.57 |
| 6 | f_uncertainty_penalty | 2,895.61 |
| ... | ... | ... |
| 16 | **f_legal_exclude_conflict** | **947.22** |
| 18 | **f_legal_include_support** | **470.25** |
| 21 | **f_legal_redirect_penalty** | **341.56** |
| **24** | **f_legal_heading_term** | **280.99** ✓ |
| 26 | **f_legal_scope_match_score** | **144.20** |

### 3.4 LegalGate Features 요약

**모든 5개 피처가 활성화됨**:
1. f_legal_exclude_conflict: 947.22 (rank #16)
2. f_legal_include_support: 470.25 (rank #18)
3. f_legal_redirect_penalty: 341.56 (rank #21)
4. **f_legal_heading_term: 280.99 (rank #24)** ✅
5. f_legal_scope_match_score: 144.20 (rank #26)

**비교 (200샘플 모델 vs 전체 데이터 모델)**:

| Feature | 200샘플 | 전체 데이터 |
|---------|---------|-------------|
| f_legal_heading_term | 0.00 (미사용) | 280.99 (사용) ✓ |
| f_legal_exclude_conflict | 19.35 | 947.22 |

---

## Step 4: 200샘플 재평가 (KB-only vs Hybrid)

### 4.1 평가 설정

- **Dataset**: all_cases_full_v7.json
- **Samples**: 200 (test split, seed=42)
- **Modes**: kb_only, hybrid

### 4.2 성능 비교

| Metric | KB-only | Hybrid | 차이 |
|--------|---------|--------|------|
| **top1_accuracy** | **0.1200** | **0.0750** | **-37.5%** ⚠️ |
| top3_accuracy | 0.2350 | 0.1400 | -40.4% |
| top5_accuracy | 0.3550 | 0.1900 | -46.5% |
| ECE | 0.7814 | 0.8293 | +6.1% (worse) |
| Brier Score | 0.7388 | 0.7643 | +3.5% (worse) |

**결론**: **Hybrid 모드가 KB-only보다 모든 지표에서 저조**

### 4.3 Usage 비교

| Component | KB-only | Hybrid |
|-----------|---------|--------|
| retriever_usage_rate | 0.0 ✓ | 1.0 ✓ |
| ranker_usage_rate | 0.0 ✓ | 1.0 ✓ |
| avg_cards_hits | 3.61 | 3.68 |
| avg_rule_hits | 0.03 | 0.07 |

**모드 분리 정상 작동**:
- KB-only는 ML retriever/ranker를 사용하지 않음
- Hybrid는 모든 샘플에서 ML retriever/ranker 적용

### 4.4 예측 차이

**top1_same_rate**: 0.82 (< 0.95 기준 충족) ✓

- 동일한 top-1: 164/200 (82%)
- 다른 top-1: 36/200 (18%)

**결론**: 두 모드가 충분히 다른 예측을 생성함

### 4.5 모델 간 비교 (200샘플 vs 전체 데이터)

**KB-only**:
- top1_accuracy: 0.12 → 0.12 (변화 없음)
- 동일한 예측: 200/200 (100%)

**Hybrid**:
- top1_accuracy: 0.075 → 0.075 (변화 없음)
- 동일한 예측: 200/200 (100%)

**결론**: f_legal_heading_term이 활성화되었으나, 이 200샘플에서는 예측을 변경하지 않음. f_lexical이 너무 dominant함 (251,890 vs 281).

---

## Step 5: Calibration 진단

### 5.1 Confidence 분석

**문제**: 모든 예측의 confidence=0.0

```python
KB-only:
  Accuracy: 12.00%
  Avg Confidence: 0.00%
  Calibration Gap: 12.00%
  Confidence range: [0.0000, 0.0000]

Hybrid:
  Accuracy: 7.50%
  Avg Confidence: 0.00%
  Calibration Gap: 7.50%
  Confidence range: [0.0000, 0.0000]
```

**원인**:
1. Classification-level confidence가 계산되지 않음
2. GRI signal confidence만 존재 (gri1_note_like, gri2a_incomplete 등)
3. score_total은 있으나 확률로 변환되지 않음

### 5.2 ECE 분석

**KB-only**: ECE=0.7814 (매우 높음)
**Hybrid**: ECE=0.8293 (더 높음)

**문제**:
- Ranker score는 확률이 아님 (3.86 같은 값)
- Calibration layer (temperature scaling, isotonic regression) 없음
- AUTO threshold가 score 기반이 아닌 임의 설정

### 5.3 권장 수정

1. **Calibration Layer 추가**:
   ```python
   from sklearn.calibration import CalibratedClassifierCV

   # Ranker 점수를 확률로 변환
   calibrated_probs = calibrator.predict_proba(ranker_scores)
   confidence = calibrated_probs[top1_idx]
   ```

2. **AUTO Threshold 재설정**:
   - 현재: 임의 threshold
   - 권장: calibrated probability > 0.7 + legal_conflict=0 + fact_sufficient=True

3. **GRI Confidence 통합**:
   - GRI signal confidence를 classification confidence에 반영
   - 불확실성이 높으면 confidence 감소

---

## 핵심 문제 및 원인 분석

### 문제 1: Hybrid 성능 저하 (KB-only 12% > Hybrid 7.5%)

**가설 1: ML Retriever 품질 문제**

```
KB-only:
  candidate_recall_5: 0.355 (35.5%)

Hybrid:
  candidate_recall_5: 0.190 (19.0%)
```

**분석**:
- ML retriever가 top-50 후보 중 정답을 포함하는 비율이 낮음
- KB retrieval (lexical + cards)이 더 효과적
- Hybrid에서 ML 후보가 KB 후보를 밀어내면서 recall 하락

**가설 2: Ranker 효과 부족**

```
Ranker NDCG@1 (학습): 0.78
실제 top1_accuracy: 0.075
```

**분석**:
- Ranker는 학습 데이터에서 78% NDCG 달성
- 하지만 실전에서는 7.5% accuracy (10배 차이)
- Train/Test split의 distribution shift 가능성

**가설 3: Feature Dominance (f_lexical)**

```
f_lexical: 251,890.46 (90% 이상 기여)
f_legal_heading_term: 280.99 (0.1% 기여)
```

**분석**:
- f_lexical이 압도적으로 강함
- 다른 피처들은 미세 조정 수준
- heading_term이 활성화되어도 예측에 큰 영향 없음

### 문제 2: f_legal_heading_term 영향력 제한

**Dataset 수준**:
- nonzero_rate: 8.27% (충분함)
- mean: 0.0087 (너무 낮음)
- range: [0, 0.5]

**모델 수준**:
- importance: 280.99 (rank #24)
- f_lexical 대비 비율: 1:896

**예측 수준**:
- 200샘플 중 예측 변경: 0개 (0%)
- 동일 예측: 200개 (100%)

**원인**:
1. **스케일 불균형**: heading_term_score 최대 0.5 vs f_lexical 최대 수십
2. **희소성**: 8.27%만 활성화 → 대부분 0
3. **신호 약함**: 매칭되어도 0.1~0.2 점수 (미미함)

### 문제 3: Calibration 미구현

**현상**:
- 모든 confidence=0.0
- ECE=0.78~0.83 (uncalibrated)
- AUTO/ASK 판단 기준 없음

**영향**:
- 사용자에게 신뢰도 표시 불가
- ASK 라우팅이 임의적
- Production 배포 불가능

---

## 권장 조치 (우선순위순)

### 1. ML Retriever 개선 또는 비활성화 (High Priority)

**옵션 A: ML Retriever 제거**
```python
# KB-only 모드를 기본으로 사용
pipeline = HSPipeline(retriever=None, ranker=None)
```

**장점**:
- 즉시 성능 12% 확보
- 안정적이고 해석 가능

**단점**:
- ML의 이점 포기
- 확장성 제한

**옵션 B: ML Retriever 재학습**
- 더 나은 embedding 모델 (jhgan/ko-sroberta → 최신 모델)
- Negative sampling 개선
- Hard negative mining

### 2. Feature 스케일 조정 (Medium Priority)

**heading_term_score 가중치 증가**:
```python
# 현재: 직접 매칭 0.1, fuzzy 0.05, 최대 1.0
# 제안: 직접 매칭 0.5, fuzzy 0.25, 최대 5.0
result.heading_term_score += 0.5  # 5배 증가
result.heading_term_score = min(result.heading_term_score, 5.0)
```

**f_lexical 정규화**:
```python
# 현재: card 매칭당 0.5~1.5 무제한 누적
# 제안: card 점수를 log scale로 변환
f_lexical = np.log1p(card_score_sum)
```

### 3. Calibration Layer 구현 (High Priority for Production)

**Temperature Scaling**:
```python
from sklearn.calibration import CalibratedClassifierCV

# Ranker 출력을 확률로 변환
calibrator = CalibratedClassifierCV(ranker, method='sigmoid')
calibrator.fit(val_scores, val_labels)

confidence = calibrator.predict_proba(test_score)[0, 1]
```

**AUTO Threshold**:
```python
# 기존: score_total > threshold (임의)
# 제안: confidence > 0.7 AND legal_conflict=False AND fact_sufficient=True
auto_eligible = (
    confidence > 0.7 and
    result.legal_conflict == False and
    result.fact_sufficient == True
)
```

### 4. 전체 데이터 재평가 (Medium Priority)

**현재**: 200샘플만 평가
**권장**: 전체 test split (1,440샘플) 평가

```bash
# 전체 test split 평가
python -m src.classifier.eval.run_eval --mode kb_only --split test --seed 42
python -m src.classifier.eval.run_eval --mode hybrid --split test --seed 42
```

**목적**:
- 200샘플은 대표성 부족 가능성
- 더 큰 샘플에서 heading_term 효과 확인
- Statistical significance 검증

---

## 결론

### ✅ 성공한 부분

1. **heading_terms 로드**: 1,239개 HS4, 8.27% 활성화율
2. **피처 구현**: LegalGate heading_term_score 계산 로직 추가
3. **모델 학습**: f_legal_heading_term importance=280.99 달성
4. **모드 분리**: KB-only vs Hybrid 정상 작동

### ⚠️ 개선 필요한 부분

1. **Hybrid 성능**: KB-only (12%) > Hybrid (7.5%)
2. **피처 영향력**: heading_term 활성화되었으나 예측 불변
3. **Calibration**: confidence=0, ECE=0.78~0.83

### 🎯 다음 단계

**즉시 조치**:
1. ML Retriever 제거 또는 개선
2. Calibration layer 구현
3. Feature scaling 조정

**중기 조치**:
1. 전체 test split (1,440샘플) 재평가
2. heading_term 가중치 증가 실험
3. f_lexical 정규화

**장기 조치**:
1. ML Retriever 재학습 (더 나은 모델)
2. Ensemble approach (KB + ML voting)
3. Active learning으로 어려운 케이스 학습

---

## 산출물

### 코드 변경

1. `src/classifier/legal_gate.py`:
   - `_load_heading_terms()` 메서드 추가
   - `_evaluate_candidate()` heading term 매칭 로직 구현

2. `artifacts/ranker_legal/`:
   - `rank_features_legal.csv` (168,677 rows, f_legal_heading_term 활성)
   - `model_legal.txt` (LightGBM 모델, importance=280.99)
   - `train_results_legal.json` (NDCG@1=0.78)

### 평가 결과

1. `artifacts/eval/kb_only_20260203_192931/`:
   - metrics_summary.json (top1=0.12)
   - usage_summary.json (retriever_used=0.0)

2. `artifacts/eval/hybrid_20260203_193049/`:
   - metrics_summary.json (top1=0.075)
   - usage_summary.json (retriever_used=1.0)

### 문서

1. `docs/HEADING_TERMS_INTEGRATION_REPORT.md` (본 문서)
2. `docs/CALIBRATION_DIAG.md` (다음 단계에서 작성 예정)
3. `docs/COMPARE_KB_ONLY_VS_HYBRID.md` (업데이트 예정)

---

**Report Generated**: 2026-02-03 19:35 KST
**Status**: Complete (with recommendations for improvement)
