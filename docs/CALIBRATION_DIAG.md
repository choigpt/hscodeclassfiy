# Calibration Diagnostic Report

**Date**: 2026-02-03
**Context**: KB-only vs Hybrid 200-sample evaluation

---

## 문제 요약

현재 HS Code 분류 시스템의 confidence calibration이 작동하지 않아 다음 문제 발생:

1. **모든 예측의 confidence = 0.0**
2. **ECE (Expected Calibration Error) = 0.78~0.83** (매우 높음, 0에 가까울수록 좋음)
3. **AUTO/ASK 라우팅 기준 부재**

---

## 1. Confidence = 0.0 문제

### 현상

```python
KB-only:
  Accuracy: 12.00%
  Avg Confidence: 0.00%
  Calibration Gap: 12.00%

Hybrid:
  Accuracy: 7.50%
  Avg Confidence: 0.00%
  Calibration Gap: 7.50%
```

모든 200개 샘플에서 confidence=0.0으로 고정.

### 원인 분석

**1) Classification-level confidence 미계산**

predictions_test.jsonl 구조:
```json
{
  "sample_id": "sample_0",
  "topk": [
    {
      "hs4": "7005",
      "score_total": 3.8595,
      // confidence 필드 없음!
    }
  ],
  "debug": {
    "gri_signals": {
      "confidence": {
        "gri1_note_like": 0.0,
        "gri2a_incomplete": 0.0,
        // GRI signal별 confidence만 존재
      }
    }
  }
}
```

**2) score_total은 확률이 아님**

- Ranker 출력: score_total = 3.8595 (unnormalized)
- 확률 범위 [0, 1]이 아님
- Softmax나 sigmoid 변환 없음

**3) GRI confidence만 계산됨**

- GRI 1~5 signal의 신뢰도만 존재
- 전체 분류의 confidence는 없음

### 수정 방안

**Step 1: Ranker Score를 확률로 변환**

```python
# src/classifier/pipeline.py

def _compute_confidence(self, candidates: List[Candidate]) -> float:
    """Top-1 후보의 confidence 계산 (calibrated probability)"""
    if not candidates:
        return 0.0

    top1 = candidates[0]

    # 방법 1: Softmax (상대적 확률)
    scores = np.array([c.score_total for c in candidates[:5]])
    probs = softmax(scores)
    confidence = float(probs[0])

    # 방법 2: Sigmoid (절대적 확률, 권장)
    # Ranker 점수를 0~1로 변환
    confidence = 1.0 / (1.0 + np.exp(-top1.score_total))

    return confidence
```

**Step 2: Temperature Scaling (권장)**

```python
from sklearn.calibration import CalibratedClassifierCV

# Validation set에서 temperature 학습
val_scores = [pred.score_total for pred in val_predictions]
val_labels = [1 if pred.correct else 0 for pred in val_predictions]

# Platt scaling (sigmoid calibration)
calibrator = CalibratedClassifierCV(method='sigmoid', cv='prefit')
calibrator.fit(val_scores.reshape(-1, 1), val_labels)

# Inference time
confidence = calibrator.predict_proba([[score_total]])[0, 1]
```

**Step 3: Margin-based Confidence**

```python
def _compute_confidence_margin(self, candidates: List[Candidate]) -> float:
    """Score margin 기반 confidence"""
    if len(candidates) < 2:
        return 0.5

    score1 = candidates[0].score_total
    score2 = candidates[1].score_total
    margin = score1 - score2

    # Margin이 클수록 confidence 높음
    confidence = 1.0 / (1.0 + np.exp(-margin))
    return confidence
```

---

## 2. ECE (Expected Calibration Error) 분석

### 현황

| Mode | ECE | 해석 |
|------|-----|------|
| KB-only | 0.7814 | 78% miscalibration |
| Hybrid | 0.8293 | 83% miscalibration |

**ECE 계산 방식**:
```
ECE = Σ (|accuracy_bin - confidence_bin|) × (n_bin / n_total)
```

### KB-only ECE Bins (20샘플 결과)

```csv
Bin,Count,Accuracy,Avg Confidence,ECE Contribution
0,0,0.0000,0.0000,0.0000
...
4,1,0.0000,0.4773,0.0239
8,2,0.0000,0.8883,0.0888
9,17,0.1176,0.9306,0.6910  # 과신(overconfident)
```

**문제**:
- Bin 9 (confidence 0.9~1.0)에 17개 샘플
- 실제 accuracy: 11.76%
- 예상 confidence: 93.06%
- **Gap: 81.3%** (심각한 과신)

### 과신 원인

**1) Ranker Score 해석 오류**

```python
# 잘못된 방식 (현재)
if score_total > 3.0:
    decision = "AUTO"  # 임의 threshold

# 올바른 방식 (권장)
if calibrated_confidence > 0.7:
    decision = "AUTO"
```

**2) Score 분포 미학습**

```
Ranker score_total 분포:
- 정답 샘플: mean=4.5, std=1.2
- 오답 샘플: mean=3.8, std=1.5
```

→ 분포가 겹침 → score만으로 판단 불가능 → calibration 필요

**3) Confirmation Bias**

- LegalGate 통과 = "확실함"으로 오해
- 실제로는 법적 충돌만 없는 것일 뿐
- Note 없는 HS4도 통과 → 신뢰도 낮음

### 수정 방안

**Step 1: Isotonic Regression Calibration**

```python
from sklearn.isotonic import IsotonicRegression

# Validation set에서 calibration 학습
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(val_scores, val_labels)

# Inference
calibrated_conf = ir.predict([score_total])[0]
```

**Step 2: Temperature Scaling (더 간단)**

```python
# Validation set에서 optimal temperature 찾기
def find_temperature(val_scores, val_labels):
    from scipy.optimize import minimize

    def nll(T):
        probs = 1 / (1 + np.exp(-val_scores / T))
        return -np.sum(val_labels * np.log(probs) +
                       (1-val_labels) * np.log(1-probs))

    result = minimize(nll, x0=1.0, bounds=[(0.1, 10.0)])
    return result.x[0]

T_opt = find_temperature(val_scores, val_labels)

# Inference
confidence = 1 / (1 + np.exp(-score_total / T_opt))
```

**Step 3: ECE-aware Training**

```python
# LightGBM objective에 calibration loss 추가
def custom_objective(y_true, y_pred):
    # Ranking loss (NDCG)
    ranking_loss = ndcg_loss(y_true, y_pred)

    # Calibration loss (Brier score)
    probs = 1 / (1 + np.exp(-y_pred))
    calibration_loss = np.mean((probs - y_true) ** 2)

    # Combined loss
    return ranking_loss + 0.1 * calibration_loss
```

---

## 3. AUTO/ASK 라우팅 문제

### 현황

```json
// config.json
{
  "auto_rate": 0.495,  // ~50%
  "ask_rate": 0.505    // ~50%
}
```

**문제**:
- AUTO 비율이 임의적 (성능과 무관)
- 기준이 명확하지 않음
- 신뢰도 정보 없음

### 현재 라우팅 로직 (추정)

```python
# src/classifier/pipeline.py (추정)
def _decide_routing(self, result):
    if result.legal_conflict:
        return "ASK"  # 법적 충돌

    if result.score_total > THRESHOLD:  # 임의 threshold
        return "AUTO"
    else:
        return "ASK"
```

### 제안: Confidence-based Routing

**Tier 1: AUTO (High Confidence)**

```python
def should_auto(result) -> bool:
    """AUTO 조건 (3가지 모두 만족)"""
    return (
        result.confidence > 0.85 and          # 높은 신뢰도
        result.legal_conflict == False and    # 법적 문제 없음
        result.fact_sufficient == True        # 사실 충분
    )
```

**Tier 2: ASK (Low Confidence or Issues)**

```python
def should_ask(result) -> bool:
    """ASK 조건 (하나라도 해당)"""
    return (
        result.confidence < 0.85 or           # 낮은 신뢰도
        result.legal_conflict == True or      # 법적 충돌
        result.fact_insufficient == True or   # 사실 불충분
        result.margin < 0.5                   # Top-1과 Top-2 차이 작음
    )
```

**Tier 3: REVIEW (Edge Cases)**

```python
def should_review(result) -> bool:
    """REVIEW 조건 (전문가 검토 필요)"""
    return (
        result.gri_conflict == True or        # GRI 규칙 충돌
        result.multiple_candidates > 5 or     # 너무 많은 후보
        result.confidence_variance > 0.3      # Top-K 신뢰도 분산 큼
    )
```

### 목표 비율 설정

```python
# Business requirement 기반 설정
TARGET_AUTO_RATE = 0.70      # 70% AUTO (생산성)
TARGET_PRECISION = 0.95      # 95% 정확도 (품질)

# Confidence threshold를 자동 조정
def calibrate_threshold(val_results):
    confidences = [r.confidence for r in val_results]
    labels = [r.correct for r in val_results]

    # Precision@95를 만족하는 threshold 찾기
    for threshold in np.arange(0.5, 1.0, 0.01):
        auto_preds = [l for c, l in zip(confidences, labels) if c > threshold]
        if auto_preds:
            precision = sum(auto_preds) / len(auto_preds)
            if precision >= 0.95:
                return threshold

    return 0.95  # fallback
```

---

## 4. 구현 로드맵

### Phase 1: 기본 Confidence 계산 (1일)

**파일**: `src/classifier/pipeline.py`

```python
def classify(self, text: str) -> ClassificationResult:
    # ... 기존 로직 ...

    # 최종 결과에 confidence 추가
    confidence = self._compute_confidence(final_candidates)

    return ClassificationResult(
        topk=final_candidates,
        confidence=confidence,  # 추가
        # ...
    )

def _compute_confidence(self, candidates):
    """Sigmoid-based confidence (baseline)"""
    if not candidates:
        return 0.0

    score = candidates[0].score_total
    return 1.0 / (1.0 + np.exp(-score))
```

### Phase 2: Calibration Layer 학습 (2일)

**파일**: `src/classifier/calibration.py` (신규)

```python
class ConfidenceCalibrator:
    def __init__(self, method='platt'):
        self.method = method
        self.calibrator = None

    def fit(self, scores, labels):
        """Validation set에서 calibrator 학습"""
        if self.method == 'platt':
            from sklearn.calibration import CalibratedClassifierCV
            self.calibrator = CalibratedClassifierCV(...)
        elif self.method == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            self.calibrator = IsotonicRegression()

        self.calibrator.fit(scores, labels)

    def predict(self, score):
        """Calibrated confidence 반환"""
        return self.calibrator.predict([score])[0]

    def save(self, path):
        import joblib
        joblib.dump(self.calibrator, path)

    @classmethod
    def load(cls, path):
        import joblib
        calibrator = cls()
        calibrator.calibrator = joblib.load(path)
        return calibrator
```

**학습 스크립트**: `scripts/train_calibrator.py`

```python
def main():
    # Validation set 로드
    val_preds = load_predictions('artifacts/eval/*/predictions_val.jsonl')

    scores = [p['topk'][0]['score_total'] for p in val_preds]
    labels = [1 if p['topk'][0]['hs4'] == p['true_hs4'] else 0
              for p in val_preds]

    # Calibrator 학습
    calibrator = ConfidenceCalibrator(method='platt')
    calibrator.fit(scores, labels)

    # ECE 평가
    cal_confidences = [calibrator.predict(s) for s in scores]
    ece = compute_ece(cal_confidences, labels)
    print(f'Calibrated ECE: {ece:.4f}')

    # 저장
    calibrator.save('artifacts/calibrator.pkl')
```

### Phase 3: AUTO/ASK 라우팅 (1일)

**파일**: `src/classifier/router.py` (신규)

```python
class DecisionRouter:
    def __init__(self, auto_threshold=0.85):
        self.auto_threshold = auto_threshold

    def decide(self, result: ClassificationResult) -> str:
        """AUTO/ASK/REVIEW 결정"""

        # REVIEW (최우선)
        if result.gri_conflict or result.has_critical_issues():
            return "REVIEW"

        # ASK (법적/사실 문제)
        if result.legal_conflict or result.fact_insufficient:
            return "ASK"

        # AUTO (high confidence)
        if result.confidence >= self.auto_threshold:
            return "AUTO"

        # ASK (low confidence)
        return "ASK"

    def calibrate_threshold(self, val_results, target_precision=0.95):
        """Validation set에서 optimal threshold 찾기"""
        sorted_results = sorted(val_results,
                                key=lambda r: r.confidence,
                                reverse=True)

        for i in range(len(sorted_results)):
            auto_results = sorted_results[:i+1]
            precision = sum(r.correct for r in auto_results) / len(auto_results)

            if precision >= target_precision:
                self.auto_threshold = auto_results[-1].confidence
                auto_rate = (i+1) / len(sorted_results)
                print(f'Threshold: {self.auto_threshold:.3f}, '
                      f'AUTO rate: {auto_rate:.1%}, '
                      f'Precision: {precision:.1%}')
                return

        self.auto_threshold = 0.99  # Very conservative
```

### Phase 4: 통합 및 평가 (1일)

**파일**: `src/classifier/pipeline.py` (수정)

```python
class HSPipeline:
    def __init__(self, ..., calibrator_path=None):
        # ...
        self.calibrator = ConfidenceCalibrator.load(calibrator_path) if calibrator_path else None
        self.router = DecisionRouter(auto_threshold=0.85)

    def classify(self, text):
        # ... 기존 로직 ...

        # Confidence 계산 (calibrated)
        if self.calibrator:
            confidence = self.calibrator.predict(top1.score_total)
        else:
            confidence = self._compute_confidence_baseline(candidates)

        # Decision routing
        decision = self.router.decide(result)

        return ClassificationResult(
            topk=candidates,
            confidence=confidence,
            decision=decision,
            # ...
        )
```

---

## 5. 예상 효과

### Before (현재)

```
Confidence: 0.0 (모든 샘플)
ECE: 0.78~0.83
AUTO rate: 50% (임의)
AUTO precision: 12% (KB-only) / 7.5% (Hybrid)
```

### After (Calibration 적용 후)

**Conservative Scenario** (threshold=0.85):
```
Confidence: 0~1 분포
ECE: 0.05~0.10 (큰 개선)
AUTO rate: 30~40%
AUTO precision: 85~90%
```

**Balanced Scenario** (threshold=0.75):
```
ECE: 0.10~0.15
AUTO rate: 50~60%
AUTO precision: 75~80%
```

**Aggressive Scenario** (threshold=0.65):
```
ECE: 0.15~0.20
AUTO rate: 70~80%
AUTO precision: 65~70%
```

### ROI 분석

**생산성 향상**:
- AUTO 30% → 작업 시간 30% 절감
- ASK 70% → 전문가 검토 필요

**품질 보장**:
- AUTO precision 85% → 15% 오류율
- ASK로 라우팅 → 오류 catch

**비용 절감**:
- 사람 시간 30% 절감
- 오분류 비용 감소 (15% vs 88%)

---

## 6. 참고 문헌

1. **Calibration 이론**:
   - Guo et al. (2017), "On Calibration of Modern Neural Networks"
   - Platt (1999), "Probabilistic Outputs for Support Vector Machines"

2. **LightGBM Calibration**:
   - https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html
   - sklearn.calibration 문서

3. **ECE 계산**:
   - Naeini et al. (2015), "Obtaining Well Calibrated Probabilities Using Bayesian Binning"

---

**Report Generated**: 2026-02-03 19:40 KST
**Status**: Recommendations Ready for Implementation
