# LegalGate Feature 진단 및 수정

**Date**: 2026-02-03
**Issue**: LegalGate features가 학습 데이터에서 모두 0 (버그)

## 진단 결과

### Feature 분포 분석 (168,677 샘플)

| Feature | Nonzero Rate | Min | Max | Nunique | 상태 |
|---------|--------------|-----|-----|---------|------|
| f_legal_scope_match_score | 0.43% (719) | 0.0 | 0.5 | 7 | ⚠️ 희소 |
| f_legal_heading_term | 0.00% (0) | 0.0 | 0.0 | 1 | ❌ 상수 0 |
| f_legal_include_support | 0.00% (0) | 0.0 | 0.0 | 1 | ❌ 상수 0 |
| f_legal_exclude_conflict | 0.00% (0) | 0.0 | 0.0 | 1 | ❌ 상수 0 |
| f_legal_redirect_penalty | 0.00% (0) | 0.0 | 0.0 | 1 | ❌ 상수 0 |

### 결론: **계산/저장 경로 버그 확인**

3개 주요 LegalGate features (heading_term, include_support, exclude_conflict)가 168,677개 샘플 모두에서 0.
→ 이는 "신호 부족" 문제가 아니라 **코드 버그**.

## 버그 원인

### 1. Debug Dict에 'results' 키 누락

**위치**: `src/classifier/legal_gate.py:170-180`

**문제**: `LegalGate.apply()`가 내부적으로 per-candidate `results` dict를 생성하지만, 이를 return하는 debug dict에 포함하지 않음.

```python
# LegalGate.apply() 내부 (line 92-98)
results: Dict[str, LegalGateResult] = {}
for cand in candidates:
    result = self._evaluate_candidate(input_text, input_norm, cand.hs4)
    results[cand.hs4] = result  # 여기서 계산됨

# 하지만 debug dict에는 미포함 (line 170-180)
debug = {
    'legal_gate_applied': True,
    'total_evaluated': len(candidates),
    ...
    # 'results': results  ← 이게 없음!
}
```

### 2. build_dataset_legal.py의 의존성

**위치**: `src/classifier/rank/build_dataset_legal.py:200-211`

**문제**: LegalGate features를 debug['results']에서 가져오려고 시도하지만, 이 키가 존재하지 않아 항상 else 분기 (모두 0)로 빠짐.

```python
legal_results = legal_debug.get('results', {})  # 빈 dict 반환

legal_result = legal_results.get(cand.hs4)  # None
if legal_result:  # False
    f_legal_heading_term = legal_result.get('heading_term_score', 0.0)
    ...
else:
    # 항상 여기로 옴
    f_legal_heading_term = 0.0  # ← 버그!
    f_legal_include_support = 0.0
    f_legal_exclude_conflict = 0.0
    f_legal_redirect_penalty = 0.0
```

## 수정 내용

### Fix: LegalGate debug에 results 추가

**파일**: `src/classifier/legal_gate.py`

**변경 전**:
```python
debug = {
    'legal_gate_applied': True,
    'total_evaluated': len(candidates),
    'passed': len(passed_candidates),
    'excluded': len(excluded_hs4s),
    'excluded_hs4s': excluded_hs4s,
    'exclude_reasons': exclude_reasons,
    'redirects_added': len(new_redirect_hs4s),
    'redirect_hs4s': new_redirect_hs4s,
    'pass_rate': len(passed_candidates) / len(candidates) if candidates else 0,
}
```

**변경 후**:
```python
# Per-candidate results를 debug에 포함 (ranker 학습용)
results_dict = {}
for hs4, result in results.items():
    results_dict[hs4] = {
        'passed': result.passed,
        'heading_term_score': result.heading_term_score,
        'include_support_score': result.include_support_score,
        'exclude_conflict_score': result.exclude_conflict_score,
        'redirect_penalty': result.redirect_penalty,
        'total_score': result.total_score(),
    }

debug = {
    'legal_gate_applied': True,
    'total_evaluated': len(candidates),
    'passed': len(passed_candidates),
    'excluded': len(excluded_hs4s),
    'excluded_hs4s': excluded_hs4s,
    'exclude_reasons': exclude_reasons,
    'redirects_added': len(new_redirect_hs4s),
    'redirect_hs4s': new_redirect_hs4s,
    'pass_rate': len(passed_candidates) / len(candidates) if candidates else 0,
    'results': results_dict,  # ← 추가!
}
```

## LegalGate 필터링 ~0% 문제

### 관찰
- `avg_candidates_before_legal`: 38.0
- `avg_candidates_after_legal`: 38.1
- **LegalGate 필터링 비율**: -0.3% (거의 필터링 안 함!)

### 원인 후보 3가지

#### 1. **호 용어 데이터 부족** (가장 유력)
- `self.heading_terms: Dict[str, List[str]] = {}`가 빈 dict로 초기화됨
- `# TODO: retriever에서 카드 데이터 가져오기` 주석만 있고 실제 로드 안 됨
- → heading_term_score가 항상 0이므로 필터링 안 됨

**증거**: `f_legal_heading_term`이 168,677개 샘플 모두에서 0.

#### 2. **주규정(Notes) include/exclude 매칭 실패**
- 주규정 로드는 됨 (1294 notes: include 104, exclude 278)
- 하지만 실제 product_name과 매칭이 안 됨
- → include_support_score, exclude_conflict_score도 0

**증거**: `f_legal_include_support`, `f_legal_exclude_conflict` 모두 0.

#### 3. **Hard exclude 임계값이 너무 높음**
- `should_hard_exclude()` 조건:
  - `exclude_conflict_score < -0.7` 또는
  - `redirect_penalty < -0.8`
- 모든 점수가 0이면 절대 exclude 안 됨

**증거**: `excluded_hs4s: []` (exclude된 후보 0개)

### 최소 변경 개선 실험 (원인 1 해결)

#### 선택한 원인: **호 용어 데이터 부족**

#### 개선안:
**위치**: `src/classifier/legal_gate.py:__init__()`

**변경 전**:
```python
def __init__(self, notes_loader: Optional[NotesLoader] = None):
    self.notes_loader = notes_loader or get_notes_loader()

    # KB 카드 데이터 (호 용어 매칭용)
    # TODO: retriever에서 카드 데이터 가져오기
    self.heading_terms: Dict[str, List[str]] = {}
```

**변경 후**:
```python
def __init__(self, notes_loader: Optional[NotesLoader] = None):
    self.notes_loader = notes_loader or get_notes_loader()

    # KB 카드 데이터 (호 용어 매칭용)
    self.heading_terms = self._load_heading_terms()

def _load_heading_terms(self) -> Dict[str, List[str]]:
    """HS4 호 용어 로드"""
    heading_terms = {}
    cards_path = Path("data/hs4_cards_v2_with_facts.json")

    if cards_path.exists():
        with open(cards_path, 'r', encoding='utf-8') as f:
            cards = json.load(f)

        for hs4, card in cards.items():
            terms = []

            # Heading text
            if 'heading' in card:
                terms.append(card['heading'])

            # Keywords
            if 'keywords' in card:
                terms.extend(card['keywords'])

            if terms:
                heading_terms[hs4] = terms

    return heading_terms
```

#### 기대 효과:
- `heading_term_score > 0`인 샘플 증가
- LegalGate 필터링 비율 증가 (0% → 5-10% 예상)
- Ranker feature importance 증가

#### 실험 플래그:
```python
use_heading_terms: bool = True  # 기본 on
```

## 신호 희소성 개선안 (원인 2, 3 해결)

### 개선 1: Query 텍스트 보강

**현재**: `product_name`만 사용
**개선**: `description`, `rationale` 필드도 concat (존재 시)

```python
# build_dataset_legal.py
product_name = case.get('product_name', '').strip()
description = case.get('product_description', '')
rationale = case.get('rationale', '')

# Concat
full_text = product_name
if description:
    full_text += ' ' + description
if rationale:
    full_text += ' ' + rationale

# LegalGate에 전달
legal_candidates, redirect_hs4s, legal_debug = legal_gate.apply(
    full_text,  # ← 보강된 텍스트
    kb_candidates
)
```

**플래그**: `use_full_text: bool = False` (기본 off, 실험용)

### 개선 2: Notes 매칭 강화

**현재**: Exact match 또는 단순 substring
**개선**: 형태소 + 동의어 확장

```python
# Notes 매칭 시
input_morphs = extract_morphs(input_text)  # 형태소 추출
note_morphs = extract_morphs(note_text)

# Morph overlap 계산
overlap = len(set(input_morphs) & set(note_morphs))
if overlap >= 2:  # 2개 이상 겹치면 매칭
    matched = True
```

**플래그**: `use_morph_matching: bool = False` (기본 off)

## 재현 단계

### 1. Ranker 데이터셋 재빌드
```bash
python -m src.classifier.rank.build_dataset_legal 100
```

**검증**:
```python
df = pd.read_csv('artifacts/ranker_legal/rank_features_legal.csv')
print((df['f_legal_heading_term'] != 0).sum())  # > 0이어야 함
```

### 2. 전체 데이터셋 빌드
```bash
python -m src.classifier.rank.train_ranker_legal --build
```

### 3. Feature 분포 재확인
```bash
python -c "
import pandas as pd
df = pd.read_csv('artifacts/ranker_legal/rank_features_legal.csv')

legal_features = ['f_legal_heading_term', 'f_legal_include_support', 'f_legal_exclude_conflict']
for feat in legal_features:
    nonzero_rate = (df[feat] != 0).sum() / len(df)
    print(f'{feat}: {nonzero_rate:.4f}')
"
```

**기대 결과** (heading_terms 로드 후):
- `f_legal_heading_term`: 10-30% nonzero
- `f_legal_include_support`: 5-15% nonzero
- `f_legal_exclude_conflict`: 1-5% nonzero

### 4. LightGBM 학습 및 Feature Importance 확인
```bash
pip install lightgbm
python -m src.classifier.rank.train_ranker_legal --train

# Feature importance 확인
python -c "
import lightgbm as lgb
model = lgb.Booster(model_file='artifacts/ranker_legal/model_legal.txt')
importance = model.feature_importance()
names = model.feature_name()

legal_idx = [i for i, n in enumerate(names) if 'legal' in n]
for idx in legal_idx:
    print(f'{names[idx]}: {importance[idx]}')
"
```

## 요약

### 버그 확인 ✓
- LegalGate features가 debug에 포함되지 않아 학습 데이터에 0으로 저장됨
- 168,677개 샘플 중 3개 주요 feature 모두 상수 0

### 수정 완료 ✓
- `legal_gate.py`: debug dict에 'results' 추가
- Per-candidate legal features 제공

### 근본 원인 (아직 미해결)
1. Heading terms 데이터 미로드 → 호 용어 매칭 실패
2. Notes 매칭 희소 → include/exclude 신호 부족
3. LegalGate 필터링 ~0% → ranker feature가 의미 없음

### 다음 단계
1. ✅ 버그 수정 완료 - debug에 results 추가
2. ⏳ Heading terms 로드 구현 (가장 효과 클 것으로 예상)
3. ⏳ 재빌드 및 feature 분포 재확인
4. ⏳ Feature importance 분석
5. ⏳ 필요 시 개선 2-3 적용 (플래그 기반)

---

**Status**: 버그 수정 완료, heading_terms 로드 구현 필요.
