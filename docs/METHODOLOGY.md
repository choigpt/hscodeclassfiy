# HS4 Classification Methodology

## 연구 질문 (Research Question)

> "구조화된 법적/해설서 기반 KB(규칙/카드/8축 속성)가, 짧은 품명 텍스트 HS4 분류에서
> 베이스라인 모델 대비 정확도와 신뢰도(calibration), 그리고 저신뢰도 라우팅 품질을 얼마나 개선하는가?"

---

## 1. 시스템 아키텍처

### 1.1 전체 파이프라인

```
입력 (품명 텍스트 or ClassificationInput)
        │
        ▼
┌───────────────────┐
│  GRI 신호 탐지    │  ← 통칙 2(a), 2(b), 3, 5 신호
│  + 8축 속성 추출  │  ← 물체본질, 재질, 가공상태, 기능용도, ...
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  ML Top-K 후보    │  ← SBert + Logistic Regression
│  + KB Top-K 후보  │  ← 카드/규칙 키워드 매칭
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Merge + Reranking │  ← KB-first merge + LightGBM ranker
└───────────────────┘
        │
        ▼
╔═══════════════════════════════╗
║ GRI 순차 오케스트레이터       ║
╠═══════════════════════════════╣
║ GRI 1: LegalGate (exclude/   ║
║        redirect)              ║
║        ↓                      ║
║ GRI 2: 미완성(2a)/혼합(2b)    ║
║        ↓                      ║
║ GRI 3: 3(a) specificity       ║
║        3(b) Essential Character║
║        3(c) last-in-order     ║
║        ↓                      ║
║ GRI 5: 용기/포장              ║
║        ↓                      ║
║ GRI 6: HS4→HS6 서브헤딩       ║
║        ↓                      ║
║ Risk: 리스크 평가              ║
╚═══════════════════════════════╝
        │
        ▼
┌───────────────────┐
│  신뢰도 판정      │  ← AUTO / ASK / REVIEW
│  + 질문 생성      │  ← 저신뢰도시 추가 정보 요청
└───────────────────┘
        │
        ▼
     최종 결과
  (Top-K + HS6 + GRI 이력 + EC + Risk + Rule Ref + Case Evidence)
```

### 1.2 주요 컴포넌트

| 컴포넌트 | 파일 | 역할 |
|----------|------|------|
| Pipeline | `pipeline.py` | 전체 파이프라인 통합 |
| **GRI Orchestrator** | `gri_orchestrator.py` | **GRI 순차 오케스트레이터 (핵심)** |
| **Essential Character** | `essential_character.py` | **GRI 3(b) 4요소 가중 합산** |
| **Subheading Resolver** | `subheading_resolver.py` | **GRI 6 HS4→HS6 해소** |
| **Risk Assessor** | `risk_assessor.py` | **리스크 평가 (LOW/MED/HIGH)** |
| Input Model | `input_model.py` | 구조화 입력 (재질 구성비, 세트 여부 등) |
| Retriever | `retriever.py` | SBert 임베딩 + LR 기반 Top-K 후보 생성 |
| Reranker | `reranker.py` | KB 카드/규칙 매칭 + rule_id 추적 |
| GRI Signals | `gri_signals.py` | 통칙 신호 탐지 |
| Attributes | `attribute_extract.py` | 8축 전역 속성 추출 |
| LegalGate | `legal_gate.py` | Tariff notes 기반 GRI 1 필터 |
| Clarifier | `clarify.py` | 저신뢰도 판정 및 질문 생성 |

---

## 2. Knowledge Base (KB) 구조

### 2.1 HS4 카드 (`hs4_cards.jsonl`)

각 HS 4자리 호에 대한 구조화된 정보:

```json
{
  "hs4": "0203",
  "title_ko": "돼지고기(신선한 것, 냉장하거나 냉동한 것으로 한정한다)",
  "includes": ["삼겹살", "등심", "안심", "갈비", ...],
  "excludes": ["가공한 것", "조리한 것"],
  "scope": ["냉장", "냉동", "신선"],
  "decision_attributes": {
    "processing_state": ["fresh", "chilled", "frozen"],
    "material": ["pork"]
  }
}
```

### 2.2 규칙 청크 (`hs4_rule_chunks.jsonl`)

해설서 기반 분류 규칙:

```json
{
  "hs4": "0203",
  "chunk_type": "include_rule",
  "text": "뼈가 있거나 없는 것으로서 염장한 것은 제외한다",
  "signals": ["뼈", "염장"],
  "polarity": "include",
  "strength": "soft",
  "quant_rule": null
}
```

### 2.3 8축 전역 속성

| 축 ID | 축 이름 | 예시 값 |
|-------|--------|---------|
| object_nature | 물체 본질 | device, substance, organism |
| material | 재질 | metal, plastic, leather |
| processing_state | 가공 상태 | fresh, frozen, processed |
| function_use | 기능/용도 | food, industrial, medical |
| physical_form | 물리적 형태 | liquid, solid, powder |
| completeness | 완성도 | complete, parts, assembly |
| quantitative_rules | 정량 규칙 | 50% 이상, 1kg 미만 |
| legal_scope | 법적 범위 | this_heading, this_chapter |

---

## 3. 평가 방법론

### 3.1 데이터 분할

- **Train (70%)**: 모델 학습
- **Validation (15%)**: 하이퍼파라미터 튜닝
- **Test (15%)**: 최종 평가

분할 전략:
- Stratified split (클래스 비율 유지)
- 최소 3개 샘플 이상 클래스만 포함

### 3.2 평가 지표

#### Core Metrics
- **Top-1/3/5 Accuracy**: 상위 K개 예측에 정답 포함 비율
- **Macro F1**: 클래스 불균형 고려 F1 점수
- **Weighted F1**: 샘플 수 가중 F1 점수
- **Coverage**: 모델이 예측한 클래스 비율

#### Calibration Metrics
- **ECE (Expected Calibration Error)**: 예측 신뢰도와 실제 정확도 차이
- **Brier Score**: 확률 예측 정확도
- **Reliability Curve**: 신뢰도 구간별 정확도 곡선

#### Routing Metrics
- **AUTO/ASK/REVIEW 비율**: 신뢰도 구간별 샘플 분포
- **라우팅별 정확도**: 각 라우팅의 Top-1 정확도
- **저신뢰도 Top-3 Hit Rate**: ASK/REVIEW에서 Top-3 정답 포함 비율

---

## 4. Ablation 실험 설계

### 4.1 모델 구성

| Model | use_gri | use_8axis | use_rules | use_ranker | Description |
|-------|---------|-----------|-----------|------------|-------------|
| B0 | - | - | - | - | TF-IDF + LR |
| B1 | - | - | - | - | SBert + LR |
| B2 | - | - | - | - | BM25 |
| P1 | X | X | X | X | ML only |
| P2 | O | X | X | X | +GRI signals |
| P3 | O | O | X | X | +8축 속성 |
| P4 | O | O | O | X | +KB 규칙 |
| P5 | O | O | O | O | +LGB ranker |
| P6 | O | O | O | O | Full (질문 포함) |

### 4.2 분석 관점

1. **GRI 신호의 효과**: P1 vs P2
2. **8축 속성의 효과**: P2 vs P3
3. **KB 규칙의 효과**: P3 vs P4
4. **Ranker의 효과**: P4 vs P5
5. **베이스라인 대비 개선**: B1 vs P5

---

## 5. 오류 분석

### 5.1 Confusion Pairs

자주 혼동되는 HS4 쌍 분석:
- 같은 류(Chapter) 내 오류
- 다른 류 간 오류

### 5.2 오류 유형

- **Chapter Error**: 류(2자리) 오류
- **Heading Error**: 같은 류, 다른 호(4자리)

### 5.3 속성별 분석

- 재질 매칭 실패 케이스
- 가공상태 매칭 실패 케이스
- 완성도 판단 실패 케이스

---

## 6. 재현성

### 6.1 환경

```
Python 3.10+
sentence-transformers
scikit-learn
lightgbm
rank-bm25
PyYAML
```

### 6.2 실행 방법

```bash
# 데이터 분할
python -m src.experiments.data_split --config configs/benchmark.yaml

# 전체 벤치마크
python -m src.experiments.run_benchmark --config configs/benchmark.yaml

# 특정 Ablation만
python -m src.experiments.run_benchmark --ablation P5_plus_ranker
```

### 6.3 출력 파일

| 파일 | 내용 |
|------|------|
| `benchmark_summary.csv` | 전체 결과 요약 |
| `ablation_table.csv` | Ablation 비교표 |
| `confusion_pairs.csv` | 상위 혼동 쌍 |
| `calibration.json` | Calibration 데이터 |
| `failure_cases.jsonl` | 실패 케이스 상세 |

---

## 7. LightGBM Ranker 학습 검증 (2026-02-08)

### 7.1 f_lexical 정규화

**변경**: `reranker.py`에서 kb_score를 `log1p(x)/log1p(30)` + clamp(1.0)으로 [0,1] 정규화

**검증 결과**:
- CSV 반영 확인: raw [0, 39] mean=2.80 → normalized [0, 1.0] mean=0.354
- **LightGBM gain 불변**: tree-based 모델은 단조 변환(monotonic transform)에 불변
  - 동일한 분할 경계 선택 → gain, NDCG, accuracy 모두 동일
- **Fallback weighted-score 경로 수정됨**: max 기여 5.85 → 0.15

### 7.2 Metric 정의 검증

- **Top-K Accuracy**: query별 정답 HS4가 예측 top-K에 포함되는 비율 (HS4 분류 정확도)
- candidate recall@K가 아닌 **분류 정확도**임을 확인
- 평가 단위: query = 결정사례 1건, label=1 if HS4 matches ground truth

### 7.3 Train/Test Leakage

| Item | Value |
|------|-------|
| Train queries | 5,757 |
| Test queries | 1,439 |
| Text overlap (exact match) | 48건 (3.4%) |
| Severity | Low (동일 품명, 다른 결정사례, query_id 분리) |

### 7.4 Feature Importance

| Rank | Feature | Gain | Ratio |
|------|---------|------|-------|
| 1 | f_lexical | 251,890 | 86.8% |
| 2 | f_specificity | 5,435 | 1.9% |
| 3 | f_form_match_score | 5,122 | 1.8% |
| 4 | f_material_match_score | 4,975 | 1.7% |
| 5 | f_card_hits | 3,307 | 1.1% |

### 7.5 Dominance 완화 실험

| | Baseline | Exp A (no f_lexical) | Exp B (regularized) |
|---|----------|---------------------|---------------------|
| Test Top-1 | 0.7661 | 0.3894 | 0.7703 |
| NDCG@5 | 0.8716 | 0.3079 | 0.8691 |
| f_lexical ratio | 86.8% | N/A | 86.3% |

**결론**: f_lexical은 정보량 자체가 지배적 (제거시 NDCG@5 0.59 drop). 단조 변환/파라미터 튜닝으로 해소 불가. `feature_interaction_constraints`, 2-stage ranker, `max_bin` 축소 등 구조적 접근 필요.

---

## 8. GRI 순차 오케스트레이터 (2026-02-20)

### 8.1 GRI 순차 적용 원칙

WCO 통칙(General Rules of Interpretation)은 반드시 **GRI 1→2→3→5→6** 순서로 적용해야 한다.
이전 파이프라인은 GRI 신호를 동시에 감지하여 ML 피처로 사용했으나, 이번 개편에서
GRI 통칙의 법적 적용 순서를 강제하는 오케스트레이터를 구현했다.

```
GRI 1: 호의 용어 + 주규정으로 Chapter/Heading 결정
       → LegalGate 기반 hard exclude + redirect
       → 단일 후보 시 즉시 확정

GRI 2: 미완성/혼합물 처리
       → 2a: 미조립/불완전 → 완성품 호에 분류 가능 확장
       → 2b: 혼합물 → 구성 재질별 후보 확장

GRI 3: 복수 후보 해소
       → 3(a): 가장 구체적 호 우선 (specificity scoring)
       → 3(b): Essential Character 4요소 모델
       → 3(c): 수 순서 최말위 (fallback)

GRI 5: 용기/포장 처리
       → 내용물 기준 분류

GRI 6: 소호 세분화
       → HS4 확정 후 HS6 결정
```

### 8.2 Essential Character (GRI 3b) 모델

4요소 가중 합산으로 복합 물품의 본질적 특성 결정:

| 요소 | 가중치 | 판단 기준 | 데이터 소스 |
|------|--------|----------|------------|
| core_function | 0.35 | 물품의 핵심 기능이 해당 호의 본래 기능과 일치하는가 | function_use 축, 카드 keywords |
| user_perception | 0.25 | 일반 사용자가 해당 물품을 어떤 호의 물품으로 인식하는가 | object_nature 축, heading title |
| area_volume | 0.20 | 구성 재질 중 면적/부피 비율이 지배적인가 | material 축, ClassificationInput.materials |
| structural | 0.20 | 해당 호의 구조적 특성을 지배하는가 | completeness 축, 카드 decision_attributes |

각 후보 HS4에 대해 4요소 점수를 산출하고, 가중 합산하여 winner 결정.
Essential Character가 적용된 경우 리스크 점수 +2.0 부여.

### 8.3 HS6 서브헤딩 분류 (GRI 6)

HS4 확정 후 2,822개 HS6 후보에서 최적 서브헤딩 결정:

1. **키워드 매칭**: 입력 텍스트 ↔ HS6 title/keywords 토큰 오버랩
2. **소호 주규정 적용**: 122개 subheading notes 중 해당 HS4 산하 주규정 매칭
3. **재질 구성비**: ClassificationInput.materials 비율과 HS6 재질 키워드 매칭
4. **판결 선례**: 정규화된 7,198건 판결 중 동일 HS4 + 유사 텍스트 선례의 HS6 참조

### 8.4 리스크 평가

5개 리스크 요인으로 분류 신뢰도 종합 판단:

| 요인 | 조건 | 점수 |
|------|------|------|
| 점수 차이 | Top1-Top2 gap < 0.15 | +3.0 |
| 점수 차이 | gap 0.15~0.30 | +1.5 |
| GRI 3(b) | Essential Character 적용됨 | +2.0 |
| 정보 누락 | 핵심 axis (material, processing_state, function_use, completeness) 2개+ 미감지 | +1.5/개 |
| 판례 충돌 | 동일 상품에 다른 코드 판결 존재 | +2.0 |
| 관할 분기 | 다른 관할권에서 다른 분류 가능성 | +1.0 |

레벨: **HIGH** (≥5.0) / **MED** (2.5~5.0) / **LOW** (<2.5)

### 8.5 판결 케이스 정규화

7,198건 관세청 결정사례를 구조화된 스키마로 변환:

```json
{
  "case_id": "KR_2025_ref_number",
  "jurisdiction": "KR",
  "hs_version": "2022",
  "final_code_4": "5515",
  "final_code_6": "551512",
  "final_code_10": "5515129000",
  "rejected_codes_6": [],
  "applied_gri": ["GRI1", "GRI6"],
  "decisive_reasoning": ["...핵심 결론문..."],
  "confidence": 0.85
}
```

두 가지 모드:
- **Mode A (규칙 기반)**: regex 패턴으로 HS코드, GRI, rejected codes, decisive reasoning 파싱
- **Mode B (하이브리드)**: Mode A + Ollama LLM 보완 (저확신 케이스)

정규화 품질: 평균 확신도 0.842, GRI 정보 96.2% 추출, HS10 99.1% 추출

### 8.6 Rule ID 추적성

모든 규칙 청크에 결정론적 rule_id 부여:
- 포맷: `{hs4}_{chunk_type}_{sha256(text[:100])[:8]}`
- source: chunk_type에서 추론 (explanatory_note, include_rule, exclude_rule 등)
- hs_version: "2022"

규칙 참조 경로: KB rules → Reranker Evidence.meta → GRI Orchestrator → ClassificationResult.rule_references

### 8.7 출력 스키마 (8개 필수 항목)

| 항목 | 타입 | 설명 |
|------|------|------|
| input_text | str | 입력 텍스트 |
| topk | List[Candidate] | Top-K 후보 (hs4 + hs6 포함) |
| decision | Decision | 분류 결정 (AUTO/ASK/REVIEW/ABSTAIN) |
| applied_gri | List[GRIApplication] | GRI 적용 이력 (순차 기록) |
| essential_character | EssentialCharacterResult | EC 판정 결과 |
| risk | RiskAssessment | 리스크 등급 + 요인 |
| rule_references | List[RuleReference] | 적용된 법적 규칙 참조 |
| case_evidence | List[CaseEvidence] | 참조된 판결 증거 |

---

## 9. 한계 및 향후 연구

### 9.1 현재 한계

1. **데이터 의존성**: 결정사례 데이터 품질에 의존
2. **언어 제한**: 한국어 품명만 지원
3. **실시간 처리**: SBert 임베딩 비용

### 9.2 향후 연구 방향

1. 다국어 확장 (영어, 중국어)
2. Few-shot/Zero-shot HS 분류
3. LLM 기반 설명 생성
4. Active Learning 통합
