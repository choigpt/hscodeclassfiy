# HS 품목분류 시스템

HS 코드 품목분류 파이프라인: 결정사례 + 구조화된 지식베이스 기반

## 구조

```
HS/
├── data/ruling_cases/           # 결정사례 (7,198개)
├── kb/
│   ├── raw/                     # 해설서 원본
│   ├── structured/              # 구조화 산출물
│   │   ├── hs4_cards.jsonl      # 1,240개 HS4 카드
│   │   ├── hs4_rule_chunks.jsonl # 11,912개 규칙 청크
│   │   ├── thesaurus_terms.jsonl # 7,098개 용어
│   │   └── disambiguation_questions.jsonl
│   └── build_scripts/           # KB 빌드 스크립트
├── artifacts/classifier/        # 학습된 모델
└── src/classifier/              # 분류 파이프라인
```

## 빠른 시작

```bash
# 분류 실행
python -m src.classifier.pipeline "냉동 돼지 삼겹살"
```

출력:
```
[1] HS 0203
    총점: 0.6031 (ML: 0.1031, Card: 0.0000, Rule: 1.0000)
    근거:
      - [include_rule] 삼겹살(streaky pork)도 분류한다

저신뢰도: True

추가 질문:
  - 가공 상태가 어떻게 되나요?
  - 물품의 형태가 어떻게 되나요?
```

## 파이프라인 구조

```
입력(품명) → Retriever(Top-50) → Reranker(카드/규칙) → Top-5 + 근거
                                                    ↓
                                          저신뢰도 → 질문 생성
```

1. **Retriever**: SentenceTransformer 임베딩 + LR로 Top-50 후보 생성
2. **Reranker**: HS4 카드/규칙 청크 기반 재정렬 + 근거 생성
3. **Clarifier**: 저신뢰도 시 추가 질문 제공

## 모델 학습

```bash
python -c "from src.classifier.retriever import HSRetriever; r = HSRetriever(); r.train_model()"
```

## KB 빌드

```bash
python kb/build_scripts/build_cards.py
python kb/build_scripts/build_chunks.py
python kb/build_scripts/build_thesaurus.py
python kb/build_scripts/build_questions.py
```

## 요구사항

```bash
pip install sentence-transformers scikit-learn joblib numpy
```

## 데이터

- **결정사례**: 관세평가분류원 품목분류 결정 7,198건
- **해설서**: WCO HS 해설서 1,240개 호
- **기준**: 2022년

---
**업데이트**: 2026-01-28
