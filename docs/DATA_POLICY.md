# Data Policy & Licensing

## 1. 데이터 분류

### 1.1 공개 가능 데이터 (MIT License)

| 데이터 | 설명 | 라이선스 |
|--------|------|----------|
| 소스 코드 | 전체 Python 코드 | MIT |
| 벤치마크 메타데이터 | `splits.json` | MIT |
| 평가 결과 | CSV, JSON 결과 파일 | MIT |
| 익명화 샘플 데이터 | 테스트용 예시 | MIT |
| 설정 파일 | YAML 설정 | MIT |

### 1.2 비공개 데이터

| 데이터 | 사유 | 대안 |
|--------|------|------|
| 결정사례 원본 | 관세청 저작권 | 요청 시 연구 목적 제공 안내 |
| KB 원본 | 해설서 파생물 | 구조 및 스키마만 공개 |
| 브랜드/회사명 포함 품명 | 개인정보/상업적 민감성 | 익명화 버전 제공 |

---

## 2. 라이선스

### 2.1 코드 라이선스 (MIT)

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 2.2 데이터 사용 조건

1. **연구 목적**: 학술 연구 및 교육 목적으로만 사용
2. **재배포 금지**: 원본 데이터 재배포 불가
3. **출처 표기**: 논문/보고서 작성 시 출처 명시
4. **상업적 이용 금지**: 상업적 목적 이용 시 별도 협의 필요

---

## 3. 익명화 정책

### 3.1 품명 익명화

원본:
```
"삼성전자 갤럭시 S24 Ultra 스마트폰 256GB"
```

익명화:
```
"[브랜드] [모델명] 스마트폰 256GB"
```

### 3.2 식별정보 제거

제거 대상:
- 회사명/브랜드명
- 모델번호
- 시리얼 번호
- 특정 개인 식별 정보

### 3.3 익명화 방법

```python
# 패턴 기반 익명화
BRAND_PATTERNS = [
    r'삼성전자?', r'LG전자?', r'현대', r'기아', ...
]

def anonymize_text(text: str) -> str:
    for pattern in BRAND_PATTERNS:
        text = re.sub(pattern, '[브랜드]', text)
    return text
```

---

## 4. 재현성 보장

### 4.1 공개 정보

| 항목 | 공개 범위 |
|------|----------|
| 학습 스크립트 | 전체 코드 |
| 하이퍼파라미터 | `configs/*.yaml` |
| 분할 설정 | `splits.json` (ID만) |
| 평가 코드 | 전체 코드 |
| 평가 결과 | 집계 통계 |

### 4.2 비공개 정보

| 항목 | 대안 |
|------|------|
| 원본 데이터 | 요청 시 제공 안내 |
| 전체 예측 결과 | 집계 통계로 대체 |
| 개별 오류 케이스 | 익명화 버전 |

---

## 5. 데이터 요청 절차

### 5.1 연구 목적 데이터 요청

1. **연락처**: [연구자 이메일]
2. **필요 정보**:
   - 소속 기관
   - 연구 목적
   - 사용 범위
   - 데이터 보호 계획

3. **제공 형태**:
   - 익명화된 샘플 데이터
   - 또는 전체 데이터 (협약 체결 후)

### 5.2 협약 내용

- 데이터 사용 목적 명시
- 재배포 금지 조항
- 결과물 공유 조건
- 데이터 삭제 기한

---

## 6. 관련 법규

### 6.1 적용 법규

- 저작권법 (해설서 파생물)
- 개인정보보호법 (품명 내 식별정보)
- 관세법 (결정사례 활용)

### 6.2 준수 사항

1. **저작권**: 해설서 직접 인용 최소화, 구조화된 정보만 추출
2. **개인정보**: 식별정보 익명화 또는 제거
3. **영업비밀**: 기업 특정 정보 비공개

---

## 7. 연락처

- **기술 문의**: GitHub Issues
- **데이터 문의**: [이메일]
- **라이선스 문의**: [이메일]

---

## 변경 이력

| 버전 | 날짜 | 내용 |
|------|------|------|
| 1.0 | 2024-01 | 최초 작성 |
