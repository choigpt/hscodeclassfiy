"""
텍스트 정규화 및 매칭 유틸리티 (개선판)
"""

import re
from typing import List, Set, Tuple


# 불용어 (범용어, 매칭에서 제외)
STOPWORDS_KO = {
    '것', '등', '및', '의', '이', '가', '을', '를', '에', '로', '으로',
    '한', '하는', '된', '되는', '있는', '없는', '같은', '위한', '대한',
    '또는', '그', '이러한', '해당', '경우', '때', '수', '더', '매우',
    '분류', '포함', '제외', '규정', '따라', '관세', '율표',
    '기타', '제품', '부품', '용', '그밖의', '밖의', '기타의',
    '물품', '상품', '품목', '종류', '형태', '방식',
}

STOPWORDS_EN = {
    'the', 'a', 'an', 'of', 'for', 'and', 'or', 'in', 'on', 'at', 'to',
    'with', 'from', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'been',
    'other', 'others', 'etc', 'nes', 'nec', 'not', 'elsewhere', 'specified',
    'including', 'excluding', 'whether', 'whether', 'such', 'those', 'these',
    'product', 'products', 'item', 'items', 'article', 'articles',
    'type', 'types', 'kind', 'kinds', 'part', 'parts',
}

STOPWORDS = STOPWORDS_KO | STOPWORDS_EN


def normalize(text: str) -> str:
    """
    텍스트 정규화 (개선판)
    - 소문자화 (영문)
    - 괄호 및 내용 제거
    - 특수문자 제거
    - 숫자/단위 분리
    - 다중 공백 정리
    """
    if not text:
        return ""

    # 소문자화
    text = text.lower()

    # 괄호 및 내용 제거
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = re.sub(r'\[[^\]]*\]', ' ', text)
    text = re.sub(r'\{[^}]*\}', ' ', text)

    # 숫자+단위 분리 (예: "55인치" -> "55 인치")
    text = re.sub(r'(\d+)(인치|cm|mm|kg|g|ml|l|개|ea|pcs)', r'\1 \2', text, flags=re.IGNORECASE)

    # 특수문자 제거 (한글, 영문, 숫자, 공백만 유지)
    text = re.sub(r'[^\w\s가-힣]', ' ', text)

    # 다중 공백 제거
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    토큰화 (한글/영문 통합)
    """
    if not text:
        return []

    # 정규화
    norm = normalize(text)

    # 공백 기준 분할
    tokens = norm.split()

    # 추가: 한글 단어 내 분절 (복합어 처리)
    expanded = []
    for token in tokens:
        expanded.append(token)
        # 한글 2글자 이상 연속이면 추가
        korean_parts = re.findall(r'[가-힣]{2,}', token)
        for part in korean_parts:
            if part != token:
                expanded.append(part)

    # 불용어 제거
    if remove_stopwords:
        tokens = [t for t in expanded if t not in STOPWORDS and len(t) >= 2]
    else:
        tokens = [t for t in expanded if len(t) >= 1]

    return tokens


def tokenize_korean(text: str) -> List[str]:
    """한글 단어만 추출"""
    if not text:
        return []
    return re.findall(r'[가-힣]{2,}', text.lower())


def tokenize_english(text: str) -> List[str]:
    """영문 단어만 추출"""
    if not text:
        return []
    return re.findall(r'[a-z]{2,}', text.lower())


def simple_contains(text: str, term: str) -> bool:
    """
    정규화 후 substring 매칭
    """
    if not text or not term:
        return False

    norm_text = normalize(text)
    norm_term = normalize(term)

    if len(norm_term) < 2:
        return False

    return norm_term in norm_text


def token_overlap(text: str, terms: List[str], min_overlap: int = 1) -> Tuple[bool, List[str]]:
    """
    토큰 교집합 매칭

    Args:
        text: 입력 텍스트
        terms: 매칭할 용어 리스트
        min_overlap: 최소 교집합 크기

    Returns:
        (매칭 여부, 매칭된 토큰 리스트)
    """
    if not text or not terms:
        return False, []

    text_tokens = set(tokenize(text, remove_stopwords=True))

    matched = []
    for term in terms:
        term_tokens = set(tokenize(term, remove_stopwords=True))
        overlap = text_tokens & term_tokens
        if overlap:
            matched.extend(list(overlap))

    matched = list(set(matched))  # 중복 제거

    return len(matched) >= min_overlap, matched


def fuzzy_match(text: str, term: str) -> Tuple[bool, str]:
    """
    퍼지 매칭 (substring + token overlap)
    둘 중 하나라도 매칭되면 True

    Returns:
        (매칭 여부, 매칭 방식)
    """
    # 1. Substring 매칭
    if simple_contains(text, term):
        return True, "substring"

    # 2. Token overlap 매칭
    is_match, matched = token_overlap(text, [term], min_overlap=1)
    if is_match:
        return True, f"token:{','.join(matched)}"

    return False, ""


def extract_keywords(text: str, min_len: int = 2, max_count: int = 10) -> List[str]:
    """
    텍스트에서 키워드 추출 (불용어 제외)
    """
    tokens = tokenize(text, remove_stopwords=True)

    # 길이 필터링
    keywords = [t for t in tokens if len(t) >= min_len]

    # 중복 제거 및 개수 제한
    seen = set()
    result = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            result.append(kw)
            if len(result) >= max_count:
                break

    return result
