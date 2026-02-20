"""
Text normalization and tokenization utilities for HS classification.
Handles Korean + English mixed text.
"""

import re
from typing import List, Set, Tuple

# Stopwords
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
    'including', 'excluding', 'whether', 'such', 'those', 'these',
    'product', 'products', 'item', 'items', 'article', 'articles',
    'type', 'types', 'kind', 'kinds', 'part', 'parts',
}

STOPWORDS = STOPWORDS_KO | STOPWORDS_EN


def normalize(text: str) -> str:
    """Normalize text: lowercase, remove brackets, clean special chars."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = re.sub(r'\[[^\]]*\]', ' ', text)
    text = re.sub(r'\{[^}]*\}', ' ', text)
    text = re.sub(r'(\d+)(인치|cm|mm|kg|g|ml|l|개|ea|pcs)', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize Korean/English mixed text."""
    if not text:
        return []
    norm = normalize(text)
    tokens = norm.split()

    expanded = []
    for token in tokens:
        expanded.append(token)
        korean_parts = re.findall(r'[가-힣]{2,}', token)
        for part in korean_parts:
            if part != token:
                expanded.append(part)

    if remove_stopwords:
        return [t for t in expanded if t not in STOPWORDS and len(t) >= 2]
    return [t for t in expanded if len(t) >= 1]


def extract_keywords(text: str, min_len: int = 2, max_count: int = 10) -> List[str]:
    """Extract keywords from text (stopwords removed, deduplicated)."""
    tokens = tokenize(text, remove_stopwords=True)
    seen = set()
    result = []
    for kw in tokens:
        if len(kw) >= min_len and kw not in seen:
            seen.add(kw)
            result.append(kw)
            if len(result) >= max_count:
                break
    return result


def simple_contains(text: str, term: str) -> bool:
    """Normalized substring match."""
    if not text or not term:
        return False
    norm_text = normalize(text)
    norm_term = normalize(term)
    return len(norm_term) >= 2 and norm_term in norm_text


def token_overlap(text: str, terms: List[str], min_overlap: int = 1) -> Tuple[bool, List[str]]:
    """Token intersection matching."""
    if not text or not terms:
        return False, []
    text_tokens = set(tokenize(text, remove_stopwords=True))
    matched = []
    for term in terms:
        term_tokens = set(tokenize(term, remove_stopwords=True))
        overlap = text_tokens & term_tokens
        if overlap:
            matched.extend(list(overlap))
    matched = list(set(matched))
    return len(matched) >= min_overlap, matched


def fuzzy_match(text: str, term: str) -> Tuple[bool, str]:
    """Fuzzy match: substring or token overlap."""
    if simple_contains(text, term):
        return True, "substring"
    is_match, matched = token_overlap(text, [term], min_overlap=1)
    if is_match:
        return True, f"token:{','.join(matched)}"
    return False, ""
