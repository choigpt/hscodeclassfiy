"""
RAG 인덱스 빌더

KB 데이터로부터 BM25 corpus, SBERT 임베딩, 사례 임베딩, 시소러스 룩업을 생성하여
artifacts/rag_index/에 저장한다.

Usage:
    python -m src.rag.index_builder
"""

import json
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

from src.classifier.utils_text import normalize, tokenize


# 경로 상수
KB_CARDS_PATH = "kb/structured/hs4_cards.jsonl"
KB_CARDS_V2_PATH = "kb/structured/hs4_cards_v2.jsonl"
KB_RULES_PATH = "kb/structured/hs4_rule_chunks.jsonl"
THESAURUS_PATH = "kb/structured/thesaurus_terms.jsonl"
CASES_PATH = "data/ruling_cases/all_cases_full_v7.json"

INDEX_DIR = "artifacts/rag_index"

SBERT_MODEL_NAME = "jhgan/ko-sroberta-multitask"


def load_hs4_cards() -> Dict[str, Dict]:
    """HS4 카드 로드 (v2 우선, 없으면 v1)"""
    cards_path = Path(KB_CARDS_V2_PATH)
    if not cards_path.exists():
        cards_path = Path(KB_CARDS_PATH)

    cards = {}
    with open(cards_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            card = json.loads(line)
            hs4 = card.get('hs4')
            if hs4:
                cards[hs4] = card
    print(f"[IndexBuilder] HS4 카드 로드: {len(cards)}건 ({cards_path})")
    return cards


def load_rule_chunks() -> Dict[str, List[Dict]]:
    """HS4별 규칙 청크 로드"""
    rules_path = Path(KB_RULES_PATH)
    if not rules_path.exists():
        print(f"[IndexBuilder] Warning: {rules_path} not found")
        return {}

    rules = {}
    count = 0
    with open(rules_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            hs4 = chunk.get('hs4')
            if hs4:
                if hs4 not in rules:
                    rules[hs4] = []
                rules[hs4].append(chunk)
                count += 1
    print(f"[IndexBuilder] 규칙 청크 로드: {count}건 ({len(rules)} HS4)")
    return rules


def load_thesaurus() -> Dict[str, List[str]]:
    """시소러스 term→aliases 룩업 테이블 로드"""
    thesaurus_path = Path(THESAURUS_PATH)
    if not thesaurus_path.exists():
        print(f"[IndexBuilder] Warning: {thesaurus_path} not found")
        return {}

    lookup = {}
    with open(thesaurus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            term = entry.get('term', '').strip()
            aliases = entry.get('aliases', [])
            if term and aliases:
                # 정규화된 term → aliases (원본 형태)
                norm_term = normalize(term)
                if norm_term:
                    lookup[norm_term] = aliases
                # aliases도 역방향으로 등록
                for alias in aliases:
                    norm_alias = normalize(alias)
                    if norm_alias and norm_alias != norm_term:
                        if norm_alias not in lookup:
                            lookup[norm_alias] = [term]
                        elif term not in lookup[norm_alias]:
                            lookup[norm_alias].append(term)
    print(f"[IndexBuilder] 시소러스 로드: {len(lookup)}개 엔트리")
    return lookup


def load_ruling_cases() -> List[Dict]:
    """결정사례 로드"""
    cases_path = Path(CASES_PATH)
    if not cases_path.exists():
        print(f"[IndexBuilder] Warning: {cases_path} not found")
        return []

    with open(cases_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    print(f"[IndexBuilder] 결정사례 로드: {len(cases)}건")
    return cases


def _flatten_field(items: list) -> List[str]:
    """카드 필드를 문자열 리스트로 변환 (str or dict 혼재 대응)"""
    result = []
    for item in items:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            # dict인 경우 reason/text/value 등에서 텍스트 추출
            for key in ('reason', 'text', 'value', 'description'):
                val = item.get(key, '')
                if val:
                    result.append(str(val))
                    break
            else:
                # 키가 없으면 전체를 문자열화
                result.append(str(item))
    return result


def build_bm25_corpus(
    cards: Dict[str, Dict],
    rules: Dict[str, List[Dict]]
) -> Tuple[List[str], List[str], List[List[str]]]:
    """
    BM25 검색을 위한 코퍼스 구축

    각 HS4 카드를 title + scope + includes + 규칙 텍스트를 결합한 하나의 문서로 만든다.

    Returns:
        hs4_ids: HS4 코드 리스트
        documents: 원본 문서 리스트
        tokenized_corpus: 토큰화된 문서 리스트 (BM25 입력용)
    """
    hs4_ids = []
    documents = []
    tokenized_corpus = []

    for hs4, card in sorted(cards.items()):
        parts = []

        # title
        title = card.get('title_ko', '')
        if title:
            parts.append(title)

        # scope keywords
        scope = card.get('scope', [])
        if scope:
            parts.append(' '.join(_flatten_field(scope)))

        # includes
        includes = card.get('includes', [])
        if includes:
            parts.append(' '.join(_flatten_field(includes)))

        # excludes (제외 정보도 검색에 포함 - 어떤 것이 이 호에서 제외되는지 아는 것이 유용)
        excludes = card.get('excludes', [])
        if excludes:
            parts.append(' '.join(_flatten_field(excludes)))

        # key_attributes
        key_attrs = card.get('key_attributes', [])
        if key_attrs:
            parts.append(' '.join(_flatten_field(key_attrs)))

        # 규칙 청크 텍스트 (상위 3개만 - 너무 길면 노이즈)
        rule_chunks = rules.get(hs4, [])
        for chunk in rule_chunks[:3]:
            chunk_text = chunk.get('text', '')
            if chunk_text:
                parts.append(chunk_text[:200])  # 청크당 최대 200자

        # 문서 결합
        doc_text = ' '.join(parts)
        tokens = tokenize(doc_text, remove_stopwords=True)

        if tokens:
            hs4_ids.append(hs4)
            documents.append(doc_text)
            tokenized_corpus.append(tokens)

    print(f"[IndexBuilder] BM25 코퍼스: {len(hs4_ids)}개 문서, 평균 토큰 수: {np.mean([len(t) for t in tokenized_corpus]):.1f}")
    return hs4_ids, documents, tokenized_corpus


def build_sbert_embeddings(
    documents: List[str],
    model_name: str = SBERT_MODEL_NAME,
    batch_size: int = 64
) -> np.ndarray:
    """SBERT 임베딩 생성"""
    from sentence_transformers import SentenceTransformer

    print(f"[IndexBuilder] SBERT 모델 로드: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[IndexBuilder] {len(documents)}개 문서 인코딩 중...")
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    print(f"[IndexBuilder] SBERT 임베딩: {embeddings.shape}")
    return embeddings


def build_case_embeddings(
    cases: List[Dict],
    model_name: str = SBERT_MODEL_NAME,
    batch_size: int = 64
) -> Tuple[np.ndarray, List[Dict]]:
    """결정사례 product_name 임베딩 생성"""
    from sentence_transformers import SentenceTransformer

    # 유효한 사례만 필터링
    valid_cases = []
    texts = []
    for case in cases:
        product_name = case.get('product_name', '').strip()
        hs_heading = case.get('hs_heading', '').strip()
        if product_name and hs_heading and len(hs_heading) == 4:
            valid_cases.append(case)
            texts.append(product_name)

    print(f"[IndexBuilder] 유효 사례: {len(valid_cases)}건 / {len(cases)}건")

    if not texts:
        return np.array([]), []

    print(f"[IndexBuilder] SBERT 모델 로드 (사례용): {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[IndexBuilder] {len(texts)}개 사례 인코딩 중...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    print(f"[IndexBuilder] 사례 임베딩: {embeddings.shape}")
    return embeddings, valid_cases


def save_index(
    index_dir: str,
    hs4_ids: List[str],
    documents: List[str],
    tokenized_corpus: List[List[str]],
    sbert_embeddings: np.ndarray,
    case_embeddings: np.ndarray,
    case_metadata: List[Dict],
    thesaurus_lookup: Dict[str, List[str]],
    cards: Dict[str, Dict],
    rules: Dict[str, List[Dict]]
):
    """인덱스 파일 저장"""
    out_dir = Path(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # BM25 corpus
    bm25_data = {
        'hs4_ids': hs4_ids,
        'documents': documents,
        'tokenized_corpus': tokenized_corpus,
    }
    with open(out_dir / 'bm25_corpus.pkl', 'wb') as f:
        pickle.dump(bm25_data, f)
    print(f"  bm25_corpus.pkl ({len(hs4_ids)} docs)")

    # SBERT embeddings
    np.save(str(out_dir / 'sbert_embeddings.npy'), sbert_embeddings)
    print(f"  sbert_embeddings.npy {sbert_embeddings.shape}")

    # Case embeddings + metadata
    np.save(str(out_dir / 'case_embeddings.npy'), case_embeddings)
    case_meta_slim = [
        {
            'product_name': c.get('product_name', ''),
            'hs_heading': c.get('hs_heading', ''),
            'rationale': c.get('rationale', '')[:300],
            'reference_number': c.get('reference_number', ''),
        }
        for c in case_metadata
    ]
    with open(out_dir / 'case_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(case_meta_slim, f, ensure_ascii=False)
    print(f"  case_embeddings.npy {case_embeddings.shape}")
    print(f"  case_metadata.json ({len(case_meta_slim)} cases)")

    # Thesaurus lookup
    with open(out_dir / 'thesaurus_lookup.pkl', 'wb') as f:
        pickle.dump(thesaurus_lookup, f)
    print(f"  thesaurus_lookup.pkl ({len(thesaurus_lookup)} entries)")

    # Cards (검색 후 컨텍스트 조립용)
    with open(out_dir / 'hs4_cards.json', 'w', encoding='utf-8') as f:
        json.dump(cards, f, ensure_ascii=False)
    print(f"  hs4_cards.json ({len(cards)} cards)")

    # Rules (검색 후 컨텍스트 조립용)
    with open(out_dir / 'hs4_rules.json', 'w', encoding='utf-8') as f:
        json.dump(rules, f, ensure_ascii=False)
    print(f"  hs4_rules.json ({len(rules)} HS4s)")

    print(f"\n[IndexBuilder] 인덱스 저장 완료: {out_dir}")


def build_all(index_dir: str = INDEX_DIR):
    """전체 인덱스 빌드"""
    print("=" * 60)
    print("RAG 인덱스 빌드 시작")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1/5] 데이터 로드")
    cards = load_hs4_cards()
    rules = load_rule_chunks()
    thesaurus = load_thesaurus()
    cases = load_ruling_cases()

    # 2. BM25 코퍼스 구축
    print("\n[2/5] BM25 코퍼스 구축")
    hs4_ids, documents, tokenized_corpus = build_bm25_corpus(cards, rules)

    # 3. SBERT 임베딩
    print("\n[3/5] SBERT 문서 임베딩")
    sbert_embeddings = build_sbert_embeddings(documents)

    # 4. 사례 임베딩
    print("\n[4/5] 사례 임베딩")
    case_embeddings, valid_cases = build_case_embeddings(cases)

    # 5. 저장
    print("\n[5/5] 인덱스 저장")
    save_index(
        index_dir=index_dir,
        hs4_ids=hs4_ids,
        documents=documents,
        tokenized_corpus=tokenized_corpus,
        sbert_embeddings=sbert_embeddings,
        case_embeddings=case_embeddings,
        case_metadata=valid_cases,
        thesaurus_lookup=thesaurus,
        cards=cards,
        rules=rules,
    )

    print("\n" + "=" * 60)
    print("RAG 인덱스 빌드 완료")
    print("=" * 60)


if __name__ == '__main__':
    build_all()
