"""
RAG Index Builder (Stage 3)

KB 데이터로부터 BM25 corpus, SBERT 임베딩, 사례 임베딩, 시소러스 룩업을 생성.
새 구조(src/text.py, src/kb/) 기반으로 인덱스 빌드.

생성 파일:
  artifacts/rag_index/
  ├── bm25_corpus.pkl        - BM25 tokenized corpus
  ├── sbert_embeddings.npy   - (N, 768) 카드별 SBERT 벡터
  ├── case_embeddings.npy    - (M, 768) 결정사례별 SBERT 벡터
  ├── case_metadata.json     - 사례 메타
  ├── thesaurus_lookup.pkl   - term -> aliases 매핑
  ├── hs4_cards.json         - 카드 전체 (context builder용)
  └── hs4_rules.json         - 규칙 전체 (context builder용)

Usage:
    python scripts/build_rag_index.py
    python scripts/build_rag_index.py --index-dir artifacts/rag_index_v2
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.text import normalize, tokenize

# Paths
KB_CARDS_V2_PATH = PROJECT_ROOT / "kb" / "structured" / "hs4_cards_v2.jsonl"
KB_CARDS_PATH = PROJECT_ROOT / "kb" / "structured" / "hs4_cards.jsonl"
KB_RULES_PATH = PROJECT_ROOT / "kb" / "structured" / "hs4_rule_chunks.jsonl"
THESAURUS_PATH = PROJECT_ROOT / "kb" / "structured" / "thesaurus_terms.jsonl"
CASES_PATH = PROJECT_ROOT / "data" / "ruling_cases" / "all_cases_full_v7.json"

SBERT_MODEL_NAME = "jhgan/ko-sroberta-multitask"
DEFAULT_INDEX_DIR = "artifacts/rag_index"


def _flatten_field(items: list) -> List[str]:
    """카드 필드를 문자열 리스트로 변환 (str or dict 혼재 대응)."""
    result = []
    for item in items:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            for key in ("reason", "text", "value", "description"):
                val = item.get(key, "")
                if val:
                    result.append(str(val))
                    break
            else:
                result.append(str(item))
    return result


def load_hs4_cards() -> Dict[str, Dict]:
    """HS4 카드 로드 (v2 우선, 없으면 v1)."""
    cards_path = KB_CARDS_V2_PATH if KB_CARDS_V2_PATH.exists() else KB_CARDS_PATH

    cards = {}
    with open(cards_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            card = json.loads(line)
            hs4 = card.get("hs4")
            if hs4:
                cards[hs4] = card
    print(f"[IndexBuilder] HS4 카드 로드: {len(cards)}건 ({cards_path.name})")
    return cards


def load_rule_chunks() -> Dict[str, List[Dict]]:
    """HS4별 규칙 청크 로드."""
    if not KB_RULES_PATH.exists():
        print(f"[IndexBuilder] Warning: {KB_RULES_PATH} not found")
        return {}

    rules = {}
    count = 0
    with open(KB_RULES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            hs4 = chunk.get("hs4")
            if hs4:
                rules.setdefault(hs4, []).append(chunk)
                count += 1
    print(f"[IndexBuilder] 규칙 청크 로드: {count}건 ({len(rules)} HS4)")
    return rules


def load_thesaurus() -> Dict[str, List[str]]:
    """시소러스 term->aliases 룩업 테이블 로드 (bidirectional)."""
    if not THESAURUS_PATH.exists():
        print(f"[IndexBuilder] Warning: {THESAURUS_PATH} not found")
        return {}

    lookup = {}
    with open(THESAURUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            term = entry.get("term", "").strip()
            aliases = entry.get("aliases", [])
            if term and aliases:
                norm_term = normalize(term)
                if norm_term:
                    lookup[norm_term] = aliases
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
    """결정사례 로드."""
    if not CASES_PATH.exists():
        print(f"[IndexBuilder] Warning: {CASES_PATH} not found")
        return []

    with open(CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    print(f"[IndexBuilder] 결정사례 로드: {len(cases)}건")
    return cases


def build_bm25_corpus(
    cards: Dict[str, Dict], rules: Dict[str, List[Dict]]
) -> Tuple[List[str], List[str], List[List[str]]]:
    """BM25 검색용 코퍼스 구축."""
    hs4_ids = []
    documents = []
    tokenized_corpus = []

    for hs4, card in sorted(cards.items()):
        parts = []

        title = card.get("title_ko", "")
        if title:
            parts.append(title)

        scope = card.get("scope", [])
        if scope:
            parts.append(" ".join(_flatten_field(scope)))

        includes = card.get("includes", [])
        if includes:
            parts.append(" ".join(_flatten_field(includes)))

        excludes = card.get("excludes", [])
        if excludes:
            parts.append(" ".join(_flatten_field(excludes)))

        key_attrs = card.get("key_attributes", [])
        if key_attrs:
            parts.append(" ".join(_flatten_field(key_attrs)))

        rule_chunks = rules.get(hs4, [])
        for chunk in rule_chunks[:3]:
            chunk_text = chunk.get("text", "")
            if chunk_text:
                parts.append(chunk_text[:200])

        doc_text = " ".join(parts)
        tokens = tokenize(doc_text, remove_stopwords=True)

        if tokens:
            hs4_ids.append(hs4)
            documents.append(doc_text)
            tokenized_corpus.append(tokens)

    avg_tokens = np.mean([len(t) for t in tokenized_corpus]) if tokenized_corpus else 0
    print(f"[IndexBuilder] BM25 코퍼스: {len(hs4_ids)}개 문서, 평균 토큰 수: {avg_tokens:.1f}")
    return hs4_ids, documents, tokenized_corpus


def build_sbert_embeddings(
    documents: List[str], batch_size: int = 64
) -> np.ndarray:
    """SBERT 임베딩 생성."""
    from sentence_transformers import SentenceTransformer

    print(f"[IndexBuilder] SBERT 모델 로드: {SBERT_MODEL_NAME}")
    model = SentenceTransformer(SBERT_MODEL_NAME)

    print(f"[IndexBuilder] {len(documents)}개 문서 인코딩 중...")
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"[IndexBuilder] SBERT 임베딩: {embeddings.shape}")
    return embeddings


def build_case_embeddings(
    cases: List[Dict], batch_size: int = 64
) -> Tuple[np.ndarray, List[Dict]]:
    """결정사례 product_name 임베딩 생성."""
    from sentence_transformers import SentenceTransformer

    valid_cases = []
    texts = []
    for case in cases:
        product_name = case.get("product_name", "").strip()
        hs_heading = case.get("hs_heading", "").strip()
        if product_name and hs_heading and len(hs_heading) == 4:
            valid_cases.append(case)
            texts.append(product_name)

    print(f"[IndexBuilder] 유효 사례: {len(valid_cases)}건 / {len(cases)}건")

    if not texts:
        return np.array([]), []

    print(f"[IndexBuilder] SBERT 모델 로드 (사례용): {SBERT_MODEL_NAME}")
    model = SentenceTransformer(SBERT_MODEL_NAME)

    print(f"[IndexBuilder] {len(texts)}개 사례 인코딩 중...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
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
    rules: Dict[str, List[Dict]],
):
    """인덱스 파일 저장."""
    out_dir = Path(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[IndexBuilder] Saving to {out_dir}...")

    # BM25 corpus
    bm25_data = {
        "hs4_ids": hs4_ids,
        "documents": documents,
        "tokenized_corpus": tokenized_corpus,
    }
    with open(out_dir / "bm25_corpus.pkl", "wb") as f:
        pickle.dump(bm25_data, f)
    print(f"  bm25_corpus.pkl ({len(hs4_ids)} docs)")

    # SBERT embeddings
    np.save(str(out_dir / "sbert_embeddings.npy"), sbert_embeddings)
    print(f"  sbert_embeddings.npy {sbert_embeddings.shape}")

    # Case embeddings + metadata
    np.save(str(out_dir / "case_embeddings.npy"), case_embeddings)
    case_meta_slim = [
        {
            "product_name": c.get("product_name", ""),
            "hs_heading": c.get("hs_heading", ""),
            "rationale": c.get("rationale", "")[:300],
            "reference_number": c.get("reference_number", ""),
        }
        for c in case_metadata
    ]
    with open(out_dir / "case_metadata.json", "w", encoding="utf-8") as f:
        json.dump(case_meta_slim, f, ensure_ascii=False)
    print(f"  case_embeddings.npy {case_embeddings.shape}")
    print(f"  case_metadata.json ({len(case_meta_slim)} cases)")

    # Thesaurus lookup
    with open(out_dir / "thesaurus_lookup.pkl", "wb") as f:
        pickle.dump(thesaurus_lookup, f)
    print(f"  thesaurus_lookup.pkl ({len(thesaurus_lookup)} entries)")

    # Cards (context builder용)
    with open(out_dir / "hs4_cards.json", "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False)
    print(f"  hs4_cards.json ({len(cards)} cards)")

    # Rules (context builder용)
    with open(out_dir / "hs4_rules.json", "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False)
    print(f"  hs4_rules.json ({len(rules)} HS4s)")


def build_all(index_dir: str = DEFAULT_INDEX_DIR):
    """전체 인덱스 빌드."""
    t0 = time.time()
    print("=" * 70)
    print("RAG Index Build")
    print("=" * 70)

    # 1. Load data
    print("\n[1/5] 데이터 로드")
    cards = load_hs4_cards()
    rules = load_rule_chunks()
    thesaurus = load_thesaurus()
    cases = load_ruling_cases()

    # 2. BM25 corpus
    print("\n[2/5] BM25 코퍼스 구축")
    t1 = time.time()
    hs4_ids, documents, tokenized_corpus = build_bm25_corpus(cards, rules)
    print(f"  -> {time.time()-t1:.1f}s")

    # 3. SBERT embeddings
    print("\n[3/5] SBERT 문서 임베딩")
    t2 = time.time()
    sbert_embeddings = build_sbert_embeddings(documents)
    print(f"  -> {time.time()-t2:.1f}s")

    # 4. Case embeddings
    print("\n[4/5] 사례 임베딩")
    t3 = time.time()
    case_embeddings, valid_cases = build_case_embeddings(cases)
    print(f"  -> {time.time()-t3:.1f}s")

    # 5. Save
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

    total_time = time.time() - t0

    # Summary
    print(f"\n{'=' * 70}")
    print(f"RAG Index Build Complete ({total_time:.1f}s)")
    print(f"{'=' * 70}")
    print(f"  Index dir: {index_dir}")
    print(f"  BM25 docs: {len(hs4_ids)}")
    print(f"  SBERT embeddings: {sbert_embeddings.shape}")
    print(f"  Case embeddings: {case_embeddings.shape}")
    print(f"  Thesaurus entries: {len(thesaurus)}")
    print(f"  Cards: {len(cards)}, Rules: {len(rules)} HS4s")


def main():
    parser = argparse.ArgumentParser(description="Build RAG Index")
    parser.add_argument(
        "--index-dir", type=str, default=DEFAULT_INDEX_DIR,
        help="Output index directory (default: artifacts/rag_index)",
    )
    args = parser.parse_args()
    build_all(index_dir=args.index_dir)


if __name__ == "__main__":
    main()
