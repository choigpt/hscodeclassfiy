"""
규칙 청크 생성 스크립트

해설서에서 규칙 단위로 분리합니다.
청크 타입:
- definition: 정의 (이 호에는 ... 분류한다)
- include_rule: 포함 규칙
- exclude_rule: 제외 규칙 (다른 호로 분류)
- example: 예시
- note: 주의사항
"""

import json
import re
from pathlib import Path
from typing import List, Dict


def extract_signals(text: str) -> List[str]:
    """텍스트에서 신호 키워드 추출"""
    # 불용어 제외
    stopwords = {
        '것', '등', '및', '의', '이', '가', '을', '를', '에', '로', '으로',
        '한', '하는', '된', '되는', '있는', '없는', '같은', '위한', '대한',
        '또는', '그', '이러한', '해당', '경우', '때', '수', '더', '매우'
    }

    # 키워드 추출
    words = re.findall(r'[가-힣]{2,10}', text)
    signals = [w for w in words if w not in stopwords]

    # 빈도 기반 상위 키워드
    from collections import Counter
    counter = Counter(signals)
    return [w for w, _ in counter.most_common(5)]


def split_into_chunks(hs_code: str, content: str) -> List[Dict]:
    """해설서 내용을 규칙 청크로 분리"""
    chunks = []

    # 문장 분리
    sentences = re.split(r'[.。]\s*', content)

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 10:
            continue

        chunk = {
            "hs4": hs_code,
            "chunk_type": None,
            "text": sent,
            "signals": [],
            "target_hs4": None
        }

        # 청크 타입 분류
        if re.search(r'이 호에는?.+분류', sent):
            chunk["chunk_type"] = "definition"
        elif re.search(r'제외|제(\d{4})호[로에]|다만|그러나', sent):
            chunk["chunk_type"] = "exclude_rule"
            # 대상 HS 추출
            match = re.search(r'제?(\d{4})호', sent)
            if match and match.group(1) != hs_code:
                chunk["target_hs4"] = match.group(1)
        elif re.search(r'포함|해당|분류[된하]', sent):
            chunk["chunk_type"] = "include_rule"
        elif re.search(r'예[:\s]|예를 들', sent):
            chunk["chunk_type"] = "example"
        elif re.search(r'주의|유의|참고|단,', sent):
            chunk["chunk_type"] = "note"
        else:
            chunk["chunk_type"] = "general"

        # 신호 키워드 추출
        chunk["signals"] = extract_signals(sent)

        if chunk["chunk_type"] != "general":  # general은 제외
            chunks.append(chunk)

    return chunks


def build_rule_chunks(commentary_path: str, output_path: str):
    """규칙 청크 생성"""
    print("=" * 60)
    print("규칙 청크 생성")
    print("=" * 60)

    with open(commentary_path, 'r', encoding='utf-8') as f:
        commentary = json.load(f)

    print(f"해설서 로드: {len(commentary)}개")

    all_chunks = []
    chunk_type_counts = {}

    for item in commentary:
        hs_code = item.get('hs_code', '')
        if not hs_code or len(hs_code) != 4:
            continue

        content = item.get('content', '')
        chunks = split_into_chunks(hs_code, content)

        for chunk in chunks:
            chunk_type = chunk["chunk_type"]
            chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1

        all_chunks.extend(chunks)

    print(f"청크 생성: {len(all_chunks)}개")

    # JSONL 형식으로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    print(f"저장: {output_path}")

    print(f"\n청크 타입별 분포:")
    for chunk_type, count in sorted(chunk_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {chunk_type}: {count}개")

    return all_chunks


if __name__ == "__main__":
    build_rule_chunks(
        commentary_path="kb/raw/hs_commentary.json",
        output_path="kb/structured/hs4_rule_chunks.jsonl"
    )
