"""
Rule ID 마이그레이션 스크립트

기존 hs4_rule_chunks.jsonl에 결정론적 rule_id를 부여하고
source, hs_version 필드를 추가하여 v2 파일로 출력.

rule_id 생성: {hs4}_{chunk_type}_{sha256(text[:100])[:8]}
source: chunk_type에서 추론
hs_version: "2022"

Usage:
    python scripts/migrate_rule_ids.py
"""

import json
import hashlib
from pathlib import Path

INPUT_PATH = "kb/structured/hs4_rule_chunks.jsonl"
OUTPUT_PATH = "kb/structured/hs4_rule_chunks_v2.jsonl"

# chunk_type → source 매핑
CHUNK_TYPE_TO_SOURCE = {
    "include_rule": "explanatory_note",
    "exclude_rule": "chapter_note",
    "definition": "heading_note",
    "example": "explanatory_note",
    "general": "general_note",
}


def generate_rule_id(hs4: str, chunk_type: str, text: str) -> str:
    """결정론적 rule_id 생성"""
    text_prefix = text[:100] if text else ""
    text_hash = hashlib.sha256(text_prefix.encode("utf-8")).hexdigest()[:8]
    return f"{hs4}_{chunk_type}_{text_hash}"


def infer_source(chunk_type: str) -> str:
    """chunk_type에서 source 추론"""
    return CHUNK_TYPE_TO_SOURCE.get(chunk_type, "general_note")


def migrate():
    input_file = Path(INPUT_PATH)
    if not input_file.exists():
        print(f"[Error] Input file not found: {input_file}")
        return

    output_file = Path(OUTPUT_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    seen_ids = set()
    duplicates = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rule = json.loads(line)
            except json.JSONDecodeError:
                continue

            hs4 = rule.get("hs4", "")
            if not hs4:
                continue

            chunk_type = rule.get("chunk_type", "general")
            text = rule.get("text", "")

            rule_id = generate_rule_id(hs4, chunk_type, text)

            # 중복 방지: suffix 추가
            if rule_id in seen_ids:
                duplicates += 1
                rule_id = f"{rule_id}_{duplicates}"
            seen_ids.add(rule_id)

            # v2 필드 추가
            rule["rule_id"] = rule_id
            rule["source"] = infer_source(chunk_type)
            rule["hs_version"] = "2022"

            fout.write(json.dumps(rule, ensure_ascii=False) + "\n")
            count += 1

    print(f"[migrate_rule_ids] Migrated {count} rules to {output_file}")
    print(f"  Duplicates resolved: {duplicates}")
    print(f"  Unique rule_ids: {len(seen_ids)}")


if __name__ == "__main__":
    migrate()
