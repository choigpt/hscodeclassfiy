"""
HS6 서브헤딩 KB 구축 스크립트

소스: data/관세청_HS부호_20260101.xlsx
- HS10 → HS6 → HS4 계층 추출
- 각 HS6에 title_ko, keywords, parent HS4 매핑
- 소호 주규정 연결
- 판결 케이스 HS6 빈도 통계

출력: kb/structured/hs6_subheadings.jsonl

Usage:
    python scripts/build_subheading_kb.py
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

EXCEL_PATH = "data/관세청_HS부호_20260101.xlsx"
NOTES_PATH = "data/tariff_notes_clean.json"
CASES_PATH = "data/ruling_cases/all_cases_full_v7.json"
OUTPUT_PATH = "kb/structured/hs6_subheadings.jsonl"


def load_hs_data_from_excel() -> list:
    """엑셀에서 HS 코드 데이터 로드"""
    try:
        import openpyxl
    except ImportError:
        print("[Warning] openpyxl not installed. Trying pandas...")
        try:
            import pandas as pd
            df = pd.read_excel(EXCEL_PATH)
            records = df.to_dict('records')
            return records
        except ImportError:
            print("[Error] pandas not installed either. Install: pip install openpyxl")
            return []

    wb = openpyxl.load_workbook(EXCEL_PATH, read_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []

    # 첫 행을 헤더로
    header = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(rows[0])]
    records = []
    for row in rows[1:]:
        record = {}
        for i, val in enumerate(row):
            if i < len(header):
                record[header[i]] = val
        records.append(record)

    wb.close()
    return records


def extract_hs_hierarchy(records: list) -> dict:
    """HS10 → HS6 → HS4 계층 추출"""
    hs6_map = defaultdict(lambda: {
        'hs6': '',
        'hs4': '',
        'title_ko': '',
        'keywords': set(),
        'hs10_codes': [],
        'subheading_notes': [],
        'case_count': 0,
    })

    for rec in records:
        # HS 코드 열 찾기 (다양한 열명 지원)
        hs_code = None
        hs_name = None

        for key in rec:
            val = str(rec[key] or '').strip()
            key_str = str(key).strip()

            # HS 코드 열
            if any(k in key_str for k in ['부호', '코드', 'HS', 'code', 'Code']):
                if re.match(r'^\d{4,10}$', val.replace('.', '').replace('-', '')):
                    hs_code = val.replace('.', '').replace('-', '')

            # 품명 열
            if any(k in key_str for k in ['품명', '한글명', '명칭', 'name', 'Name', '설명']):
                if val and len(val) > 1:
                    hs_name = val

        if not hs_code:
            continue

        # 숫자만 추출
        digits = re.sub(r'\D', '', hs_code)
        if len(digits) < 6:
            continue

        hs4 = digits[:4]
        hs6 = digits[:6]
        hs10 = digits[:10] if len(digits) >= 10 else digits

        entry = hs6_map[hs6]
        entry['hs6'] = hs6
        entry['hs4'] = hs4
        if hs10 and hs10 not in entry['hs10_codes']:
            entry['hs10_codes'].append(hs10)

        if hs_name:
            if not entry['title_ko']:
                entry['title_ko'] = hs_name
            # 키워드 추출
            tokens = re.findall(r'[가-힣]{2,}|[a-zA-Z]{3,}', hs_name)
            entry['keywords'].update(t.lower() for t in tokens if len(t) >= 2)

    return dict(hs6_map)


def load_subheading_notes() -> dict:
    """소호 주규정 로드"""
    notes_file = Path(NOTES_PATH)
    if not notes_file.exists():
        return {}

    with open(notes_file, 'r', encoding='utf-8') as f:
        notes_data = json.load(f)

    subheading_notes = defaultdict(list)

    # tariff_notes_clean.json 구조에 따라 파싱
    if isinstance(notes_data, list):
        for note in notes_data:
            level = note.get('level', '')
            if level == 'subheading' or '소호' in note.get('note_type', ''):
                hs_codes = note.get('hs_codes', [])
                for code in hs_codes:
                    if len(str(code)) >= 6:
                        hs6 = str(code)[:6]
                        subheading_notes[hs6].append(note.get('content', ''))
    elif isinstance(notes_data, dict):
        for key, entries in notes_data.items():
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict):
                        level = entry.get('level', '')
                        if '소호' in level or 'subheading' in level:
                            content = entry.get('note_content', '') or entry.get('content', '')
                            if content:
                                subheading_notes[key].append(content)

    return dict(subheading_notes)


def count_case_hs6(cases_path: str) -> dict:
    """판결 케이스에서 HS6 빈도 통계"""
    cases_file = Path(cases_path)
    if not cases_file.exists():
        return {}

    with open(cases_file, 'r', encoding='utf-8') as f:
        cases = json.load(f)

    hs6_counts = defaultdict(int)
    for case in cases:
        hs_code = case.get('decision_hs_code', '')
        # 숫자만 추출
        digits = re.sub(r'[^0-9]', '', hs_code)
        if len(digits) >= 6:
            hs6 = digits[:6]
            hs6_counts[hs6] += 1

    return dict(hs6_counts)


def build():
    print("=" * 60)
    print("HS6 서브헤딩 KB 구축")
    print("=" * 60)

    # 1. Excel에서 HS 계층 추출
    excel_path = Path(EXCEL_PATH)
    if excel_path.exists():
        print(f"\n[1] 엑셀 로드: {EXCEL_PATH}")
        records = load_hs_data_from_excel()
        print(f"  레코드 수: {len(records)}")
        hs6_map = extract_hs_hierarchy(records)
        print(f"  HS6 코드 수: {len(hs6_map)}")
    else:
        print(f"[Warning] 엑셀 파일 없음: {EXCEL_PATH}")
        print("  빈 KB 생성 (나중에 엑셀 파일 추가 후 재실행)")
        hs6_map = {}

    # 2. 소호 주규정 연결
    print(f"\n[2] 소호 주규정 로드: {NOTES_PATH}")
    subheading_notes = load_subheading_notes()
    print(f"  소호 주규정 수: {len(subheading_notes)}")

    for hs6, notes in subheading_notes.items():
        if hs6 in hs6_map:
            hs6_map[hs6]['subheading_notes'] = notes

    # 3. 판결 케이스 빈도
    print(f"\n[3] 판결 케이스 빈도: {CASES_PATH}")
    hs6_counts = count_case_hs6(CASES_PATH)
    print(f"  HS6 with cases: {len(hs6_counts)}")

    for hs6, count in hs6_counts.items():
        if hs6 in hs6_map:
            hs6_map[hs6]['case_count'] = count
        else:
            # 엑셀에 없지만 판결에 있는 HS6
            hs6_map[hs6] = {
                'hs6': hs6,
                'hs4': hs6[:4],
                'title_ko': '',
                'keywords': set(),
                'hs10_codes': [],
                'subheading_notes': [],
                'case_count': count,
            }

    # 4. 출력
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for hs6 in sorted(hs6_map.keys()):
            entry = hs6_map[hs6]
            record = {
                'hs6': entry['hs6'],
                'hs4': entry['hs4'],
                'title_ko': entry['title_ko'],
                'keywords': sorted(list(entry['keywords'])),
                'subheading_notes': entry['subheading_notes'],
                'case_count': entry['case_count'],
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1

    print(f"\n[완료] {count} HS6 서브헤딩 저장: {output_path}")

    # 통계
    with_title = sum(1 for e in hs6_map.values() if e['title_ko'])
    with_notes = sum(1 for e in hs6_map.values() if e['subheading_notes'])
    with_cases = sum(1 for e in hs6_map.values() if e['case_count'] > 0)
    print(f"  품명 있음: {with_title}")
    print(f"  주규정 있음: {with_notes}")
    print(f"  판결 있음: {with_cases}")


if __name__ == "__main__":
    build()
