"""
관세율표 TXT 파일에서 부/류별 주(註) 규정 추출

파일 구조:
- 제N부: 부 제목
  주: 1. ... 2. ...
- 제N류: 류 제목
  주: 1. ... 2. ...
  소호주: 1. ... 2. ...
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class TariffNote:
    """관세율표 주(註)"""
    level: str  # 'section', 'chapter', 'subheading'
    section_num: Optional[int]  # 부 번호
    section_title: Optional[str]  # 부 제목
    chapter_num: Optional[int]  # 류 번호 (01-99)
    chapter_title: Optional[str]  # 류 제목
    note_number: str  # 주 번호 (1, 2, 가, 나 등)
    note_content: str  # 주 내용


def roman_to_int(roman: str) -> int:
    """로마 숫자를 정수로 변환"""
    roman_map = {
        'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
        'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
        'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
        'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,
        'XXI': 21, 'XXII': 22
    }
    return roman_map.get(roman.strip())


def parse_tariff_notes(txt_path: str) -> List[TariffNote]:
    """TXT 파일에서 모든 주 규정 추출"""

    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    notes = []

    # 전체 텍스트를 부(Section) 단위로 분할
    # "제N부" 또는 "제 I 부" 패턴
    section_pattern = r'제\s*([IVX]+)\s*부([^제]+?)(?=제\s*(?:[IVX]+\s*부|\d+류)|$)'
    sections = re.finditer(section_pattern, content, re.DOTALL)

    current_section_num = None
    current_section_title = None

    for section_match in sections:
        section_roman = section_match.group(1)
        section_num = roman_to_int(section_roman)
        section_content = section_match.group(2)

        # 부 제목 추출 (첫 번째 문장)
        title_match = re.search(r'^([^\n]+)', section_content.strip())
        section_title = title_match.group(1).strip() if title_match else ""

        current_section_num = section_num
        current_section_title = section_title

        print(f"\n{'='*80}")
        print(f"제{section_roman}부 ({section_num}): {section_title[:50]}...")

        # 부 주(Section Notes) 추출
        section_notes = extract_notes(
            section_content,
            level='section',
            section_num=section_num,
            section_title=section_title,
            chapter_num=None,
            chapter_title=None
        )
        notes.extend(section_notes)
        print(f"  -> 부 주: {len(section_notes)}개")

    # 류(Chapter) 단위로 분할
    # "제N류" 패턴
    chapter_pattern = r'제\s*(\d+)\s*류([^제]+?)(?=제\s*\d+\s*류|$)'
    chapters = re.finditer(chapter_pattern, content, re.DOTALL)

    for chapter_match in chapters:
        chapter_num = int(chapter_match.group(1))
        chapter_content = chapter_match.group(2)

        # 류 제목 추출
        title_match = re.search(r'^([^\n]+?)(?:주:|$)', chapter_content.strip())
        chapter_title = title_match.group(1).strip() if title_match else ""

        print(f"\n제{chapter_num:02d}류: {chapter_title[:50]}...")

        # 류 주(Chapter Notes) 추출
        chapter_notes = extract_notes(
            chapter_content,
            level='chapter',
            section_num=current_section_num,
            section_title=current_section_title,
            chapter_num=chapter_num,
            chapter_title=chapter_title
        )
        notes.extend(chapter_notes)
        print(f"  -> 류 주: {len(chapter_notes)}개")

        # 소호주(Subheading Notes) 추출
        subheading_notes = extract_subheading_notes(
            chapter_content,
            section_num=current_section_num,
            section_title=current_section_title,
            chapter_num=chapter_num,
            chapter_title=chapter_title
        )
        notes.extend(subheading_notes)
        if subheading_notes:
            print(f"  -> 소호주: {len(subheading_notes)}개")

    return notes


def extract_notes(
    content: str,
    level: str,
    section_num: Optional[int],
    section_title: Optional[str],
    chapter_num: Optional[int],
    chapter_title: Optional[str]
) -> List[TariffNote]:
    """주 규정 추출"""

    notes = []

    # "주:" 다음부터 "소호주:" 또는 다음 섹션 전까지
    note_section_pattern = r'주:\s*(.*?)(?=소호주:|제\s*\d+\s*류|제\s*[IVX]+\s*부|\d{4}\.\d{2}|$)'
    note_section_match = re.search(note_section_pattern, content, re.DOTALL)

    if not note_section_match:
        return notes

    note_section = note_section_match.group(1)

    # 개별 주 항목 추출 (1., 2., 가., 나. 등)
    # 주 번호 패턴: "1. ", "2. ", "가. ", "나. " 등
    note_items = re.split(r'\n?(\d+|[가-힣])\.\s+', note_section)

    # 첫 번째 항목은 빈 문자열이거나 헤더이므로 건너뜀
    i = 1
    while i < len(note_items) - 1:
        note_num = note_items[i].strip()
        note_content = note_items[i + 1].strip()

        # 너무 짧은 내용은 건너뜀 (노이즈)
        if len(note_content) < 10:
            i += 2
            continue

        # 다음 주 번호 전까지만 가져오기
        # HS 코드 패턴(0000.00) 전에서 끊기
        note_content = re.split(r'\d{4}\.\d{2}', note_content)[0].strip()

        if note_content:
            notes.append(TariffNote(
                level=level,
                section_num=section_num,
                section_title=section_title,
                chapter_num=chapter_num,
                chapter_title=chapter_title,
                note_number=note_num,
                note_content=note_content
            ))

        i += 2

    return notes


def extract_subheading_notes(
    content: str,
    section_num: Optional[int],
    section_title: Optional[str],
    chapter_num: int,
    chapter_title: str
) -> List[TariffNote]:
    """소호주 추출"""

    notes = []

    # "소호주:" 패턴
    subheading_pattern = r'소호주:\s*(.*?)(?=제\s*\d+\s*류|제\s*[IVX]+\s*부|$)'
    subheading_match = re.search(subheading_pattern, content, re.DOTALL)

    if not subheading_match:
        return notes

    subheading_section = subheading_match.group(1)

    # 개별 소호주 항목 추출
    note_items = re.split(r'\n?(\d+|[가-힣])\.\s+', subheading_section)

    i = 1
    while i < len(note_items) - 1:
        note_num = note_items[i].strip()
        note_content = note_items[i + 1].strip()

        if len(note_content) < 10:
            i += 2
            continue

        # HS 코드 패턴 전에서 끊기
        note_content = re.split(r'\d{4}\.\d{2}', note_content)[0].strip()

        if note_content:
            notes.append(TariffNote(
                level='subheading',
                section_num=section_num,
                section_title=section_title,
                chapter_num=chapter_num,
                chapter_title=chapter_title,
                note_number=note_num,
                note_content=note_content
            ))

        i += 2

    return notes


def save_notes(notes: List[TariffNote], output_dir: str):
    """추출한 주를 JSON과 TXT로 저장"""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # JSON 저장
    json_path = output_path / 'tariff_notes_clean.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(note) for note in notes], f, ensure_ascii=False, indent=2)
    print(f"\n✓ JSON 저장: {json_path} ({len(notes)}개)")

    # 사람이 읽기 쉬운 TXT 저장
    txt_path = output_path / 'tariff_notes_clean.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        current_chapter = None

        for note in notes:
            # 류가 바뀔 때마다 구분선
            if note.chapter_num != current_chapter:
                f.write(f"\n{'='*80}\n")
                if note.section_num:
                    f.write(f"제{note.section_num}부: {note.section_title}\n")
                if note.chapter_num:
                    f.write(f"제{note.chapter_num:02d}류: {note.chapter_title}\n")
                f.write(f"{'='*80}\n")
                current_chapter = note.chapter_num

            level_name = {'section': '부주', 'chapter': '류주', 'subheading': '소호주'}
            f.write(f"\n[{level_name[note.level]} {note.note_number}]\n")
            f.write(f"{note.note_content}\n")

    print(f"✓ TXT 저장: {txt_path}")

    # 통계 출력
    print(f"\n{'='*80}")
    print(f"📊 추출 통계")
    print(f"{'='*80}")

    section_notes = [n for n in notes if n.level == 'section']
    chapter_notes = [n for n in notes if n.level == 'chapter']
    subheading_notes = [n for n in notes if n.level == 'subheading']

    print(f"총 주 개수: {len(notes)}")
    print(f"  - 부주: {len(section_notes)}")
    print(f"  - 류주: {len(chapter_notes)}")
    print(f"  - 소호주: {len(subheading_notes)}")

    # 류별 통계
    chapters_with_notes = {}
    for note in notes:
        if note.chapter_num:
            chapters_with_notes[note.chapter_num] = chapters_with_notes.get(note.chapter_num, 0) + 1

    print(f"\n주가 있는 류: {len(chapters_with_notes)}개")
    print(f"\n주가 많은 류 Top 10:")
    for ch, cnt in sorted(chapters_with_notes.items(), key=lambda x: x[1], reverse=True)[:10]:
        ch_note = next((n for n in notes if n.chapter_num == ch), None)
        ch_title = ch_note.chapter_title[:30] if ch_note else ""
        print(f"  제{ch:02d}류 ({ch_title}): {cnt}개")


def main():
    txt_path = "law0015562022123119186KC_000000E.txt"
    output_dir = "data"

    print("="*80)
    print("관세율표 주규정 추출 시작")
    print("="*80)

    notes = parse_tariff_notes(txt_path)
    save_notes(notes, output_dir)

    print(f"\n{'='*80}")
    print("✓ 추출 완료!")
    print("="*80)


if __name__ == "__main__":
    main()
