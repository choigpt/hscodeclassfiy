"""
관세율표 TXT 파일에서 부/류별 주(註) 규정 추출 (v2)

파일 구조 (줄바꿈으로 구분됨):
제1부
살아 있는 동물과 동물성 생산품
주:
1. ...
2. ...

제1류
살아 있는 동물
주:
1. ...
2. ...
소호주:
1. ...
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
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

    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    notes = []

    i = 0
    current_section_num = None
    current_section_title = None
    current_chapter_num = None
    current_chapter_title = None

    log = open('data/parse_log.txt', 'w', encoding='utf-8')

    def write_log(msg):
        log.write(msg + '\n')
        log.flush()

    write_log("="*80)
    write_log("관세율표 주규정 파싱 시작")
    write_log("="*80)

    while i < len(lines):
        line = lines[i].strip()

        # 부(Section) 감지: "제N부"
        section_match = re.match(r'^제(\d+)부$', line)
        if section_match:
            current_section_num = int(section_match.group(1))
            # 다음 줄이 부 제목
            if i + 1 < len(lines):
                current_section_title = lines[i + 1].strip()
            write_log(f"\n[제{current_section_num}부] {current_section_title}")

            # 부 주 추출
            section_notes = extract_notes_from_lines(
                lines, i + 2,
                level='section',
                section_num=current_section_num,
                section_title=current_section_title,
                chapter_num=None,
                chapter_title=None
            )
            notes.extend(section_notes)
            if section_notes:
                write_log(f"  -> 부 주: {len(section_notes)}개")
            i += 1
            continue

        # 류(Chapter) 감지: "제N류"
        chapter_match = re.match(r'^제(\d+)류$', line)
        if chapter_match:
            current_chapter_num = int(chapter_match.group(1))
            # 다음 줄이 류 제목
            if i + 1 < len(lines):
                current_chapter_title = lines[i + 1].strip()
            write_log(f"\n[제{current_chapter_num:02d}류] {current_chapter_title[:50]}...")

            # 류 주 추출
            chapter_notes = extract_notes_from_lines(
                lines, i + 2,
                level='chapter',
                section_num=current_section_num,
                section_title=current_section_title,
                chapter_num=current_chapter_num,
                chapter_title=current_chapter_title
            )
            notes.extend(chapter_notes)
            if chapter_notes:
                write_log(f"  -> 류 주: {len(chapter_notes)}개")

            # 소호주 추출
            subheading_notes = extract_subheading_notes_from_lines(
                lines, i + 2,
                section_num=current_section_num,
                section_title=current_section_title,
                chapter_num=current_chapter_num,
                chapter_title=current_chapter_title
            )
            notes.extend(subheading_notes)
            if subheading_notes:
                write_log(f"  -> 소호주: {len(subheading_notes)}개")

            i += 1
            continue

        i += 1

    log.close()
    return notes


def extract_notes_from_lines(
    lines: List[str],
    start_idx: int,
    level: str,
    section_num: Optional[int],
    section_title: Optional[str],
    chapter_num: Optional[int],
    chapter_title: Optional[str]
) -> List[TariffNote]:
    """주: 섹션에서 주 추출"""

    notes = []
    i = start_idx

    # "주:" 라인 찾기
    while i < len(lines):
        if lines[i].strip() == "주:":
            break
        # 다음 부/류가 시작되면 중단
        if re.match(r'^제\d+[부류]$', lines[i].strip()):
            return notes
        i += 1

    if i >= len(lines) or lines[i].strip() != "주:":
        return notes

    i += 1  # "주:" 다음 줄부터

    # 주 항목 추출
    current_note_num = None
    current_note_lines = []

    while i < len(lines):
        line = lines[i].strip()

        # 다음 섹션/류 시작하면 중단
        if re.match(r'^제\d+[부류]$', line):
            break

        # "소호주:" 시작하면 중단 (류주만 추출)
        if line == "소호주:":
            break

        # "번 호" 패턴 (HS 코드 테이블 시작)
        if line == "번 호" or re.match(r'^\d{4}$', line):
            break

        # 주 번호 패턴: "1. ", "2. ", "가. ", "나. " 등
        note_match = re.match(r'^(\d+|[가-힣])\.\s+(.+)$', line)
        if note_match:
            # 이전 주 저장
            if current_note_num and current_note_lines:
                note_content = ' '.join(current_note_lines).strip()
                if len(note_content) > 10:
                    notes.append(TariffNote(
                        level=level,
                        section_num=section_num,
                        section_title=section_title,
                        chapter_num=chapter_num,
                        chapter_title=chapter_title,
                        note_number=current_note_num,
                        note_content=note_content
                    ))

            # 새 주 시작
            current_note_num = note_match.group(1)
            current_note_lines = [note_match.group(2)]
        elif current_note_num and line:
            # 기존 주 내용 계속
            current_note_lines.append(line)
        elif not line:
            # 빈 줄은 무시
            pass

        i += 1

    # 마지막 주 저장
    if current_note_num and current_note_lines:
        note_content = ' '.join(current_note_lines).strip()
        if len(note_content) > 10:
            notes.append(TariffNote(
                level=level,
                section_num=section_num,
                section_title=section_title,
                chapter_num=chapter_num,
                chapter_title=chapter_title,
                note_number=current_note_num,
                note_content=note_content
            ))

    return notes


def extract_subheading_notes_from_lines(
    lines: List[str],
    start_idx: int,
    section_num: Optional[int],
    section_title: Optional[str],
    chapter_num: int,
    chapter_title: str
) -> List[TariffNote]:
    """소호주: 섹션에서 소호주 추출"""

    notes = []
    i = start_idx

    # "소호주:" 라인 찾기
    while i < len(lines):
        if lines[i].strip() == "소호주:":
            break
        # 다음 류가 시작되면 중단
        if re.match(r'^제\d+류$', lines[i].strip()):
            return notes
        i += 1

    if i >= len(lines) or lines[i].strip() != "소호주:":
        return notes

    i += 1  # "소호주:" 다음 줄부터

    # 소호주 항목 추출
    current_note_num = None
    current_note_lines = []

    while i < len(lines):
        line = lines[i].strip()

        # 다음 류 시작하면 중단
        if re.match(r'^제\d+류$', line):
            break

        # HS 코드 테이블 시작
        if line == "번 호" or re.match(r'^\d{4}$', line):
            break

        # 주 번호 패턴
        note_match = re.match(r'^(\d+|[가-힣])\.\s+(.+)$', line)
        if note_match:
            # 이전 소호주 저장
            if current_note_num and current_note_lines:
                note_content = ' '.join(current_note_lines).strip()
                if len(note_content) > 10:
                    notes.append(TariffNote(
                        level='subheading',
                        section_num=section_num,
                        section_title=section_title,
                        chapter_num=chapter_num,
                        chapter_title=chapter_title,
                        note_number=current_note_num,
                        note_content=note_content
                    ))

            # 새 소호주 시작
            current_note_num = note_match.group(1)
            current_note_lines = [note_match.group(2)]
        elif current_note_num and line:
            # 기존 소호주 내용 계속
            current_note_lines.append(line)

        i += 1

    # 마지막 소호주 저장
    if current_note_num and current_note_lines:
        note_content = ' '.join(current_note_lines).strip()
        if len(note_content) > 10:
            notes.append(TariffNote(
                level='subheading',
                section_num=section_num,
                section_title=section_title,
                chapter_num=chapter_num,
                chapter_title=chapter_title,
                note_number=current_note_num,
                note_content=note_content
            ))

    return notes


def save_notes(notes: List[TariffNote], output_dir: str):
    """추출한 주를 JSON과 TXT로 저장"""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # JSON 저장
    json_path = output_path / 'tariff_notes_clean.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(note) for note in notes], f, ensure_ascii=False, indent=2)
    print(f"\nJSON saved: {json_path} ({len(notes)} notes)")

    # 사람이 읽기 쉬운 TXT 저장
    txt_path = output_path / 'tariff_notes_clean.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        current_section = None
        current_chapter = None

        for note in notes:
            # 부가 바뀔 때
            if note.section_num != current_section:
                f.write(f"\n{'='*80}\n")
                f.write(f"제{note.section_num}부: {note.section_title}\n")
                f.write(f"{'='*80}\n")
                current_section = note.section_num
                current_chapter = None

            # 류가 바뀔 때
            if note.chapter_num != current_chapter:
                if note.chapter_num:
                    f.write(f"\n{'-'*80}\n")
                    f.write(f"제{note.chapter_num:02d}류: {note.chapter_title}\n")
                    f.write(f"{'-'*80}\n")
                    current_chapter = note.chapter_num

            level_name = {'section': '부주', 'chapter': '류주', 'subheading': '소호주'}
            f.write(f"\n[{level_name[note.level]} {note.note_number}]\n")
            f.write(f"{note.note_content}\n")

    print(f"TXT saved: {txt_path}")

    # 통계 출력
    print(f"\n{'='*80}")
    print(f"Statistics")
    print(f"{'='*80}")

    section_notes = [n for n in notes if n.level == 'section']
    chapter_notes = [n for n in notes if n.level == 'chapter']
    subheading_notes = [n for n in notes if n.level == 'subheading']

    print(f"Total notes: {len(notes)}")
    print(f"  - Section notes: {len(section_notes)}")
    print(f"  - Chapter notes: {len(chapter_notes)}")
    print(f"  - Subheading notes: {len(subheading_notes)}")

    # 부별 통계
    sections_with_notes = {}
    for note in section_notes:
        sections_with_notes[note.section_num] = sections_with_notes.get(note.section_num, 0) + 1

    print(f"\nSections with notes: {len(sections_with_notes)}")

    # 류별 통계
    chapters_with_notes = {}
    for note in notes:
        if note.chapter_num:
            chapters_with_notes[note.chapter_num] = chapters_with_notes.get(note.chapter_num, 0) + 1

    print(f"Chapters with notes: {len(chapters_with_notes)}")
    print(f"\nTop 10 chapters by note count:")
    for ch, cnt in sorted(chapters_with_notes.items(), key=lambda x: x[1], reverse=True)[:10]:
        ch_note = next((n for n in notes if n.chapter_num == ch), None)
        ch_title = ch_note.chapter_title[:30] if ch_note else ""
        print(f"  Chapter {ch:02d} ({ch_title}): {cnt} notes")


def main():
    txt_path = "law0015562022123119186KC_000000E.txt"
    output_dir = "data"

    print("="*80)
    print("Parsing tariff notes")
    print("="*80)

    notes = parse_tariff_notes(txt_path)
    save_notes(notes, output_dir)

    print(f"\n{'='*80}")
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()
