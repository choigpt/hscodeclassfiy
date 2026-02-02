"""
관세율표 XHTML에서 각 류(類)별 주(註) 규정 추출
"""
import re
import json
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ChapterNote:
    """류(章) 주석"""
    chapter_num: Optional[int]  # 류 번호 (부 주석의 경우 None)
    chapter_title: str  # 류 제목
    section_num: Optional[int]  # 부 번호
    section_title: Optional[str]  # 부 제목
    note_type: str  # 'chapter_note', 'section_note', 'subheading_note'
    note_number: Optional[str]  # 주 번호 (예: "1", "2", "가", "나")
    note_content: str  # 주 내용
    raw_text: str  # 원문


def extract_text_from_element(element) -> str:
    """HTML 요소에서 텍스트 추출 (공백 정규화)"""
    if element is None:
        return ""
    text = element.get_text(separator=" ", strip=True)
    # 중복 공백 제거
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_chapter_number(text: str) -> Optional[int]:
    """
    텍스트에서 류 번호 추출
    예: "제1류", "제 01류", "제01류" -> 1
    """
    match = re.search(r'제\s*0*(\d+)\s*류', text)
    if match:
        return int(match.group(1))
    return None


def parse_section_number(text: str) -> Optional[int]:
    """
    텍스트에서 부 번호 추출
    예: "제1부", "제 I 부" -> 1
    """
    # 로마 숫자 매핑
    roman_to_int = {
        'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
        'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
        'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
        'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,
        'XXI': 21, 'XXII': 22
    }

    match = re.search(r'제\s*([IVX]+)\s*부', text)
    if match:
        roman = match.group(1)
        return roman_to_int.get(roman)

    match = re.search(r'제\s*0*(\d+)\s*부', text)
    if match:
        return int(match.group(1))

    return None


def is_note_heading(text: str) -> bool:
    """주(註) 제목인지 확인"""
    patterns = [
        r'^주\s*$',
        r'^주\(註\)',
        r'^\d+\.\s*주',
        r'^소호주',
        r'^호주',
        r'^류주'
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


def extract_notes_from_xhtml(xhtml_path: str) -> List[ChapterNote]:
    """XHTML 파일에서 모든 주(註) 추출"""

    print(f"Reading XHTML file: {xhtml_path}")
    with open(xhtml_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    notes = []
    current_section_num = None
    current_section_title = None
    current_chapter_num = None
    current_chapter_title = None

    # 모든 문단과 테이블 셀 순회
    all_elements = soup.find_all(['p', 'td', 'div'])

    print(f"Total elements found: {len(all_elements)}")

    # 로그 파일에 기록 (콘솔 출력 대신)
    log_file = open("data/extraction.log", 'w', encoding='utf-8')

    def log(msg):
        """로그 출력 (파일만)"""
        log_file.write(msg + '\n')
        log_file.flush()  # 즉시 쓰기

    in_note_section = False
    note_buffer = []
    note_type = None
    note_number = None

    for idx, elem in enumerate(all_elements):
        text = extract_text_from_element(elem)

        if not text:
            continue

        # 부(Section) 감지
        section_num = parse_section_number(text)
        if section_num:
            current_section_num = section_num
            current_section_title = text
            log(f"\n[Section {section_num}] {text}")

        # 류(Chapter) 감지
        chapter_num = parse_chapter_number(text)
        if chapter_num:
            current_chapter_num = chapter_num
            current_chapter_title = text
            log(f"\n[Chapter {chapter_num}] {text}")

        # 주(註) 제목 감지
        if is_note_heading(text):
            # 이전 주 저장
            if note_buffer:
                note_content = ' '.join(note_buffer)
                notes.append(ChapterNote(
                    chapter_num=current_chapter_num,
                    chapter_title=current_chapter_title or "",
                    section_num=current_section_num,
                    section_title=current_section_title,
                    note_type=note_type or 'chapter_note',
                    note_number=note_number,
                    note_content=note_content,
                    raw_text=note_content
                ))
                note_buffer = []

            in_note_section = True
            if '소호주' in text:
                note_type = 'subheading_note'
            elif '호주' in text or '류주' in text:
                note_type = 'chapter_note'
            else:
                note_type = 'chapter_note'

            log(f"  -> Found note section: {text}")
            continue

        # 주 번호 감지 (1., 2., 가., 나. 등)
        if in_note_section:
            # 새로운 주 항목인지 확인
            note_num_match = re.match(r'^([0-9]+|[가-힣])\.\s+', text)
            if note_num_match:
                # 이전 주 저장
                if note_buffer:
                    note_content = ' '.join(note_buffer)
                    notes.append(ChapterNote(
                        chapter_num=current_chapter_num,
                        chapter_title=current_chapter_title or "",
                        section_num=current_section_num,
                        section_title=current_section_title,
                        note_type=note_type or 'chapter_note',
                        note_number=note_number,
                        note_content=note_content,
                        raw_text=note_content
                    ))
                    note_buffer = []

                note_number = note_num_match.group(1)
                note_buffer.append(text)
                log(f"    Note {note_number}: {text[:100]}...")
            elif note_buffer:
                # 주 내용 계속
                note_buffer.append(text)

            # 주 섹션 종료 조건 (다음 류/호 시작)
            if re.match(r'^\d{4}\.\d{2}', text):  # HS Code 시작
                if note_buffer:
                    note_content = ' '.join(note_buffer)
                    notes.append(ChapterNote(
                        chapter_num=current_chapter_num,
                        chapter_title=current_chapter_title or "",
                        section_num=current_section_num,
                        section_title=current_section_title,
                        note_type=note_type or 'chapter_note',
                        note_number=note_number,
                        note_content=note_content,
                        raw_text=note_content
                    ))
                    note_buffer = []
                in_note_section = False

    # 마지막 주 저장
    if note_buffer:
        note_content = ' '.join(note_buffer)
        notes.append(ChapterNote(
            chapter_num=current_chapter_num,
            chapter_title=current_chapter_title or "",
            section_num=current_section_num,
            section_title=current_section_title,
            note_type=note_type or 'chapter_note',
            note_number=note_number,
            note_content=note_content,
            raw_text=note_content
        ))

    log(f"\n\nTotal notes extracted: {len(notes)}")
    log_file.close()
    return notes


def save_notes_to_json(notes: List[ChapterNote], output_path: str):
    """추출한 주를 JSON 파일로 저장"""
    data = [asdict(note) for note in notes]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(notes)} notes to {output_path}")


def main():
    """메인 실행 함수"""
    xhtml_path = "law0015562022123119186KC_000000E/index.xhtml"
    output_json = "data/tariff_notes.json"
    output_txt = "data/tariff_notes.txt"

    # 출력 디렉토리 생성
    Path("data").mkdir(exist_ok=True)

    # 주 규정 추출
    notes = extract_notes_from_xhtml(xhtml_path)

    # JSON 저장
    save_notes_to_json(notes, output_json)

    # 텍스트 포맷으로도 저장 (사람이 읽기 쉽게)
    with open(output_txt, 'w', encoding='utf-8') as f:
        for note in notes:
            f.write(f"\n{'='*80}\n")
            if note.section_num:
                f.write(f"제{note.section_num}부: {note.section_title}\n")
            if note.chapter_num:
                f.write(f"제{note.chapter_num}류: {note.chapter_title}\n")
            f.write(f"주 유형: {note.note_type}\n")
            if note.note_number:
                f.write(f"주 번호: {note.note_number}\n")
            f.write(f"\n{note.note_content}\n")

    print(f"Saved human-readable format to {output_txt}")

    # 통계 출력
    chapter_notes = [n for n in notes if n.note_type == 'chapter_note']
    subheading_notes = [n for n in notes if n.note_type == 'subheading_note']

    print(f"\n=== Statistics ===")
    print(f"Total notes: {len(notes)}")
    print(f"Chapter notes: {len(chapter_notes)}")
    print(f"Subheading notes: {len(subheading_notes)}")

    # 류별 주 개수
    chapters_with_notes = {}
    for note in notes:
        if note.chapter_num:
            chapters_with_notes[note.chapter_num] = chapters_with_notes.get(note.chapter_num, 0) + 1

    print(f"\nChapters with notes: {len(chapters_with_notes)}")
    print(f"Top 5 chapters by note count:")
    for ch, cnt in sorted(chapters_with_notes.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Chapter {ch}: {cnt} notes")


if __name__ == "__main__":
    main()
