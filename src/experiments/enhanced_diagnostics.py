"""
Enhanced Diagnostics - 진단 시스템 강화

1. LegalGate 제외 분석: note_id/규칙타입별 집계
2. Missing Facts 분석: axis 분포, 질문 후 개선율
3. Confusion Pairs: Top20 혼동 쌍 저장
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter


@dataclass
class LegalGateExclusionCase:
    """LegalGate 제외 케이스"""
    sample_id: str
    text: str
    true_hs4: str
    exclude_conflict_score: float
    triggered_notes: List[str] = field(default_factory=list)  # note_id 리스트
    note_types: List[str] = field(default_factory=list)  # exclude/redirect 등


@dataclass
class MissingFactsCase:
    """Missing Facts 케이스"""
    sample_id: str
    text: str
    true_hs4: str
    missing_hard: List[Dict[str, Any]] = field(default_factory=list)  # [{axis, value, operator}, ...]
    questions_generated: List[str] = field(default_factory=list)


@dataclass
class ConfusionPair:
    """혼동 쌍 (정답 vs 오예측)"""
    true_hs4: str
    pred_hs4: str
    count: int = 1
    samples: List[str] = field(default_factory=list)  # sample_id 리스트


class EnhancedDiagnostics:
    """
    진단 시스템 강화

    분석 항목:
    1. LegalGate 제외: note별 집계, 규칙 타입별 집계
    2. Missing Facts: axis 분포, 질문 효과
    3. Confusion Pairs: 자주 혼동되는 HS4 쌍
    """

    def __init__(self):
        # 수집된 케이스
        self.legal_gate_exclusions: List[LegalGateExclusionCase] = []
        self.missing_facts_cases: List[MissingFactsCase] = []
        self.confusion_pairs: Dict[Tuple[str, str], ConfusionPair] = {}

    def collect_legal_gate_exclusion(
        self,
        sample_id: str,
        text: str,
        true_hs4: str,
        legal_gate_debug: Dict[str, Any]
    ) -> None:
        """
        LegalGate 제외 케이스 수집

        Args:
            sample_id: 샘플 ID
            text: 입력 텍스트
            true_hs4: 정답 HS4
            legal_gate_debug: LegalGate 디버그 정보
        """
        results = legal_gate_debug.get('results', {})
        answer_result = results.get(true_hs4)

        if not answer_result or answer_result.get('passed', True):
            # 정답이 제외되지 않음
            return

        # 제외 케이스 생성
        case = LegalGateExclusionCase(
            sample_id=sample_id,
            text=text,
            true_hs4=true_hs4,
            exclude_conflict_score=answer_result.get('exclude_conflict_score', 0.0)
        )

        # 트리거된 note 추출
        evidence = answer_result.get('evidence', [])
        for ev in evidence:
            if ev['kind'] in ['exclude_rule', 'redirect']:
                source_id = ev.get('source_id', '')
                case.triggered_notes.append(source_id)
                case.note_types.append(ev['kind'])

        self.legal_gate_exclusions.append(case)

    def collect_missing_facts(
        self,
        sample_id: str,
        text: str,
        true_hs4: str,
        fact_check_result: Dict[str, Any]
    ) -> None:
        """
        Missing Facts 케이스 수집

        Args:
            sample_id: 샘플 ID
            text: 입력 텍스트
            true_hs4: 정답 HS4
            fact_check_result: FactCheck 결과
        """
        if fact_check_result.get('sufficient', True):
            # 충분한 사실 정보 있음
            return

        # Missing hard facts 추출
        candidates_missing = fact_check_result.get('candidates_missing', {})
        answer_missing = candidates_missing.get(true_hs4)

        if not answer_missing:
            return

        missing_hard = answer_missing.get('missing_hard', [])
        if not missing_hard:
            return

        # 케이스 생성
        case = MissingFactsCase(
            sample_id=sample_id,
            text=text,
            true_hs4=true_hs4,
            missing_hard=[
                {
                    'axis': fact['axis'],
                    'operator': fact['operator'],
                    'value': fact['value']
                }
                for fact in missing_hard
            ],
            questions_generated=fact_check_result.get('questions', [])
        )

        self.missing_facts_cases.append(case)

    def collect_confusion(
        self,
        true_hs4: str,
        pred_hs4: str,
        sample_id: str
    ) -> None:
        """
        Confusion 쌍 수집 (오예측)

        Args:
            true_hs4: 정답 HS4
            pred_hs4: 예측 HS4
            sample_id: 샘플 ID
        """
        if true_hs4 == pred_hs4:
            # 정답이면 confusion 아님
            return

        pair_key = (true_hs4, pred_hs4)

        if pair_key in self.confusion_pairs:
            self.confusion_pairs[pair_key].count += 1
            self.confusion_pairs[pair_key].samples.append(sample_id)
        else:
            self.confusion_pairs[pair_key] = ConfusionPair(
                true_hs4=true_hs4,
                pred_hs4=pred_hs4,
                count=1,
                samples=[sample_id]
            )

    def analyze_legal_gate_exclusions(self) -> Dict[str, Any]:
        """
        LegalGate 제외 분석

        Returns:
            분석 결과
        """
        total = len(self.legal_gate_exclusions)

        # note_id별 집계
        note_counter = Counter()
        note_type_counter = Counter()

        for case in self.legal_gate_exclusions:
            for note_id in case.triggered_notes:
                note_counter[note_id] += 1
            for note_type in case.note_types:
                note_type_counter[note_type] += 1

        # Top 20 note
        top_notes = note_counter.most_common(20)

        return {
            'total_exclusions': total,
            'note_type_distribution': dict(note_type_counter),
            'top_triggered_notes': [
                {'note_id': note_id, 'count': count}
                for note_id, count in top_notes
            ],
        }

    def analyze_missing_facts(self) -> Dict[str, Any]:
        """
        Missing Facts 분석

        Returns:
            분석 결과
        """
        total = len(self.missing_facts_cases)

        # axis 분포
        axis_counter = Counter()
        for case in self.missing_facts_cases:
            for fact in case.missing_hard:
                axis_counter[fact['axis']] += 1

        # 질문 생성 통계
        with_questions = sum(1 for case in self.missing_facts_cases if case.questions_generated)
        avg_questions = sum(len(case.questions_generated) for case in self.missing_facts_cases) / total if total > 0 else 0

        return {
            'total_cases': total,
            'cases_with_questions': with_questions,
            'avg_questions_per_case': round(avg_questions, 2),
            'axis_distribution': dict(axis_counter),
        }

    def get_top_confusion_pairs(self, topk: int = 20) -> List[Dict[str, Any]]:
        """
        Top-K Confusion 쌍 반환

        Args:
            topk: 반환할 상위 개수

        Returns:
            Confusion 쌍 리스트
        """
        # count 내림차순 정렬
        sorted_pairs = sorted(
            self.confusion_pairs.values(),
            key=lambda x: -x.count
        )

        return [
            {
                'true_hs4': pair.true_hs4,
                'pred_hs4': pair.pred_hs4,
                'count': pair.count,
                'sample_count': len(pair.samples),
                'samples': pair.samples[:5],  # 최대 5개 샘플 예시
            }
            for pair in sorted_pairs[:topk]
        ]

    def save_diagnostics(self, output_dir: str) -> None:
        """
        진단 결과 저장

        Args:
            output_dir: 출력 디렉토리
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. LegalGate 제외 분석
        legal_analysis = self.analyze_legal_gate_exclusions()
        with open(output_path / "legal_gate_exclusions.json", 'w', encoding='utf-8') as f:
            json.dump(legal_analysis, f, ensure_ascii=False, indent=2)

        # 2. LegalGate 제외 케이스 저장 (JSONL)
        with open(output_path / "legal_gate_exclusion_cases.jsonl", 'w', encoding='utf-8') as f:
            for case in self.legal_gate_exclusions:
                f.write(json.dumps({
                    'sample_id': case.sample_id,
                    'text': case.text[:100],  # 최대 100자
                    'true_hs4': case.true_hs4,
                    'exclude_conflict_score': round(case.exclude_conflict_score, 4),
                    'triggered_notes': case.triggered_notes,
                    'note_types': case.note_types,
                }, ensure_ascii=False) + '\n')

        # 3. Missing Facts 분석
        missing_analysis = self.analyze_missing_facts()
        with open(output_path / "missing_facts_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(missing_analysis, f, ensure_ascii=False, indent=2)

        # 4. Missing Facts 케이스 저장 (JSONL)
        with open(output_path / "missing_facts_cases.jsonl", 'w', encoding='utf-8') as f:
            for case in self.missing_facts_cases:
                f.write(json.dumps({
                    'sample_id': case.sample_id,
                    'text': case.text[:100],
                    'true_hs4': case.true_hs4,
                    'missing_hard': case.missing_hard,
                    'questions_generated': case.questions_generated,
                }, ensure_ascii=False) + '\n')

        # 5. Confusion Pairs (Top 20)
        top_confusion = self.get_top_confusion_pairs(topk=20)
        with open(output_path / "confusion_pairs_top20.json", 'w', encoding='utf-8') as f:
            json.dump(top_confusion, f, ensure_ascii=False, indent=2)

        # 6. CSV 요약
        import csv

        # LegalGate 제외 note별 CSV
        with open(output_path / "legal_gate_notes.csv", 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['note_id', 'count'])
            for item in legal_analysis['top_triggered_notes']:
                writer.writerow([item['note_id'], item['count']])

        # Missing Facts axis별 CSV
        with open(output_path / "missing_facts_axis.csv", 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['axis', 'count'])
            for axis, count in sorted(missing_analysis['axis_distribution'].items(), key=lambda x: -x[1]):
                writer.writerow([axis, count])

        # Confusion Pairs CSV
        with open(output_path / "confusion_pairs.csv", 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['true_hs4', 'pred_hs4', 'count'])
            for pair in top_confusion:
                writer.writerow([pair['true_hs4'], pair['pred_hs4'], pair['count']])

        print(f"진단 결과 저장: {output_path}")

    def print_summary(self) -> None:
        """진단 요약 출력"""
        print("\n" + "=" * 60)
        print("DIAGNOSTICS SUMMARY")
        print("=" * 60)

        # 1. LegalGate 제외
        legal_analysis = self.analyze_legal_gate_exclusions()
        print(f"\n[LegalGate 제외]")
        print(f"  총 제외 케이스: {legal_analysis['total_exclusions']}")
        print(f"  규칙 타입 분포:")
        for note_type, count in legal_analysis['note_type_distribution'].items():
            print(f"    - {note_type}: {count}")
        print(f"  Top 5 트리거된 note:")
        for item in legal_analysis['top_triggered_notes'][:5]:
            print(f"    - {item['note_id']}: {item['count']}회")

        # 2. Missing Facts
        missing_analysis = self.analyze_missing_facts()
        print(f"\n[Missing Facts]")
        print(f"  총 케이스: {missing_analysis['total_cases']}")
        print(f"  질문 생성된 케이스: {missing_analysis['cases_with_questions']}")
        print(f"  평균 질문 수: {missing_analysis['avg_questions_per_case']:.2f}")
        print(f"  Axis 분포:")
        for axis, count in sorted(missing_analysis['axis_distribution'].items(), key=lambda x: -x[1])[:5]:
            print(f"    - {axis}: {count}")

        # 3. Confusion Pairs
        top_confusion = self.get_top_confusion_pairs(topk=10)
        print(f"\n[Confusion Pairs] (Top 10)")
        for i, pair in enumerate(top_confusion, 1):
            print(f"  {i}. {pair['true_hs4']} → {pair['pred_hs4']}: {pair['count']}회")


def collect_diagnostics_from_pipeline(
    samples: List[Any],
    predictions: List[List[Tuple[str, float]]],
    pipeline_debugs: List[Dict[str, Any]]
) -> EnhancedDiagnostics:
    """
    파이프라인 실행 결과에서 진단 정보 수집

    Args:
        samples: 데이터 샘플
        predictions: 예측 결과
        pipeline_debugs: 파이프라인 디버그 정보

    Returns:
        EnhancedDiagnostics
    """
    diagnostics = EnhancedDiagnostics()

    for i, sample in enumerate(samples):
        debug = pipeline_debugs[i] if i < len(pipeline_debugs) else {}
        pred = predictions[i] if i < len(predictions) else []

        # 1. LegalGate 제외 수집
        if 'legal_gate' in debug:
            diagnostics.collect_legal_gate_exclusion(
                sample.id,
                sample.text,
                sample.hs4,
                debug['legal_gate']
            )

        # 2. Missing Facts 수집
        if 'fact_check' in debug:
            diagnostics.collect_missing_facts(
                sample.id,
                sample.text,
                sample.hs4,
                debug['fact_check']
            )

        # 3. Confusion 수집 (Top-1 예측)
        if pred:
            pred_hs4 = pred[0][0]
            diagnostics.collect_confusion(
                sample.hs4,
                pred_hs4,
                sample.id
            )

    return diagnostics
