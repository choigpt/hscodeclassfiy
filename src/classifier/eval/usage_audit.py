"""
Usage Audit Module - KB/Legal 사용성 점검

목적: GRI/주규정/호용어/해설서/결정사례가 실제로 분류 과정에 활용되는지 검증
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter


@dataclass
class SampleUsageAudit:
    """샘플별 KB/Legal 사용 기록"""
    sample_id: str

    # LegalGate
    used_legal_gate: bool = False
    legal_passed_candidates: int = 0
    legal_excluded_candidates: int = 0

    # 주규정 (Notes)
    used_notes_support_count: int = 0  # include/definition/support
    used_notes_exclude_count: int = 0  # exclude/redirect

    # 호용어 (Heading Term)
    used_heading_term_match: float = 0.0

    # FactCheck
    used_factcheck: bool = False
    missing_hard_count: int = 0
    missing_soft_count: int = 0
    missing_hard_axis: List[str] = field(default_factory=list)
    questions_generated_count: int = 0
    questions_axis: List[str] = field(default_factory=list)

    # KB 자료
    used_cards_hits: int = 0
    used_rule_inc_hits: int = 0
    used_rule_exc_hits: int = 0
    used_commentary_hits: int = 0  # 해설서 근거
    used_ruling_case_hits: int = 0  # 사례 근거

    # ML Retriever
    retriever_used: bool = False

    # Ranker
    ranker_used: bool = False
    ranker_feature_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sample_id': self.sample_id,
            'used_legal_gate': self.used_legal_gate,
            'legal_passed_candidates': self.legal_passed_candidates,
            'legal_excluded_candidates': self.legal_excluded_candidates,
            'used_notes_support_count': self.used_notes_support_count,
            'used_notes_exclude_count': self.used_notes_exclude_count,
            'used_heading_term_match': round(self.used_heading_term_match, 4),
            'used_factcheck': self.used_factcheck,
            'missing_hard_count': self.missing_hard_count,
            'missing_soft_count': self.missing_soft_count,
            'missing_hard_axis': self.missing_hard_axis,
            'questions_generated_count': self.questions_generated_count,
            'questions_axis': self.questions_axis,
            'used_cards_hits': self.used_cards_hits,
            'used_rule_inc_hits': self.used_rule_inc_hits,
            'used_rule_exc_hits': self.used_rule_exc_hits,
            'used_commentary_hits': self.used_commentary_hits,
            'used_ruling_case_hits': self.used_ruling_case_hits,
            'retriever_used': self.retriever_used,
            'ranker_used': self.ranker_used,
            'ranker_feature_count': self.ranker_feature_count,
        }


@dataclass
class UsageAuditSummary:
    """Usage Audit 집계"""
    total_samples: int = 0

    # LegalGate 적용률
    legal_gate_usage_rate: float = 0.0
    avg_legal_excluded: float = 0.0

    # 주규정 사용
    avg_notes_support: float = 0.0
    avg_notes_exclude: float = 0.0

    # FactCheck
    factcheck_usage_rate: float = 0.0
    fact_insufficient_rate: float = 0.0  # missing_hard > 0
    avg_questions_generated: float = 0.0

    # KB 자료 평균 사용
    avg_cards_hits: float = 0.0
    avg_rule_hits: float = 0.0
    avg_commentary_hits: float = 0.0
    avg_ruling_case_hits: float = 0.0

    # ML Retriever 사용률
    retriever_usage_rate: float = 0.0

    # Ranker 사용률
    ranker_usage_rate: float = 0.0

    # 경고: 법적 근거 없는 예측
    no_legal_evidence_rate: float = 0.0
    no_legal_evidence_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_samples': self.total_samples,
            'legal_gate_usage_rate': round(self.legal_gate_usage_rate, 4),
            'avg_legal_excluded': round(self.avg_legal_excluded, 2),
            'avg_notes_support': round(self.avg_notes_support, 2),
            'avg_notes_exclude': round(self.avg_notes_exclude, 2),
            'factcheck_usage_rate': round(self.factcheck_usage_rate, 4),
            'fact_insufficient_rate': round(self.fact_insufficient_rate, 4),
            'avg_questions_generated': round(self.avg_questions_generated, 2),
            'avg_cards_hits': round(self.avg_cards_hits, 2),
            'avg_rule_hits': round(self.avg_rule_hits, 2),
            'avg_commentary_hits': round(self.avg_commentary_hits, 2),
            'avg_ruling_case_hits': round(self.avg_ruling_case_hits, 2),
            'retriever_usage_rate': round(self.retriever_usage_rate, 4),
            'ranker_usage_rate': round(self.ranker_usage_rate, 4),
            'no_legal_evidence_rate': round(self.no_legal_evidence_rate, 4),
            'no_legal_evidence_count': self.no_legal_evidence_count,
        }


class UsageAuditor:
    """
    KB/Legal 사용성 감사기

    샘플별로 KB/법적 자료 활용도를 기록하고 집계
    """

    def __init__(self):
        self.sample_audits: List[SampleUsageAudit] = []

    def audit_sample(
        self,
        sample_id: str,
        prediction: Dict[str, Any]
    ) -> SampleUsageAudit:
        """
        단일 샘플의 usage 감사

        Args:
            sample_id: 샘플 ID
            prediction: 예측 결과 (debug 포함)

        Returns:
            SampleUsageAudit
        """
        audit = SampleUsageAudit(sample_id=sample_id)

        debug = prediction.get('debug', {})
        topk = prediction.get('topk', [])

        # 1. LegalGate 사용
        legal_gate = debug.get('legal_gate', {})
        if legal_gate:
            audit.used_legal_gate = True

            results = legal_gate.get('results', {})
            for hs4, result in results.items():
                if result.get('passed', False):
                    audit.legal_passed_candidates += 1
                else:
                    audit.legal_excluded_candidates += 1

                # 주규정 support/exclude
                evidence = result.get('evidence', [])
                for ev in evidence:
                    if ev.get('kind') in ['include_rule', 'note_support']:
                        audit.used_notes_support_count += 1
                    elif ev.get('kind') in ['exclude_rule', 'redirect']:
                        audit.used_notes_exclude_count += 1

                # 호용어 매칭
                heading_term_score = result.get('heading_term_score', 0.0)
                audit.used_heading_term_match = max(
                    audit.used_heading_term_match,
                    heading_term_score
                )

        # 2. FactCheck 사용
        fact_check = debug.get('fact_check', {})
        if fact_check:
            audit.used_factcheck = True

            candidates_missing = fact_check.get('candidates_missing', {})
            for hs4, missing in candidates_missing.items():
                # missing_hard
                hard_facts = missing.get('missing_hard', [])
                audit.missing_hard_count += len(hard_facts)
                for fact in hard_facts:
                    axis = fact.get('axis', 'unknown')
                    if axis not in audit.missing_hard_axis:
                        audit.missing_hard_axis.append(axis)

                # missing_soft
                soft_facts = missing.get('missing_soft', [])
                audit.missing_soft_count += len(soft_facts)

            # 질문
            questions = fact_check.get('questions', [])
            audit.questions_generated_count = len(questions)
            for q in questions:
                axis = q.get('axis', 'unknown')
                if axis not in audit.questions_axis:
                    audit.questions_axis.append(axis)

        # 3. KB 자료 사용 (evidence 기반)
        for cand in topk:
            evidence = cand.get('evidence', [])
            for ev in evidence:
                kind = ev.get('kind', '')

                if kind == 'card_keyword':
                    audit.used_cards_hits += 1
                elif kind == 'include_rule':
                    audit.used_rule_inc_hits += 1
                elif kind == 'exclude_rule':
                    audit.used_rule_exc_hits += 1
                elif kind == 'commentary':
                    audit.used_commentary_hits += 1
                elif kind == 'ruling_case':
                    audit.used_ruling_case_hits += 1

        # 4. Retriever 사용
        retriever_used_from_debug = debug.get('retriever_used', False)
        if retriever_used_from_debug:
            audit.retriever_used = True

        # 5. Ranker 사용
        # debug.ranker_used 또는 debug.ranker_applied 플래그로만 판단
        # (features 존재 여부는 ranker OFF 시에도 True이므로 신뢰 불가)
        ranker_used_from_debug = debug.get('ranker_used', False)
        ranker_applied_from_debug = debug.get('ranker_applied', False)

        if ranker_used_from_debug or ranker_applied_from_debug:
            audit.ranker_used = True
            # features_count_for_ranker는 run_eval에서 ranker 사용 시에만 설정
            features_count = debug.get('features_count_for_ranker', 0)
            audit.ranker_feature_count = features_count

        return audit

    def audit_all(
        self,
        predictions: List[Dict[str, Any]]
    ) -> List[SampleUsageAudit]:
        """
        모든 샘플 감사

        Args:
            predictions: per-sample 예측 결과 리스트

        Returns:
            SampleUsageAudit 리스트
        """
        self.sample_audits = []

        for i, pred in enumerate(predictions):
            sample_id = pred.get('sample_id', f"sample_{i}")
            audit = self.audit_sample(sample_id, pred)
            self.sample_audits.append(audit)

        return self.sample_audits

    def compute_summary(self) -> UsageAuditSummary:
        """
        집계 계산

        Returns:
            UsageAuditSummary
        """
        if not self.sample_audits:
            return UsageAuditSummary()

        total = len(self.sample_audits)
        summary = UsageAuditSummary(total_samples=total)

        # LegalGate
        legal_gate_count = sum(1 for a in self.sample_audits if a.used_legal_gate)
        summary.legal_gate_usage_rate = legal_gate_count / total

        total_excluded = sum(a.legal_excluded_candidates for a in self.sample_audits)
        summary.avg_legal_excluded = total_excluded / total

        # 주규정
        total_support = sum(a.used_notes_support_count for a in self.sample_audits)
        total_exclude = sum(a.used_notes_exclude_count for a in self.sample_audits)
        summary.avg_notes_support = total_support / total
        summary.avg_notes_exclude = total_exclude / total

        # FactCheck
        factcheck_count = sum(1 for a in self.sample_audits if a.used_factcheck)
        summary.factcheck_usage_rate = factcheck_count / total

        fact_insufficient_count = sum(
            1 for a in self.sample_audits if a.missing_hard_count > 0
        )
        summary.fact_insufficient_rate = fact_insufficient_count / total

        total_questions = sum(a.questions_generated_count for a in self.sample_audits)
        summary.avg_questions_generated = total_questions / total

        # KB 자료
        total_cards = sum(a.used_cards_hits for a in self.sample_audits)
        total_rules = sum(
            a.used_rule_inc_hits + a.used_rule_exc_hits
            for a in self.sample_audits
        )
        total_commentary = sum(a.used_commentary_hits for a in self.sample_audits)
        total_ruling_case = sum(a.used_ruling_case_hits for a in self.sample_audits)

        summary.avg_cards_hits = total_cards / total
        summary.avg_rule_hits = total_rules / total
        summary.avg_commentary_hits = total_commentary / total
        summary.avg_ruling_case_hits = total_ruling_case / total

        # Retriever
        retriever_count = sum(1 for a in self.sample_audits if a.retriever_used)
        summary.retriever_usage_rate = retriever_count / total

        # Ranker
        ranker_count = sum(1 for a in self.sample_audits if a.ranker_used)
        summary.ranker_usage_rate = ranker_count / total

        # 경고: 법적 근거 없는 예측
        # (LegalGate 사용 안 했고, 주규정 근거도 없음)
        no_legal_count = sum(
            1 for a in self.sample_audits
            if not a.used_legal_gate and
               a.used_notes_support_count == 0 and
               a.used_notes_exclude_count == 0
        )
        summary.no_legal_evidence_rate = no_legal_count / total
        summary.no_legal_evidence_count = no_legal_count

        return summary

    def save_audit(
        self,
        output_dir: str
    ) -> None:
        """
        감사 결과 저장

        Args:
            output_dir: 출력 디렉토리
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. usage_audit.jsonl (샘플별)
        with open(output_path / 'usage_audit.jsonl', 'w', encoding='utf-8') as f:
            for audit in self.sample_audits:
                f.write(json.dumps(audit.to_dict(), ensure_ascii=False) + '\n')

        # 2. usage_summary.json (집계)
        summary = self.compute_summary()
        with open(output_path / 'usage_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)

        # 3. usage_summary.csv
        with open(output_path / 'usage_summary.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Samples', summary.total_samples])
            writer.writerow(['LegalGate Usage Rate', f"{summary.legal_gate_usage_rate:.4f}"])
            writer.writerow(['Avg Legal Excluded', f"{summary.avg_legal_excluded:.2f}"])
            writer.writerow(['Avg Notes Support', f"{summary.avg_notes_support:.2f}"])
            writer.writerow(['Avg Notes Exclude', f"{summary.avg_notes_exclude:.2f}"])
            writer.writerow(['FactCheck Usage Rate', f"{summary.factcheck_usage_rate:.4f}"])
            writer.writerow(['Fact Insufficient Rate', f"{summary.fact_insufficient_rate:.4f}"])
            writer.writerow(['Avg Questions Generated', f"{summary.avg_questions_generated:.2f}"])
            writer.writerow(['Avg Cards Hits', f"{summary.avg_cards_hits:.2f}"])
            writer.writerow(['Avg Rule Hits', f"{summary.avg_rule_hits:.2f}"])
            writer.writerow(['Avg Commentary Hits', f"{summary.avg_commentary_hits:.2f}"])
            writer.writerow(['Avg Ruling Case Hits', f"{summary.avg_ruling_case_hits:.2f}"])
            writer.writerow(['Ranker Usage Rate', f"{summary.ranker_usage_rate:.4f}"])
            writer.writerow(['No Legal Evidence Rate', f"{summary.no_legal_evidence_rate:.4f}"])
            writer.writerow(['No Legal Evidence Count', summary.no_legal_evidence_count])

        print(f"Usage audit saved to {output_path}")


def audit_predictions(
    predictions: List[Dict[str, Any]],
    output_dir: Optional[str] = None
) -> UsageAuditSummary:
    """
    Usage audit 편의 함수

    Args:
        predictions: per-sample 예측 결과
        output_dir: 출력 디렉토리 (선택적)

    Returns:
        UsageAuditSummary
    """
    auditor = UsageAuditor()
    auditor.audit_all(predictions)

    if output_dir:
        auditor.save_audit(output_dir)

    return auditor.compute_summary()
