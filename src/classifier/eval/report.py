"""
Report Generator - 평가 리포트 자동 생성

artifacts/eval/<run_id>/ 하위 산출물을 읽어 마크다운 리포트 생성
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class ReportGenerator:
    """
    평가 리포트 생성기

    산출물:
    - docs/EVAL_REPORT_<run_id>.md
    """

    def __init__(self, run_dir: str):
        """
        Args:
            run_dir: artifacts/eval/<run_id> 경로
        """
        self.run_dir = Path(run_dir)
        self.run_id = self.run_dir.name

        # 데이터 로드
        self.config = self._load_json('config.json')
        self.metrics = self._load_json('metrics_summary.json')
        self.usage = self._load_json('usage_summary.json')

    def _load_json(self, filename: str) -> Dict[str, Any]:
        """JSON 파일 로드"""
        file_path = self.run_dir / filename
        if not file_path.exists():
            return {}

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate(self, output_path: Optional[str] = None) -> str:
        """
        리포트 생성

        Args:
            output_path: 출력 경로 (None이면 docs/EVAL_REPORT_<run_id>.md)

        Returns:
            리포트 경로
        """
        if output_path is None:
            output_path = f"docs/EVAL_REPORT_{self.run_id}.md"

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 리포트 생성
        report_lines = []

        # 헤더
        report_lines.extend(self._generate_header())
        report_lines.append("")

        # 섹션 1: 실험 설정
        report_lines.extend(self._generate_config_section())
        report_lines.append("")

        # 섹션 2: 지표 정의
        report_lines.extend(self._generate_metrics_definition())
        report_lines.append("")

        # 섹션 3: 정량 결과
        report_lines.extend(self._generate_quantitative_results())
        report_lines.append("")

        # 섹션 4: Bucket 분석
        report_lines.extend(self._generate_bucket_analysis())
        report_lines.append("")

        # 섹션 5: Usage Audit
        report_lines.extend(self._generate_usage_audit())
        report_lines.append("")

        # 푸터
        report_lines.extend(self._generate_footer())

        # 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"리포트 생성: {output_file}")
        return str(output_file)

    def _generate_header(self) -> List[str]:
        """헤더 생성"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode = self.config.get('mode', 'unknown')

        return [
            f"# HS4 Classification Evaluation Report",
            "",
            f"**Run ID**: `{self.run_id}`",
            f"**Mode**: {mode}",
            f"**Generated**: {timestamp}",
            "",
            "---",
        ]

    def _generate_config_section(self) -> List[str]:
        """실험 설정 섹션"""
        mode = self.config.get('mode', 'unknown')
        seed = self.config.get('seed', 42)
        topk = self.config.get('topk', 100)
        dataset_path = self.config.get('dataset_path', 'N/A')
        limit = self.config.get('limit')

        pipeline_config = self.config.get('pipeline_config', {})

        lines = [
            "## 1. 실험 설정",
            "",
            "### 데이터",
            f"- **Dataset**: `{dataset_path}`",
            f"- **Split**: Train {self.config.get('train_ratio', 0.8)*100:.0f}% / "
            f"Val {self.config.get('val_ratio', 0.1)*100:.0f}% / "
            f"Test {self.config.get('test_ratio', 0.1)*100:.0f}%",
            f"- **Seed**: {seed}",
        ]

        if limit:
            lines.append(f"- **샘플 제한**: {limit} (테스트용)")

        lines.extend([
            "",
            "### 모델 설정",
            f"- **Mode**: `{mode}`",
            f"- **Top-K**: {topk}",
            "",
            "### Pipeline 구성",
        ])

        for key, value in pipeline_config.items():
            status = "✅" if value else "❌"
            lines.append(f"- **{key}**: {status}")

        if mode == 'kb_only':
            lines.extend([
                "",
                "**KB-only 모드**:",
                "- ML retriever: OFF",
                "- Ranker: OFF",
                "- LegalGate + FactCheck + KB 카드/규칙만 사용",
            ])
        elif mode == 'hybrid':
            lines.extend([
                "",
                "**Hybrid 모드**:",
                "- ML retriever: ON (가능한 경우)",
                "- LegalGate + FactCheck + Ranker: ON",
                "- 전체 파이프라인 사용",
            ])

        return lines

    def _generate_metrics_definition(self) -> List[str]:
        """지표 정의 섹션"""
        return [
            "## 2. 지표 정의",
            "",
            "### A) Top-k Accuracy",
            "정답 HS4가 Top-K 예측에 포함된 비율",
            "- **Top-1**: 정확도 (첫 번째 예측이 정답)",
            "- **Top-3**: 상위 3개 중 정답 포함",
            "- **Top-5**: 상위 5개 중 정답 포함",
            "",
            "### B) Macro F1",
            "클래스별 F1 score의 평균 (불균형 데이터에 강건)",
            "",
            "### C) Candidate Recall@K",
            "Retrieval 단계 후보에 정답이 포함되는 비율",
            "- Reranking 효과 측정 지표",
            "",
            "### D) Calibration",
            "- **ECE (Expected Calibration Error)**: 모델 신뢰도와 실제 정확도의 차이",
            "- **Brier Score**: 확률 예측의 MSE",
            "",
            "### E) Routing",
            "결정 상태별 비율",
            "- **AUTO**: 자동 분류 가능",
            "- **ASK**: 추가 정보 필요",
            "- **REVIEW**: 전문가 검토 필요",
            "- **ABSTAIN**: 분류 불가",
            "",
            "### F) Legal Conflict Rate",
            "최종 예측이 LegalGate의 hard-exclude/redirect 규범과 충돌하는 비율",
            "- **목표**: 0에 가깝게 유지 (법 준수)",
            "",
            "### G) Fact Missing Stats",
            "FactCheck에서 missing_hard가 발생한 비율 및 axis 분포",
            "",
            "### H) Confusion Pairs",
            "자주 혼동되는 HS4 쌍 (y_true → y_pred)",
        ]

    def _generate_quantitative_results(self) -> List[str]:
        """정량 결과 섹션"""
        total = self.metrics.get('total_samples', 0)

        lines = [
            "## 3. 정량 결과",
            "",
            f"**Total Samples**: {total}",
            "",
            "### 성능 지표",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Top-1 Accuracy | {self.metrics.get('top1_accuracy', 0):.4f} |",
            f"| Top-3 Accuracy | {self.metrics.get('top3_accuracy', 0):.4f} |",
            f"| Top-5 Accuracy | {self.metrics.get('top5_accuracy', 0):.4f} |",
            f"| Macro F1 | {self.metrics.get('macro_f1', 0):.4f} |",
            f"| Weighted F1 | {self.metrics.get('weighted_f1', 0):.4f} |",
            "",
            "### Candidate Recall",
            "",
            "| K | Recall |",
            "|---|--------|",
            f"| 5 | {self.metrics.get('candidate_recall_5', 0):.4f} |",
            f"| 10 | {self.metrics.get('candidate_recall_10', 0):.4f} |",
            f"| 20 | {self.metrics.get('candidate_recall_20', 0):.4f} |",
            "",
            "### Calibration",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| ECE | {self.metrics.get('ece', 0):.4f} |",
            f"| Brier Score | {self.metrics.get('brier_score', 0):.4f} |",
            "",
            "### Routing",
            "",
            "| Status | Rate |",
            "|--------|------|",
            f"| AUTO | {self.metrics.get('auto_rate', 0)*100:.1f}% |",
            f"| ASK | {self.metrics.get('ask_rate', 0)*100:.1f}% |",
            f"| REVIEW | {self.metrics.get('review_rate', 0)*100:.1f}% |",
            f"| ABSTAIN | {self.metrics.get('abstain_rate', 0)*100:.1f}% |",
            "",
            "### Legal Compliance",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Legal Conflict Rate | {self.metrics.get('legal_conflict_rate', 0):.4f} |",
            f"| Legal Conflict Count | {self.metrics.get('legal_conflict_count', 0)} |",
        ]

        # 경고
        conflict_rate = self.metrics.get('legal_conflict_rate', 0)
        if conflict_rate > 0.01:  # 1% 이상
            lines.extend([
                "",
                f"⚠️ **Warning**: Legal conflict rate가 {conflict_rate*100:.2f}%로 높습니다. "
                f"LegalGate 규범 준수를 확인하세요.",
            ])

        return lines

    def _generate_bucket_analysis(self) -> List[str]:
        """Bucket 분석 섹션"""
        fact_missing_rate = self.metrics.get('fact_missing_rate', 0)
        hard_axis_dist = self.metrics.get('missing_hard_axis_dist', {})
        soft_axis_dist = self.metrics.get('missing_soft_axis_dist', {})

        lines = [
            "## 4. Bucket 분석",
            "",
            "### Fact-Insufficient Bucket",
            f"- **비율**: {fact_missing_rate*100:.1f}%",
            "",
            "**Missing Hard Axis 분포**:",
            "",
        ]

        if hard_axis_dist:
            for axis, count in sorted(hard_axis_dist.items(), key=lambda x: -x[1]):
                lines.append(f"- {axis}: {count}")
        else:
            lines.append("- (없음)")

        lines.extend([
            "",
            "**Missing Soft Axis 분포**:",
            "",
        ])

        if soft_axis_dist:
            for axis, count in sorted(soft_axis_dist.items(), key=lambda x: -x[1]):
                lines.append(f"- {axis}: {count}")
        else:
            lines.append("- (없음)")

        lines.extend([
            "",
            "### Legal-Conflict Bucket",
            f"- **비율**: {self.metrics.get('legal_conflict_rate', 0)*100:.1f}%",
            f"- **샘플 수**: {self.metrics.get('legal_conflict_count', 0)}",
            "",
            "정답이 LegalGate에서 hard-exclude/redirect된 케이스",
            "",
            "### Confusion Pairs (Top 10)",
            "",
        ])

        confusion_pairs = self.metrics.get('confusion_pairs_top20', [])[:10]
        if confusion_pairs:
            lines.append("| True HS4 | Pred HS4 | Count |")
            lines.append("|----------|----------|-------|")
            for pair in confusion_pairs:
                lines.append(f"| {pair['true_hs4']} | {pair['pred_hs4']} | {pair['count']} |")
        else:
            lines.append("(없음)")

        return lines

    def _generate_usage_audit(self) -> List[str]:
        """Usage Audit 섹션"""
        lines = [
            "## 5. Usage Audit 요약",
            "",
            "법/자료의 실제 활용 수치",
            "",
            "### LegalGate 사용",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| 적용률 | {self.usage.get('legal_gate_usage_rate', 0)*100:.1f}% |",
            f"| 평균 제외 후보 | {self.usage.get('avg_legal_excluded', 0):.2f} |",
            f"| 평균 주규정 Support | {self.usage.get('avg_notes_support', 0):.2f} |",
            f"| 평균 주규정 Exclude | {self.usage.get('avg_notes_exclude', 0):.2f} |",
            "",
            "### FactCheck 사용",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| 적용률 | {self.usage.get('factcheck_usage_rate', 0)*100:.1f}% |",
            f"| Fact Insufficient 비율 | {self.usage.get('fact_insufficient_rate', 0)*100:.1f}% |",
            f"| 평균 생성 질문 수 | {self.usage.get('avg_questions_generated', 0):.2f} |",
            "",
            "### KB 자료 사용 (평균 근거 개수)",
            "",
            "| 자료 | 평균 |",
            "|------|------|",
            f"| 카드 (Cards) | {self.usage.get('avg_cards_hits', 0):.2f} |",
            f"| 규칙 (Rules) | {self.usage.get('avg_rule_hits', 0):.2f} |",
            f"| 해설서 (Commentary) | {self.usage.get('avg_commentary_hits', 0):.2f} |",
            f"| 결정사례 (Ruling Cases) | {self.usage.get('avg_ruling_case_hits', 0):.2f} |",
            "",
            "### Ranker 사용",
            "",
            f"- **사용률**: {self.usage.get('ranker_usage_rate', 0)*100:.1f}%",
            "",
            "### 경고: 법적 근거 없는 예측",
            "",
            f"- **비율**: {self.usage.get('no_legal_evidence_rate', 0)*100:.1f}%",
            f"- **샘플 수**: {self.usage.get('no_legal_evidence_count', 0)}",
            "",
        ]

        # 경고
        no_legal_rate = self.usage.get('no_legal_evidence_rate', 0)
        if no_legal_rate > 0.05:  # 5% 이상
            lines.extend([
                f"⚠️ **Warning**: 법적 근거 없는 예측이 {no_legal_rate*100:.1f}%로 높습니다. "
                f"LegalGate/FactCheck 활용도를 높이세요.",
            ])

        return lines

    def _generate_footer(self) -> List[str]:
        """푸터"""
        return [
            "---",
            "",
            "## 참고",
            "",
            "- **산출물 경로**: `" + str(self.run_dir) + "`",
            "- **상세 데이터**:",
            "  - `predictions_test.jsonl`: 샘플별 예측 결과",
            "  - `metrics_summary.json`: 지표 요약",
            "  - `ece_bins.csv`: ECE bin별 데이터",
            "  - `confusion_pairs.csv`: Confusion 쌍",
            "  - `usage_audit.jsonl`: 샘플별 usage 기록",
            "  - `usage_summary.json`: Usage 집계",
            "",
            "**라이선스 고려**: 이 리포트는 note_id/source_ref만 기록하며 원문을 대량 인용하지 않습니다.",
        ]


def generate_report(run_dir: str, output_path: Optional[str] = None) -> str:
    """
    리포트 생성 편의 함수

    Args:
        run_dir: artifacts/eval/<run_id> 경로
        output_path: 출력 경로 (선택적)

    Returns:
        리포트 파일 경로
    """
    generator = ReportGenerator(run_dir)
    return generator.generate(output_path)


def main():
    """CLI 엔트리포인트"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument('run_dir', help='artifacts/eval/<run_id> 경로')
    parser.add_argument('--output', help='출력 경로 (default: docs/EVAL_REPORT_<run_id>.md)')

    args = parser.parse_args()

    generate_report(args.run_dir, args.output)


if __name__ == '__main__':
    main()
