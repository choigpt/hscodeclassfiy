"""
후처리 모듈

GRI 신호 검출 + LegalGate 검증 + 신뢰도 보정
기존 src.classifier 모듈을 재사용한다.
"""

from typing import List, Dict, Any, Tuple, Optional

from .types import RAGResult, RAGCandidate, RAGEvidence, Confidence


class PostProcessor:
    """RAG 후처리기"""

    def __init__(self):
        self._gri_loaded = False
        self._legal_gate = None

    def detect_gri(self, text: str) -> Dict[str, Any]:
        """GRI 신호 검출 (기존 모듈 재사용)"""
        try:
            from src.classifier.gri_signals import analyze_text_for_gri
            return analyze_text_for_gri(text)
        except ImportError:
            return {'active_gri': [], 'signals': {}, 'any_signal': False}

    def validate_legal_gate(
        self,
        input_text: str,
        candidate_hs4s: List[str]
    ) -> Dict[str, Any]:
        """
        LegalGate 검증 (기존 모듈 재사용)

        Returns:
            {
                'passed_hs4s': [...],
                'excluded_hs4s': [...],
                'redirect_hs4s': [...],
                'results': {hs4: {passed, scores...}},
            }
        """
        try:
            from src.classifier.types import Candidate
            from src.classifier.legal_gate import LegalGate

            # 후보를 Candidate 객체로 변환
            candidates = [
                Candidate(hs4=hs4, score_total=1.0)
                for hs4 in candidate_hs4s
            ]

            if self._legal_gate is None:
                self._legal_gate = LegalGate()

            passed, redirects, debug = self._legal_gate.apply(input_text, candidates)

            passed_hs4s = [c.hs4 for c in passed]
            return {
                'passed_hs4s': passed_hs4s,
                'excluded_hs4s': debug.get('excluded_hs4s', []),
                'redirect_hs4s': redirects,
                'debug': debug,
            }
        except ImportError as e:
            # LegalGate 사용 불가 시 모든 후보 통과
            return {
                'passed_hs4s': candidate_hs4s,
                'excluded_hs4s': [],
                'redirect_hs4s': [],
                'debug': {'error': str(e)},
            }

    def calibrate_confidence(
        self,
        raw_llm_conf: float,
        top_retrieval_score: float,
        legal_gate_passed: bool,
        gri_any_signal: bool,
        is_fallback: bool
    ) -> Confidence:
        """
        신뢰도 보정

        LLM 자가 신뢰도를 검색 점수, LegalGate, GRI 신호로 보정한다.

        보정 공식:
            calibrated = w_llm * raw_llm + w_ret * retrieval_score + w_legal * legal_bonus
            - GRI 신호가 있으면 0.9를 상한으로 클리핑 (복잡한 케이스)
            - fallback이면 상한 0.5
        """
        if is_fallback:
            # LLM 실패, 검색만으로 판단
            retrieval_conf = min(top_retrieval_score * 3.0, 0.5)  # RRF 점수 스케일링
            return Confidence(
                raw_llm=0.0,
                retrieval=retrieval_conf,
                legal_gate=1.0 if legal_gate_passed else 0.3,
                calibrated=retrieval_conf * (0.8 if legal_gate_passed else 0.5),
            )

        # 정상 LLM 응답
        w_llm = 0.5
        w_ret = 0.3
        w_legal = 0.2

        retrieval_scaled = min(top_retrieval_score * 3.0, 1.0)
        legal_bonus = 1.0 if legal_gate_passed else 0.3

        calibrated = (
            w_llm * raw_llm_conf
            + w_ret * retrieval_scaled
            + w_legal * legal_bonus
        )

        # GRI 신호가 있으면 상한 0.9 (복잡한 케이스는 100% 확신 불가)
        if gri_any_signal:
            calibrated = min(calibrated, 0.9)

        calibrated = max(0.0, min(1.0, calibrated))

        return Confidence(
            raw_llm=raw_llm_conf,
            retrieval=retrieval_scaled,
            legal_gate=legal_bonus,
            calibrated=calibrated,
        )

    def apply(
        self,
        result: RAGResult,
        retrieval_results: List[Tuple[str, float]]
    ) -> RAGResult:
        """
        전체 후처리 적용

        Args:
            result: LLM 분류 결과 (또는 fallback 결과)
            retrieval_results: 검색 결과 [(hs4, score), ...]

        Returns:
            후처리 적용된 RAGResult
        """
        # 1. GRI 신호 검출
        gri_analysis = self.detect_gri(result.input_text)
        result.gri_signals = gri_analysis

        # 2. LegalGate 검증
        candidate_hs4s = [c.hs4 for c in result.candidates]
        if candidate_hs4s:
            legal_result = self.validate_legal_gate(result.input_text, candidate_hs4s)
            result.legal_gate_debug = legal_result

            # LegalGate에서 제외된 후보 마킹
            excluded = set(legal_result.get('excluded_hs4s', []))
            if excluded:
                # 제외된 후보를 뒤로 보내기
                passed = [c for c in result.candidates if c.hs4 not in excluded]
                failed = [c for c in result.candidates if c.hs4 in excluded]
                for c in failed:
                    c.evidence.append(RAGEvidence(
                        kind='legal_gate',
                        source_id=c.hs4,
                        text='LegalGate 제외',
                        score=-1.0,
                    ))
                result.candidates = passed + failed

                # best_hs4가 제외되었으면 교체
                if result.best_hs4 in excluded and passed:
                    result.best_hs4 = passed[0].hs4

        # 3. 신뢰도 보정
        top_retrieval_score = retrieval_results[0][1] if retrieval_results else 0.0
        legal_gate_passed = result.best_hs4 not in set(
            result.legal_gate_debug.get('excluded_hs4s', [])
        )

        result.confidence = self.calibrate_confidence(
            raw_llm_conf=result.confidence.raw_llm,
            top_retrieval_score=top_retrieval_score,
            legal_gate_passed=legal_gate_passed,
            gri_any_signal=gri_analysis.get('any_signal', False),
            is_fallback=result.is_fallback,
        )

        return result
