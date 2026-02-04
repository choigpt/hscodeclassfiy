"""
KB-Only Model

ML 없이 KB (Knowledge Base) 만 사용하는 모델:
- LegalGate (GRI 1) 필터링
- HS4 카드 키워드 매칭
- 주규정 기반 점수 계산

Layer 1 (Law) 만 사용하는 순수 KB 기반 모델
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .baselines import BaselineModel, Prediction
from src.classifier.legal_gate import LegalGate
from src.classifier.reranker import HSReranker
from src.classifier.types import Candidate


class KBOnlyModel(BaselineModel):
    """
    KB-Only 베이스라인 (ML 없음, KB만 사용)

    파이프라인:
    1. KB에서 카드 키워드 기반 후보 생성 (ML retrieval 없음)
    2. LegalGate (GRI 1) 필터링
    3. 카드/규칙 점수로 재정렬

    특징:
    - ML 모델 없음 (학습 불필요)
    - 법 규범 (Layer 1) 만 사용
    - 빠른 추론 속도
    - 해석 가능성 높음
    """

    def __init__(
        self,
        use_legal_gate: bool = True,
        kb_topk: int = 30,
        verbose: bool = False
    ):
        """
        Args:
            use_legal_gate: LegalGate 사용 여부
            kb_topk: KB 검색 top-K
            verbose: 상세 출력
        """
        self.use_legal_gate = use_legal_gate
        self.kb_topk = kb_topk
        self.verbose = verbose

        # KB 컴포넌트
        self.reranker = HSReranker()
        self.legal_gate = LegalGate() if use_legal_gate else None

        # 학습 불필요 (KB 기반)
        self.fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """
        KB-Only는 학습 불필요 (KB는 미리 구축됨)

        Args:
            texts: 학습 텍스트 (사용 안 함)
            labels: 학습 라벨 (사용 안 함)
        """
        if self.verbose:
            print(f"[KB-Only] 학습 불필요 (KB 기반, {len(texts)} 샘플 무시)")

        self.fitted = True

    def predict(self, text: str, topk: int = 5) -> List[Prediction]:
        """
        KB 기반 예측

        Args:
            text: 입력 텍스트
            topk: 반환할 top-K

        Returns:
            예측 결과 (상위 K개)
        """
        # 1. KB에서 후보 생성 (ML 없이 카드 키워드 매칭)
        candidates = self.reranker.retrieve_from_kb(
            text,
            topk=self.kb_topk,
            gri_signals=None  # GRI 신호도 사용 안 함 (순수 KB)
        )

        if not candidates:
            # 후보 없으면 빈 리스트
            return []

        # 2. LegalGate 필터링 (선택적)
        if self.legal_gate:
            candidates, redirect_hs4s, _ = self.legal_gate.apply(text, candidates)

            # 리다이렉트 후보 추가
            if redirect_hs4s:
                for rhs4 in redirect_hs4s:
                    candidates.append(Candidate(hs4=rhs4, score_card=0.5))

        # 3. 카드/규칙 점수로 재정렬
        # (reranker.rerank()를 사용하지 않고 단순 점수 계산)
        scored_candidates = []

        for cand in candidates:
            # score_card와 score_rule 합산
            total_score = cand.score_card + cand.score_rule
            scored_candidates.append((cand.hs4, total_score))

        # 점수 내림차순 정렬
        scored_candidates.sort(key=lambda x: -x[1])

        # 4. Top-K 반환
        predictions = []
        for rank, (hs4, score) in enumerate(scored_candidates[:topk], start=1):
            predictions.append(Prediction(
                hs4=hs4,
                score=score,
                rank=rank
            ))

        return predictions

    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        확률 예측

        Args:
            text: 입력 텍스트

        Returns:
            {hs4: score} 딕셔너리
        """
        predictions = self.predict(text, topk=100)  # 전체 후보

        # 점수를 확률로 정규화 (softmax)
        import numpy as np
        scores = np.array([p.score for p in predictions])

        if len(scores) == 0:
            return {}

        # Softmax
        exp_scores = np.exp(scores - np.max(scores))  # 수치 안정성
        probs = exp_scores / exp_scores.sum()

        return {
            p.hs4: float(prob)
            for p, prob in zip(predictions, probs)
        }

    def save(self, path: str) -> None:
        """
        모델 저장 (KB-Only는 설정만 저장)

        Args:
            path: 저장 경로
        """
        import pickle

        config = {
            'type': 'kb_only',
            'use_legal_gate': self.use_legal_gate,
            'kb_topk': self.kb_topk,
            'fitted': self.fitted,
        }

        with open(path, 'wb') as f:
            pickle.dump(config, f)

    def load(self, path: str) -> None:
        """
        모델 로드

        Args:
            path: 모델 경로
        """
        import pickle

        with open(path, 'rb') as f:
            config = pickle.load(f)

        self.use_legal_gate = config['use_legal_gate']
        self.kb_topk = config['kb_topk']
        self.fitted = config['fitted']

        # KB 컴포넌트 재초기화
        self.legal_gate = LegalGate() if self.use_legal_gate else None

    def get_classes(self) -> List[str]:
        """
        KB에 있는 모든 HS4 클래스 반환

        Returns:
            HS4 리스트
        """
        # reranker에서 카드 HS4 추출
        if hasattr(self.reranker, 'cards'):
            return list(self.reranker.cards.keys())
        return []


def create_kb_only_model(
    use_legal_gate: bool = True,
    kb_topk: int = 30,
    verbose: bool = False
) -> KBOnlyModel:
    """
    KB-Only 모델 생성 편의 함수

    Args:
        use_legal_gate: LegalGate 사용 여부
        kb_topk: KB 검색 top-K
        verbose: 상세 출력

    Returns:
        KBOnlyModel 인스턴스
    """
    return KBOnlyModel(
        use_legal_gate=use_legal_gate,
        kb_topk=kb_topk,
        verbose=verbose
    )
