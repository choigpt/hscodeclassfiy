"""
HS Retriever - 임베딩 기반 Top-K 후보 생성
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Set
import joblib

from .types import Candidate


class HSRetriever:
    """
    Sentence Transformer 임베딩 + Logistic Regression 기반 후보 생성
    """

    def __init__(
        self,
        st_model_name: str = "jhgan/ko-sroberta-multitask",
        lr_path: Optional[str] = None,
        label_encoder_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.st_model_name = st_model_name
        self.lr_path = lr_path or "artifacts/classifier/model_lr.joblib"
        self.label_encoder_path = label_encoder_path or "artifacts/classifier/label_encoder.joblib"
        self.device = device

        self.st_model = None
        self.lr_model = None
        self.label_encoder = None

        self._load_models()

    def _load_models(self):
        """모델 로드"""
        # Sentence Transformer
        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(self.st_model_name, device=self.device)
            print(f"[Retriever] Sentence Transformer 로드: {self.st_model_name}")
        except Exception as e:
            raise RuntimeError(f"Sentence Transformer 로드 실패: {e}")

        # Logistic Regression
        lr_file = Path(self.lr_path)
        le_file = Path(self.label_encoder_path)

        if lr_file.exists() and le_file.exists():
            self.lr_model = joblib.load(lr_file)
            self.label_encoder = joblib.load(le_file)
            print(f"[Retriever] LR 모델 로드: {lr_file}")
            print(f"[Retriever] 클래스 수: {len(self.label_encoder.classes_)}")
        else:
            print(f"[Retriever] 경고: LR 모델 없음. train_model()로 학습 필요")
            print(f"  - {lr_file}")
            print(f"  - {le_file}")

    def is_ready(self) -> bool:
        """모델이 준비됐는지 확인"""
        return self.lr_model is not None and self.label_encoder is not None

    def get_model_classes(self) -> Set[str]:
        """모델이 알고 있는 HS4 클래스 set 반환"""
        if self.label_encoder is None:
            return set()
        return set(self.label_encoder.classes_)

    def has_hs4_in_model(self, hs4: str) -> bool:
        """특정 HS4가 모델에 있는지 확인"""
        return hs4 in self.get_model_classes()

    def get_num_classes(self) -> int:
        """모델의 클래스 수 반환"""
        if self.label_encoder is None:
            return 0
        return len(self.label_encoder.classes_)

    def embed(self, text: str) -> np.ndarray:
        """텍스트 임베딩 (768차원)"""
        if self.st_model is None:
            raise RuntimeError("Sentence Transformer 모델이 로드되지 않았습니다.")

        embedding = self.st_model.encode(text, convert_to_numpy=True)
        return embedding

    def predict_topk(self, text: str, k: int = 50) -> List[Candidate]:
        """
        Top-K 후보 생성

        Args:
            text: 입력 텍스트
            k: 후보 개수

        Returns:
            Candidate 리스트 (score_ml 내림차순)
        """
        if self.lr_model is None or self.label_encoder is None:
            raise RuntimeError("LR 모델이 로드되지 않았습니다. train_model()로 학습하세요.")

        # 임베딩
        embedding = self.embed(text).reshape(1, -1)

        # 확률 예측
        proba = self.lr_model.predict_proba(embedding)[0]

        # Top-K 인덱스
        top_indices = np.argsort(proba)[-k:][::-1]

        # 후보 생성
        candidates = []
        for idx in top_indices:
            hs4 = self.label_encoder.inverse_transform([idx])[0]
            score = float(proba[idx])
            candidates.append(Candidate(hs4=hs4, score_ml=score))

        return candidates

    def train_model(
        self,
        cases_path: str = "data/ruling_cases/all_cases_full_v7.json",
        output_dir: str = "artifacts/classifier",
        test_size: float = 0.1
    ) -> Tuple[float, int]:
        """
        결정사례로 LR 모델 학습

        Args:
            cases_path: 결정사례 JSON 경로
            output_dir: 아티팩트 저장 디렉토리
            test_size: 테스트셋 비율

        Returns:
            (accuracy, num_classes)
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split

        print("[Retriever] 모델 학습 시작...")

        # 데이터 로드
        with open(cases_path, 'r', encoding='utf-8') as f:
            cases = json.load(f)

        print(f"  결정사례: {len(cases)}개")

        # 유효한 케이스만 필터링
        valid_cases = []
        for c in cases:
            hs4 = c.get('hs_heading', '')
            name = c.get('product_name', '').strip()
            if hs4 and len(hs4) == 4 and name:
                valid_cases.append({
                    'text': name,
                    'hs4': hs4
                })

        print(f"  유효 케이스: {len(valid_cases)}개")

        # 최소 3개 이상 샘플이 있는 클래스만 (train/test 분할을 위해)
        from collections import Counter
        hs4_counts = Counter(c['hs4'] for c in valid_cases)
        valid_hs4 = {hs4 for hs4, cnt in hs4_counts.items() if cnt >= 3}
        valid_cases = [c for c in valid_cases if c['hs4'] in valid_hs4]

        print(f"  필터 후: {len(valid_cases)}개, {len(valid_hs4)}개 클래스")

        # 텍스트와 라벨 분리
        texts = [c['text'] for c in valid_cases]
        labels = [c['hs4'] for c in valid_cases]

        # 라벨 인코딩
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)

        # 임베딩
        print("  임베딩 생성 중...")
        embeddings = self.st_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        # Train/Test 분할 (클래스 수가 많아 stratify 사용 안 함)
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, y, test_size=test_size, random_state=42
        )

        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

        # 학습
        print("  LR 학습 중...")
        lr_model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            n_jobs=-1,
            verbose=1
        )
        lr_model.fit(X_train, y_train)

        # 평가
        accuracy = lr_model.score(X_test, y_test)
        print(f"  Test Accuracy: {accuracy:.4f}")

        # 저장
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        lr_file = output_path / "model_lr.joblib"
        le_file = output_path / "label_encoder.joblib"

        joblib.dump(lr_model, lr_file)
        joblib.dump(label_encoder, le_file)

        print(f"  저장: {lr_file}")
        print(f"  저장: {le_file}")

        # 내부 상태 업데이트
        self.lr_model = lr_model
        self.label_encoder = label_encoder
        self.lr_path = str(lr_file)
        self.label_encoder_path = str(le_file)

        return accuracy, len(label_encoder.classes_)


# 단독 실행 시 학습
if __name__ == "__main__":
    retriever = HSRetriever()
    if retriever.lr_model is None:
        retriever.train_model()
