"""
Baseline Models - B0 TF-IDF+LR, B1 SBert+LR, B2 BM25

베이스라인 모델 구현 및 학습/평가 인터페이스
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter, defaultdict
from abc import ABC, abstractmethod
import pickle


@dataclass
class Prediction:
    """모델 예측 결과"""
    hs4: str
    score: float
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hs4": self.hs4,
            "score": round(self.score, 6),
            "rank": self.rank
        }


@dataclass
class PredictionResult:
    """전체 예측 결과"""
    sample_id: str
    text: str
    true_hs4: str
    predictions: List[Prediction]
    top1_correct: bool = False
    top3_correct: bool = False
    top5_correct: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "text": self.text,
            "true_hs4": self.true_hs4,
            "predictions": [p.to_dict() for p in self.predictions[:10]],
            "top1_correct": self.top1_correct,
            "top3_correct": self.top3_correct,
            "top5_correct": self.top5_correct,
        }


class BaselineModel(ABC):
    """베이스라인 모델 추상 클래스"""

    @abstractmethod
    def fit(self, texts: List[str], labels: List[str]) -> None:
        """모델 학습"""
        pass

    @abstractmethod
    def predict(self, text: str, topk: int = 5) -> List[Prediction]:
        """예측 수행"""
        pass

    @abstractmethod
    def predict_proba(self, text: str) -> Dict[str, float]:
        """확률 예측"""
        pass

    def predict_batch(self, texts: List[str], topk: int = 5) -> List[List[Prediction]]:
        """배치 예측"""
        return [self.predict(text, topk) for text in texts]

    @abstractmethod
    def save(self, path: str) -> None:
        """모델 저장"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """모델 로드"""
        pass

    def get_classes(self) -> List[str]:
        """학습된 클래스 목록"""
        return []


class TFIDFBaseline(BaselineModel):
    """
    B0: TF-IDF + Logistic Regression

    전통적인 텍스트 분류 베이스라인
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        C: float = 1.0
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C

        self.vectorizer = None
        self.classifier = None
        self.classes_ = []

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """TF-IDF + LR 학습"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        print(f"[TF-IDF] 학습 시작: {len(texts)} 샘플")

        # 라벨 인코딩
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        self.classes_ = list(self.label_encoder.classes_)

        print(f"  클래스: {len(self.classes_)}")

        # TF-IDF 벡터화
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            analyzer='char_wb',  # 한국어에 적합
            min_df=2
        )
        X = self.vectorizer.fit_transform(texts)

        print(f"  피처: {X.shape[1]}")

        # 로지스틱 회귀 학습
        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=1000,
            class_weight='balanced',
            n_jobs=-1,
            verbose=0
        )
        self.classifier.fit(X, y)

        print("[TF-IDF] 학습 완료")

    def predict(self, text: str, topk: int = 5) -> List[Prediction]:
        """Top-K 예측"""
        if self.vectorizer is None or self.classifier is None:
            raise RuntimeError("모델이 학습되지 않았습니다")

        X = self.vectorizer.transform([text])
        proba = self.classifier.predict_proba(X)[0]

        # Top-K 인덱스
        top_indices = np.argsort(proba)[-topk:][::-1]

        predictions = []
        for rank, idx in enumerate(top_indices, 1):
            hs4 = self.classes_[idx]
            score = float(proba[idx])
            predictions.append(Prediction(hs4=hs4, score=score, rank=rank))

        return predictions

    def predict_proba(self, text: str) -> Dict[str, float]:
        """전체 확률 반환"""
        if self.vectorizer is None or self.classifier is None:
            raise RuntimeError("모델이 학습되지 않았습니다")

        X = self.vectorizer.transform([text])
        proba = self.classifier.predict_proba(X)[0]

        return {hs4: float(p) for hs4, p in zip(self.classes_, proba)}

    def save(self, path: str) -> None:
        """모델 저장"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'label_encoder': self.label_encoder,
                'classes_': self.classes_,
                'config': {
                    'max_features': self.max_features,
                    'ngram_range': self.ngram_range,
                    'C': self.C
                }
            }, f)
        print(f"[TF-IDF] 저장: {path}")

    def load(self, path: str) -> None:
        """모델 로드"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vectorizer = data['vectorizer']
        self.classifier = data['classifier']
        self.label_encoder = data['label_encoder']
        self.classes_ = data['classes_']
        print(f"[TF-IDF] 로드: {path}")

    def get_classes(self) -> List[str]:
        return self.classes_


class SBertBaseline(BaselineModel):
    """
    B1: Sentence-BERT + Logistic Regression

    사전학습 임베딩 기반 베이스라인 (기존 retriever.py 래핑)
    """

    def __init__(
        self,
        st_model: str = "jhgan/ko-sroberta-multitask",
        C: float = 1.0,
        device: Optional[str] = None
    ):
        self.st_model_name = st_model
        self.C = C
        self.device = device

        self.st_model = None
        self.classifier = None
        self.label_encoder = None
        self.classes_ = []

    def _load_st_model(self):
        """Sentence Transformer 로드"""
        if self.st_model is None:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(self.st_model_name, device=self.device)
            print(f"[SBert] ST 모델 로드: {self.st_model_name}")

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """SBert + LR 학습"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        print(f"[SBert] 학습 시작: {len(texts)} 샘플")

        # ST 모델 로드
        self._load_st_model()

        # 라벨 인코딩
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        self.classes_ = list(self.label_encoder.classes_)

        print(f"  클래스: {len(self.classes_)}")

        # 임베딩 생성
        print("  임베딩 생성 중...")
        embeddings = self.st_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print(f"  임베딩 차원: {embeddings.shape}")

        # 로지스틱 회귀 학습
        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=1000,
            class_weight='balanced',
            n_jobs=-1,
            verbose=1
        )
        self.classifier.fit(embeddings, y)

        print("[SBert] 학습 완료")

    def predict(self, text: str, topk: int = 5) -> List[Prediction]:
        """Top-K 예측"""
        if self.classifier is None:
            raise RuntimeError("모델이 학습되지 않았습니다")

        self._load_st_model()

        embedding = self.st_model.encode(text, convert_to_numpy=True).reshape(1, -1)
        proba = self.classifier.predict_proba(embedding)[0]

        # Top-K 인덱스
        top_indices = np.argsort(proba)[-topk:][::-1]

        predictions = []
        for rank, idx in enumerate(top_indices, 1):
            hs4 = self.classes_[idx]
            score = float(proba[idx])
            predictions.append(Prediction(hs4=hs4, score=score, rank=rank))

        return predictions

    def predict_proba(self, text: str) -> Dict[str, float]:
        """전체 확률 반환"""
        if self.classifier is None:
            raise RuntimeError("모델이 학습되지 않았습니다")

        self._load_st_model()

        embedding = self.st_model.encode(text, convert_to_numpy=True).reshape(1, -1)
        proba = self.classifier.predict_proba(embedding)[0]

        return {hs4: float(p) for hs4, p in zip(self.classes_, proba)}

    def save(self, path: str) -> None:
        """모델 저장 (ST 모델 제외)"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'label_encoder': self.label_encoder,
                'classes_': self.classes_,
                'config': {
                    'st_model': self.st_model_name,
                    'C': self.C
                }
            }, f)
        print(f"[SBert] 저장: {path}")

    def load(self, path: str) -> None:
        """모델 로드"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.classifier = data['classifier']
        self.label_encoder = data['label_encoder']
        self.classes_ = data['classes_']
        self.st_model_name = data['config']['st_model']
        print(f"[SBert] 로드: {path}")

    def get_classes(self) -> List[str]:
        return self.classes_


class BM25Baseline(BaselineModel):
    """
    B2: BM25 검색 기반

    전통적인 정보 검색 베이스라인
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        topk: int = 50
    ):
        self.k1 = k1
        self.b = b
        self.topk = topk

        self.documents: List[str] = []
        self.labels: List[str] = []
        self.classes_ = []
        self.bm25 = None

        # 클래스별 문서 인덱스
        self.class_doc_indices: Dict[str, List[int]] = defaultdict(list)

    def _tokenize(self, text: str) -> List[str]:
        """간단한 토크나이저 (공백 + 문자 n-gram)"""
        tokens = text.lower().split()

        # 2-gram, 3-gram 추가
        ngrams = []
        for token in tokens:
            if len(token) >= 2:
                for i in range(len(token) - 1):
                    ngrams.append(token[i:i+2])
            if len(token) >= 3:
                for i in range(len(token) - 2):
                    ngrams.append(token[i:i+3])

        return tokens + ngrams

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """BM25 인덱스 구축"""
        from rank_bm25 import BM25Okapi

        print(f"[BM25] 인덱스 구축: {len(texts)} 문서")

        self.documents = texts
        self.labels = labels
        self.classes_ = list(set(labels))

        # 클래스별 문서 인덱스 구축
        self.class_doc_indices = defaultdict(list)
        for i, label in enumerate(labels):
            self.class_doc_indices[label].append(i)

        # 토큰화
        tokenized_docs = [self._tokenize(doc) for doc in texts]

        # BM25 인덱스 구축
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)

        print(f"  클래스: {len(self.classes_)}")
        print("[BM25] 인덱스 구축 완료")

    def predict(self, text: str, topk: int = 5) -> List[Prediction]:
        """Top-K 예측"""
        if self.bm25 is None:
            raise RuntimeError("인덱스가 구축되지 않았습니다")

        query_tokens = self._tokenize(text)
        scores = self.bm25.get_scores(query_tokens)

        # 클래스별 최대 점수 집계
        class_scores: Dict[str, float] = defaultdict(float)
        for i, score in enumerate(scores):
            label = self.labels[i]
            class_scores[label] = max(class_scores[label], score)

        # 정렬
        sorted_classes = sorted(class_scores.items(), key=lambda x: -x[1])[:topk]

        # 점수 정규화 (softmax 근사)
        if sorted_classes:
            max_score = sorted_classes[0][1]
            total = sum(np.exp(s - max_score) for _, s in sorted_classes)

            predictions = []
            for rank, (hs4, score) in enumerate(sorted_classes, 1):
                normalized_score = np.exp(score - max_score) / total if total > 0 else 0
                predictions.append(Prediction(hs4=hs4, score=float(normalized_score), rank=rank))
        else:
            predictions = []

        return predictions

    def predict_proba(self, text: str) -> Dict[str, float]:
        """전체 확률 반환"""
        if self.bm25 is None:
            raise RuntimeError("인덱스가 구축되지 않았습니다")

        query_tokens = self._tokenize(text)
        scores = self.bm25.get_scores(query_tokens)

        # 클래스별 최대 점수
        class_scores: Dict[str, float] = defaultdict(float)
        for i, score in enumerate(scores):
            label = self.labels[i]
            class_scores[label] = max(class_scores[label], score)

        # Softmax 정규화
        if class_scores:
            max_score = max(class_scores.values())
            exp_scores = {k: np.exp(v - max_score) for k, v in class_scores.items()}
            total = sum(exp_scores.values())
            return {k: v / total for k, v in exp_scores.items()}

        return {}

    def save(self, path: str) -> None:
        """모델 저장"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'labels': self.labels,
                'classes_': self.classes_,
                'class_doc_indices': dict(self.class_doc_indices),
                'config': {
                    'k1': self.k1,
                    'b': self.b,
                    'topk': self.topk
                }
            }, f)
        print(f"[BM25] 저장: {path}")

    def load(self, path: str) -> None:
        """모델 로드"""
        from rank_bm25 import BM25Okapi

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.documents = data['documents']
        self.labels = data['labels']
        self.classes_ = data['classes_']
        self.class_doc_indices = defaultdict(list, data['class_doc_indices'])

        # BM25 재구축
        tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)

        print(f"[BM25] 로드: {path}")

    def get_classes(self) -> List[str]:
        return self.classes_


def create_baseline(model_type: str, params: Dict[str, Any]) -> BaselineModel:
    """베이스라인 모델 팩토리"""
    if model_type == "tfidf_lr":
        return TFIDFBaseline(
            max_features=params.get('max_features', 10000),
            ngram_range=tuple(params.get('ngram_range', [1, 2])),
            C=params.get('C', 1.0)
        )
    elif model_type == "sbert_lr":
        return SBertBaseline(
            st_model=params.get('st_model', 'jhgan/ko-sroberta-multitask'),
            C=params.get('C', 1.0)
        )
    elif model_type == "bm25":
        return BM25Baseline(
            k1=params.get('k1', 1.5),
            b=params.get('b', 0.75),
            topk=params.get('topk', 50)
        )
    else:
        raise ValueError(f"Unknown baseline type: {model_type}")


# 테스트
if __name__ == "__main__":
    # 간단한 테스트
    texts = [
        "냉동 돼지고기 삼겹살",
        "냉동 돼지고기 등심",
        "신선 쇠고기",
        "LED TV 55인치",
        "액정 모니터",
    ]
    labels = ["0203", "0203", "0201", "8528", "8528"]

    print("=== TF-IDF Baseline ===")
    tfidf = TFIDFBaseline()
    tfidf.fit(texts, labels)
    preds = tfidf.predict("냉동 돼지고기", topk=3)
    for p in preds:
        print(f"  {p.hs4}: {p.score:.4f}")

    print("\n=== BM25 Baseline ===")
    bm25 = BM25Baseline()
    bm25.fit(texts, labels)
    preds = bm25.predict("냉동 돼지고기", topk=3)
    for p in preds:
        print(f"  {p.hs4}: {p.score:.4f}")
