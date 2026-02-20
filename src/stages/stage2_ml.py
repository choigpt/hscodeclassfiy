"""
Stage 2: ML (SBERT + LR) Classifier

Pre-trained SBERT embedding + Logistic Regression. No KB.

Pipeline:
  Input -> SBERT embedding (768d) -> LR predict_proba -> Top-K -> StageResult
"""

from typing import Optional
from pathlib import Path

import numpy as np

from ..types import BaseClassifier, StageResult, Prediction, StageID


LR_PATH = "artifacts/classifier/model_lr.joblib"
LE_PATH = "artifacts/classifier/label_encoder.joblib"
ST_MODEL = "jhgan/ko-sroberta-multitask"


class MLClassifier(BaseClassifier):
    """ML classifier: SBERT embeddings + Logistic Regression."""

    def __init__(
        self,
        st_model_name: str = ST_MODEL,
        lr_path: str = LR_PATH,
        le_path: str = LE_PATH,
        device: Optional[str] = None,
    ):
        self.st_model_name = st_model_name
        self._st_model = None
        self._lr_model = None
        self._label_encoder = None
        self._device = device

        self._load_models(lr_path, le_path)

    def _load_models(self, lr_path: str, le_path: str):
        import joblib

        # SBERT
        from sentence_transformers import SentenceTransformer
        self._st_model = SentenceTransformer(self.st_model_name, device=self._device)
        print(f"[ML] SBERT loaded: {self.st_model_name}")

        # LR + LabelEncoder
        lr_file = Path(lr_path)
        le_file = Path(le_path)
        if lr_file.exists() and le_file.exists():
            self._lr_model = joblib.load(lr_file)
            self._label_encoder = joblib.load(le_file)
            print(f"[ML] LR loaded: {len(self._label_encoder.classes_)} classes")
        else:
            raise FileNotFoundError(f"ML model not found: {lr_file} / {le_file}")

    @property
    def name(self) -> str:
        return "ML (SBERT+LR)"

    @property
    def stage_id(self) -> StageID:
        return StageID.ML

    def embed(self, text: str) -> np.ndarray:
        return self._st_model.encode(text, convert_to_numpy=True)

    def predict_topk(self, text: str, k: int = 50):
        """Return [(hs4, prob), ...] top-k."""
        embedding = self.embed(text).reshape(1, -1)
        proba = self._lr_model.predict_proba(embedding)[0]
        top_indices = np.argsort(proba)[-k:][::-1]
        return [
            (self._label_encoder.inverse_transform([idx])[0], float(proba[idx]))
            for idx in top_indices
        ]

    def classify(self, text: str, topk: int = 5) -> StageResult:
        results = self.predict_topk(text, k=topk)

        predictions = [
            Prediction(hs4=hs4, score=prob, rank=i + 1)
            for i, (hs4, prob) in enumerate(results)
        ]

        confidence = predictions[0].score if predictions else 0.0

        return StageResult(
            input_text=text,
            predictions=predictions,
            confidence=confidence,
            metadata={'stage': 'ml'},
        )
