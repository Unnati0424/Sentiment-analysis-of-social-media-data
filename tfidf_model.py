import joblib
import numpy as np
from typing import Tuple, List
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=50000
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                n_jobs=1
            )),
        ]
    )


def train(texts: List[str], labels: List[str]) -> Pipeline:
    pipe = build_pipeline()
    pipe.fit(texts, labels)
    return pipe


def predict(pipe: Pipeline, texts: List[str]) -> Tuple[List[str], np.ndarray]:
    probs = pipe.predict_proba(texts)
    labels = pipe.classes_[np.argmax(probs, axis=1)].tolist()
    return labels, probs


def save_model(pipe: Pipeline, path: str) -> None:
    joblib.dump(pipe, path)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)
