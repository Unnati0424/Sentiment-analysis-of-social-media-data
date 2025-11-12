import argparse
import os
import pandas as pd
import numpy as np
from typing import List
from tqdm import tqdm

from .tfidf_model import train as tfidf_train, save_model as tfidf_save, load_model as tfidf_load, predict as tfidf_predict
from .transformer_infer import TwitterSentiment
from .metrics import evaluate_predictions, confusion


LABELS = ["negative", "neutral", "positive"]


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have columns: text,label")
    return df


def cmd_train_tfidf(args: argparse.Namespace) -> None:
    df = _read_csv(args.train)
    print(f"Training TF-IDF model on {len(df)} examples...")
    pipe = tfidf_train(df["text"].tolist(), df["label"].tolist())
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    tfidf_save(pipe, args.model_out)
    print(f"Saved TF-IDF model → {args.model_out}")


def _eval_one(name: str, y_true: List[str], y_pred: List[str]) -> None:
    print(f"\n{name} — Classification Report")
    rep = evaluate_predictions(y_true, y_pred, LABELS)
    print(rep.to_string())
    cm = confusion(y_true, y_pred, LABELS)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))


def cmd_evaluate(args: argparse.Namespace) -> None:
    df = _read_csv(args.test)

    # TF-IDF
    if args.tfidf_model and os.path.exists(args.tfidf_model):
        pipe = tfidf_load(args.tfidf_model)
        tfidf_preds, _ = tfidf_predict(pipe, df["text"].tolist())
        _eval_one("TF-IDF + LogisticRegression", df["label"].tolist(), tfidf_preds)
    else:
        print("Skipping TF-IDF: model file not provided or not found.")

    # Transformer
    transformer = TwitterSentiment()
    t_preds, _ = transformer.predict(df["text"].tolist())
    _eval_one("Transformer (twitter-roberta-base-sentiment)", df["label"].tolist(), t_preds)


def cmd_predict(args: argparse.Namespace) -> None:
    text = args.text
    if not text:
        raise ValueError("Provide --text")

    if args.method == "tfidf":
        if not args.tfidf_model or not os.path.exists(args.tfidf_model):
            raise ValueError("TF-IDF model path required and must exist.")
        pipe = tfidf_load(args.tfidf_model)
        labels, probs = tfidf_predict(pipe, [text])
        print(f"Method: TF-IDF | Label: {labels[0]} | Probs: {probs[0]}")
    else:
        transformer = TwitterSentiment()
        labels, probs = transformer.predict([text])
        print(f"Method: Transformer | Label: {labels[0]} | Probs [neg,neu,pos]: {probs[0]}")


def main():
    parser = argparse.ArgumentParser(description="Social Media Sentiment Analysis")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train-tfidf", help="Train TF-IDF baseline")
    p_train.add_argument("--train", required=True, help="Path to train CSV (text,label)")
    p_train.add_argument("--model_out", default="models/tfidf_logreg.joblib")
    p_train.set_defaults(func=cmd_train_tfidf)

    p_eval = sub.add_parser("evaluate", help="Evaluate both methods on a test CSV")
    p_eval.add_argument("--test", required=True, help="Path to test CSV (text,label)")
    p_eval.add_argument("--tfidf_model", default="models/tfidf_logreg.joblib")
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = sub.add_parser("predict", help="Predict a single text")
    p_pred.add_argument("--text", required=True)
    p_pred.add_argument("--method", choices=["tfidf", "transformer"], default="transformer")
    p_pred.add_argument("--tfidf_model", default="models/tfidf_logreg.joblib")
    p_pred.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
