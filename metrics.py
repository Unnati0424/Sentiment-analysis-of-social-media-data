import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_predictions(true: List[str], pred: List[str], labels: List[str]) -> pd.DataFrame:
    report = classification_report(true, pred, target_names=labels, output_dict=True, digits=4)
    return pd.DataFrame(report).transpose()


def confusion(true: List[str], pred: List[str], labels: List[str]) -> np.ndarray:
    return confusion_matrix(true, pred, labels=labels)
