from typing import List, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline


class TwitterSentiment:
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipeline = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True
        )
        # Default order: ['negative', 'neutral', 'positive']
        self.labels = ["negative", "neutral", "positive"]

    def predict(self, texts: List[str]) -> Tuple[List[str], np.ndarray]:
        outputs = self.pipeline(texts, truncation=True)
        probs = []
        preds = []
        for out in outputs:
            # Convert to [neg, neu, pos] probabilities
            p = np.array([s["score"] for s in out], dtype=np.float32)
            probs.append(p)
            preds.append(self.labels[int(np.argmax(p))])
        return preds, np.stack(probs, axis=0)
