# Sentiment-analysis-of-social-media-data
# Social Media Sentiment Analysis

End-to-end sentiment classification for social media text:
- Trainable baseline: TF‑IDF + LogisticRegression
- Pretrained transformer: twitter-roberta-base-sentiment (3 classes)

## Features
- Clean, minimal project with CLI and Streamlit app
- Training/evaluation with metrics and confusion matrix
- Ready-to-run sample dataset; drop in your own CSVs

## Install
```bash
python -m pip install -r requirements.txt
```

## Train TF‑IDF Baseline
```bash
python src\\cli.py train-tfidf --train data\\train.csv --model_out models\\tfidf_logreg.joblib
```

## Evaluate (TF‑IDF and Transformer)
```bash
python src\\cli.py evaluate --test data\\test.csv --tfidf_model models\\tfidf_logreg.joblib
```

## Predict
- TF‑IDF:
```bash
python src\\cli.py predict --text "This update is terrible." --tfidf_model models\\tfidf_logreg.joblib --method tfidf
```
- Transformer:
```bash
python src\\cli.py predict --text "Absolutely love this!" --method transformer
```

## Streamlit App
```bash
streamlit run app.py
```
- Paste posts or upload a CSV with columns: `text`
- See label and probabilities with both methods

## Dataset
- Included: `data/train.csv`, `data/test.csv` (small curated sample with `text,label`).
- Labels: `negative`, `neutral`, `positive`.
- Replace with your data; keep the same schema.

## Model Notes
- TF‑IDF model saves to `models/tfidf_logreg.joblib`.
- Transformer uses `cardiffnlp/twitter-roberta-base-sentiment` directly — no training required.
- For large data, consider stratified splits and more robust regularization.

## License
MIT — feel free to fork and adapt.
