import streamlit as st
import pandas as pd
import numpy as np
from src.tfidf_model import load_model as tfidf_load, predict as tfidf_predict
from src.transformer_infer import TwitterSentiment

st.set_page_config(page_title="Social Media Sentiment", page_icon="💬", layout="wide")
st.title("💬 Social Media Sentiment Analysis")

st.markdown("Classify posts into negative, neutral, positive using a trainable TF‑IDF baseline and a pretrained transformer model.")

with st.sidebar:
    st.header("Settings")
    method = st.selectbox("Method", ["Transformer (twitter-roberta)", "TF-IDF (trained)"])
    tfidf_path = st.text_input("TF-IDF model path", "models/tfidf_logreg.joblib")

col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("Single Post")
    text = st.text_area("Enter post text", height=150, placeholder="Type or paste a social post...")
    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please enter text.")
        else:
            if method.startswith("Transformer"):
                model = TwitterSentiment()
                labels, probs = model.predict([text])
                label = labels[0]
                p = probs[0]
            else:
                try:
                    pipe = tfidf_load(tfidf_path)
                    labels, probs = tfidf_predict(pipe, [text])
                    label = labels[0]
                    p = probs[0]
                except Exception as e:
                    st.error(f"Failed to load TF-IDF model: {e}")
                    p = None
                    label = None

            if label is not None:
                st.success(f"Predicted: {label}")
                st.progress(float(p[["negative","neutral","positive"].index(label)]) if isinstance(p, np.ndarray) else 0.0)

with col_output:
    st.subheader("Batch CSV")
    st.caption("Upload a CSV with a 'text' column.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        if "text" not in df.columns:
            st.error("CSV must have a 'text' column.")
        else:
            if st.button("Run Batch"):
                try:
                    if method.startswith("Transformer"):
                        model = TwitterSentiment()
                        labels, probs = model.predict(df["text"].tolist())
                    else:
                        pipe = tfidf_load(tfidf_path)
                        labels, probs = tfidf_predict(pipe, df["text"].tolist())
                    out = pd.DataFrame({
                        "text": df["text"],
                        "pred": labels,
                        "neg": probs[:, 0],
                        "neu": probs[:, 1],
                        "pos": probs[:, 2],
                    })
                    st.dataframe(out.head(50), use_container_width=True)
                    st.download_button("Download Results", out.to_csv(index=False), "sentiment_results.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error processing batch: {e}")
