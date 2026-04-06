import streamlit as st
import joblib
import os
import re
import string

st.set_page_config(
    page_title="Airline Sentiment Analyzer",
    page_icon="✈️",
    layout="centered",
)

st.markdown("""
    <style>
    .main-title { font-size: 2.4rem; font-weight: 800; text-align: center; color: #38bdf8; }
    .subtitle   { text-align: center; color: #94a3b8; margin-bottom: 2rem; }
    .result-positive { background: linear-gradient(135deg,#052e16,#166534); border-left: 4px solid #22c55e;
        padding: 1rem 1.5rem; border-radius: 8px; color: #86efac; font-size: 1.3rem; font-weight: bold; }
    .result-negative { background: linear-gradient(135deg,#450a0a,#991b1b); border-left: 4px solid #ef4444;
        padding: 1rem 1.5rem; border-radius: 8px; color: #fca5a5; font-size: 1.3rem; font-weight: bold; }
    .result-neutral  { background: linear-gradient(135deg,#0c1a2e,#1e3a5f); border-left: 4px solid #38bdf8;
        padding: 1rem 1.5rem; border-radius: 8px; color: #7dd3fc; font-size: 1.3rem; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">✈️ Airline Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Classify tweet sentiment about airlines — Positive, Neutral, or Negative</div>', unsafe_allow_html=True)

# Search for model in multiple locations (works locally and on Streamlit Cloud)
ROOT = os.path.dirname(__file__)
MODEL_CANDIDATES = [
    os.path.join(ROOT, "models", "sentiment_model.pkl"),
    os.path.join(ROOT, "src", "models", "sentiment_model.pkl"),
]

@st.cache_resource
def load_model():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return joblib.load(path)
    return None

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()

model = load_model()

examples = [
    "Thank you @SouthwestAir for getting me home safe! Best airline ever!",
    "Flight delayed AGAIN. @united has the worst customer service I've ever experienced.",
    "@AmericanAir my bag didn't arrive. Nobody at the desk could help. Horrible.",
    "Just landed on time. Pretty average flight with @Delta.",
    "I love how @JetBlue always has extra legroom. Worth every penny!",
]

st.subheader("Try an example")
selected = st.selectbox("Pick a sample tweet:", ["— choose —"] + examples)

st.subheader("Or type your own")
user_input = st.text_area(
    "Tweet text:",
    value=selected if selected != "— choose —" else "",
    height=120,
    placeholder="e.g. Flight delayed again, terrible service!"
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_btn = st.button("Analyze Sentiment", use_container_width=True, type="primary")

if predict_btn:
    if not user_input.strip():
        st.warning("Please enter some text first.")
    elif model is None:
        st.info("⚠️ Model not trained yet. Run `cd src && python train.py` first. Showing keyword demo:")
        text_lower = user_input.lower()
        if any(w in text_lower for w in ["love","great","best","amazing","thank","awesome","wonderful"]):
            label = "positive"
        elif any(w in text_lower for w in ["hate","worst","delayed","terrible","horrible","bad","cancel","awful"]):
            label = "negative"
        else:
            label = "neutral"
        emoji = {"positive": "😊", "negative": "😠", "neutral": "😐"}[label]
        st.markdown(f'<div class="result-{label}">{emoji} Predicted Sentiment: <strong>{label.upper()}</strong> (demo mode)</div>', unsafe_allow_html=True)
    else:
        processed = preprocess(user_input)
        prediction = model.predict([processed])[0]
        proba = model.predict_proba([processed])[0]
        classes = model.classes_
        confidence = max(proba) * 100
        emoji = {"positive": "😊", "negative": "😠", "neutral": "😐"}.get(prediction, "🤔")
        st.markdown(f'<div class="result-{prediction}">{emoji} Predicted Sentiment: <strong>{prediction.upper()}</strong> ({confidence:.1f}% confidence)</div>', unsafe_allow_html=True)
        st.markdown("**Confidence breakdown:**")
        for cls, prob in zip(classes, proba):
            st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

st.markdown("---")
st.markdown("**About:** NLP pipeline using Logistic Regression + TF-IDF on the US Airline Twitter dataset. Class imbalance handled with SMOTE. Experiments tracked with MLflow.")
