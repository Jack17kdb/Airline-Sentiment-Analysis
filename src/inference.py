import mlflow
import mlflow.sklearn
import spacy
import re

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    return text

def spacy_preprocess(text):
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct
    ]
    return " ".join(tokens)

with open("latest_run.txt", "r") as f:
    RUN_ID = f.read().strip()

model_pipeline = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/sentiment_model_pipeline")

def predict_sentiment(text):
    cleaned = clean_text(text)
    processed = spacy_preprocess(cleaned)
    probs = model_pipeline.predict_proba([processed])[0]

    labels = model_pipeline.named_steps['classifier'].classes_
    prediction = labels[probs.argmax()]
    confidence = probs.max()

    return {
        "text": text,
        "sentiment": prediction,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    print(predict_sentiment("This airline ruined my trip"))
    print(predict_sentiment("Amazing service and friendly staff"))
