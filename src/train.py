import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from preprocess import preprocess_dataset

df = preprocess_dataset("../data/raw/Tweets.csv")

X = df["processed_text"]
y = df["airline_sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

custom_strategy = {
    "negative": 9178,
    "neutral": 7000,
    "positive": 7000
}

smote = SMOTE(sampling_strategy=custom_strategy, random_state=42)

X_resampled, y_resampled = smote.fit_resample(X_train_vec, y_train)

mlflow.set_experiment("sentiment_tfidf_logreg")

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"Active Run ID: {run_id}")
    
    mlflow.log_params({
        "sampling_strategy": str(custom_strategy),
        "vectorizer": "tfidf",
        "max_features": 10000,
        "ngram_range": "(1, 2)",
        "classifier": "logistic_regression"
    })

    pipe = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
 
    pipe.named_steps['classifier'].fit(X_resampled, y_resampled)

    y_pred = pipe.named_steps['classifier'].predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)

    print("Accuracy:", accuracy)

    mlflow.sklearn.log_model(pipe, "sentiment_model_pipeline")

    with open("latest_run.txt", "w") as f:
        f.write(run_id)
        
