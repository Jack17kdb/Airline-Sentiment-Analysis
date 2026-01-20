import pandas as pd
import mlflow
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import preprocess_dataset

df = preprocess_dataset("../data/raw/Tweets.csv")

X = df["processed_text"]
y = df["airline_sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

with open("latest_run.txt", "r") as f:
    RUN_ID = f.read().strip()

model_pipeline = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/sentiment_model_pipeline")

y_pred = model_pipeline.predict(X_test)

labels = model_pipeline.named_steps['classifier'].classes_

cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

report = classification_report(y_test, y_pred)
print(report)

with mlflow.start_run(run_name="evaluation", nested=True):
    mlflow.log_text(report, "classification_report.txt")

    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
