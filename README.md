# ✈️ Airline Sentiment Analysis

A machine learning project that analyzes Twitter sentiment about airlines using Natural Language Processing (NLP) and Logistic Regression. This project demonstrates end-to-end ML workflow including data preprocessing, model training with class imbalance handling, evaluation, and inference.

## 🎯 Project Overview

This project classifies airline-related tweets into three sentiment categories: **positive**, **neutral**, and **negative**. It addresses the challenge of imbalanced datasets using SMOTE (Synthetic Minority Over-sampling Technique) and leverages MLflow for experiment tracking and model management.

## 🔑 Key Features

- **Advanced Text Preprocessing**: SpaCy-based lemmatization, stopword removal, and text cleaning
- **TF-IDF Vectorization**: Feature extraction with bigrams (1,2) and 10,000 max features
- **Class Imbalance Handling**: Custom SMOTE strategy to balance sentiment classes
- **MLflow Integration**: Complete experiment tracking, parameter logging, and model versioning
- **Production-Ready Inference**: Pipeline for real-time sentiment prediction with confidence scores

## 📊 Dataset

The project uses the Twitter US Airline Sentiment dataset containing customer tweets about major US airlines. The dataset exhibits significant class imbalance which is addressed during training.

## 🛠️ Technology Stack

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and preprocessing
- **spaCy**: Advanced NLP text processing
- **imbalanced-learn**: SMOTE implementation for handling class imbalance
- **MLflow**: Experiment tracking and model management

## 📁 Project Structure

```
airline_sentiment_analysis/
├── data/
│   ├── raw/                 # Raw tweet data
│   └── processed/           # Processed datasets
├── src/
│   ├── preprocess.py        # Data cleaning and preprocessing
│   ├── train.py             # Model training pipeline
│   ├── evaluate.py          # Model evaluation and metrics
│   └── inference.py         # Prediction script
├── mlruns/                  # MLflow experiment tracking
├── config.py                # Configuration settings
├── requirements.txt         # Project dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.7+ installed on your system.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/airline_sentiment_analysis.git
cd airline_sentiment_analysis
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the SpaCy language model:
```bash
python -m spacy download en_core_web_sm
```

### Usage

#### 1. Preprocess the Data
```python
from src.preprocess import preprocess_dataset

df = preprocess_dataset("data/raw/Tweets.csv")
```

#### 2. Train the Model
```bash
cd src
python train.py
```

This will:
- Preprocess the dataset
- Apply SMOTE for class balancing
- Train a Logistic Regression model
- Log experiments to MLflow
- Save the trained pipeline

#### 3. Evaluate the Model
```bash
python evaluate.py
```

Generates:
- Classification report with precision, recall, and F1-scores
- Confusion matrix visualization
- Logs metrics to MLflow

#### 4. Make Predictions
```bash
python inference.py
```

Or use in your code:
```python
from src.inference import predict_sentiment

result = predict_sentiment("Amazing service and friendly staff!")
print(result)
# Output: {'text': '...', 'sentiment': 'positive', 'confidence': 0.89}
```

## 📈 Model Performance

The model uses a custom SMOTE sampling strategy to address class imbalance:
- **Negative class**: 9,178 samples (majority - kept as is)
- **Neutral class**: Upsampled to 7,000 samples
- **Positive class**: Upsampled to 7,000 samples

This balanced approach ensures the model doesn't simply predict the majority class.

## 🔬 Pipeline Architecture

1. **Text Cleaning**: Lowercase conversion, URL removal, mention removal, hashtag cleaning
2. **SpaCy Preprocessing**: Lemmatization, stopword filtering, punctuation removal
3. **TF-IDF Vectorization**: Unigrams and bigrams with 10,000 features
4. **SMOTE Resampling**: Balanced training data
5. **Logistic Regression**: Classification with 1000 max iterations

## 📊 MLflow Tracking

All experiments are tracked using MLflow:
- **Parameters**: Sampling strategy, vectorizer settings, model hyperparameters
- **Metrics**: Accuracy, precision, recall, F1-score
- **Artifacts**: Trained model pipeline, confusion matrices, classification reports

View experiments:
```bash
mlflow ui
```

Navigate to `http://localhost:5000` to explore experiment runs.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/jack17kdb)
- LinkedIn: [my LinkedIn](https://linkedin.com/in/johnson-ndung-u-7b527b38b)

## 🙏 Acknowledgments

- Twitter US Airline Sentiment dataset
- SpaCy for powerful NLP capabilities
- MLflow for experiment tracking
- scikit-learn community

---

⭐ If you found this project helpful, please consider giving it a star!
