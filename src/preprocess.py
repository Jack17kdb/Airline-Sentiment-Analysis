import numpy as np
import pandas as pd
import spacy
import re

nlp = spacy.load("en_core_web_sm", disable=["ner"])

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

def preprocess_dataset(dataset):
    df = pd.read_csv(dataset)
    df = pd.concat([df['airline_sentiment'], df['text']], axis=1)

    df['Clean_text'] = df['text'].apply(clean_text)
    df['processed_text'] = df['Clean_text'].apply(spacy_preprocess)
    
    df = df.drop(['text', 'Clean_text'], axis=1)
    
    return df
