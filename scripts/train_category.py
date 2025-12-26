#!/usr/bin/env python3
"""Train category classifier and save as final1.pickle

This script extracts the training steps from `Category.ipynb` and
saves a reproducible model artifact.
"""
import os
import pickle
import string
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score


def punctuation_removal(text):
    return ''.join([c for c in text if c not in string.punctuation])


def preprocess(df):
    df = df.astype(str).dropna().reset_index(drop=True)
    df['text'] = df['text'].apply(lambda x: x.lower())
    df['text'] = df['text'].apply(punctuation_removal)
    # remove stopwords lazily (nltk required)
    try:
        import nltk
        from nltk.corpus import stopwords
        stop = set(stopwords.words('english'))
        df['text'] = df['text'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop]))
    except Exception:
        # if stopwords not available, skip this step
        pass
    return df


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, '..')
    data_path = os.path.join(root, 'data', 'bbc-text.csv')
    print('Reading', data_path)
    news = pd.read_csv(data_path)
    from sklearn.utils import shuffle as _shuffle
    news = _shuffle(news, random_state=42).reset_index(drop=True)
    news = preprocess(news)

    X = news['text']
    y = news['category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])

    print('Training category model...')
    model = pipe.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f'Category model accuracy: {acc:.4f}')

    out_path = os.path.join(root, 'final1.pickle')
    with open(out_path, 'wb') as f:
        pickle.dump(model, f)
    print('Saved model to', out_path)


if __name__ == '__main__':
    main()
