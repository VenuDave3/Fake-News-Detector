#!/usr/bin/env python3
"""Train fake/real classifier and save as final2.pickle

Extracted and cleaned from `FakeNews.ipynb`.
"""
import os
import pickle
import string
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def punctuation_removal(text):
    return ''.join([c for c in text if c not in string.punctuation])


def preprocess(df):
    df = df.astype(str).dropna().reset_index(drop=True)
    df['text'] = df['text'].apply(lambda x: x.lower())
    df['text'] = df['text'].apply(punctuation_removal)
    try:
        import nltk
        from nltk.corpus import stopwords
        stop = set(stopwords.words('english'))
        df['text'] = df['text'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop]))
    except Exception:
        pass
    return df


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    fake = pd.read_csv(os.path.join(root, 'data', 'Fake.csv'))
    true = pd.read_csv(os.path.join(root, 'data', 'True.csv'))

    fake['target'] = 'FAKE'
    true['target'] = 'REAL'
    data = pd.concat([fake, true]).reset_index(drop=True)

    data = preprocess(data)

    X = data['text']
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('model', PassiveAggressiveClassifier(max_iter=50, random_state=42))
    ])

    print('Training fake/real model...')
    model = pipe.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f'Fake/Real model accuracy: {acc:.4f}')

    out_path = os.path.join(root, 'final2.pickle')
    with open(out_path, 'wb') as f:
        pickle.dump(model, f)
    print('Saved model to', out_path)


if __name__ == '__main__':
    import os
    main()
