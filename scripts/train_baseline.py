"""Train a scikit-learn baseline (TF-IDF + LogisticRegression).

Saves model to `artifacts/baseline.pkl` and metrics to
`reports/baseline_metrics.json`.
"""
from __future__ import annotations
import os
import csv
import json
import pickle
from typing import Tuple

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_texts(min_samples_per_class: int = 5000) -> Tuple[list, list]:
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")
    texts = []
    labels = []
    if os.path.exists(fake_path) and os.path.exists(true_path):
        try:
            with open(fake_path, newline="", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for i, r in enumerate(reader):
                    if i >= min_samples_per_class:
                        break
                    t = (r.get("text") or r.get("content") or "").strip()
                    if t:
                        texts.append(t)
                        labels.append(0)
            with open(true_path, newline="", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for i, r in enumerate(reader):
                    if i >= min_samples_per_class:
                        break
                    t = (r.get("text") or r.get("content") or "").strip()
                    if t:
                        texts.append(t)
                        labels.append(1)
            if texts:
                return texts, labels
        except Exception:
            pass

    # fallback tiny dataset
    texts = [
        "President signs new bill into law",
        "Scientists discover water on Mars",
        "Click here to win a million dollars now",
        "Celebrity caught in shocking scandal — you won't believe it",
    ]
    labels = [1, 1, 0, 0]
    return texts, labels


def train_save(min_samples_per_class: int = 5000):
    X, y = load_texts(min_samples_per_class=min_samples_per_class)
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        import numpy as np

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
        Xt = tfidf.fit_transform(X_train)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xt, y_train)

        Xte = tfidf.transform(X_test)
        ypred = clf.predict(Xte)
        acc = accuracy_score(y_test, ypred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, ypred, average='binary')
        try:
            proba = clf.predict_proba(Xte)[:,1]
            roc = roc_auc_score(y_test, proba)
        except Exception:
            roc = None

        metrics = {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc) if roc is not None else None,
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
        }

        # Save model bundle
        bundle = {
            'vectorizer': tfidf,
            'classifier': clf,
        }
        model_path = os.path.join(ARTIFACTS_DIR, 'baseline.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(bundle, f)

        metrics_path = os.path.join(REPORTS_DIR, 'baseline_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        print('Baseline training complete. Model saved to', model_path)
        print('Metrics:', metrics)
    except Exception as e:
        print('Baseline training failed (scikit-learn missing or error):', e)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Train TF-IDF + LogisticRegression baseline')
    p.add_argument('--samples-per-class', type=int, default=5000)
    args = p.parse_args()
    train_save(min_samples_per_class=args.samples_per_class)
