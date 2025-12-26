"""A tiny, original summarizer used by the Flask app.

This is a simple sentence-scoring summarizer (TF-IDF sentence scores).
It's intentionally lightweight to avoid vendoring the upstream summarizer.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import tokenize


def GetText(title, text, n_sentences=3):
    """Return n_sentences summary for given text.

    Args:
        title: article title (unused by default but kept for compatibility)
        text: full article text
        n_sentences: number of sentences to return

    Returns:
        summary string (n_sentences joined by space)
    """
    if not text or len(text.strip()) == 0:
        return ''

    sents = tokenize.sent_tokenize(text)
    if len(sents) <= n_sentences:
        return ' '.join(sents)

    # score sentences using TF-IDF over sentences
    try:
        vect = TfidfVectorizer(stop_words='english')
        X = vect.fit_transform(sents)
        scores = X.sum(axis=1).A.ravel()
        ranked_ix = scores.argsort()[::-1]
        selected = sorted(ranked_ix[:n_sentences])
        summary = ' '.join([sents[i] for i in selected])
        return summary
    except Exception:
        # fallback: naive first-n sentences
        return ' '.join(sents[:n_sentences])
