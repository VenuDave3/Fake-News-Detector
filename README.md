# Fake news detection — News Interpreter

This repository contains code and notebooks for building simple NLP pipelines
to classify news text by category and to detect fake vs. real news. The project
includes a Flask demo that accepts a URL or pasted text, predicts labels, and
returns a short summary.

## Data

This project uses the public Kaggle dataset "Fake and Real News":

https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

The dataset contains separate CSV files for fake and real news and totals
on the order of 40–50k articles. See `data/` for the CSV files used in
the notebooks (keep local copies for reproducibility).

## Third-party components and attribution

- The demo uses a small, original summarizer implemented in `summarizer_custom.py`.
- This project uses public libraries: scikit-learn, nltk, newspaper3k, and
	others listed in `requirements.txt`.

If you used code from external tutorials or examples while developing these
notebooks, keep attributions in the notebook markdown or the project README.
When relying on third-party packages in your environment, follow their
respective license requirements (install via pip when the upstream package is
required).

## Reproducing the models

Two training scripts are provided under `scripts/`:

- `scripts/train_category.py` → trains a category classifier and saves
	`final1.pickle`.
- `scripts/train_fake_real.py` → trains a fake/real classifier and saves
	`final2.pickle`.

Install dependencies and run the scripts (use a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_category.py
python scripts/train_fake_real.py
```

After training, start the demo with `python app.py`.

## Additional analysis

- `notebooks/ngram_analysis.ipynb` — a small, original notebook that compares top unigrams in the
	`Fake.csv` and `True.csv` datasets and produces side-by-side bar charts. This demonstrates a
	lightweight, original visualization added to the repository.



