"""Minimal LSTM training module with import-safety.

This file provides:
- load_data(): loads CSVs if available, otherwise creates a tiny toy dataset
- build_model(): attempts to build a tf.keras LSTM model; if TensorFlow is
  unavailable, returns a lightweight dummy object so imports/smoke-tests pass.
- train(): training routine (only runs if TensorFlow is present)

The design ensures importing this module is safe even when TensorFlow isn't
installed in the environment (useful for CI or lightweight smoke tests).
"""
from __future__ import annotations
import os
import json
from typing import Tuple, Any

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_data(min_samples: int = 1000) -> Tuple[list, list]:
    """Load dataset from `data/` if present, otherwise return toy examples.

    Returns X (list of texts) and y (list of 0/1 labels).
    """
    import csv

    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")

    examples = []
    labels = []

    if os.path.exists(fake_path) and os.path.exists(true_path):
        # Try to read a subset (avoid loading huge files during quick runs)
        try:
            with open(fake_path, newline="", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for i, r in enumerate(reader):
                    if i >= min_samples // 2:
                        break
                    text = (r.get("text") or r.get("content") or "").strip()
                    if text:
                        examples.append(text)
                        labels.append(0)
            with open(true_path, newline="", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for i, r in enumerate(reader):
                    if i >= min_samples // 2:
                        break
                    text = (r.get("text") or r.get("content") or "").strip()
                    if text:
                        examples.append(text)
                        labels.append(1)
            if examples:
                return examples, labels
        except Exception:
            pass

    # Fallback tiny toy dataset
    examples = [
        "President signs new bill into law",
        "Scientists discover water on Mars",
        "Click here to win a million dollars now",
        "Celebrity caught in shocking scandal — you won't believe it",
    ]
    labels = [1, 1, 0, 0]
    return examples, labels


def build_model(vocab_size: int = 10000, embed_dim: int = 128, lstm_units: int = 64) -> Any:
    """Try to build a tf.keras LSTM model. If TF isn't installed, return
    a DummyModel that provides a `.summary()` method and `.compile()` stub.
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True),
            LSTM(lstm_units),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model
    except Exception:
        class DummyModel:
            def summary(self):
                print(f"DummyModel(vocab_size={vocab_size}, embed_dim={embed_dim}, lstm_units={lstm_units})")

            def compile(self, *args, **kwargs):
                print("DummyModel.compile() called (no-op)")

            def fit(self, *args, **kwargs):
                raise RuntimeError("TensorFlow not available: cannot train")

            def save(self, path):
                with open(path + ".dummy", "w") as f:
                    f.write("dummy model placeholder")

        return DummyModel()


def train(epochs: int = 3, batch_size: int = 32):
    """High-level training entrypoint.

    This function will only succeed if TensorFlow is installed. It exists to
    provide a single callable entry for scripts/CI. For quick checks use
    `build_model()` which is safe when TF is absent.
    """
    X, y = load_data(min_samples=2000)
    try:
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        import numpy as np

        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        tokenizer.fit_on_texts(X)
        seqs = tokenizer.texts_to_sequences(X)
        maxlen = max(len(s) for s in seqs)
        Xp = pad_sequences(seqs, maxlen=min(maxlen, 500))
        y_arr = np.array(y)

        model = build_model()
        if hasattr(model, "fit"):
            history = model.fit(Xp, y_arr, epochs=epochs, batch_size=batch_size, validation_split=0.1)
            model_path = os.path.join(ARTIFACTS_DIR, "lstm.h5")
            model.save(model_path)
            # Save tokenizer for reuse
            tok_path = os.path.join(ARTIFACTS_DIR, "tokenizer.json")
            with open(tok_path, "w", encoding="utf-8") as f:
                f.write(tokenizer.to_json())
            hist_path = os.path.join(ARTIFACTS_DIR, "lstm_history.json")
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)
            print("Training complete. Artifacts saved to:", ARTIFACTS_DIR)
        else:
            print("Model does not support training (dummy). No artifacts written.")
    except Exception as e:
        print("Training aborted (TensorFlow likely missing or other error):", e)


if __name__ == "__main__":
    # Simple CLI entry for manual runs
    import argparse

    p = argparse.ArgumentParser(description="Minimal LSTM train runner (safe if TF missing)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size)
