"""Quick smoke test for the LSTM module.

This script imports `scripts.train_lstm`, loads data (toy fallback ok),
builds the model (or dummy), and prints a short confirmation. It's safe to
run in CI and doesn't require TensorFlow to be installed.
"""
from scripts import train_lstm


def main():
    X, y = train_lstm.load_data(min_samples=10)
    print(f"Loaded {len(X)} examples (labels: {set(y)})")
    model = train_lstm.build_model()
    print("Model summary (or dummy):")
    try:
        model.summary()
    except Exception as e:
        print("Model summary failed:", e)
    print("Smoke test OK — import and build succeeded.")


if __name__ == "__main__":
    main()
