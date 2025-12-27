Colab recipe — run LSTM experiments on free/paid GPU
---------------------------------------------------

This guide helps you run the repository's LSTM experiments on Google Colab (recommended if you don't have a local GPU). It uses the code in `scripts/train_lstm.py` but shows how to run an end-to-end notebook-friendly workflow.

Steps (one-click style):

1. Open a new Colab notebook: https://colab.research.google.com/
2. In the top menu choose "Runtime -> Change runtime type" and select "GPU" (e.g., NVIDIA T4).
3. In the first cell, clone or upload the repository. If you have the repo on GitHub run:

```python
!git clone https://github.com/VenuDave3/Fake-News-Detector.git
%cd Fake-News-Detector
```

4. Install dependencies (CPU/GPU TF will be installed by pip in Colab):

```python
!pip install -r requirements.txt
# if TF not in requirements, install explicitly (Colab uses GPU-enabled TF):
!pip install --upgrade tensorflow
```

5. Option A — run the LSTM training script directly (uses the dataset in `data/` if present):

```python
!python3 scripts/train_lstm.py --epochs 5 --batch-size 64
```

6. Option B — open a notebook cell and paste a smaller training loop (gives more control):

```python
from scripts.train_lstm import load_data, build_model
X,y = load_data(min_samples=20000)
print('Loaded', len(X))
# proceed to Tokenizer, pad_sequences, model.fit as shown in the repo
```

7. After training, download artifacts from `artifacts/` (e.g., `lstm.h5`, `tokenizer.json`) to your machine or push them to a cloud storage bucket.

Notes
- Colab free tier provides limited GPU time and preemptible sessions; for long experiments use Colab Pro or a cloud VM.
- Adjust `--epochs` and `--batch-size` to fit within Colab VRAM (start with 32 or 64). Use EarlyStopping in notebooks to avoid long wasted runs.

If you'd like, I can generate a ready-to-run Colab notebook file (.ipynb) with cells for each step above and sample hyperparameters. Say "generate Colab notebook" and I'll add it to the repo.
