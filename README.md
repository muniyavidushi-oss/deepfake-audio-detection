# 🎙️ Deepfake Audio Detection — MARS Open Projects 2026

Classify a speech recording as **Genuine (Human)** or **Deepfake (AI-Generated)**
using classic audio features (librosa) and a compact Keras MLP.

**Live demo:** https://deepfake-audio-detection-jsdbhghef6utxtrhxmw4tf.streamlit.app/#b2057d24
**Labels:** `real = 0` (Genuine), `fake = 1` (Deepfake)

##  Deliverables — where to find each one

| Required deliverable | Where it is |
|---|---|
| 1. ipynb notebook with full running code | [`notebook.ipynb`](notebook.ipynb) |
| 2. Trained model | `deepfake_model.keras` (+ `.h5` / `.weights.h5`), `scaler.pkl`, `threshold.json` |
| 3. Python script to test new audio samples | [`predict.py`](predict.py) — run `python predict.py your_clip.wav` |
| 4. Performance report (accuracy, EER, F1, confusion matrix) | See the **Results** section below |
| 5. Preprocessing, feature extraction & model architecture | See the **Dataset & Preprocessing**, **Feature Extraction**, and **Model** sections below |
| 6. Clear, detailed README | This file |
| Bonus: live web app | [Open the Streamlit app](PASTE-YOUR-URL-HERE) |

## Dataset & Preprocessing
- **Fake-or-Real (FoR)** dataset, **for-norm** (normalized) variant.
- Structure: `for-norm/{training,testing,validation}/{real,fake}/*.wav`
- Trained on `training`; evaluated on the held-out `testing` split.
- A discovery routine walks the dataset and auto-selects the folder named with
  "norm" that has a `training/{real,fake}` layout (no hardcoded paths).
- Audio: loaded **mono @ 16 kHz**, padded/truncated to **4 s** (64,000 samples).
- Up to 4,000 clips per class were used for training; features are cached to disk.

## Feature Extraction (`features.py`, 124-dim vector)
Defined once and reused by the notebook, training script, CLI, and app:

| Feature | Stats | Dims |
|---|---|---|
| MFCC (n=40) | mean + std | 80 |
| Chroma STFT | mean + std | 24 |
| Spectral contrast | mean + std | 14 |
| Zero-crossing rate | mean + std | 2 |
| RMS energy | mean + std | 2 |
| Spectral rolloff | mean + std | 2 |
| **Total** | | **124** |

Features are standardized with a `StandardScaler` fit on the training split
(saved as `scaler.pkl`); inference reuses the exact same scaler.

## Model
Compact MLP — binary cross-entropy, Adam, EarlyStopping, ReduceLROnPlateau,
balanced class weights:

```
Input(124)
 → Dense(256) + BatchNorm + ReLU + Dropout(0.4)
 → Dense(128) + BatchNorm + ReLU + Dropout(0.3)
 → Dense(64)  + BatchNorm + ReLU + Dropout(0.2)
 → Dense(1, sigmoid)
```

**Decision threshold.** The detector operates at its **Equal-Error-Rate (EER)
operating point** (cutoff ≈ 0.0059) — the balanced operating point standard for
spoof/deepfake detectors. The cutoff is saved in `threshold.json` and reused by
the app and CLI.

## Results (for-norm `testing` split — 4,634 clips)

| Metric | Target | Achieved | Status |
|---|---|---|---|
| Accuracy | ≥ 80% | **88.78%** |  
| EER | ≤ 12% | **11.22%** |  
| F1 score | ≥ 80% | **89.01%** |  
| Per-class accuracy (real) | ≥ 75% | **88.74%** |  
| Per-class accuracy (fake) | ≥ 75% | **88.82%** | 

Confusion matrix (at the EER cutoff):

```
                 Predicted real   Predicted fake
   True real          2009              255
   True fake           265             2105
```

Training / validation curves: `training_history.png`.

## How to Run

### Notebook (Google Colab) — training
Open `notebook.ipynb` and run the cells top to bottom (install → mount Drive →
discover dataset → extract features → train → evaluate → save model + scaler +
threshold).

### CLI inference
```bash
pip install -r requirements.txt
python predict.py path/to/clip.wav
# -> Verdict: Genuine (Human) | Confidence: 96.3%
```

### Streamlit app
```bash
streamlit run app.py
```
Deployed on Streamlit Community Cloud (Python 3.12). The repo includes
`deepfake_model.keras`, `scaler.pkl`, and `threshold.json` so the app loads
the trained model directly.

## Repo Layout
```
app.py               # Streamlit web app
features.py          # shared feature extractor (single source of truth)
predict.py           # CLI inference
train_pipeline.py    # standalone training pipeline
requirements.txt
notebook.ipynb
deepfake_model.keras / .h5 / .weights.h5
scaler.pkl
threshold.json
training_history.png
```

## Notes & Future Work
- The same feature-extraction function is used in training and inference;
  changing it in one place only would silently break accuracy.
- The FoR `testing` split uses different AI voice generators than `training`,
  which makes cross-distribution generalization hard. A planned improvement is a
  **CNN over log-mel spectrograms** (instead of summary-statistic features) to
  capture fine time–frequency artifacts and generalize better to unseen
  generators.

## License
MIT
