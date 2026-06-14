"""Streamlit app — Deepfake Audio Detector. Needs: features.py + deepfake_model.keras
(or .h5/.weights.h5) + scaler.pkl + threshold.json in the repo."""
from __future__ import annotations
import os, json, tempfile
import joblib
import numpy as np
import streamlit as st
from tensorflow import keras

from features import extract_features_from_file, N_FEATURES

st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="centered")
CANDIDATE_DIRS = [".", "artifacts", "model", "models"]

def _find(name):
    for d in CANDIDATE_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None

# Must match the architecture trained in the notebook
def build_model(input_dim):
    m = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, name="dense_1"), keras.layers.BatchNormalization(name="bn_1"),
        keras.layers.Activation("relu", name="act_1"), keras.layers.Dropout(0.4, name="drop_1"),
        keras.layers.Dense(128, name="dense_2"), keras.layers.BatchNormalization(name="bn_2"),
        keras.layers.Activation("relu", name="act_2"), keras.layers.Dropout(0.3, name="drop_2"),
        keras.layers.Dense(64, name="dense_3"), keras.layers.BatchNormalization(name="bn_3"),
        keras.layers.Activation("relu", name="act_3"), keras.layers.Dropout(0.2, name="drop_3"),
        keras.layers.Dense(1, activation="sigmoid", name="out"),
    ], name="deepfake_mlp")
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

@st.cache_resource(show_spinner="Loading model …")
def load_artifacts():
    sp = _find("scaler.pkl")
    if sp is None:
        raise FileNotFoundError("scaler.pkl not found in repo")
    scaler = joblib.load(sp)

    threshold = 0.5
    tp = _find("threshold.json")
    if tp:
        try: threshold = float(json.load(open(tp))["threshold"])
        except Exception: pass

    kp, hp, wp = _find("deepfake_model.keras"), _find("deepfake_model.h5"), _find("deepfake_model.weights.h5")
    model = None
    for path in (kp, hp):
        if path and model is None:
            try: model = keras.models.load_model(path)
            except Exception: model = None
    if model is None and wp:
        model = build_model(N_FEATURES); model.load_weights(wp)
    if model is None:
        raise FileNotFoundError("No loadable model file found")
    return model, scaler, threshold

def predict_bytes(file_bytes, suffix, model, scaler):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes); path = tmp.name
    try:
        feats = extract_features_from_file(path).reshape(1, -1)
        return float(model.predict(scaler.transform(feats), verbose=0).ravel()[0])
    finally:
        os.unlink(path)

st.title("🎙️ Deepfake Audio Detector")
st.caption("Genuine (Human) vs Deepfake (AI-Generated) — librosa features + a Keras MLP.")
with st.sidebar:
    st.header("About")
    st.write("Trained on the Fake-or-Real (for-norm) dataset. real = Genuine (0), fake = Deepfake (1).")
    st.write(f"Feature vector length: {N_FEATURES}")

try:
    model, scaler, THRESHOLD = load_artifacts()
    ok = True
    st.sidebar.write(f"Decision cutoff: {THRESHOLD:.4f}")
except Exception as e:
    ok = False
    st.error(f"Could not load model: {e}")

uploaded = st.file_uploader("Upload an audio clip", type=["wav", "mp3", "flac", "ogg", "m4a"])
if uploaded is not None:
    st.audio(uploaded)
    if ok and st.button("Analyze", type="primary"):
        suffix = os.path.splitext(uploaded.name)[1] or ".wav"
        with st.spinner("Analyzing …"):
            p_fake = predict_bytes(uploaded.getvalue(), suffix, model, scaler)
        if p_fake >= THRESHOLD:
            st.error("### 🤖 Deepfake (AI-Generated)"); conf = p_fake
        else:
            st.success("### 🧑 Genuine (Human)"); conf = 1 - p_fake
        st.metric("Confidence", f"{conf*100:.1f}%")
        st.caption(f"P(deepfake) = {p_fake:.4f}  ·  cutoff = {THRESHOLD:.4f}")
else:
    st.info("⬆️ Upload a short speech clip to get a verdict.")
