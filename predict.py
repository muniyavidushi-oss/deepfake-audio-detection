"""CLI: python predict.py clip.wav   (needs features.py + model + scaler.pkl + threshold.json)"""
from __future__ import annotations
import os, sys, json, argparse
import joblib, numpy as np
from tensorflow import keras
from features import extract_features_from_file, N_FEATURES

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
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]); return m

def load_all(art):
    scaler = joblib.load(os.path.join(art, "scaler.pkl"))
    threshold = 0.5
    tp = os.path.join(art, "threshold.json")
    if os.path.exists(tp):
        try: threshold = float(json.load(open(tp))["threshold"])
        except Exception: pass
    model = None
    for f in ("deepfake_model.keras", "deepfake_model.h5"):
        p = os.path.join(art, f)
        if os.path.exists(p) and model is None:
            try: model = keras.models.load_model(p)
            except Exception: model = None
    wp = os.path.join(art, "deepfake_model.weights.h5")
    if model is None and os.path.exists(wp):
        model = build_model(N_FEATURES); model.load_weights(wp)
    return model, scaler, threshold

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio"); ap.add_argument("--artifacts", default=".")
    a = ap.parse_args()
    if not os.path.isfile(a.audio): print("file not found:", a.audio); sys.exit(1)
    model, scaler, thr = load_all(a.artifacts)
    feats = extract_features_from_file(a.audio).reshape(1, -1)
    p = float(model.predict(scaler.transform(feats), verbose=0).ravel()[0])
    verdict, conf = ("Deepfake (AI-Generated)", p) if p >= thr else ("Genuine (Human)", 1 - p)
    print(f"\n{a.audio}\nVerdict: {verdict}\nConfidence: {conf*100:.2f}%  (P_fake={p:.4f}, cutoff={thr:.4f})")
