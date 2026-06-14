"""
train_pipeline.py
=================
Standalone end-to-end training pipeline for the Deepfake Audio Detector.

Usage:
    python train_pipeline.py --data_root /path/to/search --out_dir ./artifacts \
                             --max_per_class 4000

`--data_root` is a directory under which the for-norm folder lives; the
script walks it and auto-discovers training/{real,fake}. Defaults to '.'.

Outputs in --out_dir:
    deepfake_model.keras / .h5 / .weights.h5   (3 formats for portability)
    scaler.pkl, confusion_matrix.png, training_history.png, feature_cache/*.npz

NOTE: build_model is imported by predict.py and app.py for the weights-only
fallback, so the module-level code here must stay import-safe (no heavy work
outside __main__).
"""
from __future__ import annotations
import argparse
import glob
import os
import random

import joblib
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from features import extract_features_from_file, N_FEATURES

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

CLASSES = [("real", 0), ("fake", 1)]           # real=Genuine=0, fake=Deepfake=1
AUDIO_EXTS = ("*.wav", "*.WAV", "*.flac", "*.mp3")


# --------------------------- discovery ---------------------------
def _has_real_fake(d: str) -> bool:
    return os.path.isdir(os.path.join(d, "real")) and os.path.isdir(os.path.join(d, "fake"))


def find_dataset_root(search_root: str) -> str:
    norm_match = any_match = None
    for dp, _, _ in os.walk(search_root):
        train = os.path.join(dp, "training")
        if os.path.isdir(train) and _has_real_fake(train):
            if "norm" in os.path.basename(dp).lower() and norm_match is None:
                norm_match = dp
            if any_match is None:
                any_match = dp
    chosen = norm_match or any_match
    if chosen is None:
        raise FileNotFoundError(
            f"No folder with training/real and training/fake under: {search_root}")
    return chosen


def pick_eval_split(root: str) -> str:
    for name in ("testing", "validation"):
        cand = os.path.join(root, name)
        if os.path.isdir(cand) and _has_real_fake(cand):
            return cand
    raise FileNotFoundError(f"No 'testing'/'validation' split with real/fake under {root}")


# --------------------------- features ---------------------------
def _list_files(class_dir: str):
    files = []
    for pat in AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(class_dir, pat)))
    return sorted(files)


def build_split(split_dir: str, max_per_class: int, cache_path: str):
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        print(f"  loaded cache {cache_path}  X={d['X'].shape}")
        return d["X"], d["y"]
    X, y = [], []
    for cls, label in CLASSES:
        files = _list_files(os.path.join(split_dir, cls))
        if max_per_class:
            files = files[:max_per_class]
        for f in tqdm(files, desc=f"  {os.path.basename(split_dir)}/{cls}"):
            try:
                X.append(extract_features_from_file(f)); y.append(label)
            except Exception as e:
                print(f"    skip {f}: {e}")
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.int64)
    assert X.ndim == 2 and X.shape[1] == N_FEATURES, f"got {X.shape}"
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y)
    print(f"  cached {cache_path}  X={X.shape}")
    return X, y


# --------------------------- model ---------------------------
def build_model(input_dim: int) -> keras.Model:
    """⚠️ Keep IDENTICAL to the build_model cell in notebook.ipynb."""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, name="dense_1"),
        keras.layers.BatchNormalization(name="bn_1"),
        keras.layers.Activation("relu", name="act_1"),
        keras.layers.Dropout(0.4, name="drop_1"),
        keras.layers.Dense(128, name="dense_2"),
        keras.layers.BatchNormalization(name="bn_2"),
        keras.layers.Activation("relu", name="act_2"),
        keras.layers.Dropout(0.3, name="drop_2"),
        keras.layers.Dense(64, name="dense_3"),
        keras.layers.BatchNormalization(name="bn_3"),
        keras.layers.Activation("relu", name="act_3"),
        keras.layers.Dropout(0.2, name="drop_3"),
        keras.layers.Dense(1, activation="sigmoid", name="out"),
    ], name="deepfake_mlp")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


# --------------------------- metrics ---------------------------
def compute_eer(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0), float(thr[i])


def evaluate(model, scaler, X_eval, y_eval, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    proba = model.predict(scaler.transform(X_eval), verbose=0).ravel()
    y_pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred)
    eer, eer_thr = compute_eer(y_eval, proba)
    cm = confusion_matrix(y_eval, y_pred)
    real_acc, fake_acc = (cm.diagonal() / cm.sum(axis=1))

    print("\n================  EVALUATION  ================")
    print(classification_report(y_eval, y_pred,
          target_names=["real (Genuine)", "fake (Deepfake)"], digits=4))
    print(f"Accuracy           : {acc*100:.2f}%")
    print(f"F1 score           : {f1*100:.2f}%")
    print(f"EER                : {eer*100:.2f}%  (thr={eer_thr:.3f})")
    print(f"Per-class accuracy : real={real_acc*100:.2f}%  fake={fake_acc*100:.2f}%")
    print(f"Confusion matrix   :\n{cm}")

    checks = {
        "Accuracy >= 80%": acc >= 0.80, "EER <= 12%": eer <= 0.12, "F1 >= 80%": f1 >= 0.80,
        "real acc >= 75%": real_acc >= 0.75, "fake acc >= 75%": fake_acc >= 0.75,
    }
    print("\n----------------  PASS / FAIL  ----------------")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}]  {k}")
    print(f"\nOVERALL: {'ALL PASSED' if all(checks.values()) else 'SOME FAILED'}")

    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["real", "fake"], yticklabels=["real", "fake"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    return dict(acc=acc, f1=f1, eer=eer, real_acc=real_acc, fake_acc=fake_acc)


def plot_history(history, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h["accuracy"], label="train"); ax[1].plot(h["val_accuracy"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "training_history.png"), dpi=150)
    plt.close()


# --------------------------- orchestration ---------------------------
def run(data_root, out_dir, max_per_class, epochs, batch_size):
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "feature_cache"); os.makedirs(cache_dir, exist_ok=True)

    root = find_dataset_root(data_root)
    train_dir, eval_dir = os.path.join(root, "training"), pick_eval_split(root)
    print(f"Dataset root : {root}\nTrain split  : {train_dir}\nEval split   : {eval_dir}")

    print("\nExtracting TRAIN features ...")
    X_full, y_full = build_split(train_dir, max_per_class,
                                 os.path.join(cache_dir, f"train_{max_per_class}.npz"))
    print("\nExtracting EVAL features ...")
    X_eval, y_eval = build_split(eval_dir, max_per_class,
                                 os.path.join(cache_dir, f"eval_{max_per_class}.npz"))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=0.15, stratify=y_full, random_state=SEED)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_val_s = scaler.transform(X_tr), scaler.transform(X_val)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))

    classes = np.unique(y_tr)
    cw = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print(f"Class weights: {class_weight}")

    model = build_model(X_tr_s.shape[1]); model.summary()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                                          min_lr=1e-6, verbose=1),
    ]
    history = model.fit(X_tr_s, y_tr, validation_data=(X_val_s, y_val),
                        epochs=epochs, batch_size=batch_size,
                        class_weight=class_weight, callbacks=callbacks, verbose=2)

    model.save(os.path.join(out_dir, "deepfake_model.keras"))
    try:
        model.save(os.path.join(out_dir, "deepfake_model.h5"))
    except Exception as e:
        print(f"(.h5 save skipped: {e})")
    model.save_weights(os.path.join(out_dir, "deepfake_model.weights.h5"))

    plot_history(history, out_dir)
    evaluate(model, scaler, X_eval, y_eval, out_dir)
    print(f"\nArtifacts written to: {os.path.abspath(out_dir)}")


def parse_args():
    p = argparse.ArgumentParser(description="Train the deepfake audio detector")
    p.add_argument("--data_root", default=".", help="dir to search for the for-norm folder")
    p.add_argument("--out_dir", default="./artifacts")
    p.add_argument("--max_per_class", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.data_root, a.out_dir, a.max_per_class, a.epochs, a.batch_size)
