# realtime_mi_ssvep_gate.py

import json
import pickle
import numpy as np
import time

from scipy.signal import welch
from pylsl import StreamInlet, resolve_byprop

# =========================
# CONFIG
# =========================

FS = 256
WINDOW = 512
UPDATE = 64

MODEL_DIR = "gate_model"

# =========================
# LOAD MODEL
# =========================

with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(f"{MODEL_DIR}/lda.pkl", "rb") as f:
    clf = pickle.load(f)

with open(f"{MODEL_DIR}/model_info.json", "r") as f:
    info = json.load(f)

# =========================
# FEATURE FUNCTION
# =========================

def extract_psd_features(epoch):

    feats = []

    for ch in range(epoch.shape[0]):

        freqs, psd = welch(epoch[ch], fs=FS, nperseg=256)

        bands = info["bands"]

        for low, high in bands:
            idx = (freqs >= low) & (freqs < high)
            feats.append(np.mean(psd[idx]))

    return np.array(feats)

# =========================
# EEG STREAM
# =========================

print("Waiting for EEG stream...")

streams = resolve_byprop("name", "EmotivEEG", timeout=10)
inlet = StreamInlet(streams[0])

buffer = np.zeros((14, 0))

print("Connected.")

# =========================
# REAL-TIME LOOP
# =========================

while True:

    sample, _ = inlet.pull_sample()
    sample = np.array(sample).reshape(14, 1)

    buffer = np.concatenate([buffer, sample], axis=1)

    if buffer.shape[1] > WINDOW:
        buffer = buffer[:, -WINDOW:]

    if buffer.shape[1] < WINDOW:
        continue

    feat = extract_psd_features(buffer)
    feat = scaler.transform([feat])

    score = clf.decision_function(feat)[0]

    prob = clf.predict_proba(feat)[0]

    pred = clf.predict(feat)[0]

    print(
        f"SCORE={score:.4f} "
        f"P(MI)={prob[0]:.4f} "
        f"P(SSVEP)={prob[1]:.4f}"
    )

    if pred == 0:
        state = "MI"
    else:
        state = "SSVEP"

    print(
        f"STATE: {state} | "
        f"P(MI)={prob[0]:.3f} | "
        f"P(SSVEP)={prob[1]:.3f}"
    )

    time.sleep(0.05)